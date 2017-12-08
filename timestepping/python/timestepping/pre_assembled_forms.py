#!/usr/bin/env python3

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2017 University of Edinburgh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import OrderedDict
import copy

import dolfin
import ufl

from .caches import *
from .exceptions import *
from .fenics_overrides import *
from .fenics_utils import *
from .statics import *
from .versions import *
from . import fenics_utils
import six

__all__ = \
  [
    "NonPAFilter",
    "PAFilter",
    "PAForm",
    "PAMatrixFilter",
    "PAStaticFilter"
  ]

def preprocess_integral(integral):
    """
    Given an Integral associated with the given Form, return the integrand and
    a list of arguments which can be used to construct an Integral from the
    integrand.
    """

    integrand = integral.integrand()
    integral_type, domain, subdomain_id, metadata, subdomain_data = \
      integral.integral_type(), integral.domain(), integral.subdomain_id(), integral.metadata(), integral.subdomain_data()
    return integrand, [integral_type, domain, subdomain_id, metadata, subdomain_data]

def matrix_optimisation(form):
    """
    Attempt to convert a linear form into the action of a bi-linear form.
    Return a (bi-linear Form, Function) pair on success, and None on failure.
    """

    if not isinstance(form, ufl.form.Form):
        raise InvalidArgumentException("form must be a Form")

    # Find the test function
    args = ufl.algorithms.extract_arguments(form)
    if not len(args) == 1:
        # This is not a linear form
        return None

    # Look for a single non-static Function dependency
    tcs = extract_non_static_coefficients(form)
    if not len(tcs) == 1:
        # Found too many non-static coefficients
        return None
    elif not isinstance(tcs[0], dolfin.Function):
        # The only non-static coefficient is not a Function
        return None
    # Found a single non-static Function dependency
    fn = tcs[0]

    # Look for a static bi-linear form whose action can be used to construct
    # the linear form
    mat_form = derivative(form, fn,
      # Hack to work around an obscure FEniCS bug
      expand = dolfin.MPI.size(dolfin.mpi_comm_world()) == 1 or
        (not is_r0_function_space(args[0].function_space()) and not is_r0_function(fn)))
    if n_non_static_coefficients(mat_form) > 0:
        # The form is non-linear
        return None
    try:
        rhs_form = dolfin.rhs(dolfin.replace(form, {fn:dolfin.TrialFunction(fn.function_space())}))
    except ufl.log.UFLException:
        # The form might be inhomogeneous
        return None
    if not is_empty_form(rhs_form):
        # The form might be inhomogeneous
        return None

    # Success
    return mat_form, fn

class PAFilter(object):
    """
    A pre-assembly filter. This processes input forms for pre-assembly.

    Constructor arguments:
      quadrature_degree: Quadrature degree used for form assembly.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, quadrature_degree, pre_assembly_parameters):
        if not isinstance(quadrature_degree, int) or quadrature_degree < 0:
            raise InvalidArgumentException("quadrature_degree must be a non-negative integer")
        if not isinstance(pre_assembly_parameters, dolfin.Parameters):
            raise InvalidArgumentException("default_pre_assembly_parameters must be a Parameters")

                                                                # No real need to copy here
        self.pre_assembly_parameters = pre_assembly_parameters#.copy()
        self._quadrature_degree = quadrature_degree
        self._form = ufl.form.Form([])
        self._n = 0
        self._rank = None

        return

    def name(self):
        """
        Return the name of the PAFilter type.
        """

        return self.__class__.__name__

    def _add(self, form):
        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")
        elif is_empty_form(form):
            return

        if self._n == 0:
            self._form = form
            self._n += 1
            self._rank = form_rank(form)
        else:
            rank = form_rank(form)
            if not rank == self._rank:
                raise InvalidArgumentException("Unexpected form rank: %i" % rank)
            self._form += form
            self._n += 1

        return

    def _cache_assemble(self, form):
        return assembly_cache.assemble(form,
          form_compiler_parameters = {"quadrature_degree":self._quadrature_degree})

    def _assemble(self, form, tensor = None):
        return assemble(form, tensor = tensor,
          form_compiler_parameters = {"quadrature_degree":self._quadrature_degree})

    def add(self, form):
        """
        Attempt to add a Form to the PAFilter. Return a (form, n_added) pair,
        where form is the remaining part of the input Form *not* added to the
        PAFilter, and n_added indicates the number of terms which *are* added to the
        PAFilter.
        """

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")

        return form, 0

    def n(self):
        """
        Return the number of terms added to the PAFilter.
        """

        return self._n

    def rank(self):
        """
        Return the rank of the Form associated with the PAFilter.
        """

        return self._rank

    def pre_assemble(self):
        """
        Pre-assemble the PAFilter.
        """

        return

    def match_tensor(self, tensor = None):
        """
        Addition of GenericMatrix s can be more efficient if the sparsity patterns
        match. Extend the sparsity pattern of the input GenericMatrix and any
        relevant internally stored GenericMatrix so that their sparsity patterns
        match. Return the expanded GenericMatrix if tensor is not None. Otherwise
        return a copy of an internal GenericMatrix if one exists, or None otherwise.
        Must be called after the pre_assemble method.

        Arguments:
          tensor: The GenericMatrix whose sparsity pattern is to match any relevant
                  internally stored GenericMatrix s.
        """

        return tensor

    def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
        """
        Return the result of assembling the Form associated with the PAFilter.

        Arguments:
          tensor: If not None, add the result of assembing the form to tensor.
          same_nonzero_pattern: If tensor is a GenericMatrix, whether tensor has the
                                same sparsity pattern as the GenericMatrix which
                                will be added to it.
          copy: Whether the result should be a copy. If copy is False then the
                result should not be modified.
        """

        raise StateException("Cannot assemble empty Form")

class PAStaticFilter(PAFilter):
    """
    A pre-assembly filter which processes static terms.

    Constructor arguments:
      quadrature_degree: Quadrature degree used for form assembly.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, quadrature_degree, pre_assembly_parameters):
        PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
        self.__pre_assembled = None

        return

    def add(self, form):
        """
        Attempt to add a Form to the PAFilter. Return a (form, n_added) pair,
        where form is the remaining part of the input Form *not* added to the
        PAFilter, and n_added indicates the number of terms which *are* added to the
        PAFilter.
        """

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")
        elif is_empty_form(form):
            return form, 0

        if is_static_form(form):
            self._add(form)
            self.__pre_assembled = None
            return ufl.form.Form([]), 1
        else:
            return form, 0

    def pre_assemble(self):
        """
        Pre-assemble the PAFilter.
        """

        if self._n == 0:
            return

        self.__pre_assembled = self._cache_assemble(self._form)

        return

    def match_tensor(self, tensor = None):
        """
        Addition of GenericMatrix s can be more efficient if the sparsity patterns
        match. Extend the sparsity pattern of the input GenericMatrix and any
        relevant internally stored GenericMatrix so that their sparsity patterns
        match. Return the expanded GenericMatrix if tensor is not None. Otherwise
        return a copy of an internal GenericMatrix if one exists, or None otherwise.
        Must be called after the pre_assemble method.

        Arguments:
          tensor: The GenericMatrix whose sparsity pattern is to match any relevant
                  internally stored GenericMatrix s.
        """

        if self.__pre_assembled is None:
            if self._n == 0:
                return tensor
            else:
                raise StateException("Cannot call match_tensor method when not pre-assembled")
        elif not self._rank == 2:
            raise StateException("Unexpected form rank: %i" % self._rank)
        if not tensor is None and not isinstance(tensor, dolfin.GenericMatrix):
            raise InvalidArgumentException("tensor must be a GenericMatrix")

        if tensor is None:
            return self.__pre_assembled.copy()
        else:
            assert(isinstance(self.__pre_assembled, dolfin.GenericMatrix))
            self.__pre_assembled.axpy(0.0, tensor, False)
            tensor.axpy(0.0, self.__pre_assembled, False)
            return tensor

    def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
        """
        Return the result of assembling the Form associated with the PAFilter.

        Arguments:
          tensor: If not None, add the result of assembing the form to tensor.
          same_nonzero_pattern: If tensor is a GenericMatrix, whether tensor has the
                                same sparsity pattern as the GenericMatrix which
                                will be added to it.
          copy: Whether the result should be a copy of internal data. If copy is
                False and tensor is None then the result should not be modified.
        """

        if self.__pre_assembled is None:
            if self._n == 0:
                raise StateException("Cannot assemble empty Form")
            else:
                raise StateException("Cannot call assemble method when not pre-assembled")

        L = self.__pre_assembled
        if tensor is None:
            if copy:
                L = L.copy()
            return L
        else:
            if isinstance(L, dolfin.GenericMatrix) and same_nonzero_pattern:
                tensor.axpy(1.0, L, True)
            else:
                tensor += L
            return tensor

class PAMatrixFilter(PAFilter):
    """
    A pre-assembly filter which processes terms which can be converted to a
    static bi-linear form action.

    Constructor arguments:
      quadrature_degree: Quadrature degree used for form assembly.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, quadrature_degree, pre_assembly_parameters):
        PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
        self.__L = OrderedDict()
        self.__pre_assembled = None

        return

    def add(self, form):
        """
        Attempt to add a Form to the PAFilter. Return a (form, n_added) pair,
        where form is the remaining part of the input Form *not* added to the
        PAFilter, and n_added indicates the number of terms which *are* added to the
        PAFilter.
        """

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")
        elif is_empty_form(form):
            return form, 0

        mform = matrix_optimisation(form)
        if mform is None:
            return form, 0
        mat_form, fn = mform

        self._add(form)
        if fn in self.__L:
            self.__L[fn] += mat_form
        else:
            self.__L[fn] = mat_form
        self.__pre_assembled = None
        return ufl.form.Form([]), 1

    def pre_assemble(self):
        """
        Pre-assemble the PAFilter.
        """

        if self._n == 0:
            return

        self.__pre_assembled = OrderedDict()
        for fn in self.__L:
            self.__pre_assembled[fn] = self._cache_assemble(self.__L[fn])

        return

    def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
        """
        Return the result of assembling the Form associated with the PAFilter.

        Arguments:
          tensor: If not None, add the result of assembing the form to tensor.
          same_nonzero_pattern: If tensor is a GenericMatrix, whether tensor has the
                                same sparsity pattern as the GenericMatrix which
                                will be added to it.
          copy: Whether the result should be a copy of internal data. If copy is
                False and tensor is None then the result should not be modified.
        """

        if self.__pre_assembled is None:
            if self._n == 0:
                raise StateException("Cannot assemble empty Form")
            else:
                raise StateException("Cannot call assemble method when not pre-assembled")

        if tensor is None:
            fn, mat = list(self.__pre_assembled.items())[0]
            L = mat * fn.vector()
            for fn, mat in list(self.__pre_assembled.items())[1:]:
                L += mat * fn.vector()
            return L
        else:
            for fn, mat in six.iteritems(self.__pre_assembled):
                tensor += mat * fn.vector()
            return tensor

class NonPAFilter(PAFilter):
    """
    A dummy pre-assembly filter which accepts any Form, and uses standard
    assembly.

    Constructor arguments:
      quadrature_degree: Quadrature degree used for form assembly.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, quadrature_degree, pre_assembly_parameters):
        PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
        self.__tensor = None

        return

    def add(self, form):
        """
        Attempt to add a Form to the PAFilter. Return a (form, n_added) pair,
        where form is the remaining part of the input Form *not* added to the
        PAFilter, and n_added indicates the number of terms which *are* added to the
        PAFilter.
        """

        if not isinstance(form, ufl.form.Form):
            raise InvalidArgumentException("form must be a Form")
        elif is_empty_form(form):
            return form, 0

        self._add(form)
        self.__tensor = None
        return ufl.form.Form([]), 1

    def match_tensor(self, tensor = None):
        """
        Addition of GenericMatrix s can be more efficient if the sparsity patterns
        match. Extend the sparsity pattern of the input GenericMatrix and any
        relevant internally stored GenericMatrix so that their sparsity patterns
        match. Return the expanded GenericMatrix if tensor is not None. Otherwise
        return a copy of an internal GenericMatrix if one exists, or None otherwise.
        Must be called after the pre_assemble method.

        Arguments:
          tensor: The GenericMatrix whose sparsity pattern is to match any relevant
                  internally stored GenericMatrix s.
        """

        if self._n == 0:
            return tensor
        elif not self._rank == 2:
            raise StateException("Unexpected form rank: %i" % self._rank)
        if not tensor is None and not isinstance(tensor, dolfin.GenericMatrix):
            raise InvalidArgumentException("tensor must be a GenericMatrix")

        one = dolfin.Constant(1.0)
        form = dolfin.replace(self._form, {c:one for c in ufl.algorithms.extract_coefficients(self._form)})
        if self._rank == 1:
            tensor = self._assemble(form)
        elif self._rank == 2:
            if self.__tensor is None:
                self.__tensor = self._assemble(form)
            if tensor is None:
                tensor = self.__tensor.copy()
            else:
                self.__tensor.axpy(0.0, tensor, False)
                tensor.axpy(0.0, self.__tensor, False)
        else:
            raise StateException("Unexpected form rank: %i" % self._rank)

        return tensor

    def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
        """
        Return the result of assembling the Form associated with the PAFilter.

        Arguments:
          tensor: If not None, add the result of assembing the form to tensor.
          same_nonzero_pattern: If tensor is a GenericMatrix, whether tensor has the
                                same sparsity pattern as the GenericMatrix which
                                will be added to it.
          copy: Whether the result should be a copy of internal data. If copy is
                False and tensor is None then the result should not be modified.
        """

        if self._n == 0:
            raise StateException("Cannot assemble empty Form")

        if self._rank == 2:
            if self.__tensor is None:
                self.__tensor = L = self._assemble(self._form)
            else:
                L = self._assemble(self._form, tensor = self.__tensor)
            if tensor is None:
                if copy:
                    L = L.copy()
                return L
            else:
                if same_nonzero_pattern:
                    tensor.axpy(1.0, L, True)
                else:
                    tensor += L
                return tensor
        else:
            if tensor is None:
                return self._assemble(self._form)
            else:
                tensor += self._assemble(self._form)
                return tensor

class PAForm(object):
    """
    A pre-assembled form. Given a form of arbitrary rank, this finds and
    pre-assembles static terms.

    Constructor arguments:
      form: The Form to be pre-assembled.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, form, pre_assembly_parameters = {}):
        if not isinstance(form, ufl.form.Form) or is_empty_form(form):
            raise InvalidArgumentException("form must be a non-empty Form")

        rank = form_rank(form)
        if rank == 0:
            default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["integrals"]
        elif rank == 1:
            default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"]
        elif rank == 2:
            default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"]
        else:
            raise InvalidArgumentException("Unexpected form rank: %i" % rank)
        npre_assembly_parameters = default_pre_assembly_parameters.copy()
        npre_assembly_parameters.update(pre_assembly_parameters)
        pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)

        pa_filters = [PAStaticFilter]
        if rank == 1 and pre_assembly_parameters["matrix_optimisation"]:
            pa_filters.append(PAMatrixFilter)

        self.__form = form
        self.__rank = rank
        self.__quadrature_degree = form_quadrature_degree(form)
        self.__deps = set(ufl.algorithms.extract_coefficients(form))
        self.__is_static = is_static_form(form)
        self.pre_assembly_parameters = pre_assembly_parameters
        self.__set_optimise(form, pa_filters)

        return

    def __set_optimise(self, form, pa_filters):
        pa_filters = [filter(self.__quadrature_degree, self.pre_assembly_parameters) for filter in pa_filters]
        non_pa_filter = NonPAFilter(self.__quadrature_degree, self.pre_assembly_parameters)

        for filter in pa_filters:
            form, n_added = filter.add(form)

        if not is_empty_form(form):
            if self.pre_assembly_parameters["expand_form"]:
                expand_expr = fenics_utils.expand_expr
            else:
                def expand_expr(e):
                    if isinstance(e, ufl.algebra.Sum):
                        return e.operands()
                    else:
                        return e

            if self.pre_assembly_parameters["term_optimisation"]:
                terms = []
                for integral in form.integrals():
                    integrand, iargs = preprocess_integral(integral)
                    def integrand_to_form(expr):
                        return ufl.form.Form([ufl.Integral(expr, *iargs)])
                    terms.append((integrand_to_form(integrand), [integrand_to_form(e) for e in expand_expr(integrand)]))
            else:
                terms = [(form, [form])]
            for term in terms:
                term_n_pa = 0
                non_pa_terms = []
                for sterm in term[1]:
                    for filter in pa_filters:
                        sterm, n_added = filter.add(sterm)
                        term_n_pa += n_added
                    non_pa_terms.append(sterm)
                if term_n_pa == 0:
                    non_pa_filter.add(term[0])
                else:
                    for sterm in non_pa_terms:
                        non_pa_filter.add(sterm)

        n = {}
        n_pa = 0
        filters = []
        for filter in pa_filters:
            n_filter = filter.n()
            n[filter.name()] = n_filter
            n_pa += n_filter
            if n_filter > 0:
                filter.pre_assemble()
                filters.append(filter)
        n_non_pa = non_pa_filter.n()
        if n_non_pa > 0:
            non_pa_filter.pre_assemble()
            filters.insert(0, non_pa_filter)

        if self.__rank == 2 and len(filters) > 1:
            tensor = filters[0].match_tensor()
            for filter in filters[1:] + filters[:-1]:
                tensor = filter.match_tensor(tensor = tensor)
            same_nonzero_pattern = True
        else:
            same_nonzero_pattern = False

        self.__filters = filters
        self.__n = n
        self.__n_pre_assembled = n_pa
        self.__n_non_pre_assembled = n_non_pa
        self.__same_nonzero_pattern = same_nonzero_pattern

        return

    def assemble(self, copy = False):
        """
        Return the result of assembling the Form associated with the PAForm.

        Arguments:
          copy: Whether the result should be a copy of internal data. If copy is
                False then the result should not be modified.
        """

        if len(self.__filters) == 0:
            raise StateException("Cannot assemble empty form")
        elif len(self.__filters) == 1:
            return self.__filters[0].assemble(copy = copy)
        else:
            tensor = self.__filters[0].assemble(copy = True)
            for filter in self.__filters[1:]:
                tensor = filter.assemble(tensor = tensor, same_nonzero_pattern = self.__same_nonzero_pattern, copy = False)
            return tensor

    def rank(self):
        """
        Return the PAForm rank.
        """

        return self.__rank

    def is_static(self):
        """
        Return whether the PAForm is static.
        """

        return self.__is_static

    def n(self):
        """
        Return a dictionary of name:n_added pairs, indicating the number of terms
        added to each considered PAFilter.
        """

        return self.__n

    def n_pre_assembled(self):
        """
        Return the number of pre-assembled terms.
        """

        return self.__n_pre_assembled

    def n_non_pre_assembled(self):
        """
        Return the number of non-pre-assembled terms.
        """

        return self.__n_non_pre_assembled

    def dependencies(self, non_symbolic = False):
        """
        Return PAForm dependencies. The optional non_symbolic has no effect.
        """

        return self.__deps

_assemble_classes.append(PAForm)
