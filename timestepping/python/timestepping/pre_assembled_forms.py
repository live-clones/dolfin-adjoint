#!/usr/bin/env python2

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014 University of Edinburgh
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

from caches import *
from exceptions import *
from fenics_overrides import *
from fenics_utils import *
from statics import *
from versions import *

__all__ = \
  [
    "PABilinearForm",
    "PAForm",
    "PALinearForm"
  ]

if dolfin_version() < (1, 4, 0):
    def preprocess_integral(integral):
        """
        Given an Integral associated with the given Form, return the integrand and
        a list of arguments which can be used to construct an Integral from the
        integrand.
        """

        integrand = integral.integrand()
        domain_type, domain_description, compiler_data, domain_data = \
          integral.domain_type(), integral.domain_description(), integral.compiler_data(), integral.domain_data()
        return integrand, [domain_type, domain_description, compiler_data, domain_data]

    def _assemble_tensor(form, tensor):
        return assemble(form, tensor = tensor, reset_sparsity = False)
else:
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

    def _assemble_tensor(form, tensor):
        return assemble(form, tensor = tensor)

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
      expand = dolfin.MPI.num_processes() == 1 or
        (not is_r0_function_space(args[0].function_space()) and not is_r0_function(fn)))
    if n_non_static_coefficients(mat_form) > 0:
        # The form is non-linear
        return None
    try:
        rhs_form = rhs(replace(form, {fn:dolfin.TrialFunction(fn.function_space())}))
    except ufl.log.UFLException:
        # The form might be inhomogeneous
        return None
    if not is_empty_form(rhs_form):
        # The form might be inhomogeneous
        return None

    # Success
    return mat_form, fn

class PAForm(object):
    """
    A pre-assembled form. Given a form of arbitrary rank, this finds and
    pre-assembles static terms.

    Constructor arguments:
      form: The Form to be pre-assembled.
      pre_assembly_parameters: Parameters defining detailed optimisation options.
    """

    def __init__(self, form, pre_assembly_parameters = {},
      default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["forms"]):
        if not isinstance(default_pre_assembly_parameters, dolfin.Parameters):
            raise InvalidArgumentException("default_pre_assembly_parameters must be a Parameters")

        npre_assembly_parameters = default_pre_assembly_parameters.copy()
        npre_assembly_parameters.update(pre_assembly_parameters)
        pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)

        self.pre_assembly_parameters = pre_assembly_parameters
        self.__set(form)

        return

    def __set(self, form):
        if not isinstance(form, ufl.form.Form) or is_empty_form(form):
            raise InvalidArgumentException("form must be a non-empty Form")

        self._set_optimise(form)

        self.__rank = form_rank(form)
        self.__deps = ufl.algorithms.extract_coefficients(form)

        return

    def _set_optimise(self, form):
        if is_static_form(form):
            self._set_pa([form], [])
        elif self.pre_assembly_parameters["term_optimisation"]:
            quadrature_degree = form_quadrature_degree(form)

            pre_assembled_L = []
            non_pre_assembled_L = []

            for integral in form.integrals():
                integrand, iargs = preprocess_integral(integral)
                if self.pre_assembly_parameters["expand_form"]:
                    if isinstance(integrand, ufl.algebra.Sum):
                        terms = [(term, expand_expr(term)) for term in integrand.operands()]
                    else:
                        terms = [(integrand, expand_expr(integrand))]
                else:
                    if isinstance(integrand, ufl.algebra.Sum):
                        terms = [(term, [term]) for term in integrand.operands()]
                    else:
                        terms = [(integrand, [integrand])]
                for term in terms:
                    pterm = [], []
                    for sterm in term[1]:
                        stform = QForm([ufl.Integral(sterm, *iargs)], quadrature_degree = quadrature_degree)
                        if is_static_form(stform):
                            pterm[0].append(stform)
                        else:
                            pterm[1].append(stform)
                    if len(pterm[0]) == 0:
                        tform = QForm([ufl.Integral(term[0], *iargs)], quadrature_degree = quadrature_degree)
                        non_pre_assembled_L.append(tform)
                    else:
                        pre_assembled_L += pterm[0]
                        non_pre_assembled_L += pterm[1]

            self._set_pa(pre_assembled_L, non_pre_assembled_L)
        else:
            self._set_pa([], [form])

        return

    def _set_pa(self, pre_assembled_L, non_pre_assembled_L):
        if len(pre_assembled_L) == 0:
            self._pre_assembled_L = None
        else:
            l_L = pre_assembled_L[0]
            for L in pre_assembled_L[1:]:
                l_L += L
            self._pre_assembled_L = assembly_cache.assemble(l_L, compress = self.pre_assembly_parameters["compress_matrices"])

        if len(non_pre_assembled_L) == 0:
            self._non_pre_assembled_L = None
        else:
            self._non_pre_assembled_L = non_pre_assembled_L[0]
            for L in non_pre_assembled_L[1:]:
                self._non_pre_assembled_L += L

        self._n_pre_assembled = len(pre_assembled_L)
        self._n_non_pre_assembled = len(non_pre_assembled_L)

        return

    def assemble(self, copy = False):
        """
        Return the result of assembling the Form associated with the PAForm. If
        copy is False then existing data may be returned -- it is expected in this
        case that the return value will never be modified.
        """

        if self._non_pre_assembled_L is None:
            if self._pre_assembled_L is None:
                raise StateException("Cannot assemble empty form")
            else:
                if copy:
                    L = self._pre_assembled_L.copy()
                else:
                    L = self._pre_assembled_L
        else:
            L = assemble(self._non_pre_assembled_L)
            if not self._pre_assembled_L is None:
                L += self._pre_assembled_L

        return L

    def rank(self):
        """
        Return the Form rank.
        """

        return self.__rank

    def is_static(self):
        """
        Return whether the PAForm is static.
        """

        return self._n_non_pre_assembled == 0

    def n_pre_assembled(self):
        """
        Return the number of pre-assembled terms.
        """

        return self._n_pre_assembled

    def n_non_pre_assembled(self):
        """
        Return the number of non-pre-assembled terms.
        """

        return self._n_non_pre_assembled

    def n(self):
        """
        Return the total number of terms.
        """

        return self.n_pre_assembled() + self.n_non_pre_assembled()

    def dependencies(self, non_symbolic = False):
        """
        Return Form dependencies. The optional non_symbolic has no effect.
        """

        return self.__deps

_assemble_classes.append(PAForm)

class PABilinearForm(PAForm):
    """
    A pre-assembled bi-linear form. This is similar to PAForm, but applies
    additional optimisations specific to bi-linear forms. Also has different
    default parameters.
    """

    def __init__(self, form, pre_assembly_parameters = {}):
        if not form_rank(form) == 2:
            raise InvalidArgumentException("form must be a rank 2 form")

        PAForm.__init__(self, form,
          pre_assembly_parameters = pre_assembly_parameters,
          default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"])

        if not self._non_pre_assembled_L is None and not self._pre_assembled_L is None:
            # Work around a (rare) reproducibility issue:
            # _assemble_tensor(form, tensor = tensor) can give (very slightly)
            # different results if given matrices with different sparsity patterns.

            # The coefficients may contain invalid data, or be non-wrapping
            # WrappedFunction s. Replace the coefficients with Constant(1.0).
            form = self._non_pre_assembled_L
            one = dolfin.Constant(1.0)
            repl = {c:one for c in ufl.algorithms.extract_coefficients(form)}
            form = replace(form, repl)

            # Set up the matrices
            L = self.__non_pre_assembled_L_tensor = assemble(form)
            L.axpy(1.0, self._pre_assembled_L, False)
            self._pre_assembled_L = self._pre_assembled_L.copy()
            self._pre_assembled_L.axpy(0.0, L, False)

        return

    def assemble(self, copy = False):
        """
        Return the result of assembling the Form associated with the PABilinearForm.
        If copy is False then existing data may be returned -- it is expected in
        this case that the return value will never be modified.
        """

        if self._non_pre_assembled_L is None:
            if self._pre_assembled_L is None:
                raise StateException("Cannot assemble empty form")
            else:
                if copy:
                    L = self._pre_assembled_L.copy()
                else:
                    L = self._pre_assembled_L
        else:
            if hasattr(self, "_PABilinearForm__non_pre_assembled_L_tensor"):
                L = self.__non_pre_assembled_L_tensor
                _assemble_tensor(self._non_pre_assembled_L, tensor = L)
                if not self._pre_assembled_L is None:
                    # The pre-assembled matrix and the non-pre-assembled matrix have
                    # previously been configured so as to have the same sparsity pattern
                    # (below)
                    L.axpy(1.0, self._pre_assembled_L, True)
                # A copy is not really required here
                #if copy:
                #  L = L.copy()
            else:
                L = self.__non_pre_assembled_L_tensor = assemble(self._non_pre_assembled_L)
                assert(self._pre_assembled_L is None)

        return L

class PALinearForm(PAForm):
    """
    A pre-assembled linear form. This is similar to PAForm, but applies additional
    optimisations specific to linear forms. Also has different default parameters.
    """

    def __init__(self, form, pre_assembly_parameters = {}):
        if not form_rank(form) == 1:
            raise InvalidArgumentException("form must be a rank 1 form")

        PAForm.__init__(self, form,
          pre_assembly_parameters = pre_assembly_parameters,
          default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"])

        return

    def _set_optimise(self, form):
        pre_assembled_L = []
        mult_assembled_L = OrderedDict()
        non_pre_assembled_L = []

        if is_static_form(form):
            pre_assembled_L.append(form)
        elif self.pre_assembly_parameters["term_optimisation"]:
            quadrature_degree = form_quadrature_degree(form)

            for integral in form.integrals():
                integrand, iargs = preprocess_integral(integral)
                if self.pre_assembly_parameters["expand_form"]:
                    if isinstance(integrand, ufl.algebra.Sum):
                        terms = [(term, expand_expr(term)) for term in integrand.operands()]
                    else:
                        terms = [(integrand, expand_expr(integrand))]
                else:
                    if isinstance(integrand, ufl.algebra.Sum):
                        terms = [(term, [term]) for term in integrand.operands()]
                    else:
                        terms = [(integrand, [integrand])]
                for term in terms:
                    pterm = [], [], []
                    for sterm in term[1]:
                        stform = QForm([ufl.Integral(sterm, *iargs)], quadrature_degree = quadrature_degree)
                        if is_static_form(stform):
                            pterm[0].append(stform)
                        elif self.pre_assembly_parameters["matrix_optimisation"]:
                            mform = matrix_optimisation(stform)
                            if mform is None:
                                pterm[2].append(stform)
                            else:
                                pterm[1].append(mform)
                        else:
                            pterm[2].append(stform)
                    if len(pterm[0]) == 0 and len(pterm[1]) == 0:
                        tform = QForm([ufl.Integral(term[0], *iargs)], quadrature_degree = quadrature_degree)
                        non_pre_assembled_L.append(tform)
                    else:
                        pre_assembled_L += pterm[0]
                        for mform in pterm[1]:
                            if mform[1] in mult_assembled_L:
                                mult_assembled_L[mform[1]].append(mform[0])
                            else:
                                mult_assembled_L[mform[1]] = [mform[0]]
                        non_pre_assembled_L += pterm[2]
        elif self.pre_assembly_parameters["matrix_optimisation"]:
            mform = matrix_optimisation(form)
            if mform is None:
                non_pre_assembled_L.append(form)
            else:
                mult_assembled_L[mform[1]] = [mform[0]]
        else:
            non_pre_assembled_L.append(form)

        self._set_pa(pre_assembled_L, mult_assembled_L, non_pre_assembled_L)

        return

    def _set_pa(self, pre_assembled_L, mult_assembled_L, non_pre_assembled_L):
        if len(pre_assembled_L) == 0:
            self._pre_assembled_L = None
        else:
            l_L = pre_assembled_L[0]
            for L in pre_assembled_L[1:]:
                l_L += L
            self._pre_assembled_L = assembly_cache.assemble(l_L)

        n_mult_assembled_L = 0
        if len(mult_assembled_L) == 0:
            self._mult_assembled_L = None
        else:
            self._mult_assembled_L = []
            for fn in mult_assembled_L:
                mat_forms = mult_assembled_L[fn]
                n_mult_assembled_L += len(mat_forms)
                mat_form = mat_forms[0]
                for lmat_form in mat_forms[1:]:
                    mat_form += lmat_form
                self._mult_assembled_L.append([assembly_cache.assemble(mat_form, compress = self.pre_assembly_parameters["compress_matrices"]), fn])

        if len(non_pre_assembled_L) == 0:
            self._non_pre_assembled_L = None
        else:
            self._non_pre_assembled_L = non_pre_assembled_L[0]
            for L in non_pre_assembled_L[1:]:
                self._non_pre_assembled_L += L

        self._n_pre_assembled = len(pre_assembled_L) + n_mult_assembled_L
        self._n_non_pre_assembled = len(non_pre_assembled_L)
        self.__static = (self._n_non_pre_assembled == 0) and (n_mult_assembled_L == 0)

        return

    def assemble(self, copy = False):
        """
        Return the result of assembling the Form associated with this PALinearForm.
        If copy is False then an existing GenericVector may be returned -- it is
        expected in this case that the return value will never be modified.
        """

        if self._non_pre_assembled_L is None:
            if self._mult_assembled_L is None:
                if self._pre_assembled_L is None:
                    raise StateException("Cannot assemble empty form")
                else:
                    if copy:
                        L = self._pre_assembled_L.copy()
                    else:
                        L = self._pre_assembled_L
            else:
                L = self._mult_assembled_L[0][0] * self._mult_assembled_L[0][1].vector()
                for i in xrange(1, len(self._mult_assembled_L)):
                    L += self._mult_assembled_L[i][0] * self._mult_assembled_L[i][1].vector()
                if not self._pre_assembled_L is None:
                    L += self._pre_assembled_L
        else:
            L = assemble(self._non_pre_assembled_L)
            if not self._mult_assembled_L is None:
                for i in xrange(len(self._mult_assembled_L)):
                    L += self._mult_assembled_L[i][0] * self._mult_assembled_L[i][1].vector()
            if not self._pre_assembled_L is None:
                L += self._pre_assembled_L

        return L

    def is_static(self):
        """
        Return whether the PALinearForm is static.
        """

        return self.__static
