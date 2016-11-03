#!/usr/bin/env python2

# Copyright (C) 2007 Anders Logg
# Copyright (C) 2007-2011 Anders Logg and Garth N. Wells
# Copyright (C) 2008-2013 Martin Sandve Alnes
# Copyright (C) 2010-2012 Anders Logg
# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014-2016 University of Edinburgh
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

# Copyright (C) 2007 Anders Logg from FFC file ffc/codesnippets.py, bzr branch
# 1.1.x 1771
# Code first added: 2013-04-10

# Copyright (C) 2007-2011 Anders Logg and Garth N. Wells from DOLFIN file
# dolfin/fem/DirichletBC.cpp, bzr branch 1.2.x 7509
# Code first added: 2013-06-03

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file ufl/form.py, bzr
# branch 1.1.x 1484
# Code first added: 2013-05-08

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file ufl/coefficient.py,
# bzr trunk 1571
# Code first added: 2012-11-20

# Copyright (C) 2010-2012 Anders Logg from DOLFIN file
# dolfin/la/GenericMatrix.cpp, bzr 1.2.x branch 7509
# Code first added: 2013-05-23

import copy

import dolfin
import ffc
import instant
import numpy
import types
import ufl

from .embedded_cpp import *
from .versions import *

__all__ = []

# Only versions 1.5.x and 1.6.x have been tested.
if dolfin_version() < (1, 5, 0) or dolfin_version() >= (1, 7, 0):
    dolfin.warning("DOLFIN version %s not supported" % dolfin.__version__)
if ufl_version() < (1, 5, 0) or ufl_version() >= (1, 7, 0):
    dolfin.warning("UFL version %s not supported" % ufl.__version__)
if ffc_version() < (1, 5, 0) or ffc_version() >= (1, 7, 0):
    dolfin.warning("FFC version %s not supported" % ffc.__version__)
if instant_version() < (1, 5, 0) or instant_version() >= (1, 7, 0):
    dolfin.warning("Instant version %s not supported" % instant.__version__)

# DOLFIN patches.
if dolfin_version() < (1, 1, 0):
    __all__ += \
      [
        "GenericLinearSolver",
        "GenericLUSolver",
        "FacetFunction",
        "MeshFunction",
        "UnitCubeMesh",
        "UnitIntervalMesh",
        "UnitSquareMesh",
        "RectangleMesh",
        "has_krylov_solver_method",
        "has_krylov_solver_preconditioner",
        "has_lu_solver_method"
      ]

    GenericLinearSolver = dolfin.GenericLinearSolver = dolfin.cpp.GenericLinearSolver
    GenericLUSolver = dolfin.GenericLUSolver = dolfin.cpp.GenericLUSolver
    UnitCubeMesh = dolfin.UnitCube
    UnitIntervalMesh = dolfin.UnitInterval
    UnitSquareMesh = dolfin.UnitSquare
    RectangleMesh = dolfin.Rectangle

    def has_krylov_solver_method(method):
        return method in [k_method[0] for k_method in dolfin.krylov_solver_methods()]
    dolfin.has_krylov_solver_method = has_krylov_solver_method

    def has_krylov_solver_preconditioner(pc):
        return pc in [k_pc[0] for k_pc in dolfin.krylov_solver_preconditioners()]
    dolfin.has_krylov_solver_preconditioner = has_krylov_solver_preconditioner

    def has_lu_solver_method(method):
        return method in [lu_method[0] for lu_method in dolfin.lu_solver_methods()]
    dolfin.has_lu_solver_method = has_lu_solver_method

    class FacetFunction(dolfin.FacetFunction):
        def __new__(cls, tp, mesh, value = 0):
            if tp == "size_t":
                tp = "uint"
            return dolfin.FacetFunction.__new__(cls, tp, mesh, value = value)

    class MeshFunction(dolfin.MeshFunction):
        def __new__(cls, tp, *args):
            if tp == "size_t":
                tp = "uint"
            return dolfin.MeshFunction.__new__(cls, tp, *args)

    __GenericVector_gather_orig = dolfin.GenericVector.gather
    __GenericVector_gather_code = EmbeddedCpp(
      code =
        """
        Array< double > lx(n);
        Array< unsigned int > lnodes(n);
        for(size_t i = 0;i < n;i++){
          lnodes[i] = nodes[i];
        }
        v->gather(lx, lnodes);
        for(size_t i = 0;i < n;i++){
          x[i] = lx[i];
        }
        """,
      n = int, nodes = int_arr, v = dolfin.GenericVector, x = double_arr)
    def GenericVector_gather(self, *args):
        if len(args) == 1 and isinstance(args[0], numpy.ndarray) and len(args[0].shape) == 1 and args[0].dtype == "int32":
            x = numpy.empty(args[0].shape[0])
            __GenericVector_gather_code.run(n = args[0].shape[0], nodes = args[0], v = self, x = x)
            return x
        else:
            return __GenericVector_gather_orig(self, *args)
    dolfin.GenericVector.gather = GenericVector_gather
    del(GenericVector_gather)

    __Vector_gather_orig = dolfin.Vector.gather
    def Vector_gather(self, *args):
        if len(args) == 1 and isinstance(args[0], numpy.ndarray) and len(args[0].shape) == 1 and args[0].dtype == "int32":
            x = numpy.empty(args[0].shape[0])
            __GenericVector_gather_code.run(n = args[0].shape[0], nodes = args[0], v = self, x = x)
            return x
        else:
            return __Vector_gather_orig(self, *args)
    dolfin.Vector.gather = Vector_gather
    del(Vector_gather)

    def Constant__getattribute__(self, key):
        if key == "gather":
            raise AttributeError
        else:
            return object.__getattribute__(self, key)
    dolfin.Constant.__getattribute__ = Constant__getattribute__
    del(Constant__getattribute__)
if dolfin_version() < (1, 2, 0):
    __all__ += \
      [
        "grad",
        "FacetNormal"
      ]

    def grad(f):
        g = dolfin.grad(f)
        if len(g.shape()) == 0:
            return dolfin.as_vector([g])
        else:
            return g

    def FacetNormal(mesh):
        nm = dolfin.FacetNormal(mesh)
        if len(nm.shape()) == 0:
            return dolfin.as_vector([nm])
        else:
            return nm
if dolfin_version() < (1, 3, 0):
    __name_counter = [0]
    __Constant__init__orig = dolfin.Constant.__init__
    def Constant__init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        if "label" in kwargs:
            label = kwargs["label"]
            del(kwargs["label"])
        else:
            label = "a Constant"
        if "name" in kwargs:
            name = kwargs["name"]
            del(kwargs["name"])
        else:
            name = "f_%i" % __name_counter[0]
            __name_counter[0] += 1

        __Constant__init__orig(self, *args, **kwargs)
        self.rename(name, label)

        return
    dolfin.Constant.__init__ = Constant__init__
    del(Constant__init__)

    __Function__init__orig = dolfin.Function.__init__
    def Function__init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        if "label" in kwargs:
            label = kwargs["label"]
            del(kwargs["label"])
        else:
            label = "a Function"
        if "name" in kwargs:
            name = kwargs["name"]
            del(kwargs["name"])
        else:
            name = "f_%i" % __name_counter[0]
            __name_counter[0] += 1

        __Function__init__orig(self, *args, **kwargs)
        self.rename(name, label)

        return
    dolfin.Function.__init__ = Function__init__
    del(Function__init__)
if dolfin_version() < (1, 1, 0):
    # Modified version of code from GenericMatrix.cpp, DOLFIN bzr 1.2.x branch
    # revision 7509
    __GenericMatrix_compress_code = EmbeddedCpp(
      code =
        """
        Timer timer("Compress matrix");

        // Create new sparsity pattern
        GenericSparsityPattern* new_sparsity_pattern = mat->factory().create_pattern();
        // Check that we get a full sparsity pattern
        if (!new_sparsity_pattern)
        {
          warning("Linear algebra backend does not supply a sparsity pattern, "
                  "ignoring call to compress().");
          return 0;
        }

        // Retrieve global and local matrix info
        std::vector<unsigned int> global_dimensions(2);
        global_dimensions[0] = mat->size(0);
        global_dimensions[1] = mat->size(1);
        std::vector<std::pair<unsigned int, unsigned int> > local_range(2);
        local_range[0] = mat->local_range(0);
        local_range[1] = mat->local_range(0);

        // With the row-by-row algorithm used here there is no need for inserting non_local
        // rows and as such we can simply use a dummy for off_process_owner
        std::vector<const boost::unordered_map<unsigned int, unsigned int>* > off_process_owner(2);
        const boost::unordered_map<unsigned int, unsigned int> dummy;
        off_process_owner[0] = &dummy;
        off_process_owner[1] = &dummy;
        const std::pair<unsigned int, unsigned int> row_range = local_range[0];
        const unsigned int m = row_range.second - row_range.first;

        // Initialize sparsity pattern
        new_sparsity_pattern->init(global_dimensions, local_range, off_process_owner);

        // Declare some variables used to extract matrix information
        std::vector<unsigned int> columns;
        std::vector<double> values;
        std::vector<double> allvalues; // Hold all values of local matrix
        std::vector<dolfin::la_index> allcolumns;  // Hold column id for all values of local matrix
        std::vector<dolfin::la_index> offset(m + 1); // Hold accumulated number of cols on local matrix
        offset[0] = 0;
        std::vector<dolfin::la_index> thisrow(1);
        std::vector<dolfin::la_index> thiscolumn;
        std::vector<const std::vector<dolfin::la_index>* > dofs(2);
        dofs[0] = &thisrow;
        dofs[1] = &thiscolumn;

        // Iterate over rows
        for (std::size_t i = 0; i < m; i++)
        {
          // Get row and locate nonzeros. Store non-zero values and columns for later
          const unsigned int global_row = i + row_range.first;
          mat->getrow(global_row, columns, values);
          std::size_t count = 0;
          thiscolumn.clear();
          for (std::size_t j = 0; j < columns.size(); j++)
          {
            // Store if non-zero or diagonal entry. PETSc solvers require this
            if (std::abs(values[j]) > DOLFIN_EPS || columns[j] == global_row)
            {
              thiscolumn.push_back(columns[j]);
              allvalues.push_back(values[j]);
              allcolumns.push_back(columns[j]);
              count++;
            }
          }

          thisrow[0] = global_row;
          offset[i + 1] = offset[i] + count;

          // Build new compressed sparsity pattern
          new_sparsity_pattern->insert(dofs);
        }

        // Finalize sparsity pattern
        new_sparsity_pattern->apply();

        // Recreate matrix with the new sparsity pattern
        mat->init(*new_sparsity_pattern);

        // Put the old values back in the newly compressed matrix
        for (std::size_t i = 0; i < m; i++)
        {
          const dolfin::la_index global_row = i + row_range.first;
          mat->set(&allvalues[offset[i]], 1, &global_row,
              offset[i+1] - offset[i], &allcolumns[offset[i]]);
        }

        mat->apply("insert");
        """,
      mat = dolfin.GenericMatrix)
    def GenericMatrix_compress(self):
        if dolfin.MPI.num_processes() == 1 or self.size(0) == self.size(1):
            __GenericMatrix_compress_code.run(mat = self)
        return
    dolfin.GenericMatrix.compress = GenericMatrix_compress
    del(GenericMatrix_compress)
elif dolfin_version() < (1, 4, 0):
    __GenericMatrix_compress_orig = dolfin.GenericMatrix.compress
    def GenericMatrix_compress(self):
        if dolfin.MPI.num_processes() == 1 or self.size(0) == self.size(1):
            __GenericMatrix_compress_orig(self)
        return
    dolfin.GenericMatrix.compress = GenericMatrix_compress
    del(GenericMatrix_compress)
elif dolfin_version() < (1, 5, 0):
    def GenericMatrix_compress(self):
        if dolfin.MPI.num_processes() == 1 or self.size(0) == self.size(1):
            self.compressed(self)
        return
    dolfin.GenericMatrix.compress = GenericMatrix_compress
    del(GenericMatrix_compress)
if dolfin_version() < (1, 1, 0):
    # Modified version of code from DirichletBC.cpp, DOLFIN bzr 1.2.x branch
    # revision 7509
    __DirichletBC_zero_columns_code = EmbeddedCpp(
      includes =
        """
        #include <queue>
        """,
      code =
        """
        DirichletBC::Map bv_map;
        bc->get_boundary_values(bv_map, bc->method());

        // Create lookup table of dofs
        //const unsigned int nrows = A->size(0); // should be equal to b->size()
        const unsigned int ncols = A->size(1); // should be equal to max possible dof+1

        std::pair<unsigned int, unsigned int> rows = A->local_range(0);

        std::vector<char> is_bc_dof(ncols);
        std::vector<double> bc_dof_val(ncols);
        for (DirichletBC::Map::const_iterator bv = bv_map.begin();  bv != bv_map.end();  ++bv)
        {
          is_bc_dof[bv->first] = 1;
          bc_dof_val[bv->first] = bv->second;
        }

        // Scan through all columns of all rows, setting to zero if is_bc_dof[column]
        // At the same time, we collect corrections to the RHS

        std::vector<unsigned int> cols;
        std::vector<double> vals;
        std::queue<std::vector<double> > A_vals;
        std::queue<unsigned int> A_rows;
        std::queue<std::vector<unsigned int> > A_cols;
        std::vector<double> b_vals;
        std::vector<dolfin::la_index> b_rows;

        for (unsigned int row = rows.first; row < rows.second; row++)
        {
          // If diag_val is nonzero, the matrix is a diagonal block (nrows==ncols),
          // and we can set the whole BC row
          if (diag_val != 0.0 && is_bc_dof[row])
          {
            A->getrow(row, cols, vals);
            for (std::size_t j = 0; j < cols.size(); j++)
              vals[j] = (cols[j] == row)*diag_val;
            A_vals.push(std::vector<double>(vals));
            A_rows.push(row);
            A_cols.push(std::vector<unsigned int>(cols));
            b->setitem(row, bc_dof_val[row]*diag_val);
          }
          else // Otherwise, we scan the row for BC columns
          {
            A->getrow(row, cols, vals);
            bool row_changed = false;
            for (std::size_t j = 0; j < cols.size(); j++)
            {
              const unsigned int col = cols[j];

              // Skip columns that aren't BC, and entries that are zero
              if (!is_bc_dof[col] || vals[j] == 0.0)
                continue;

              // We're going to change the row, so make room for it
              if (!row_changed)
              {
                row_changed = true;
                b_rows.push_back(row);
                b_vals.push_back(0.0);
              }

              b_vals.back() -= bc_dof_val[col]*vals[j];
              vals[j] = 0.0;
            }
            if (row_changed)
            {
              A_vals.push(std::vector<double>(vals));
              A_rows.push(row);
              A_cols.push(std::vector<unsigned int>(cols));
            }
          }
        }

        while(!A_vals.empty())
        {
          A->setrow(A_rows.front(), A_cols.front(), A_vals.front());
          A_vals.pop();  A_rows.pop();  A_cols.pop();
        }
        A->apply("insert");
        b->add(&b_vals.front(), b_rows.size(), &b_rows.front());
        b->apply("add");
        """, bc = dolfin.DirichletBC, A = dolfin.GenericMatrix, b = dolfin.GenericVector, diag_val = float)
    def DirichletBC_zero_columns(self, A, b, diag_val):
        __DirichletBC_zero_columns_code.run(bc = self, A = A, b = b, diag_val = diag_val)
        return
    dolfin.DirichletBC.zero_columns = DirichletBC_zero_columns
    del(DirichletBC_zero_columns)
elif dolfin_version() < (1, 5, 0):
    # Modified version of code from DirichletBC.cpp, DOLFIN bzr 1.2.x branch
    # revision 7509
    __DirichletBC_zero_columns_code = EmbeddedCpp(
      includes =
        """
        #include <queue>
        """,
      code =
        """
        DirichletBC::Map bv_map;
        bc->get_boundary_values(bv_map, bc->method());

        // Create lookup table of dofs
        //const std::size_t nrows = A->size(0); // should be equal to b->size()
        const std::size_t ncols = A->size(1); // should be equal to max possible dof+1

        std::pair<std::size_t, std::size_t> rows = A->local_range(0);

        std::vector<char> is_bc_dof(ncols);
        std::vector<double> bc_dof_val(ncols);
        for (DirichletBC::Map::const_iterator bv = bv_map.begin();  bv != bv_map.end();  ++bv)
        {
          is_bc_dof[bv->first] = 1;
          bc_dof_val[bv->first] = bv->second;
        }

        // Scan through all columns of all rows, setting to zero if is_bc_dof[column]
        // At the same time, we collect corrections to the RHS

        std::vector<std::size_t> cols;
        std::vector<double> vals;
        std::queue<std::vector<double> > A_vals;
        std::queue<size_t> A_rows;
        std::queue<std::vector<size_t> > A_cols;
        std::vector<double> b_vals;
        std::vector<dolfin::la_index> b_rows;

        for (std::size_t row = rows.first; row < rows.second; row++)
        {
          // If diag_val is nonzero, the matrix is a diagonal block (nrows==ncols),
          // and we can set the whole BC row
          if (diag_val != 0.0 && is_bc_dof[row])
          {
            A->getrow(row, cols, vals);
            for (std::size_t j = 0; j < cols.size(); j++)
              vals[j] = (cols[j] == row)*diag_val;
            A_vals.push(std::vector<double>(vals));
            A_rows.push(row);
            A_cols.push(std::vector<size_t>(cols));
            b->setitem(row, bc_dof_val[row]*diag_val);
          }
          else // Otherwise, we scan the row for BC columns
          {
            A->getrow(row, cols, vals);
            bool row_changed = false;
            for (std::size_t j = 0; j < cols.size(); j++)
            {
              const std::size_t col = cols[j];

              // Skip columns that aren't BC, and entries that are zero
              if (!is_bc_dof[col] || vals[j] == 0.0)
                continue;

              // We're going to change the row, so make room for it
              if (!row_changed)
              {
                row_changed = true;
                b_rows.push_back(row);
                b_vals.push_back(0.0);
              }

              b_vals.back() -= bc_dof_val[col]*vals[j];
              vals[j] = 0.0;
            }
            if (row_changed)
            {
              A_vals.push(std::vector<double>(vals));
              A_rows.push(row);
              A_cols.push(std::vector<size_t>(cols));
            }
          }
        }

        while(!A_vals.empty())
        {
          A->setrow(A_rows.front(), A_cols.front(), A_vals.front());
          A_vals.pop();  A_rows.pop();  A_cols.pop();
        }
        A->apply("insert");
        b->add(&b_vals.front(), b_rows.size(), &b_rows.front());
        b->apply("add");
        """, bc = dolfin.DirichletBC, A = dolfin.GenericMatrix, b = dolfin.GenericVector, diag_val = float)
    def DirichletBC_zero_columns(self, A, b, diag_val):
        __DirichletBC_zero_columns_code.run(bc = self, A = A, b = b, diag_val = diag_val)
        return
    dolfin.DirichletBC.zero_columns = DirichletBC_zero_columns
    del(DirichletBC_zero_columns)
if dolfin_version() >= (1, 4, 0) and dolfin_version < (1, 5, 0):
    __all__ += \
      [
        "info_blue",
        "info_green",
        "info_red"
      ]

    info_blue = dolfin.info_blue = lambda message : dolfin.info(ufl.log.BLUE % message)
    info_green = dolfin.info_green = lambda message : dolfin.info(ufl.log.GREEN % message)
    info_red = dolfin.info_red = lambda message : dolfin.info(ufl.log.RED % message)
if dolfin_version() < (1, 6, 0):
    __all__ += \
      [
        "RectangleMesh"
      ]

    def RectangleMesh(*args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], dolfin.Point) and isinstance(args[1], dolfin.Point):
            return dolfin.RectangleMesh(args[0].x(), args[0].y(), args[1].x(), args[1].y(), *args[2:], **kwargs)
        elif len(args) >= 3 and isinstance(args[1], dolfin.Point) and isinstance(args[2], dolfin.Point):
            return dolfin.RectangleMesh(args[0], args[1].x(), args[1].y(), args[2].x(), args[2].y(), *args[3:], **kwargs)
        else:
            return dolfin.RectangleMesh(*args, **kwargs)

# UFL patches.
if ufl_version() < (1, 1, 0):
    # Modified version of code from coefficient.py, UFL bzr trunk revision 1463
    def Coefficient__eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ufl.coefficient.Coefficient):
            return False
        return self._count == other._count and self._element == other._element
    ufl.coefficient.Coefficient.__eq__ = Coefficient__eq__
    del(Coefficient__eq__)

    # Modified version of code from form.py, UFL bzr 1.1.x branch revision 1484
    def Form__mul__(self, coefficient):
        if isinstance(coefficient, ufl.expr.Expr):
            return ufl.formoperators.action(self, coefficient)
        return NotImplemented
    ufl.Form.__mul__ = Form__mul__
    del(Form__mul__)

# FFC patches.
if ffc_version() < (1, 2, 0):
    # Patch to code from FFC file codesnippets.py, FFC bzr 1.1.x branch revision
    # 1771
    ffc.codesnippets._circumradius_1D = """\
    // Compute circumradius, in 1D it is equal to the cell volume/2.0.
    const double circumradius%(restriction)s = std::abs(detJ%(restriction)s)/2.0;"""
    ffc.codesnippets.circumradius[1] = ffc.codesnippets._circumradius_1D
