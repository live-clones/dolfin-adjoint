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
import numpy

from .exceptions import *
from .fenics_overrides import *

__all__ = \
  [
    "write_vtu"
  ]

def write_vtu(filename, fns, index = None, t = None):
    """
    Write the supplied Function or Function s to a vtu or pvtu file with the
    supplied filename base. If the Function s are defined on multiple function
    spaces then separate output files are written for each function space. The
    optional integer index can be used to add an index to the output filenames.
    If t is supplied then a scalar field equal to t and with name "time" is added
    to the output files.

    All Function s should be on the same mesh and have unique names. In 1D all
    Function s must have Lagrange basis functions (continuous or discontinous)
    with degree 0 to 3. In 2D and 3D all Function s must have Lagrange basis
    functions (continuous or discontinuous) with degree 0 to 2.
    """

    import vtk

    if isinstance(fns, dolfin.Function):
        return write_vtu(filename, [fns], index = index, t = t)
    if not isinstance(filename, str):
        raise InvalidArgumentException("filename must be a string")
    if not isinstance(fns, list):
        raise InvalidArgumentException("fns must be a Function or a list of Function s")
    if len(fns) == 0:
        raise InvalidArgumentException("fns must include at least one Function")
    for fn in fns:
        if not isinstance(fn, dolfin.Function):
            raise InvalidArgumentException("fns must be a Function or a list of Function s")
    if not index is None and not isinstance(index, int):
        raise InvalidArgumentException("index must be an integer")
    if not t is None and not isinstance(t, (float, dolfin.Constant)):
        raise InvalidArgumentException("t must be a float or Constant")

    mesh = fns[0].function_space().mesh()
    dim = mesh.geometry().dim()
    if not dim in [1, 2, 3]:
        raise NotImplementedException("Mesh dimension %i not supported by write_vtu" % dim)

    def expand_sub_fns(fn):
        n_sub_spaces = fn.function_space().num_sub_spaces()
        if n_sub_spaces > 1:
            fns = fn.split(deepcopy = True)
            for i, sfn in enumerate(copy.copy(fns)):
                sfn.rename("%s_%i" % (fn.name(), i + 1), "%s_%i" % (fn.name(), i + 1))
                if sfn.function_space().num_sub_spaces() > 0:
                    fns.remove(sfn)
                    fns += expand_sub_fns(sfn)
            return fns
        elif n_sub_spaces == 1:
            e = fn.function_space().ufl_element().extract_component(0)[1]
            space = dolfin.FunctionSpace(mesh, e.family(), e.degree())
            sfn = dolfin.Function(space, name = "%s_1" % fn.name())
            sfn.vector()[:] = fn.vector()
            return [sfn]
        else:
            return [fn]

    nfns = []
    for i, fn in enumerate(fns):
        space = fn.function_space()
        if not space.mesh().id() == mesh.id():
            raise InvalidArgumentException("Require exactly one mesh in write_vtu")
        n_sub_spaces = space.num_sub_spaces()
        nfns += expand_sub_fns(fn)
    fns = nfns;  del(nfns)

    spaces = []
    lfns = OrderedDict()
    for fn in fns:
        space = fn.function_space()
        assert(space.mesh().id() == mesh.id())
        e = space.ufl_element()
        assert(e.cell().geometric_dimension() == dim)
        assert(e.cell().topological_dimension() == dim)
        if (not e.family() in ["Lagrange", "Discontinuous Lagrange"]
            or not dim in [1, 2, 3]
            or (dim == 1 and not e.degree() in [1, 2, 3])
            or (dim in [2, 3] and not e.degree() in [1, 2])) and \
          (not e.family() == "Discontinuous Lagrange"
            or not dim in [1, 2, 3]
            or not e.degree() == 0):
            raise NotImplementedException('Element family "%s" with degree %i in %i dimension(s) not supported by write_vtu' % (e.family(), e.degree(), dim))
        if e in lfns:
            lfns[e].append(fn)
        else:
            spaces.append(space)
            lfns[e] = [fn]
    fns = lfns

    if len(spaces) == 1:
        filenames = [filename]
    else:
        filenames = []
        for space in spaces:
            e = space.ufl_element()
            lfilename = "%s_P%i" % (filename, e.degree())
            if e.family() == "Discontinuous Lagrange":
                lfilename = "%s_DG" % lfilename
            filenames.append(lfilename)

    if not t is None:
        for space in spaces:
            lt = dolfin.Function(space, name = "time")
            if isinstance(t, float):
                lt.vector()[:] = t
            else:
                lt.assign(t)
            fns[space.ufl_element()].append(lt)

    names = {e:[] for e in fns}
    for e in fns:
        for fn in fns[e]:
            name = fn.name()
            if name in names[e]:
                raise InvalidArgumentException("Duplicate Function name: %s" % name)
            names[e].append(name)

    for filename, space in zip(filenames, spaces):
        e = space.ufl_element()
        degree = e.degree()

        vtu = vtk.vtkUnstructuredGrid()

        dof = space.dofmap()
        nodes = set()
        for i in range(mesh.num_cells()):
            cell =  dof.cell_dofs(i)
            for node in cell:
                nodes.add(node)
        nodes = numpy.array(sorted(list(nodes)), dtype = numpy.intc)

        if degree == 0:
            xspace = dolfin.FunctionSpace(mesh, "CG", 1)
            xdof = xspace.dofmap()
            xnodes = set()
            for i in range(mesh.num_cells()):
                cell =  xdof.cell_dofs(i)
                for node in cell:
                    xnodes.add(node)
            xnodes = numpy.array(sorted(list(xnodes)), dtype = numpy.intc)
        else:
            xspace = space
            xdof = dof
            xnodes = nodes
        xnode_map = {node:i for i, node in enumerate(xnodes)}

        x = dolfin.interpolate(dolfin.Expression("x[0]", element = xspace.ufl_element()), xspace).vector().gather(xnodes)
        if dim > 1:
            y = dolfin.interpolate(dolfin.Expression("x[1]", element = xspace.ufl_element()), xspace).vector().gather(xnodes)
        if dim > 2:
            z = dolfin.interpolate(dolfin.Expression("x[2]", element = xspace.ufl_element()), xspace).vector().gather(xnodes)
        n = x.shape[0]

        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        points.SetNumberOfPoints(n)
        if dim == 1:
            for i in range(n):
                points.SetPoint(i, x[i], 0.0, 0.0)
        elif dim == 2:
            for i in range(n):
                points.SetPoint(i, x[i], y[i], 0.0)
        else:
            for i in range(n):
                points.SetPoint(i, x[i], y[i], z[i])
        vtu.SetPoints(points)

        id_list = vtk.vtkIdList()
        if dim == 1:
            if degree in [0, 1]:
                cell_type = vtk.vtkLine().GetCellType()
                id_list.SetNumberOfIds(2)
                cell_map = None
            elif degree == 2:
                cell_type = vtk.vtkQuadraticEdge().GetCellType()
                id_list.SetNumberOfIds(3)
                cell_map = None
            else:
                cell_type = vtk.vtkCubicLine().GetCellType()
                id_list.SetNumberOfIds(4)
                cell_map = None
        elif dim == 2:
            if degree in [0, 1]:
                cell_type = vtk.vtkTriangle().GetCellType()
                id_list.SetNumberOfIds(3)
                cell_map = None
            else:
                cell_type = vtk.vtkQuadraticTriangle().GetCellType()
                id_list.SetNumberOfIds(6)
                cell_map = {0:0, 1:1, 2:2, 3:5, 4:3, 5:4}
        else:
            if degree in [0, 1]:
                cell_type = vtk.vtkTetra().GetCellType()
                id_list.SetNumberOfIds(4)
                cell_map = None
            else:
                cell_type = vtk.vtkQuadraticTetra().GetCellType()
                id_list.SetNumberOfIds(10)
                cell_map = {0:0, 1:1, 2:2, 3:3, 4:9, 5:6, 6:8, 7:7, 8:5, 9:4}
        for i in range(mesh.num_cells()):
            cell = xdof.cell_dofs(i)
            assert(len(cell) == id_list.GetNumberOfIds())
            if not cell_map is None:
                cell = [cell[cell_map[j]] for j in range(len(cell))]
            for j in range(len(cell)):
                id_list.SetId(j, xnode_map[cell[j]])
            vtu.InsertNextCell(cell_type, id_list)

        if degree == 0:
            for fn in fns[e]:
                if not fn.value_rank() == 0:
                    raise NotImplementedException("Function rank %i not supported by write_vtu" % fn.value_rank())
                data = fn.vector().gather(nodes)
                cell_data = vtk.vtkDoubleArray()
                cell_data.SetNumberOfComponents(1)
                cell_data.SetNumberOfValues(mesh.num_cells())
                cell_data.SetName(fn.name())
                for i, datum in enumerate(data):
                    cell_data.SetValue(i, datum)
                vtu.GetCellData().AddArray(cell_data)
            vtu.GetCellData().SetActiveScalars(names[e][0])
        else:
            for fn in fns[e]:
                if not fn.value_rank() == 0:
                    raise NotImplementedException("Function rank %i not supported by write_vtu" % fn.value_rank())
                data = fn.vector().gather(nodes)
                point_data = vtk.vtkDoubleArray()
                point_data.SetNumberOfComponents(1)
                point_data.SetNumberOfValues(n)
                point_data.SetName(fn.name())
                for i, datum in enumerate(data):
                    point_data.SetValue(i, datum)
                vtu.GetPointData().AddArray(point_data)
            vtu.GetPointData().SetActiveScalars(names[e][0])

        if dolfin.MPI.size(dolfin.mpi_comm_world()) > 1:
            writer = vtk.vtkXMLPUnstructuredGridWriter()
            writer.SetNumberOfPieces(dolfin.MPI.size(dolfin.mpi_comm_world()))
            writer.SetStartPiece(dolfin.MPI.rank(dolfin.mpi_comm_world()))
            writer.SetEndPiece(dolfin.MPI.rank(dolfin.mpi_comm_world()))
            ext = ".pvtu"
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            ext = ".vtu"
        if index is None:
            filename = "%s%s" % (filename, ext)
        else:
            filename = "%s_%i%s" % (filename, index, ext)
        writer.SetFileName(filename)
        writer.SetDataModeToAppended()
        writer.EncodeAppendedDataOff()
        writer.SetCompressorTypeToZLib()
        writer.GetCompressor().SetCompressionLevel(9)
        writer.SetBlockSize(2 ** 15)
        writer.SetInputData(vtu)
        writer.Write()
        if not writer.GetProgress() == 1.0 or not writer.GetErrorCode() == 0:
            raise IOException("Failed to write vtu file: %s" % filename)

    return
