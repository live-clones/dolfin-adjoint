#!/usr/bin/env python3

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014, 2017 University of Edinburgh
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

# Modified version of shallow_water test from dolfin-adjoint bzr trunk 636
# Code first added: 2013-04-26

from dolfin import Mesh, Expression
from math import exp, sqrt, pi

import sw_lib

params=sw_lib.parameters({
    'depth' : 5.,
    'g' : 10.,
    'f' : 1.0313e-4,
    'dump_period' : 10
    })

# Basin radius.
r0=250000
# Long wave celerity.
c=sqrt(params["g"]*params["depth"])


params["finish_time"]=4*2*pi*r0/c
params["dt"]=params["finish_time"]/4000.

# Rossby radius.
LR=c/params["f"]

class InitialConditions(Expression):
    def __init__(self):
        pass
    def eval(self, values, X):
        r=(X[0]**2+X[1]**2)**0.5
        if r>0.0001:
            values[0]=-0.05*c*exp((r-r0)/LR)*X[0]/r*X[1]/r
            values[1]= 0.05*c*exp((r-r0)/LR)*X[0]/r*X[0]/r
            values[2]= 0.05*exp((r-r0)/LR)*X[0]/r
        else:
            values[0]=0.
            values[1]=0.
            values[2]=0.
    def value_shape(self):
        return (3,)

#try:
#  mesh=Mesh("basin.xml")
#except RuntimeError:
#  import sys
#  import os.path
#
#  mesh=Mesh(os.path.dirname(sys.argv[0]) + os.path.sep + "basin.xml")
#
#mesh.order()
#mesh.init()
