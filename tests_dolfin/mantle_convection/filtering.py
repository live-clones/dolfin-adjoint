__author__ = "Lyudmyla Vynnytska"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *
from numpy import *

def filter_properties(rho):

    rho1 = rho.vector().array()

    num = len(rho1)
    Cmin = min(rho1)
    """
    def boundMin(x):
         if abs(x - Cmin) < DOLFIN_EPS:
             x = 0.0
         return x
    rho1 = array(map(boundMin, rho1))
    Cmin = min(rho1)

    Cmax = max(rho1)
    def boundMax(x):
         if abs(x - Cmax) < DOLFIN_EPS:
             x = 1.0
         return x
    rho1 = array(map(boundMax, rho1))
    """
    Cmax = max(rho1)
    Csum0 = sum(rho1)

    # Add test first:
    if (2.0 - Cmax) <= abs(Cmin):
        print "No internal consistency!"
        exit()

    # Do nothing if all values are in range [0.0, 1.0]
    if Cmin >= 0.0 and Cmax <= 1.0:
        return

    # Filter values below 0 and above 1.
    def bounds(x):
        if x < DOLFIN_EPS:
            x = 0.0
        if x > 1.0 - DOLFIN_EPS:
            x = 1.0
        return x
    rho1 = array(map(bounds, rho1))

    # Move stuff that is almost small and almost big
    for i in range(num):
        if rho1[i] <= abs(Cmin):
            rho1[i] = 0.0
        if rho1[i] >= (2.0 - Cmax):
            rho1[i] = 1.0

    def perturb(x):
        return (abs(x) > DOLFIN_EPS and abs(x - 1.0) > DOLFIN_EPS)
    num1 = len(filter(perturb, rho1))

    Csum1 = sum(rho1)

    if num1 > 0:
        Dist = (Csum0 - Csum1) / num1
    else:
        Dist = 0.0

    for i in range(num):
        if rho1[i] > 0.0 and rho1[i] < 1.0:
            rho1[i] += Dist

    rho.vector()[:] = rho1
