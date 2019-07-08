#-*- coding: utf-8 -*-
# module finite_d.py
# MIT License

# Copyright (c) 2019 Jacob Wilkins

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__doc__ = """
Module to simply, but flexibly calculate finite differences on numpy grids or functions.

Note: This module is neither optimised nor parallelised and is not intended to be used for high-performance applications.
"""
__author__ = "Jacob S. Wilkins"

import math
import numpy
import itertools
from collections.abc import Iterable

_stored_coeffs  = []
_stored_points  = None
_stored_N       = None

def finite_d_coeffs( stencil, derivOrders, direction = 0, x0 = 0., points = None, store = True, storeN = (0,4) ):
    """
    Generate the coefficients for finite differences
    Using Fornberg method:
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
    Mathematics of Computation, 51 (184): 699–706, doi:10.1090/S0025-5718-1988-0935077-0, ISSN 0025-5718.

    In:
    stencil       (int)              : Order of stencil to generate
    derivOrders (tuple(int) | int)   : Orders of derivative to generate
    direction     (-1,0,1)           : Backward, Centre, Forward difference
    x0                               : centre-point of derivative
    store         (bool)             : Store calculated coeffs in array for reuse
    storeN        (tuple(int))       : range of derivative orders to keep stored in array

    Returns:
    list of lists of coefficients for finite differences of derivOrders in order specified by tuple
    """
    global _stored_coeffs
    global _stored_points
    global _stored_N


    if isinstance(derivOrders, (list, tuple)):
        maxDerivOrder = max(derivOrders)
    elif isinstance(derivOrders, int):
        maxDerivOrder = derivOrders
        derivOrders = (derivOrders,)
    else:
        raise TypeError("derivOrders must be of type int or tuple/list of ints")

    if any(derivOrder < 1 for derivOrder in derivOrders):
        raise ValueError("Cannot caclulate negative or 0th order derivatives")

    if not isinstance(maxDerivOrder, int):
        raise TypeError("derivOrders must be of type int")

    if points is None:
        if direction == 0:
            points = range (-stencil, stencil + 1)
        elif direction == 1:
            points = list(range ( 0, stencil + 1))
        elif direction == -1:
            points = list(range (-stencil, 1))
        else:
            raise ValueError("direction must be one of -1, 0, 1")
    else:
        points = sorted(points)

    diffs = numpy.zeros(len(points))
    for i in range(0,len(points)-1):
        diffs[i] = 1. / (points[i+1] - points[i])
    diffs[-1] = 1. / (points[-1] - points[-2])
    
    # Reuse old coeffs if available
    if points == _stored_points and \
       maxDerivOrder < len(_stored_coeffs) and \
       all( order in range(*_stored_N) for order in derivOrders) and \
       points is None:
        return points, [_stored_coeffs[order] for order in derivOrders]


    if len(points) < maxDerivOrder:
        raise ValueError("Requested stencil larger than possible")

    coeffs = numpy.zeros( (maxDerivOrder + 1, len(points), len(points)) ) # Need to account for zero points

    coeffs[0][0][0] = 1
    c1 = 1.
    for n in range (1, len(points)):
        c2 = 1.
        for v in range(n):
            c3 = points[n] - points[v]
            c2 *= c3

            for m in range(0, min(n,maxDerivOrder)+1):
                coeffs[m][n][v] = (( points[n] - x0 ) * coeffs[m][n-1][v] - m * coeffs[m-1][n-1][v])/c3

        for m in range (0, min(n,maxDerivOrder) + 1):
            coeffs[m][n][n] = (c1/c2)*(m * coeffs[m-1][n-1][n-1] - (points[n-1] - x0)*coeffs[m][n-1][n-1])

        c1 = c2

    if store:
        storeN = (storeN[0], min(maxDerivOrder, storeN[1]) )
        _stored_coeffs  = [coeffs[i][-1][:] * (diffs**i) for i in range(*storeN) ]
        _stored_points  = points
        _stored_N       = storeN

    return points, tuple(coeffs[i][-1][:] * (diffs**i) for i in derivOrders)


def _finite_d_point_func( F, x, h, order, points, prefacs, args, kwargs):
    return sum ( prefac * F( x + point*h, *args, **kwargs )  for point, prefac in zip(points, prefacs)) / (h**order)

def _finite_d_point_grid_1D( grid, x, h, order, points, prefacs, boundary = None ):
    if boundary is None: # No boundary == Periodic
        return sum ( prefac * grid[ (x + point)%len(grid) ] for point, prefac in zip(points, prefacs)) / (h**order)
    else:
        return sum ( (prefac * grid[ (x + point) ] if 0 <= x + point < len(grid) else prefac*boundary)
                     for point, prefac in zip(points, prefacs)) / (h**order)

def finite_d( F, h, order = None, stencil = None, x = None, points = None, prefacs = None, *funcArgs, **funcKwargs ):
    """
    Perform a general N-order n-stencil finite difference derivative of a function over a range or a grid.

    One of either order and stencil or points and prefacs must be defined. If both are defined, points and prefacs takes priority.
    If order and stencil are specified the code can generate the desired coefficients and points automatically otherwise they can be passed in if manually generated via finite_d_coeffs or other means.

    For numpy arrays:
        x determines the slice of a numpy array to be operated over.
        funcArgs and funcKwargs do nothing.
    For functions:
        x is either an iterable of arguments which are looped over or a single point to differentiate.

    In:
    F       (function | numpy.ndarray) : Function or grid to differentiate
    stencil (int)                      : Order of stencil
    h       (list(float) | float)      : step length for derivative
    order   (int)                      : Derivative order for automatically generated stencils
    stencil (int)                      : Stencil order for automatically generated stencils
    x       (function arg | slice)     : For grids numpy slice to be acted upon
                                         For functions the values of x to be acted upon
    points  (list(float))              : List of stencil points to run over
    prefacs (list(float))              : List of finite difference coefficients
    *funcArgs (Any)                    : Extra arguments to be passed to F if function
    **funcKwargs (Any)                 : Extra keyword arguments to be passed to F if function

    Returns:
    Generator of calculated finite differences
    """

    if points is not None and prefacs is not None:
        pass
    elif order is not None and stencil is not None:
        points, (prefacs,) = finite_d_coeffs(stencil, order)
    else:
        raise ValueError("Must specify either points and prefacs or stencil and derivative order")

    if callable( F ):

        if isinstance(x, Iterable):
            for xVal in x:
                yield _finite_d_point_func( F, xVal, h, order, points, prefacs, funcArgs, funcKwargs)
        else:
            yield _finite_d_point_func( F, x, h, order, points, prefacs, funcArgs, funcKwargs)

    elif isinstance( F, numpy.ndarray ):

        if x is None:
            x = F
        else:          # x to be resolved by numpy arrays
            x = F[x]

        ranges = map(range, x.shape)
        for indices in itertools.product(*ranges):

            point = x[indices]
            dx = 0.
            for dim in range(x.ndim):

                currSlice = tuple([index if i != dim else ... for i,index in enumerate(indices) ])
                dimSlice = x[currSlice]

                dx += _finite_d_point_grid_1D( dimSlice, indices[dim], h, order, points, prefacs)

            yield dx
    else:
        raise TypeError

if __name__ == "__main__":
    # Run Unit tests

    ### Stencil tests
    
    # Centred test
    _, (testVal,) = finite_d_coeffs( stencil = 5, derivOrders = 6, direction = 0, store = False )
    expectVal = numpy.array ([ 13/240, -19/24, 87/16, -39/2, 323/8, -1023/20, 323/8, -39/2, 87/16, -19/24, 13/240 ])
    print("Maximum deviation, Centred, order 6, stencil 5: ", max( abs(testVal - expectVal)))
    assert all( abs(testVal - expectVal) < 1e-10 ), "Not all values within error"

    # Forward Test
    _, (testVal,) = finite_d_coeffs( stencil = 8, derivOrders = 4, direction = 1, store = False)
    expectVal = numpy.array([1069/80, -1316/15, 15289/60, -2144/5, 10993/24, -4772/15, 2803/20, -536/15, 967/240])
    print("Maximum deviation, Forward, order 4, stencil 8: ", max( abs(testVal - expectVal)))
    assert all( abs(testVal - expectVal) < 1e-10 ), "Not all values within error"

    # Backward test
    _, (testVal,) = finite_d_coeffs( stencil = 8, derivOrders = 4, direction = -1, store = False )
    expectVal = expectVal[::-1]
    print("Maximum deviation, Backward, order 4, stencil 8: ", max( abs(testVal - expectVal)))
    assert all( abs(testVal - expectVal) < 1e-10 ), "Not all values within error"

    ### Derivative tests
    
    # Test first, second and third derivative quartic
    f = lambda x : x**3

    # Point tests
    *testVal, = finite_d(f, 0.01, order = 1, stencil = 5, x = 4)
    assert abs(testVal[0] - 3*(4**2)) < 1e-5, "Point test order 1 derivative failed"

    *testVal, = finite_d(f, 0.01, order = 2, stencil = 5, x = 4)
    assert abs(testVal[0] - 6*(4**1)) < 1e-5, "Point test order 2 derivative failed"

    *testVal, = finite_d(f, 0.01, order = 3, stencil = 5, x = 4)
    assert abs(testVal[0] - 6) < 1e-5, "Point test order 3 derivative failed"

    # Line tests
    vals = numpy.array([4,5,6])

    *testVal, = finite_d(f, 0.01, order = 1, stencil = 5, x = vals)
    testVal = numpy.array(testVal)
    assert all(abs(testVal - 3*(vals**2)) < 1e-5), "Line test order 1 derivative failed"

    *testVal, = finite_d(f, 0.01, order = 2, stencil = 5, x = vals)
    testVal = numpy.array(testVal)
    assert all(abs(testVal - 6*(vals**1)) < 1e-5), "Line test order 2 derivative failed"

    *testVal, = finite_d(f, 0.01, order = 3, stencil = 5, x = vals)
    testVal = numpy.array(testVal)
    assert all(abs(testVal - 6) < 1e-5), "Line test order 3 derivative failed"
