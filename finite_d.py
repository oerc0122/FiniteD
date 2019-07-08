import math
import numpy
import itertools

_stored_coeffs  = []
_stored_stencil = None
_stored_N       = None

def finite_d_coeffs( stencil, derivOrders, direction = 0, x0 = 0., points = None, store = True, storeN = (0,4) ):
    """
    Generate the coefficients for finite differences
    Using Fornberg method:
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
    Mathematics of Computation, 51 (184): 699–706, doi:10.1090/S0025-5718-1988-0935077-0, ISSN 0025-5718.

    In:
    stencil (int) : Order of stencil to generate
    maxDerivOrder (tuple(int) | int) : Highest order of derivative to generate
    direction (-1,0,1) : Backward, Centre, Forward difference
    x0 : centre-point of derivative
    store (bool) : Store calculated coeffs in array for reuse
    storeN (tuple(int)) : range of values to store in array

    Returns:
    list of coefficients for finite differences
    """
    global _stored_coeffs
    global _stored_stencil
    global _stored_N


    if isinstance(derivOrders, (list, tuple)):
        maxDerivOrder = max(derivOrders)
    elif isinstance(derivOrders, int):
        maxDerivOrder = derivOrders
        derivOrders = (derivOrders,)
    else: raise TypeError("derivOrders must be of type int or tuple/list of ints")

    if not isinstance(maxDerivOrder, int):raise TypeError("derivOrders must be of type int")

    if points is None:
        if direction == 0:
            points = range (-stencil, stencil + 1) #, key = lambda x:abs(x)
        elif direction == 1:
            points = list(range ( 0, stencil + 1))
        elif direction == -1:
            points = list(range ( 0, -stencil-1, -1))
        else: raise ValueError("direction must be one of -1, 0, 1")

    # Reuse old coeffs if available
    if stencil == _stored_stencil and \
       maxDerivOrder < len(_stored_coeffs) and \
       all( order in range(*_stored_N) for order in derivOrders) and \
       points is None:
        return points, [_stored_coeffs[order] for order in derivOrders]


    if len(points) < maxDerivOrder: raise ValueError("Stencil larger than possible")

    coeffs = numpy.zeros( (maxDerivOrder + 1, len(points), len(points)) ) # Need to account for zero points

    coeffs[0][0][0] = 1
    c1 = 1.
    for n in range (1, len(points)):
        c2 = 1.
        for v in range(n):
            c3 = points[n] - points[v]
            c2 *= c3

            if n < maxDerivOrder: coeffs[n][n-1][v] = 0

            for m in range(0, min(n,maxDerivOrder)+1):
                coeffs[m][n][v] = (( points[n] - x0 ) * coeffs[m][n-1][v] - m * coeffs[m-1][n-1][v])/c3

        for m in range (0, min(n,maxDerivOrder) + 1):
            coeffs[m][n][n] = (c1/c2)*(m * coeffs[m-1][n-1][n-1] - (points[n-1] - x0)*coeffs[m][n-1][n-1])

        c1 = c2

    if store:
        storeN = (storeN[0], min(maxDerivOrder, storeN[1]) )
        _stored_coeffs  = [coeffs[i][-1][:] for i in range(*storeN) ]
        _stored_stencil = stencil
        _stored_N       = storeN

    return points, [coeffs[i][-1][:] for i in derivOrders ]


def _finite_d_point_func( F, x, h, order, points, prefacs, args):
    return sum ( prefac * F( x + point*h, *args )  for point, prefac in zip(points, prefacs)) / (h**order)

def _finite_d_point_grid_1D( grid, x, h, order, points, prefacs, boundary = None ):
    if boundary is None: # No boundary == Periodic
        print("BEEP:", sum ( prefac * grid[ (x + point)%len(grid) ] for point, prefac in zip(points, prefacs)) / (h**order))
        return sum ( prefac * grid[ (x + point)%len(grid) ] for point, prefac in zip(points, prefacs)) / (h**order)
    else:
        return sum ( (prefac * grid[ (x + point) ] if 0 <= x + point < len(grid) else prefac*boundary)
                     for point, prefac in zip(points, prefacs)) / (h**order)

def finite_d( F, h, order, stencil, x = None, points = None, *funcArgs ):
    """
    Perform a general N-order n-stencil finite difference derivative of a function over a range or a grid
    """
    points, prefacs = finite_d_coeffs(stencil, order)

    prefacs = prefacs[0]

    if callable( F ):

        assert (x is not None), "X is undefined in function call"

        if isinstance( x, (int, float) ):
            yield _finite_d_point_func( F, x, h, order, points, prefacs, funcArgs)
        elif isinstance( x, (list, tuple)) :
            for xVal in x:
                yield _finite_d_point_func( F, xVal, h, order, points, prefacs, funcArgs)
        else:
            raise TypeError

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
