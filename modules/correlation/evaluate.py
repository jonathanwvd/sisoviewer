"""
Functions to evaluate a time series in time domain.
"""

functions = {
    'max': {
        'description': 'Compute the maximum value. Run the function numpy.max',
        'par': {
        },
    },

    'min': {
        'description': 'Compute the minimum value. Run the function numpy.min',
        'par': {
        },
    },

    'std': {
        'description': 'Compute the standard deviation. Run the function numpy.std',
        'par': {
            'n_std': {'description': 'multiply n_std by the standard deviation',
                      'default': 1,
                      'type': 'float'},
        },
    },
}


def max(x, y, par):
    from numpy import max
    return max(y)


def min(x, y, par):
    from numpy import min
    return min(y)


def std(x, y, par):
    from numpy import std
    return std(y) * par[0]
