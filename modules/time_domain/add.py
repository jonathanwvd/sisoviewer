"""
Functions to add lines to the time domain plot.
"""

functions = {
    'mean': {
        'description': 'Add the arithmetic mean. Run the function numpy.mean',
        'par': {
        },
    },
    'max': {
        'description': 'Add the maximum value. Run the function numpy.max',
        'par': {
        },
    },
    'min': {
        'description': 'Add the minimum value. Run the function numpy.min',
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


def mean(x, y, par):
    from numpy import mean, ones
    mn = mean(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add


def max(x, y, par):
    from numpy import max, ones
    mn = max(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add


def min(x, y, par):
    from numpy import min, ones
    mn = min(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add


def std(x, y, par):
    from numpy import std, ones, mean
    mn = std(y) * float(par[0]) + mean(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add