"""
Functions to add lines to the frequency domain plot.
"""

functions = {
    'max': {
        'description': 'Add the maximum value. Run function numpy.max',
        'par': {
        },
    },
    'std': {
        'description': 'Add the standard deviation. Run function numpy.std',
        'par': {
            'n_std': {'description': 'multiply n_std by the standard deviation',
                      'default': 1,
                      'type': 'float'},
        },
    },
}


def max(x, y, par):
    from numpy import max, ones
    mn = max(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add


def std(x, y, par):
    from numpy import std, ones
    mn = std(y) * par[0]
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add
