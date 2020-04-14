"""
Functions to evaluate a time series in time domain.
"""

functions = {
    'mean': {
        'description': 'Compute the arithmetic mean. Run the function numpy.mean',
        'par': {
        },
    },

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

    'mean_max_min_std': {
        'description': 'Compute the mean, maximum value, minimum value, and standard deviation.',
        'par': {
        },
    },

    'zeros': {
        'description': 'Find the zero positions. The given values are the position after the zero-crossing.',
        'par': {
        },
    },
}


def mean(x, y, par):
    from numpy import mean
    return mean(y)


def max(x, y, par):
    from numpy import max
    return max(y)


def min(x, y, par):
    from numpy import min
    return min(y)


def std(x, y, par):
    from numpy import std
    return std(y) * par[0]


def mean_max_min_std(x, y, par):
    from numpy import mean, max, min, std
    return [mean(y), max(y), min(y), std(y)]


def zeros(x, y, par):
    from numpy import sign
    zeros_pos = []
    for i in range(1, len(y)):
        if sign(y[i]) != sign(y[i - 1]):
            zeros_pos.append(x[i])
    return zeros_pos
