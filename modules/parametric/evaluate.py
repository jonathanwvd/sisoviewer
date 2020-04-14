"""
Functions to evaluate a time series in time domain.
"""

functions = {
    'mean': {
        'description': 'Compute the arithmetic mean of both axes (x, y). Run the function numpy.mean',
        'par': {
        },
    },

    'max': {
        'description': 'Compute the maximum value of both axes (x, y). Run the function numpy.max',
        'par': {
        },
    },

    'min': {
        'description': 'Compute the minimum value of both axes (x, y). Run the function numpy.min',
        'par': {
        },
    },

    'std': {
        'description': 'Compute the standard deviation of both axes (x, y). Run the function numpy.std',
        'par': {
            'n_std': {'description': 'multiply n_std by the standard deviation',
                      'default': 1,
                      'type': 'float'},
        },
    },

    'ellipse_dist': {
        'description': 'Give the x axis length of the ellipse',
        'par': {
        },
    },
}


def mean(x, y, par):
    from numpy import mean
    return mean(x), mean(y)


def max(x, y, par):
    from numpy import max
    return max(x), max(y)


def min(x, y, par):
    from numpy import min
    return min(x), min(y)


def std(x, y, par):
    from numpy import std
    return std(x) * par[0], std(y) * par[0]


def ellipse_dist(x, y, par):
    import numpy as np
    from numpy.linalg import eig, inv

    def fitEllipse(x, y):
        x = np.asarray(x)[:, np.newaxis]
        y = np.asarray(y)[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2;
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    a = fitEllipse(x, y)
    axes = ellipse_axis_length(a)
    AS = axes[0] * 2

    return AS
