"""
Functions to evaluate a spectrum in frequency domain.
"""

functions = {
    'max': {
        'description': 'Compute the maximum amplitude and its frequency in length(data)/Ts',
        'par': {
        },
    },
    'std': {
        'description': 'Compute the standard deviation. Run the function numpy.std',
        'par': {
            'n_std': {'description':  'multiply n_std by the standard deviation',
                      'default': 1,
                      'type': 'float'}
        }
    }
}


def max(x, y, par):
    from numpy import max, argmax
    half = int(len(x) / 2)
    return [max(y[:half]), x[argmax(y[:half])]]


def std(x, y, par):
    from numpy import std
    ev = std(y) * par[0]
    return ev


