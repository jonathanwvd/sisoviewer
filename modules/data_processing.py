"""
Functions to apply data processing to a time series
"""

functions = {
    'medfilt': {
        'description': 'Perform a median filter on an N-dimensional array. Run the function scipy.signal.medfilt',
        'par': {
            'kernel_size': {'description': 'A scalar giving the size of the median filter window. Elements of '
                                           'kernel_size should be odd.',
                            'default': 3,
                            'type': 'int'},
        },
    },
    'filtfilt': {
        'description': 'Apply a digital filter forward and backward to a signal. Rum the function scipy.signal.filtfilt'
                       ' with a Butterworth filter',
        'par': {
            'N': {'description': 'The order of the filter.',
                  'default': 8,
                  'type': 'int'},
            'Wn': {'description': 'The critical frequency or frequencies. For lowpass and highpass filters, Wn is a '
                                  'scalar; for bandpass and bandstop filters, Wn is a length-2 sequence.',
                   'default': '[0.125]',
                   'type': 'array_like'},
            'btype ': {'description': 'The type of filter',
                       'default': 'lowpass',
                       'type': 'lowpass, highpass, bandpass, or bandstop'},
        },
    },
    'normalize': {
        'description': 'Normalize time series to mean equal to zero and standard deviation equal to one.',
        'par': {
        },
    },
}


def medfilt(x, y, par):
    import logging
    if (~par[0] % 2) or (par[0] < 0):
        logging.warning('kernel_size must be odd and positive')
        x_pre, y_pre = x, y

    else:
        from scipy.signal import medfilt
        y_pre = medfilt(y, par[0])
        x_pre = x
    return x_pre, y_pre


def filtfilt(x, y, par):
    from scipy.signal import butter, filtfilt
    b, a = butter(par[0], par[1], par[2])
    y_pre = filtfilt(b, a, y)
    x_pre = x
    return x_pre, y_pre


def normalize(x, y, par):
    from numpy import mean, std
    y_zero_mean = y - mean(y)
    y_pre = y_zero_mean / std(y_zero_mean)
    x_pre = x
    return x_pre, y_pre
