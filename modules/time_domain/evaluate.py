"""
Functions to evaluate a time series in time domain.
"""

functions = {
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
    'miao': {
        'description': 'Oscillation detection method proposed by Miao and Seborg. The function returns 1 for oscillatory time series and 0 for non-oscillatory. The functions returns, also, the r index.',
        'par': {
        },
    },
    'thornhill': {
        'description': 'Oscillation detection method proposed by Thornhill and Hagglund. The function returns 1 for oscillatory time series or 0 for non-oscillatory. The functions also returns the regularity factor r.',
        'par': {
        },
    },
    'li': {
        'description': 'Oscillation detection method proposed by Li et al. The function returns 1 for oscillatory time series or 0 for non-oscillatory.',
        'par': {
        },
    },
}


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


def miao(x, y, par):
    from numpy import correlate, var, sign

    def peak_g(y):
        peaks = [0]
        for i in range(2, len(y)):
            if sign((y[i]) - y[i - 1]) != sign((y[i - 1]) - y[i - 2]):
                peaks.append(i)
        return peaks

    n = y.size
    r = correlate(y, y, mode='full')[-n:]
    auto_co = r / (var(y) * n)

    peaks = peak_g(auto_co)

    if len(peaks) < 4:
        R = 0

    else:
        b = (auto_co[peaks[0]] + auto_co[peaks[2] - 1]) / 2 - auto_co[peaks[1] - 1]
        a = -(auto_co[peaks[1] - 1] + auto_co[peaks[3] - 1]) / 2 + auto_co[peaks[2] - 1]
        R = a / b

    if R > 0.5:
        flag = 1
    else:
        flag = 0

    return [flag, R]


def thornhill(x, y, par):
    from numpy import mean, var, correlate, std, diff, arange

    def zero_g(y):
        from numpy import sign
        zero = []
        for i in range(1, y.size):
            if sign(y[i]) != sign(y[i - 1]):
                zero.append(i)
        return zero

    def reg(y):
        t = arange(0, len(y))
        zero = zero_g(y)

        if len(zero) < 2:
            r = 0
        else:
            Tp = mean(diff(t[zero[0:]]))
            Sp = std(diff(t[zero[0:]]))
            if Sp == 0:
                r = float('inf')
            else:
                r = Tp / (3 * Sp)
        return r

    # completar com os filtros
    y = y - mean(y)
    variance = var(y)

    n = y.size
    r = correlate(y, y, mode='full')[-n:]
    auto_co = r / (variance * n)
    r = reg(auto_co)

    if r > 1:
        flag = 1
    else:
        flag = 0
    return [flag, r]


def li(x, y, par):
    from numpy import argwhere, std, copy, zeros, arange, diff, nan, isnan, mean
    from scipy.fftpack import dct, idct

    def zero_g(y):
        from numpy import sign
        zero = []
        for i in range(1, y.size):
            if sign(y[i]) != sign(y[i - 1]):
                zero.append(i)
        return zero

    def reg(y):
        t = arange(0, len(y))
        zero = zero_g(y)

        if len(zero) < 2:
            r = 0
        else:
            Tp = mean(diff(t[zero[0:]]))
            Sp = std(diff(t[zero[0:]]))
            if Sp == 0:
                r = float('inf')
            else:
                r = Tp / (3 * Sp)
        return r

    def seg_select(y):
        # por tipo: geral, li, window
        seg = []
        ks, ke = nan, nan

        if y[0] != 0:
            ks = 0

        for i in range(1, y.size):
            if isnan(ks) & (y[i - 1] == 0) & (y[i] != 0):
                ks = i
            if (i + 4) < len(y):
                if (y[i + 1] == 0) & (y[i + 2] == 0) & (y[i + 3] == 0) & (y[i + 4] == 0):
                    ke = i
            if (not isnan(ks)) & (not isnan(ke)):
                if ke > ks:
                    seg.append(arange(ks, ke + 1))
                    ks = nan
                    ke = nan
        return seg

    low = 1
    up = 3

    # Apply DCT
    y_spc = dct(y - mean(y), norm='ortho')

    # Restricted to values above 1 std
    SL = argwhere(abs(y_spc) < low * std(y_spc))
    yl = copy(y_spc)
    yl[SL] = 0

    # Restricted to values above 3 std
    SL = argwhere(abs(y_spc) < up * std(y_spc))
    yu = copy(y_spc)
    yu[SL] = 0

    flags = [0]

    if sum(abs(yu)) != 0:
        seg = seg_select(yu)
        segl = seg_select(yl)

        # Test oscillation
        for i in range(0, len(seg)):
            for j in range(0, len(segl)):
                if max(y_spc[segl[j]]) == max(y_spc[seg[i]]):
                    # Oscila para o de maior desvio
                    yu = zeros(len(y_spc))
                    yu[seg[i]] = y_spc[seg[i]]
                    xu = idct(yu)
                    ru = reg(xu)

                    # Oscila para o de menor desvio
                    yl = zeros(len(y_spc))
                    yl[segl[j]] = y_spc[segl[j]]
                    xl = idct(yl)
                    rl = reg(xl)

                    # Se oscila para os dois desvios
                    if ru > 1 and rl > 1:
                        flags.append(1)

    return max(flags)
