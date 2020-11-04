import numpy as np

def zero_cross(y, zero=None):
    "Regularidadeno tempo"
    y = y - np.mean(y)
    r = reg(y)

    if r > 1:
        flag = 1
    else:
        flag = 0
    return [flag, r]


def miao(y):
    n = y.size
    r = np.correlate(y, y, mode='full')[-n:]
    auto_co = r / (np.var(y) * n)

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


def thornhill(y):
    # completar com os filtros
    y = y - np.mean(y)
    variance = np.var(y)

    n = y.size
    r = np.correlate(y, y, mode='full')[-n:]
    auto_co = r / (variance * n)
    r = reg(auto_co)

    if r > 1:
        flag = 1
    else:
        flag = 0
    return [flag, r]


def forsman(y, alpha=0.6, gamma=0.75):
    y = y - np.mean(y)

    zero = zero_g(y)

    summ = []
    dis = []

    for i in range(1, len(zero)):
        summ.append(np.sum(y[zero[i - 1]:zero[i]]))
        dis.append(zero[i] - zero[i - 1])

    count = 0
    for i in range(2, len(summ)):
        summ_r = summ[i] / summ[i - 2]
        dis_r = dis[i] / dis[i - 2]
        if summ_r > alpha and summ_r < 1 / alpha and dis_r > gamma and dis_r < 1 / gamma:
            count += 1

    N = len(summ) - 2
    h = count / N

    if h > 0.4:
        flag = 1
    else:
        flag = 0
    return [flag, h]


def li(y):
    from scipy.fftpack import dct, idct
    low = 1
    up = 3

    # Apply DCT
    y_spc = dct(y - np.mean(y), norm='ortho')

    # Restricted to values ​​above 1 std
    SL = np.argwhere(abs(y_spc) < low * np.std(y_spc))
    yl = np.copy(y_spc)
    yl[SL] = 0

    # Restricted to values ​​above 3 std
    SL = np.argwhere(abs(y_spc) < up * np.std(y_spc))
    yu = np.copy(y_spc)
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
                    yu = np.zeros(len(y_spc))
                    yu[seg[i]] = y_spc[seg[i]]
                    xu = idct(yu)
                    ru = reg(xu)

                    # Oscila para o de menor desvio
                    yl = np.zeros(len(y_spc))
                    yl[segl[j]] = y_spc[segl[j]]
                    xl = idct(yl)
                    rl = reg(xl)

                    # Se oscila para os dois desvios
                    if ru > 1 and rl > 1:
                        flags.append(1)

    return [max(flags), sum(flags)]


def karra(y):
    # Step 1
    y_n = norm_one(y)

    # Step 2
    y_filt = cut_freq(y_n, [0.02, 0.99])

    # Step 3 e 4
    # não é 0.1 desvios padroes, é 0.1 normalizado
    y_seg = split_signal(y_filt, 0.1, thre_type='abs', seg_type='general')

    # Step 5
    regu = np.zeros(len(y_seg))
    decay = np.zeros(len(y_seg))

    for ind, i in enumerate(y_seg):
        regu[ind] = thornhill(i)[1]
        decay[ind] = miao(i)[1]

    # Step 6
    flags = np.zeros(len(y_seg), dtype=np.int)
    for ind, i in enumerate(y_seg):
        if decay[ind] > 0.5 and regu[ind] > 1:
            flags[ind] = 1
        else:
            flags[ind] = 0

    if len(flags) == 0:
        flag = 0
    else:
        flag = max(flags)
    return [flag, np.sum(flags)]


def depizzol(y):
    # Step 1
    y_n = norm_one(y)

    # Step 2
    low = 2 / y.size
    y_filt = cut_freq(y_n, [low, 0.99])

    # Step 3 e 4
    y_seg = split_signal(y_filt, 2, thre_type='std', seg_type='general', window=True)

    # Step 5
    regu = np.zeros(len(y_seg))
    decay = np.zeros(len(y_seg))

    for ind, i in enumerate(y_seg):
        regu[ind] = thornhill(i)[1]
        decay[ind] = miao(i)[1]

    # Step 6
    flags = np.zeros(len(y_seg), dtype=np.int)
    for ind, i in enumerate(y_seg):
        if decay[ind] > 0.5 and regu[ind] > 1:
            flags[ind] = 1
        else:
            flags[ind] = 0

    if len(flags) == 0:
        flag = 0
    else:
        flag = max(flags)
    return [flag, np.sum(flags)]


######################################
# Funções adicionais
def norm_one(y):
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))
    return y


def norm_std(y):
    y = (y - np.mean(y)) / (np.std(y))
    return y


def norm_spec(y):
    y = y / max(y)
    return y


def zero_g(y):
    from numpy import sign
    zero = []
    for i in range(1, y.size):
        if sign(y[i]) != sign(y[i - 1]):
            zero.append(i)
    return zero


def peak_g(y):
    peaks = [0]
    for i in range(2, len(y)):
        if np.sign((y[i]) - y[i - 1]) != np.sign((y[i - 1]) - y[i - 2]):
            peaks.append(i)
    return peaks


def Mm(y):
    peaks_M = []
    peaks_m = []

    for i in range(1, len(y)):
        if np.sign((y[i]) - y[i - 1]) > 0 and np.sign((y[i - 1]) - y[i - 2]) < 0:
            peaks_m.append(i)
        if np.sign((y[i]) - y[i - 1]) < 0 and np.sign((y[i - 1]) - y[i - 2]) > 0:
            peaks_M.append(i)
    return peaks_M, peaks_m


def reg(y):
    t = np.arange(0, len(y))
    zero = zero_g(y)

    if len(zero) < 2:
        r = 0
    else:
        Tp = np.mean(np.diff(t[zero[0:]]))
        Sp = np.std(np.diff(t[zero[0:]]))
        if Sp == 0:
            r = float('inf')
        else:
            r = Tp / (3 * Sp)
    return r


def sparseness(y):
    N = int(np.ceil(y.size / 2))
    x = np.abs(np.fft.fft(y))[:N]

    sparse = (np.sqrt(N) - (np.sum(np.abs(x)) / np.sqrt(np.sum(x ** 2)))) / (np.sqrt(N) - 1)
    return sparse


def power(y, rang):
    # Usar power spectrum
    N = int(np.ceil(y.size / 2))
    x = np.abs(np.fft.fft(y))[:N]
    po = np.sum(x[rang[0]:rang[1] + 1]) / np.sum(x)
    return po


def seg_select(y, seg_type='li'):
    # por tipo: geral, li, window

    if seg_type == 'li':
        seg = []
        ks, ke = np.nan, np.nan

        if y[0] != 0:
            ks = 0

        for i in range(1, y.size):
            if np.isnan(ks) & (y[i - 1] == 0) & (y[i] != 0):
                ks = i
            if (i + 4) < len(y):
                if (y[i + 1] == 0) & (y[i + 2] == 0) & (y[i + 3] == 0) & (y[i + 4] == 0):
                    ke = i
            if (not np.isnan(ks)) & (not np.isnan(ke)):
                if ke > ks:
                    seg.append(np.arange(ks, ke + 1))
                    ks = np.nan
                    ke = np.nan

    elif seg_type == 'general':
        seg = []
        ks, ke = np.nan, np.nan

        if y[0] != 0:
            ks = 0

        for i in range(1, y.size):
            if np.isnan(ks) & (y[i - 1] == 0) & (y[i] != 0):
                ks = i
            if (i + 1) < len(y):
                if (y[i + 1] == 0):
                    ke = i
            if (not np.isnan(ks)) & (not np.isnan(ke)):
                if ke > ks:
                    seg.append(np.arange(ks, ke + 1))
                    ks = np.nan
                    ke = np.nan

    return seg


def split_signal(y, thre=1, thre_type='std', seg_type='li', window=False):
    y = norm_one(y)

    N = y.size // 2 + 1
    y_spec = np.fft.fft(y)
    y_spec_f = abs(y_spec)[:N]

    # Para o método de Depizzol utilizar janela Gaussiana
    if window == True:
        N = y_spec_f.size / 16  # Esse parâmetro não foi dado
        alpha = 6
        n = np.arange(-N / 2 + 1, N / 2)
        L = N - 1
        win = np.exp(-1 / 2 * (alpha * n / (L / 2)) ** 2)
        y_spec_f = np.convolve(win, y_spec_f, 'same')

    # Zerar coeficientes não-significativos
    if thre_type == 'std':
        SL = np.argwhere(abs(y_spec_f) < thre * np.std(y_spec_f))
        y_spec_f[SL] = 0

    elif thre_type == 'abs':
        y_spec_f = norm_spec(abs(y_spec_f))
        SL = np.argwhere(abs(y_spec_f) < thre)
        y_spec_f[SL] = 0

    # Segmenta o espectro
    seg = seg_select(y_spec_f, seg_type=seg_type)

    x = []
    for ind, i in enumerate(seg):
        y_seg = np.zeros(len(y_spec), dtype='complex128')
        y_seg[i] = y_spec[i]
        y_seg[-i] = y_spec[-i]
        x.append(np.fft.ifft(y_seg).real)
    return x


def cut_freq(y, bounds):
    bounds_nn = np.asarray(bounds) / 2

    N = y.size
    x = np.fft.fft(y)

    ini = np.int(np.ceil(bounds_nn[0] * N))
    fim = np.int(np.ceil(bounds_nn[1] * N))

    x[:ini + 1] = 0
    x[y.size - ini:] = 0

    x[fim:N // 2 - 1] = 0
    x[N // 2 - 1:N - fim + 1] = 0

    y = np.fft.ifft(x).real
    return y
