"""
Functions to add lines to the time domain plot.
"""

functions = {
    'mean_x': {
        'description': 'Add the arithmetic mean to the x axis. Run the function numpy.mean',
        'par': {
        },
    },
    'mean_y': {
        'description': 'Add the arithmetic mean to the y axis. Run the function numpy.mean',
        'par': {
        },
    },
    'ellipse_fit': {
        'description': 'fit and add an ellipse to the plot',
        'par': {
        },
    },
}


def mean_x(x, y, par):
    from numpy import mean, ones
    mn = mean(x)
    x_add = ones(len(x)) * mn
    y_add = y
    return x_add, y_add


def mean_y(x, y, par):
    from numpy import mean, ones
    mn = mean(y)
    y_add = ones(len(y)) * mn
    x_add = x
    return x_add, y_add


def ellipse_fit(x, y, par):
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

    def ellipse_center(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])

    def ellipse_angle_of_rotation2(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        if b == 0:
            if a > c:
                return 0
            else:
                return np.pi / 2
        else:
            if a > c:
                return np.arctan(2 * b / (a - c)) / 2
            else:
                return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    arc = 2
    R = np.arange(0, arc * np.pi+0.05, 0.05)
    a = fitEllipse(x, y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)

    a, b = axes
    x_add = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    y_add = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)

    return x_add, y_add
