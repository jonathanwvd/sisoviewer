# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
import os
from pathlib import Path

os.chdir(Path('data/data_conversion/jelali_huang_to_csv/'))

# load data
data = sio.loadmat('isdb10')

ind_book = ['BAS', 'CHEM', 'PAP', 'POW', 'MIN', 'MET', 'TEST']
var = ['PV', 'OP', 'SP', 't']
info = ['Comments', 'BriefComments', 'Type', 'Ts']

# prepare dictionary to save information
info_dict = {}
for i in info:
    info_dict[i] = []

if not os.path.isdir('jelali_huang'):
    os.mkdir('jelali_huang')

os.chdir(Path('jelali_huang'))

# get industry type
ind = data['cdata'][0][0].dtype.names
for ind_i, i in enumerate(ind):
    # get loops in industry type
    loop = data['cdata'][0][0][i][0][0].dtype.names
    for ind_l, l in enumerate(loop):
        data_l = data['cdata'][0][0][i][0][0][l]
        dict_l = {}
        # for each variable
        for v in var:
            data_l = data['cdata'][0][0][i][0][0][l][0][0][v].squeeze()
            dict_l[v] = data_l

        # loop without time information
        if (ind_i == 2) and (ind_l == 6):
            dict_l['t'] = np.arange(0, 0.2 * len(dict_l['OP']), 0.2)

        df = pd.DataFrame(dict_l)
        df = df.rename(columns={'t': 'Time'})
        df = df.set_index('Time')

        # save csv
        loop_name = ind_book[ind_i] + str(ind_l + 1)
        df.to_csv(loop_name + '.csv')
