import pandas as pd
import datetime as dt
from pathlib import Path
import os


def hdf_to_csv(path, name=None):
    """
    Convert the HDF5 files in csv

    Parameter
    ---------
    path: string
        Path to the HDF5 file.
    name: string
        Name of the dataset. The default is the name of the HDF5 file.
    """

    path = Path(path)
    csv_path = Path('data/data_conversion/data_csv/')

    if name is None:
        name = path.stem

    if not os.path.isdir(csv_path / name):
        os.mkdir(csv_path / name)

    store = pd.HDFStore(path)

    info = store['info']
    general_info = store['general_info']

    for l in store.keys():
        if l in ['/general_info', '/info']:
            pass

        else:
            data_l = store[l]
            if general_info['standard_type'].values[0] == 'same_file_standard':
                data_l.to_csv(csv_path / f'{name}\\{l[1:]}.csv')

            else:
                loop, var = l.split('/')[1:]
                if not os.path.isdir(csv_path / f'{name}\\{loop}'):
                    os.mkdir(csv_path / f'{name}\\{loop}')

                data_l.to_csv(csv_path / f'{name}\\{loop}\\{var}.csv')

    info.to_excel(csv_path / f'{name}\\info.xlsx')
