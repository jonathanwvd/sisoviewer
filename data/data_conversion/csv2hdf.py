import pandas as pd
import datetime as dt
from pathlib import Path


def csv_to_hdf(path, name, time_stamp=False, time_stamp_form='%Y-%m-%d %H:%M:%S:%f', read_only=True):
    """
    Convert the csv files in standard format into hdf5 file useble by the tool

    Parameter
    ---------
    path: string
        Path to the dataset.
    name: string
        Name of the dataset. The default is the name of the csv folder.
    time_stamp_form:
        Form of the time-stamp. The default is %Y-%m-%d %H:%M:%S:%f.
    read_only: boolean
        True if the dataset can not be changed inside the tool. The default is true.
    """
    # Path as a pathlib instance
    path = Path(path)

    # read infos
    info = pd.read_excel(path / 'info.xlsx', index_col=0)
    tags = info.index.drop('default')

    # check the type of standard the files are following. If csv files in main folder, all_in is True
    files = [f.name for f in list(path.glob('*.csv'))]
    standard = 'same_file_standard' if bool(files) else 'separated_files_standard'
    measurements = [['MV', 'OP', 'PV', 'SP']]

    if standard == 'same_file_standard':
        store = pd.HDFStore(name + '.h5')

        for ind_t, t in enumerate(tags):
            print(f'Working on loop {t}. {ind_t + 1} of {len(tags)}')

            data = pd.read_csv(path / (t + '.csv'))
            if time_stamp:
                data = time_stamp_reader(data, time_stamp_form)
            data = data.set_index('Time')

            store.put(t, data)
            measurements.append(data.columns.values)

    else:
        store = pd.HDFStore(name + '.h5')

        for ind_t, t in enumerate(tags):
            print(f'Working on loop {t}. {ind_t + 1} of {len(tags)}')

            path_local = path / t
            measurements_local = list(path_local.glob('*.csv'))
            measurements.append([m.stem for m in measurements_local])

            for ind_m, m in enumerate(measurements_local):
                data = pd.read_csv(m)
                if time_stamp:
                    data = time_stamp_reader(data, time_stamp_form)
                data = data.set_index('Time')

                store.put(t + '/' + m.stem, data)

    # save loop info
    # complete NaN values with default
    for cl in info.keys():
        fill = info[cl][0]
        info[cl] = info[cl].fillna(fill)

    info['measurements'] = measurements
    store.put('info', info)

    # save general info
    general_info = {'standard_type': standard, 'time_stamp': time_stamp, 'read_only': read_only}
    general_info = pd.DataFrame(general_info, index=[1])
    store.put('general_info', general_info)
    store.close()


def new_dataset(path, name):
    """
    Create a new empty dataset to save data inside the tool

    Parameter
    ---------
    path: string
        Path to the info_new_dataset.xlsx file.
    name: string
        Name of the dataset.
    """
    # Path as a pathlib instance
    path = Path(path)
    path_ex = path / 'info_new_dataset.xlsx'

    store_new = pd.HDFStore(name + '.h5')
    info_new = pd.read_excel(path_ex, index_col=0)
    store_new.put('info', info_new)

    general_info = {'standard_type': 'same_file_standard', 'time_stamp': False, 'read_only': False}
    general_info = pd.DataFrame(general_info, index=[1])
    store_new.put('general_info', general_info)

    store_new.close()


def time_stamp_reader(data, form):
    Time = data['Time'].values
    Time = [dt.datetime.strptime(time, form) for time in Time]
    data['Time'] = Time
    return data
