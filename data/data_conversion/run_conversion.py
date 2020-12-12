"""
Examples of inputs to run the conversions
"""

# import local functions
# the import below may require modification depending on where you are executing the code. 
# the following commented lines will work if you are running from the folder of this file.
# from csv2hdf import csv_to_hdf, new_dataset
# from hdf2csv import hdf_to_csv
from data.data_conversion.csv2hdf import csv_to_hdf, new_dataset
from data.data_conversion.hdf2csv import hdf_to_csv


# # conversion from csv to hdf5
path = 'data/data_conversion/jelali_huang_to_csv/jelali_huang'
name = 'jelali_huang'
read_only = True
csv_to_hdf(path, name, read_only=read_only)


# # creates an empty dataset to save data
# path = ''
# name = 'saved_data'
# new_dataset(path, name)


# # conversion HDF into csv
# path = 'data/SISO-SAMP.h5'
# name = None
# hdf_to_csv(path, name)
