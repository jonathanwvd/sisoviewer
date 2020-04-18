"""
Run csv2hdf function to generate the HDF5 file.
"""

# import local functions
from csv2hdf import csv_to_hdf, new_dataset

# jelali_huang dataset
path = 'jelali_huang_to_csv/jelali_huang'
name = 'jelali_huang'
read_only = True
csv_to_hdf(path, name, read_only=read_only)


# # saved_data dataset
# path = ''
# name = 'saved_data'
# new_dataset(path, name)
