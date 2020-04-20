# SISO Viewer

This application aims to facilitate the development and testing of CPM techniques. By using the application, researchers can verify weak points and work on modifications to improve the technique performance. The application was internally coded in Python and was designed based on the experience of the authors on CPM techniques development.

More information on https://www.ufrgs.br/gimscop/repository/siso-viewer/

![alt text](https://raw.githubusercontent.com/jonathanwvd/sisoviewer/master/assets/screenshot.png "screeshot")


## Installing and running the application

### Requirements
[Python 3.7](https://www.python.org/downloads/) or superior.

### 1. Clone GitHub repository
Download and extract the project repository or clone from GitHub page.

### 2. Download the libraries
Open Command Prompt (Windows users) or Terminal (Linux and macOS users), navigate to the project folder and run

`pip install -r requirements.txt`


> Make sure Python is in Windows Path ([solution](https://datatofish.com/add-python-to-windows-path/)).

> We recommend working with [Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/).

### 3. Open the application
In the project folder, run

`python app.py`

Wait until the server is set up and paste `http://localhost:8000/` on your browser.

## Downloading and loading the datasets
The datasets can be downloaded from the following [link](https://www.ufrgs.br/gimscop/repository/siso-viewer/datasets/).

Extract the dataset from the file and move the HDF5 files into the *data* folder in the project root. 
The application will recognize these files automatically after rerunning.

### Downloading and converting the dataset provided by Jelali et al. [1] 
This is a well-know dataset on CPM. To download the dataset, go to the [Jelali's book page](https://sites.ualberta.ca/~bhuang/Stiction-Book.htm). To convert the dataset to HDF5 format, move the _.mat_ file to the data/data_conversion/jelali_huang_to_csv folder and run (from the project root)

```
python data\data_conversion\jelali_huang_to_csv\mat_to_csv.py
python data\data_conversion\run_conversion.py
```

and then move the _jelali_huang.h5_ file from the _root_ to the _data_ folder

## Working with the application
1. **Load the data** - in the *Load data* section, select the dataset and the loop.
2. **Plot and process the data** - in the *Data processing* section, select the sampling time, plot the data, select the range and apply data processing.
3. **Analyze the data and techniques** - in the *Time domain*, *Frequency domain*, *Correlation*, and *Parametric plot* sections, 
the processed data can be analyzed and the techniques evaluated.
These sections are divided into two subsections. The *Add to plot* subsections contain functions that add new lines to the chart.
the *Evaluate from data* section contains functions that print outputs.

### Tips
1. Hover the question mark or the parameters name to see their description.
2. The charts are interactive. To have more information about the interactions, check [plotly documentation](https://plotly.com/python/).

## Adding new functions
New functions can easily be added to the application. These functions must follow a standard, so they can be read properly by the application.
The best way to follow the standard is by copying and changing one of the default functions, which are located in the *modules* folder. 
Functions for data processing, for example, are found on the *data_processing.py* file.

A function is divided into two sections. The first section presents the function description, parameters descriptions, default values, and types.
The second section presents the code that is run by the application. 
New functions must be copied to the corresponding Python file in the *modules* folder. For example, a function that adds a new line to the 
time domain chart must be added to the *data/modules/time_domain/add.py* file

## Adding new datasets
New datasets can also be added. These must be in the default HDF5 format supported by the application, which are generated from
CSV files by running the function *csv_to_hdf* in the *data/data_conversion/csv_to_hdf.py* file. Together with the CSV files, an Excel 
file with the loops information must be provided.

For writing the default CSV and Excel file, check the folder *data/data_conversion/jelali_huang_to_csv/jelali_huang* after running the 
command `python data\data_conversion\jelali_huang_to_csv\mat_to_csv.py` mentioned above.



## References
[1] JELALI, Mohieddine; HUANG, Biao (Ed.). **Detection and diagnosis of stiction in control loops: state of the art and advanced methods.** 
Springer Science & Business Media, 2009.
