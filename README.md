# SISO Viewer

This application aims to facilitate the development and testing of CPM techniques. By using the application, researchers can verify techniques weak points and work on improvements. The application was entirely coded in Python and was designed based on the experience of the authors on CPM techniques development.

More information on https://www.ufrgs.br/gimscop/repository/sisoviewer/

![alt text](https://raw.githubusercontent.com/jonathanwvd/sisoviewer/master/assets/screenshot.png "screeshot")

If you used this app in your work, please cite:
DAMBROS, Jônathan WV; TRIERWEILER, Jorge O.; FARENZENA, Marcelo. 
INDUSTRIAL DATASETS AND A TOOL FOR SISO DATA VISUALIZATION AND ANALYSIS. 
Computers & Chemical Engineering, p. 107198, 2020.

## Running the portable version (maybe not the latest version)

The portable version includes all that is necessary to run the application (Python, libraries, and data). To run the portable version:
1. [Download](https://www.ufrgs.br/gimscop/repository/siso-viewer/) and extract the *7zip* file.
2. Run the *run_app.bat* file.
3. Wait until the server is set up and paste `http://localhost:8050/` to your browser.

## Installing and running the application (latest version)

### Requirements
[Python 3.7](https://www.python.org/downloads/) or superior.

### 1. Clone GitHub repository
Download and extract GitHub repository or clone from GitHub page.

### 2. Download the libraries
Open Command Prompt (Windows users) or Terminal (Linux and macOS users), navigate to the project folder and run

`pip install -r requirements.txt`


> Make sure Python is in Windows Path ([solution](https://datatofish.com/add-python-to-windows-path/)).

> We recommend working with [Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/).

### 3. Open the application
In the project folder, run

`python app.py`

Wait until the server is set up and paste `http://localhost:8050/` to your browser.

## Downloading and loading the datasets
The datasets can be downloaded from the following [link](https://www.ufrgs.br/gimscop/repository/siso-viewer/datasets/).

Extract the datasets from the downloaded file and move the HDF5 files into the *data* folder in the project root. 
The application will recognize these files automatically after rerunning.

### Downloading and converting the dataset provided by Jelali et al. [1] 
This is a well-know dataset on CPM. To download the dataset, go to the [Jelali's book page](https://sites.ualberta.ca/~bhuang/Stiction-Book.htm). To convert the dataset to HDF5 format, move the _.mat_ file to the data/data_conversion/jelali_huang_to_csv folder.

If using the portable version, run the run_jelali_conversion.bat file in the project root.
If not, run (from the project root)

```
python data\data_conversion\jelali_huang_to_csv\mat_to_csv.py
python data\data_conversion\run_conversion.py
```

Finnaly, move the _jelali_huang.h5_ file from the project root to the _data_ folder.

## Working with the application
1. **Load the data** - in the *Load data* section, select the dataset and the loop.
2. **Plot and process the data** - in the *Data processing* section, select the sampling time, plot the data, select the range and apply data processing.
3. **Analyze the data and techniques** - in the *Time domain*, *Frequency domain*, *Correlation*, and *Parametric plot* sections, 
the processed data can be analyzed and the techniques evaluated.
These sections are divided into two subsections. The *Add to plot* subsection contains functions that add new lines to the chart.
the *Evaluate from data* section contains functions that print outputs.

### Tips
1. Hover the question mark or the parameters name to see their description.
2. The charts are interactive. To have more information about the interactions, check [plotly documentation](https://plotly.com/python/).

## Adding new functions
New functions can easily be added to the application. These functions must follow a standard, so they can be read properly by the application.
The best way to follow the standard is by copying and changing one of the default functions, which are located in the *modules* folder. 
Functions for data processing, for example, are found on the *data_processing.py* file.

A function is divided into two sections. The first section contains the function description, parameters descriptions, default values, and types.
The second section contains the code that is run by the application. 
New functions must be copied to the corresponding Python file in the *modules* folder. For example, a function that adds a new line to the time domain chart must be added to the *data/modules/time_domain/add.py* file

## Adding new datasets
New datasets can also be added. These must be in the default HDF5 format supported by the application, which are generated from
CSV files by running the function *csv_to_hdf* in the *data/data_conversion/csv_to_hdf.py* file. Together with the CSV files, an Excel 
file with the loops information must be provided.

To write the default CSV and Excel files, check the Jelali's dataset example in folder *data/data_conversion/jelali_huang_to_csv/jelali_huang* after running the 
`python data\data_conversion\jelali_huang_to_csv\mat_to_csv.py` mentioned above.

## Converting HDF5 into CSV
To convert an HDF5 file into CSV files follow the example on `data/data_conversion\run_ceonversion.py`.

## References
[1] JELALI, Mohieddine; HUANG, Biao (Ed.). **Detection and diagnosis of stiction in control loops: state of the art and advanced methods.** 
Springer Science & Business Media, 2009.
