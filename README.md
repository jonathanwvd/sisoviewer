# SISO Viewer

This application aims to facilitate the development and testing of CPM techniques. By using the application, researchers can verify weak points and work on modifications to improve the technique performance. The application was internally coded in Python and was designed based on the experience of the authors on CPM techniques development.

Tutorial video, citation, about and other information on https://www.ufrgs.br/gimscop/repository/siso-viewer/

![alt text](https://raw.githubusercontent.com/jonathanwvd/sisoviewer/master/assets/screenshot.png "screeshot")


## Installing and running the application

### Requirements
[Python 3.7](https://www.python.org/downloads/) or superior.

### 1. Clone GitHub repository
Download and extract the project repository or clone from GitHub page.

### 2. Download the libraries
Open Command Prompt (Windows users) or Terminal (Linux and macOS users), navigate to the project folder and run

`pip install -r requirements.txt`

> Make sure Python is in Windows Path ([solution](https://datatofish.com/add-python-to-windows-path/)) 

> We recommend working with [Virtual Environment](https://realpython.com/python-virtual-environments-a-primer/).

### 3. Open the application
In the project folder, run

`python app.py`

Click in the printed HTML link to open the application

## Downloading and loading the datasets
The datasets can be downloaded from the following [link](https://www.ufrgs.br/gimscop/repository/siso-viewer/datasets/).

Extract the dataset from the file and move the HDF5 files into the /data folder in the project root. The application will recognize these files automatically after opening.

### Downloadng and converting the datased provided by Jelali et al.
This is a well-know dataset on CPM. To download the dataset, go to the [Jelali's book page](https://sites.ualberta.ca/~bhuang/Stiction-Book.htm). To convert the dataset to HDF5 format, move the _.mat_ file to the data/data_conversion/jelali_huang_to_csv folder and run (from the project root)

`python data\data_conversion\jelali_huang_to_csv\mat_to_csv.py`

`python data\data_conversion\run_conversion.py`

and then move the _jelali_huang.h5_ file from the _root_ to the _data_ folder

## Working with the tool
1. **Load the data** - in the *Load data* section, select the dataset and the loop.
2. **Plot and process the data** - in the *Data processing* section, select the sampling time, plot the data, select the range and apply data processing.
3. **Analise the data and techniques** - in the *Time domain*, *Frequency domain*, *Correlation*, and *Parametric plot* sections, analise the data and CPM techniques.

## Adding new functions
New functions can be easily added to the tool. The 

## Adding new datasets
