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

`pip3 install -r requirements.txt`

> Make sure Python is in Windows Path ([solution](https://datatofish.com/add-python-to-windows-path/)) 

> We recommend working with [Virtual Environment](https://realpython.com/python-virtual-environments-a-primer/).

### 3. Open the application
In the project folder, run

`python3 app.py`

Click in the printed HTML link to open the application

## Downloading and loading the dataset
The dataset can be downloaded from the following [link](https://www.ufrgs.br/gimscop/repository/siso-viewer/datasets/).

Extract the dataset from the file and move the HDF5 files into the /data folder in the project root. The application will recognize these files automatically after opening.
