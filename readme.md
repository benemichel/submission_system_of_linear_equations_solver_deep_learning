# System of linear equations solver with deep learning

## Packages
non standard Python libraries used:
* cv2
* numpy
* pandas
* matplotlib
* PIL
* torchvision
* torch

## Data
The kaggle mathematical images dataset can be downloaded from: https://www.kaggle.com/xainano/handwrittenmathsymbols
Data directory has to be set accordingly in the create_synthetic_sle_datasets notebook 
and in the classification_with_convolutional_net notebook  in code cell 2
`
#set accordingly

KAGGLE_SYMBOL_DIRECTORY = 'C:\\Users\\bened\\ML_datasets\\kaggle_handwritten_symbols'
`

Additionally some example files are included but can also be created executing the notebooks.

## Command line tool
Tool can be started according the following logic (no setup required):

` python sle_solver.py .\data\sle_synthetic_2020-10-21-18-14-16985.jpg .\data\model_dict.pth `

## Setup
Download and unzip kaggle dataset (see Data).

## Additional information
Notebooks were saved after a run with a dataset of only 30 synthetically created SLE images to keep data size low. Evaluation was performed with a dataset of 2000 images.
More information can be found in project documentation pdf file.