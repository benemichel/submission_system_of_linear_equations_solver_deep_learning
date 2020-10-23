"""
used as command line application

sle_solver:
1. accepts an user scanned image of an sle
2. parses the image to get images representing the single
    mathematical symbols in the equations of the sle
3. uses a pretrained pytroch convolutional neural net to label
    all of these images
4. creates a digital representation of the sle and solves it
5. outputs the solution if available to the user


Usage:
$python sle_solver.py .\data\example.jpg .\data\model_dict.pth

Arguments:
image_path [str] - path to image file
model_path [str] - path to pth model file
"""

print("### Hello from the SLE Solver command line tool ###")
import numpy as np
from PIL import Image
import os
import argparse
import image_parsing_helpers as pars_helpers
import image_classification
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
import torch
import time

#add arguments
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to the image file")
parser.add_argument("model_path", help="Pytorch model file")
args = parser.parse_args()

#extract arguments
image_path = args.image_path
model_path = args.model_path

PARSED_DIRECTORY = './data/sle_symbols_parsed_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

#parse the images
print("### parsing image ###")
src = cv.imread(image_path)
image = src.copy()
print("#transforming image")
image_transformed = pars_helpers.transform_image(image, False)
print("#cleaning image from fragments")
image_cleaned = pars_helpers.remove_fragments(image_transformed, fraction = 0.2, plot = False )
print("#searching for borders")
borders_x, borders_y = pars_helpers.find_borders(image_cleaned, threshold = 1, dilation = 30, padding = 1, plot = False)
print("#extracting equations as sections")
sections = pars_helpers.extract_sections(image_cleaned, borders_x, borders_y, plot = False)
print("#parsing single symbols and storing information in DataFrame")
figures_df = pars_helpers.create_figures_df(sections, 3, 30, 10, 1)
print("#remove empty rows")
figures_df['image'] = figures_df['image'].apply(pars_helpers.remove_empty_rowscols)
print("#resize images")
figures_df['image'] = figures_df['image'].apply(pars_helpers.resize_28x28)
print("#add padding to get quadratic image")
figures_df['image'] = figures_df['image'].apply(pars_helpers.pad_fill, args = ((28,28), 0))
print("#saving the parsed images to subfolder")
figures_df.apply(lambda row : pars_helpers.save_image_from_df(row, PARSED_DIRECTORY), axis = 1)
print(f"### single images were parsed and saved to directory {PARSED_DIRECTORY}")


#image classification
print("### image classifcation ###")
print("#loading convolutional neural net")
conv = image_classification.ConvNN(28, 256, 16, 0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
conv = conv.to(device)
conv.load_state_dict(torch.load(model_path, map_location = device))
print('#create dataloader')
dataset = image_classification.create_pytorch_dataset_from_folder(PARSED_DIRECTORY)
dataloader = torch.utils.data.DataLoader(dataset)
print('#perform inference')
class_mappings = {'+': 0, '-': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, '=': 12, 'x': 13, 'y': 14, 'z': 15}
class_mappings_reverse = {0: '+', 1: '-', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9', 12: '=', 13: 'x', 14: 'y', 15: 'z'}
predictions = image_classification.predict_sle_symbols(conv, dataloader, class_mappings, class_mappings_reverse )
print("#create sle and calculate solution")
line_symbols, A, b, var_list = image_classification.create_sle_parts(predictions)

print("#equations found by inference")


x = image_classification.calc_solution(A, b)

#final outputs
print("### results ###")
for line in line_symbols:
    print("".join(line))
print(f"A {A}")
print(f"b {b}")
print("### the suggested solution is: ###")
print(f"x {x}")
