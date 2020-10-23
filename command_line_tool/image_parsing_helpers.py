import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os
from PIL import Image

def transform_image(image, plot = False):
    """convert to grayscale for thresholding and then to binary image"""
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, image_threshed = cv.threshold(image_gray, 150, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

    if plot:
        fig, axes = plt.subplots(1,3, figsize = (20,40))
        axes[0].imshow(image)
        axes[0].set_title("original image")
        axes[1].imshow(image_gray, cmap = "gray")
        axes[1].set_title("grayscale image")
        axes[2].imshow(image_threshed, cmap = "gray")
        axes[2].set_title("binary image")
        plt.show()

    print(f"image_orig shape: {image.shape}")
    print(f"image_gray shape: {image_gray.shape}")
    print(f"image_threshed shape: {image_threshed.shape}")
    print(f"unique pixel values: {np.unique(image_threshed)}")

    return image_threshed

def remove_fragments(image, fraction = 0.1, plot = False, debug = False):
    """removes unwanted ragments from an image identified by their area in
    relation to the mean contour area of the whole image
    params:
        fraction: fraction of mean contour area used as threshold
    """

    print("only parent contours are inspected")
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #show all found contours
    image_all_contours = image.copy()
    image_all_contours = np.expand_dims(image_all_contours, axis=2).repeat(3, axis=2)
    for k, _ in enumerate(contours):
        image_all_contours = cv.drawContours(image_all_contours, contours, k, (0, 125, 255), math.ceil(image.shape[0]/50))

    #create prefilled mask
    mask = np.ones(image.shape[:2], dtype="uint8")

    #show small contours
    contour_areas = []
    power = 1
    for c, contour in enumerate(contours):
        contour_areas.append(cv.contourArea(contour) / image.shape[0] / image.shape[1])

    contour_areas_p = fraction * np.mean(np.array(contour_areas))

    if debug:
        print(f" {fraction}% of mean used to cut areas below this value ({contour_areas_p})")
        print(f"{power} power of contour area used for thresholding")

    image_small_contours = image.copy()
    image_small_contours = np.expand_dims(image_small_contours, axis=2).repeat(3, axis=2)

    dropped_indices = []
    for c, contour in enumerate(contours):
        if((cv.contourArea(contour) / image.shape[0] / image.shape[1]) ** power < contour_areas_p):
            dropped_indices.append(c)
            image_small_contours = cv.drawContours(image_small_contours, contours, c, (0, 0, 255), math.ceil(image.shape[0]/50))
            #add the unwanted contours
            cv.drawContours(mask, [contour], -1, 0, -1)
    print(f"indices of areas dropped {dropped_indices}")

    #apply the mask
    image_cleaned = cv.bitwise_and(image, image, mask = mask)

    if plot:
        plt.boxplot(np.array(contour_areas) ** power)
        plt.axhline(y=contour_areas_p)
        plt.title(f"distribution of contour areas power {power}")
        plt.show()

        fig, axes = plt.subplots(1,4, figsize = (30,10))
        axes[0].imshow(image)
        axes[0].set_title("original")
        axes[1].imshow(image_all_contours)
        axes[1].set_title("all contours")
        axes[2].imshow(image_small_contours)
        axes[2].set_title("contours with area below threshold (blue)")
        axes[3].imshow(image_cleaned)
        axes[3].set_title("cleaned")
        plt.show()

    return image_cleaned


def find_borders(image, threshold = 1, dilation = 0, padding = 0, plot = False):
    """find borders in an image. An upper border is a line (one value in histogram)
    where the value is lower than a predefined threshold `threshold` and
    the next line has a value > `threshold`
    The image can be dilated. In combination with a higher threshold value dilation can
    ensure that only borders dividing very dense areas from nearly blank areas are found.
    Otherwise fractals in blank areas would create borders too.
    params:
        image: image array
        threshold: threshold value see above
        dilation: dilation steps to perform before identifiying borders
        padding: amount of pixels to add to the borders (increases later extracted line images)
    """

    image_local = image.copy()
    #only dilate in x direction
    kernel = np.ones([1,2])
    image_dilated_x = cv.dilate(image_local, kernel, iterations = dilation)

    #only dilate in y direction
    kernel = np.ones([2,1])
    image_dilated_y = cv.dilate(image_local, kernel, iterations = dilation)

    hist_row = cv.reduce(image_dilated_x, 1, cv.REDUCE_AVG).reshape(-1)
    hist_col = cv.reduce(image_dilated_y, 0, cv.REDUCE_AVG).reshape(-1)

    borders_0_row = [i - padding for i in range(len(hist_row)-1)  if hist_row[i] <= threshold and hist_row[i+1] >  threshold]
    borders_1_row = [i + padding for i in range(len(hist_row)-1)  if hist_row[i] > threshold and hist_row[i+1] <= threshold]
    print(borders_0_row)
    print(borders_1_row)

    borders_0_col = [i for i in range(len(hist_col)-1)  if hist_col[i] <= threshold and hist_col[i+1] > threshold]
    borders_1_col = [i for i in range(len(hist_col)-1)  if hist_col[i] > threshold and hist_col[i+1] <= threshold]

    #if dimension == 0 and len(borders_0) > 1:
    if len(borders_0_col) > 1:
        borders_0_col = [borders_0_col[0]]
        borders_1_col = [borders_1_col[-1]]

    #print(f"borders found for dimension {dimension} (0: vertical borders, 1 : horizontal borders)")
    print(f"{len(list(zip(borders_0_row, borders_1_row)))} row border pairs were found")
    print(list(zip(borders_0_row, borders_1_row)))
    print(f"{len(list(zip(borders_0_col, borders_1_col)))} col border pairs were found")
    print(list(zip(borders_0_col, borders_1_col)))

    borders_all_row = borders_0_row + borders_1_row
    borders_all_col = borders_0_col + borders_1_col
    image_borders_row = image.copy()
    image_borders_col = image.copy()
    image_borders_all = image.copy()

    line_width = math.ceil(image.shape[1]/500)
    #print(line_width)
    for border in borders_all_row:
        for i in range(line_width):
            image_borders_row[min(image.shape[0] - 1, border + i),:] = 128
            #image_borders_all[max(image.shape[0]border + i,:] = 128

    for border in borders_all_col:
        for i in range(line_width):
            image_borders_col[:, min(image.shape[1] - 1, border + i)] = 128
            #image_borders_all[:,border + i] = 128

    #plotting
    if plot:
        fig, axes = plt.subplots(3,3, figsize=(15, 15))

        axes[0][1].barh(range(len(hist_row)), hist_row)
        axes[1][0].bar(range(len(hist_col)), hist_col)
        axes[0][0].imshow(image_local, aspect = "auto")
        axes[0][2].imshow(image_borders_row, aspect = "auto")
        axes[2][0].imshow(image_borders_col, aspect = "auto")
        axes[1][1].imshow(image_borders_all, aspect = "auto")
        axes[1][2].imshow(image_dilated_x, aspect = "auto")
        axes[2][1].imshow(image_dilated_y, aspect = "auto")
        plt.show()

    assert len(borders_0_row) + len(borders_1_row), 'no row borders found'
    assert len(borders_0_row) == len(borders_1_row), 'uneven number of row borders'
    assert len(borders_0_col) + len(borders_1_col), 'no col borders found'
    assert len(borders_0_col) == len(borders_1_col), 'uneven number of col borders'

    return (list(zip(borders_0_col,borders_1_col)), list(zip(borders_0_row,borders_1_row)))

def extract_sections(image, borders_x, borders_y, plot = False):
    """
    params:
        image: image array
        borders_x, boders_y: pixel values as locations of lower/upper and
        left/right borders used for cutting the sections
    returns:
        list of section coordinates as tuples (y1, y2, x1, x2)
    """
    line_width = math.ceil(image.shape[1]/200)
    image_borders = image.copy()
    x1, x2 = borders_x[0]
    for i in range(line_width):
        image_borders[:, min(x1 + i, image.shape[1] - 1)] = 128
        image_borders[:, min(x2 + i, image.shape[1] - 1)] = 128

    sections = []

    for borders in borders_y:
        y1, y2 = borders
        sections.append(image[y1:y2,x1:x2])
        for i in range(line_width):
            image_borders[min(y1 + i, image.shape[0] - 1),:] = 128
            image_borders[min(y2 + i, image.shape[0] - 1),:] = 128

    #plot
    if plot:
        fig = plt.figure(figsize = (15, 5))
        grid = plt.GridSpec(len(borders_y), 3, wspace = 0.4, hspace = 0.3, figure = fig)
        plt.subplot(grid[:len(borders_y), 0])
        plt.imshow(image, aspect = "auto")
        plt.subplot(grid[:len(borders_y), 1])
        plt.imshow(image_borders, aspect = "auto")
        for i in range(len(borders_y)):
            plt.subplot(grid[i, 2])
            plt.imshow(sections[i], aspect = "auto")

        plt.show()

    return sections

def extract_figures(image, dila_iterations_x = 1, dila_iterations_y = 1, contour_thresh = 1, pixel_increase = 0, plot = False):
    """ extract a single image for each symbol in the line images. The areas resembling the symbols
    are found via contours (borders of areas of same pixel values)
    params:
        image: image array
        dila_iterations: number of dilation iterations used
        contour_tresh: minimum contour area a contour must have to be extracted
        pixel_increase: some padding for the extracted image
    returns: list of 2D image matrices sorted by their horizontal position in the image
    """

    kernel_x = np.ones([1,2])
    image_dilation = cv.dilate(image, kernel_x, iterations = dila_iterations_x)

    kernel_y = np.ones([2,1])
    image_dilation = cv.dilate(image_dilation, kernel_y, iterations = dila_iterations_y)

    #cv.RETR_EXTERNAL, only parents, all childs are discarded
    contours,hierarchy = cv.findContours(image_dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    images_figures = []
    images_coords = []
    image_height, image_width  = image.shape

    for contour in contours:

        if(cv.contourArea(contour)**2 > contour_thresh):
            x_coords = [coord[0][0] for coord in contour]
            y_coords = [coord[0][1] for coord in contour]

            x1 = max(min(x_coords) - pixel_increase, 0)
            x2 = min(max(x_coords) + pixel_increase, image_width)
            y1 = max(min(y_coords) - pixel_increase, 0)
            y2 = min(max(y_coords) + pixel_increase, image_height)


            images_figures.append(image[y1:y2,x1:x2])
            images_coords.append([x1,x2,y1,y2])

    print(f"{len(images_figures)} figures found")

    #sort
    x1s = [coord[0] for coord in images_coords]
    print(x1s)
    zipped = sorted(zip(x1s,images_figures))
    print(len(x1s))
    print(len(images_figures))
    images_figures_sorted = [image for _, image in (sorted(zip(x1s,images_figures)))]

    if plot:
        plt.imshow(image_dilation)
        plt.show()
        fig, axes = plt.subplots(1,len(images_figures_sorted))
        for i, figure  in enumerate(images_figures_sorted):
            axes[i].imshow(figure)

        plt.show()

    return images_figures_sorted

def create_figures_df(sections, dila_iterations=1, contour_thresh = 1, pixel_increase = 0, plot = False):
    """create a dataframe containing position (line and position in line) and
    image array data for each symbol
    for more information see extract_figures function
    params:
        sections: borders of sections to be extracted
        dila_iterations: number of dilation iterations used
        contour_tresh: minimum contour area a contour must have to be extracted
        pixel_increase: some padding for the extracted image
    returns:
        pandas Dataframe
    """

    images = []
    positions = []
    lines = []
    for (i, section) in enumerate(sections):
        images_figures_sorted = extract_figures(section, dila_iterations, contour_thresh, pixel_increase,  plot)
        for (j,one_image) in enumerate(images_figures_sorted):
            positions.append(j)
            lines.append(i)
            images.append(one_image)
    figures_df = pd.DataFrame( {'line' : lines, 'position' : positions, 'image' : images })
    figures_df.head()

    return figures_df

def remove_empty_rowscols(image,  plot = False):
    """removes all completely empty lines and columns at the edge of an image (padding)"""

    image_orig = image.copy()

    threshold = 2
    #first filled row
    row_0 = [i for i in range(0, image.shape[0]-1) if np.sum(image[i, :]) <= threshold and np.sum(image[i+1, :]) > threshold]
    row_1 = [i for i in range(0, image.shape[0]-1) if np.sum(image[i, :]) >= threshold and np.sum(image[i+1, :]) < threshold]

    if row_0:
        row_0 = min(row_0)
        image = np.delete(image,range(0, row_0),0)

    if row_1:
        row_1 = max(row_1)
        image = np.delete(image,range(row_1,image.shape[0]),0)

    col_0 = [i for i in range(0, image.shape[1]-1)  if np.sum(image[:, i]) <= threshold and np.sum(image[:, i + 1]) > threshold]
    col_1 = [i for i in range(0, image.shape[1]-1)  if np.sum(image[:, i]) >= threshold and np.sum(image[:, i + 1]) < threshold]

    if col_0:
        col_0 = min(col_0)
        image = np.delete(image, range(0, col_0), 1)

    if col_1:
        col_1 = max(col_1)
        image = np.delete(image, range(col_1, image.shape[1]), 1)

    if plot:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(image_orig)
        axes[1].imshow(image)
        plt.show()

    return image


def resize_28x28(image, plot = False):
    """resize to 28x28 pixel image preserving asepct ratio"""

    image_orig = image.copy()

    #important: preserve AR! -> resize dimension with higher size to 20, other will be <=20
    #ascpect ratio = width/height
    n_cols = image.shape[1]
    n_rows = image.shape[0]

    n_rows_new = 28
    n_cols_new = 28

    if (n_cols > n_rows):
        ar = float(n_cols) / n_rows
        n_rows_new = int(n_cols_new / ar)
    else:
        ar = float(n_cols) / n_rows
        n_cols_new = int(n_rows_new * ar)

    image = cv.resize(image,(n_cols_new, n_rows_new))

    ret, image = cv.threshold(image, 200, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    if plot:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(image_orig)
        axes[1].imshow(image)
        plt.show()

    return image

def pad_fill(image, target_size, fill_value = 0, plot = False):
    """add padding to resize to a quadratic image file
    params:
        image: image array
        target_size: width/height of returned image
        fill_value: added pixels will be filled with this value
    """
    image_orig = image.copy()
    target_mat = np.full(
        shape=target_size,
        fill_value=fill_value,
        dtype=np.int)

    pad_x = (target_size[1] - image.shape[1]) / 2.0
    pad_y = (target_size[0] - image.shape[0]) / 2.0

    target_mat[math.floor(pad_y) : target_size[0] - math.ceil(pad_y),
               math.floor(pad_x) : target_size[1] - math.ceil(pad_x)] = image
    image = target_mat

    if plot:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(image_orig)
        axes[1].imshow(image)
        plt.show()

    return image

def save_image_from_df(row, directory, filename = None):
    """saves images stored in a dataframe as single files in one folder"""
    os.makedirs(directory, exist_ok = True)
    image = Image.fromarray(row['image'])
    image = image.convert( 'RGB')

    if filename:
        os.makedirs(os.path.join(directory, filename.replace('.jpg', '')), exist_ok = True)
        image.save(os.path.join(directory, filename.replace('.jpg', ''), str(row['line']) + '_' + str(row['position']) + '.jpg'))

    else:
        image.save(os.path.join(directory, str(row['line']) + '_' + str(row['position']) + '.jpg'))
