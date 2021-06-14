# -*- coding: utf-8 -*-
"""This script removes table data from documents

Example:
    To test:
        $ python remove_table.py [input_folder]
    To produce an output:
        $ python remove_table.py [input_folder] [output_folder]
"""

__version__ = '0.1'
__author__ = 'Neosh Sheikh'

import pydicom
import nibabel as nib  # nNIfTI loader
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import cv2
import math
import time
from multiprocessing import Process
#  TODO remove if not used later
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

# In hounsfield units
BINARIZE_THRESHOLD = 100
SURROUNDING_THRESHOLD = -10


binarize_threshold_pixel = None
air_replace_value_pixel = None
surrounding_threshold_pixel = None


def pixel_HU_calculate(pixel_value, slope, intercept):
    return pixel_value * slope + intercept


def HU_pixel_calculate(HU_value, slope, intercept):
    return (HU_value - intercept) / slope


def binarize(img_3d):
    '''
    Returns a binarized form of the 3d data array

        Parameters:
            img_3d (ndarray): 3d data array

        Returns:
            img_3dB (ndarray): binarized 3d array
    '''
    thresh = binarize_threshold_pixel
    img_3dB = np.where(img_3d > thresh, 1, 0)
    return img_3dB


def y_value_from_polar(line, x):
    '''
    Returns a the y value

        Parameters:
            line (any): line data returned from opencv
            x (int): 

        Returns:
            y (float): y value
    '''
    theta = line[0][1]
    rho = line[0][0]
    return - math.cos(theta)/math.sin(theta) + rho/math.sin(theta)


def check_air_around_line(line, img_2d):

    # For all points above and below the line, look for white space

    number_of_hits = 0

    for j in range(0, len(img_2d[0])):
        #  j is the x value

        y_val = int(y_value_from_polar(line, j))

        # If no positive hit is detected, assume that metallic part could be above
        # move up until solid data found. keep removing til unfound
        upward_found = False

        if(y_val - 10 < 0 or y_val + 10 > len(img_2d[0])):
            return False

        for i in range(y_val+1, y_val-10, -1):
            if(img_2d[i][j] < surrounding_threshold_pixel):
                upward_found = True
            elif upward_found:
                break

        downward_found = False
        for i in range(y_val, y_val+10):
            if(img_2d[i][j] < surrounding_threshold_pixel):
                downward_found = True
            elif downward_found:
                break

        if(downward_found and upward_found):
            number_of_hits += 1

    if(number_of_hits/len(img_2d[0]) > 0.97):
        return True
    else:
        return False


def hough_transform_return_table(_img2d, neighboring_point=None, debug=False, top_zero=True):
    '''
    Returns the line that defines the top of the table

        Parameters:
            _img2d(ndarray): 2d array of the sagittal slice
            neighboring_point (any): point form line data from the neighbor (optional)
            debug (bool): shows the output to check if everything is working

        Returns:
            min_line (any): the line that defines the top of the table
    '''
    temp_3d = np.zeros((len(_img2d), len(_img2d[0]), 3), "int16")
    temp_3d[:, :, 0] = _img2d[:, :] * 254
    temp_3d[:, :, 1] = _img2d[:, :]*254
    temp_3d[:, :, 2] = _img2d[:, :]*254

    temp8 = np.uint8(temp_3d)

    dst = cv2.Canny(temp8, 0, 254)
    # TODO figue out the best values to use here

    thresh_val = len(_img2d[0])//2
    # print(thresh_val)
    lines = cv2.HoughLines(dst, 1, np.pi / (500), thresh_val,
                           None, 0, 0, np.pi/180*88, np.pi/180*92)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    min_rho = -1
    min_line = None

    if lines is not None:
        for i in range(0, len(lines)):

            dist_from_bottom = y_value_from_polar(lines[i], 0)
            range_condition = neighboring_point is None or (
                neighboring_point is not None and abs(neighboring_point-dist_from_bottom) < 2)
            if(range_condition):
                # check on 2d image if there is low density space on both sides of each line
                if(top_zero and lines[i][0][0] > len(_img2d)//2):
                    # if(True):

                    if(min_rho == -1):
                        min_line = lines[i]
                        min_rho = dist_from_bottom
                    elif(min_rho > lines[i][0][0]):
                        min_line = lines[i]
                        min_rho = dist_from_bottom

                    rho = lines[i][0][0]
                    theta = lines[i][0][1]

                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    # Show results
    if(debug):
        cv2.imshow("Pre detection", dst)
        cv2.imshow(
            "Detected Lines (in red) - Standard Hough Line Transform", cdst)

    # TODO detect which line is the top of the table.
    # Hardcoding for now to test

    return min_line


# TODO this method may need to be removed in the future and functions using this need to use air_values_under_line
# - as is risky
def air_values_under_point(img_3d, min_rho, slice_loc):

    if(min_rho > 0):
        temp_2d = img_3d[:, slice_loc, :]

        # TODO move up towards the body until air is detected
        # to make up for error in line identification
        for j in range(0, len(temp_2d[0])):
            # TODO remove
            # print("slice point {} {} {}".format(slice_loc, j, min_rho))

            # If no positive hit is detected, assume that metallic part could be above
            # move up until solid data found. keep removing til unfound
            positive_found = False
            for i in range(min_rho+1, min_rho-10, -1):
                # TODO check if also under skin levels? current issue is that
                # - some skin data can be removed if body part is touching the table
                if(temp_2d[i][j] > surrounding_threshold_pixel):
                    temp_2d[i][j] = air_replace_value_pixel
                    positive_found = True
                # neg after becoming positive
                elif positive_found:
                    break

            for i in range(min_rho, len(temp_2d)):
                temp_2d[i][j] = air_replace_value_pixel

        img_3d[:, slice_loc, :] = temp_2d


def air_values_under_line(img_3d, min_line, slice_loc):
    if(min_line is not None):
        temp_2d = img_3d[:, slice_loc, :]

        # TODO move up towards the body until air is detected
        # to make up for error in line identification
        for j in range(0, len(temp_2d[0])):

            yMin = math.ceil(y_value_from_polar(min_line, j))
            # TODO remove
            # print("slice point {} {} {}".format(slice_loc, j, yMin))

            # If no positive hit is detected, assume that metallic part could be above
            # move up until solid data found. keep removing til unfound
            positive_found = False
            for i in range(yMin+2, yMin-10, -1):
                # TODO check if also under skin levels? current issue is that
                # - some skin data can be removed if body part is touching the table
                if(temp_2d[i][j] > surrounding_threshold_pixel):
                    temp_2d[i][j] = air_replace_value_pixel
                    positive_found = True
                # neg after becoming positive
                elif positive_found:
                    break

            for i in range(yMin-1, len(temp_2d)):
                temp_2d[i][j] = air_replace_value_pixel

        img_3d[:, slice_loc, :] = temp_2d


def table_remove_sagittal(img_3d, img_3dB, slice_loc, neighboring_point=None):
    # find line defining table top on sagittal plane
    min_line = hough_transform_return_table(
        img_3dB[:, slice_loc, :], neighboring_point=neighboring_point)

    if(min_line is None):
        return None

    min_rho = int(min_line[0][0])
    air_values_under_line(img_3d, min_line, slice_loc)

    # Returned so can be used by next object
    return min_line


# TODO Defined spearately in order to run in parallel
# #######################################################
def table_remove_right(img_3d, img_3dB, min_line):
    # going through sagittal planes
    img_shape = img_3d.shape
    lenOfBod = len(img_3dB[0])/2
    midpoint = img_shape[1]//2

    if min_line is None:
        neighboring_point = None
    else:
        neighboring_point = y_value_from_polar(min_line, 0)
    neighboring_line = min_line

    for i in range(midpoint, 0, -1):
        if(i % (int(lenOfBod/10)) == 0):
            print("{} %".format((lenOfBod-i)/lenOfBod*100))
        line = table_remove_sagittal(img_3d, img_3dB, i, neighboring_point)
        if(line is None and neighboring_line is not None):
            air_values_under_line(img_3d, neighboring_line, i)
            # air_values_under_point(img_3d, neighboring_point, i)

        else:
            neighboring_point = int(y_value_from_polar(line, 0))
            neighboring_line = line


def table_remove_left(img_3d, img_3dB, min_line):
    # going through sagittal planes
    img_shape = img_3d.shape
    lenOfBod = len(img_3dB[0])/2
    midpoint = img_shape[1]//2
    if min_line is None:
        neighboring_point = None
    else:
        neighboring_point = y_value_from_polar(min_line, 0)
    neighboring_line = min_line
    print("down")

    for i in range(midpoint, len(img_3dB[0]), 1):
        if(i % (int(lenOfBod/10)) == 0):
            print("{} %".format((i-lenOfBod)/lenOfBod*100))
        line = table_remove_sagittal(img_3d, img_3dB, i, neighboring_point)
        if(line is None and neighboring_line is not None):
            air_values_under_line(img_3d, neighboring_line, i)
            # air_values_under_point(img_3d, neighboring_point-1, i)
        else:
            neighboring_point = int(line[0][0])
            neighboring_line = line

# #######################################################
#  TODO run left and right in parallel, leaving for now to avoid premature optimization


def table_remove(img_3d, debug=False, debug_slice=None):
    '''
    Removes the table from the data
        Parameters:
            img_3d(ndarray): 3d array defining the scan
    '''

    img_3dB = binarize(img_3d)
    img_shape = img_3d.shape

    # print("HERE")

    if(debug):
        if(debug_slice is None):
            print("Missing debug_slice")
            return

        min_line = hough_transform_return_table(
            img_3dB[:, debug_slice, :], debug=True)

        return

    # If not debug

    # going through sagittal planes
    lenOfBod = len(img_3dB[0])
    midpoint = img_shape[1]//2

    if(midpoint % (int(lenOfBod/10)) == 0):
        print("{} %".format(midpoint/lenOfBod*100))
    min_line = table_remove_sagittal(img_3d, img_3dB, midpoint)

    table_remove_right(img_3d, img_3dB, min_line)
    table_remove_left(img_3d, img_3dB, min_line)


def load_slices_dcm(folder_path):
    # load the DICOM files
    files = []
    data_glob = glob.glob(os.path.join(folder_path, "*.dcm"))

    print("loading files")
    for fname in data_glob:
        files.append(pydicom.dcmread(fname))

    print("file count: {}".format(len(files)))

    print(files[0])

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    return slices


def load_slices_nii(file_name):
    # load the DICOM files

    # Get nibabel image object
    img = nib.load(file_name)
    img_data = img.get_fdata()
    img_data_arr = np.asanyarray(img_data)

    return (img_data_arr, img.dataobj.slope, img.dataobj.inter, nib.aff2axcodes(img.affine), img)


def create_3dform(slices):

    # create 3D array 0,1 = xy ; 2 = z (slice)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img_3d = np.zeros(img_shape, "int16")

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img_3d[:, :, i] = img2d
    return (img_3d, img_shape)


def save_dicom(slices, img_3d, index, folder=None):
    # We know that img_3d[:, :, i] = img2d in the simple case

    slices[index].PixelData = img_3d[:, :, index].tostring()

    # # Create some temporary filenames
    file_name = f'CT{index:06}.dcm'

    filename_little_endian = os.path.join(folder, file_name)

    slices[index].save_as(filename_little_endian)
    print("Writing test file", filename_little_endian)
    print("File saved.")


def display_data(img_3d, img_shape, ax_aspect, sag_aspect, cor_aspect):
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 3, 1)
    plt.imshow(img_3d[:, :, img_shape[2]//2])
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 3, 2)
    plt.imshow(img_3d[:, img_shape[1]//2, :])
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 3, 3)
    plt.imshow(img_3d[:, :, img_shape[2]//2-1])
    a3.set_aspect(ax_aspect)

    a4 = plt.subplot(2, 3, 4)
    plt.imshow(img_3d[:, :, img_shape[2]//2-2])
    a4.set_aspect(ax_aspect)

    a5 = plt.subplot(2, 3, 5)
    plt.imshow(img_3d[:, img_shape[1]//2+50, :])
    a5.set_aspect(sag_aspect)

    a6 = plt.subplot(2, 3, 6)
    plt.imshow(img_3d[:, :, img_shape[2]//2+2])
    a6.set_aspect(ax_aspect)

    plt.show()


def debugger(img_3d, img_shape, ax_aspect, sag_aspect, cor_aspect):

    # img_3d = binarize(img_3d)
    # img_shape[1]//2
    global debugNum
    debugNum = img_shape[1]//2
    table_remove(img_3d, debug=True, debug_slice=img_shape[1]//2)
    display_data(img_3d, img_shape, ax_aspect, sag_aspect, cor_aspect)


def transform_data_LR_I_P(img_3d, orientation=None):
    # positive - SA LR??
    # want orientation IS = [2]
    # LR = [1]
    # AP = [0]
    print(img_3d.shape)
    print(orientation)
    lr_index, ap_index, is_index = (None, None, None)
    lr_flip, ap_flip, is_flip = (False, False, False)

    for i, o in enumerate(orientation):
        # Do not care about flipping RL because plane is unused
        if(o == 'L' or o == 'R'):
            lr_index = i

        elif(o == 'A'):
            ap_index = i
            ap_flip = True
        elif(o == 'P'):
            ap_index = i

        elif(o == 'S'):
            is_index = i
            is_flip = True
        elif(o == 'I'):
            is_index = i

    print("{} {} {}".format(lr_index, ap_index, is_index))
    if(is_index == 0):
        img_3d = np.swapaxes(img_3d, ap_index, 0)
        if(ap_index != 2):
            img_3d = np.swapaxes(img_3d, ap_index, 2)

    elif(lr_index == 0):
        img_3d = np.swapaxes(img_3d,  0, ap_index)
        if(ap_index != 1):
            img_3d = np.swapaxes(img_3d, ap_index, 1)

    elif(ap_index == 0):
        if(lr_index != 1):
            img_3d = np.swapaxes(img_3d, lr_index, is_index)
    # want SA as 0

    if(ap_flip):
        # do some stuff
        img_3d = img_3d[::-1, :, :]
    if(is_flip):
        # do some stuff
        img_3d = img_3d[:, :, ::-1]
    return img_3d


def hello():
    print("hello")


def process_dicom(f_path, out_folder_path = None):
    slices = load_slices_dcm(f_path)

    (img_3d, img_shape) = create_3dform(slices)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness

    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]

    # TODO remove
    #########################################################
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if(intercept is None or slope is None):
        print("No slope or no intercept in data")
        sys.exit()

    # Assign correct values
    global binarize_threshold_pixel
    global air_replace_value_pixel
    global surrounding_threshold_pixel

    binarize_threshold_pixel = HU_pixel_calculate(
        BINARIZE_THRESHOLD, slope, intercept)
    air_replace_value_pixel = img_3d.min()
    surrounding_threshold_pixel = HU_pixel_calculate(
        SURROUNDING_THRESHOLD, slope, intercept)

    table_remove(img_3d)

    if(out_folder_path):
        # Save dicom files
        for i, s in enumerate(slices):
            save_dicom(slices, img_3d, i, out_folder_path)
    else:
        display_data(img_3d, img_shape, ax_aspect, sag_aspect, cor_aspect)


if __name__ == "__main__":

    if(len(sys.argv) == 3 and sys.argv[2] is not None):
        out_folder_path = sys.argv[2]

    print('glob: {}'.format(sys.argv[1]))
    f_path = sys.argv[1]

    # TODO check if type is folder, if yes, try as dcm
    if os.path.isdir(f_path):
        process_dicom(f_path, out_folder_path)

    elif os.path.isfile(f_path):
        (img_3d, slope, intercept, orientation,
         file_data) = load_slices_nii(f_path)

        # transformation
        img_3d = transform_data_LR_I_P(img_3d, orientation)
        img_shape = img_3d.shape

        # Assign correct values
        binarize_threshold_pixel = HU_pixel_calculate(
            BINARIZE_THRESHOLD, 1, 1)
        air_replace_value_pixel = img_3d.min()
        surrounding_threshold_pixel = HU_pixel_calculate(
            SURROUNDING_THRESHOLD, 1, 1)

        # table_remove(img_3d)
        table_remove(img_3d, debug=True, debug_slice=img_shape[1]//2)

        # img_3d = transform_data_LR_I_P(img_3d, orientation)
        img_shape = img_3d.shape

        if(len(sys.argv) == 3 and sys.argv[2] is not None):
            out_folder_path = sys.argv[2]
            # Save file
            clipped_img = nib.Nifti1Image(
                img_3d, file_data.affine, file_data.header)
            f = nib.save(clipped_img, os.path.join(out_folder_path, "bac.nii"))
            print(f)
        else:
            slices_data = [img_3d[img_shape[0]//2, :, :], img_3d[:,
                                                                 img_shape[1]//2, :], img_3d[:, :, img_shape[2]//2]]

            """ Function to display row of image slices """
            fig, axes = plt.subplots(1, len(slices_data))
            for i, s in enumerate(slices_data):
                axes[i].imshow(s)

            plt.suptitle("Center slices for NIfTI image")
            plt.show()

    else:
        print("It is a special file (socket, FIFO, device file)")

    # TODO else load as .nii
