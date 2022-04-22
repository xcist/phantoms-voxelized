######################################################################################################################
# Program name: DICOM_to_voxelized_phantom.py
#
# This program reads in a series of DICOM images and several parameters specified in a config file. It then generates a 
# voxelized phantom based on those images and parameters.
#
# Inputs (specified in a config file):
#   phantom.dicom_path    (string)           Path where the DICOM images are located.
#   phantom.phantom_path  (string)           Path where the phantom files are to be written
#                                                (the last folder name will be the phantom files' base name).
#   phantom.materials     [list of strings]  Material names; must be specified in the CatSim material data folder.
#   phantom.mu_energy     (float)            Energy (keV) at which mu is to be calculated for all materials.
#                                                Note that this should correcspond to the estimated "effective energy"
#                                                for the kVp used for the scan that produced the DICOM images.
#   phantom.thresholds    [list of floats]   Lower threshold (HU) for each material.
#   phantom.slice_range   [list of ints]	 DICOM image numbers to include.
#   phantom.show_phantom  (logical)          Flag to turn on/off image display.
#   phantom.overwrite     (logical)          Flag to overwrite existing files without warning.
# 
#  Outputs
#    a .json file that describes the phantom
#    a .raw file containing the input (HU) images in one big file, easily read by ImageJ.
#    a .raw file for each material, containing the "volume fraction" of that material in each voxel.
#
# Author: Chad Bircher 8/18/2021
# Revised: Paul FitzGerald 3/13/2022
#
# Copyright 2020, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE
#
######################################################################################################################

from pathlib import Path
import os
import numpy as np
import re
import pydicom
import copy
import json
import matplotlib.pyplot as plt
from catsim.GetMu  import GetMu
from catsim.CommonTools import source_cfg


class IndexTracker:
    # Tracker for allowing scroll wheel to move through a large number of images
    # The images are assumed to be stored as [image_number, x, y]
    # In the event that the images are stored as [x, y, image number] update [self.ind, :, :] -> [:, :, self.ind]
    def __init__(self, ax, X):
        self.ax = ax
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[self.ind, :, :]) # initialize the image number
        self.update() # draw the default image (typically the central slice)

    def on_scroll(self, event): # when the wheel has scrolled
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices # increment image number by one
        else:
            self.ind = (self.ind - 1) % self.slices # decrement image number by one
        self.update() # update the image

    def update(self): # When updating
        self.im.set_data(self.X[self.ind, :, :]) # select the image
        self.ax.set_ylabel('slice %s' % self.ind) # title the y axis with the image number
        self.im.axes.figure.canvas.draw() # update the figure with the selected image
        

def initialize(phantom):

    # This function:
    #   Checks for the existence of the DICOM files.
    #   Creates a list of DICOM filenames that will be read.
    #   Initializes several variables.
    #   Checks for the existence of the phantom folder/files.
    #   Creates a list of material volume fraction filenames that will be written.
    #   Allocates an array for the volume fraction data.
    #   If they are not specified in the config file, calculates segmentation thresholds.
    #   Creates a dictionary for material variables.
    # Inputs:
    #   phantom.dicom_path
    #   phantom.slice_range
    #   phantom.phantom_path
    #   phantom.materials
    # Outputs:
    #   phantom.basename
    #   phantom.num_materials
    #   phantom.num_cols
    #   phantom.num_rows
    #   phantom.num_slices
    #   phantom.pixel_size_x
    #   phantom.pixel_size_y
    #   phantom.pixel_size_z
    #   dicom_filenames - list of DICOM filenames in numerical order
    #   volume_fraction_filenames - list of material volume fraction filenames in order specified
    #   volume_fraction_array - a 4D array (column, row, slice, material)
    #   materials_dict - includes material names, volume fraction file names, mu values, and threshold values. 

    # Find all files and DICOM files in the specified DICOM directory
    if not os.path.exists(phantom.dicom_path):
        raise Exception('******** Error! {:s} not found..'.format(phantom.dicom_path))
    allfiles = [f for f in os.listdir(phantom.dicom_path) if os.path.isfile(os.path.join(phantom.dicom_path, f))]
    dcmfiles = [j for j in allfiles if '.dcm' in j]
    if len(dcmfiles) == 0:
        raise Exception('******** Error! No DICOM files exist in {:s}.'.format(phantom.dicom_path))

    # Sort DICOM files: Initial order is alphabetical - switch to numeric.
    dcm_num = []
    for dicom_filename in dcmfiles:
        # dicom number in the file name - alphabetical order - convert to number
        dcm_num.append(int(re.findall(r'\d+', dicom_filename)[0]))

    # sort the dicom numbers by numeric order
    indices = sorted(range(len(dcm_num)),key=dcm_num.__getitem__)

    # find the index associated with the new sorted number to open files in numeric rather than alphabetic order
    dicom_filenames = [dcmfiles[index] for index in indices]

    # If a range of slices was specified, reduce the list of files.
    try:
        # check if ranges for slices are given
        try:
            # If a single range is given use that range of slices
            test = list(range(phantom.slice_range[0], 1 + phantom.slice_range[1]))
        except:
            test = []
            for j in phantom.slice_range:
                # if a series of ranges are given then include the slices in each range
                test += list(range(j[0], 1+j[1]))
        # filter the dicom data on the selected range(s)
        dicom_filenames = [dicom_filenames[i] for i in test]
    except:
        # keep the original if the ranges are inappropriate or if slice_range is not specified.
        dicom_filenames = dicom_filenames

    # Read in one DICOM file to use as a sample.
    dicom_filename = os.path.join(phantom.dicom_path, dcmfiles[0])
    # sample_dicom = []
    sample_dicom = pydicom.dcmread(dicom_filename)

    # Display basic information about the DICOM dataset.
    try:
        PatientAge = sample_dicom.PatientAge
    except:
        PatientAge = 'Unavailable'

    try:
        PatientSex = sample_dicom.PatientSex
    except:
        PatientSex = 'Unavailable'

    try:
        Manufacturer = sample_dicom.Manufacturer
    except:
        Manufacturer = 'Unavailable'

    try:
        ManufacturerModelName = sample_dicom.ManufacturerModelName
    except:
        ManufacturerModelName = 'Unavailable'

    try:
        StudyDate = sample_dicom.StudyDate
    except:
        StudyDate = 'Unavailable'

    try:
        StudyDescription = sample_dicom.StudyDescription
    except:
        StudyDescription = 'Unavailable'

    try:
        KVP = sample_dicom.KVP
    except:
        KVP = 'Unavailable'

    try:
        XRayTubeCurrent = sample_dicom.XRayTubeCurrent
    except:
        XRayTubeCurrent = 'Unavailable'

    try:
        ConvolutionKernel = sample_dicom.ConvolutionKernel
    except:
        ConvolutionKernel = 'Unavailable'

    try:
        ReconstructionDiameter = sample_dicom.ReconstructionDiameter
    except:
        ReconstructionDiameter = 'Unavailable'

    try:
        PixelSpacing = sample_dicom.PixelSpacing[0]
    except:
        PixelSpacing = 'Unavailable'

    try:
        SliceThickness = sample_dicom.SliceThickness
    except:
        SliceThickness = 'Unavailable'

    print('*')
    print('*********************************************')
    print('* DICOM dataset information:')
    print('* Patient age: {:s}, sex: {:s}'.format(PatientAge, PatientSex))
    print('* Scanner: {:s} {:s}'.format(Manufacturer, ManufacturerModelName))
    print('* Study date & desciption: {:s}, {:s}'.format(StudyDate, StudyDescription))
    print('* Technique: {} kVp, {} mA'.format(KVP, XRayTubeCurrent))
    print('* Reconstruction: {:s} kernel, {}-mm FOV'.format(ConvolutionKernel, ReconstructionDiameter))
    print('* Image: {}-mm pixels (XY), {}-mm slices (Z)'.format(PixelSpacing, SliceThickness))
    print('*********************************************')

    # Initialize variables.
    phantom.basename = os.path.basename(phantom.phantom_path)
    phantom.num_materials = len(phantom.materials)
    phantom.num_cols = sample_dicom.Columns
    phantom.num_rows = sample_dicom.Rows
    phantom.num_slices = len(dicom_filenames)
    phantom.pixel_size_x = sample_dicom.PixelSpacing[0]
    phantom.pixel_size_y = sample_dicom.PixelSpacing[1]
    phantom.pixel_size_z = sample_dicom.SliceThickness
    phantom.mu_water = GetMu('water', phantom.mu_energy)
    phantom.mu_water = phantom.mu_water[-1]
    phantom.json_filename = phantom.phantom_path + '/' + phantom.basename + '.json'

    filenames_first_part = phantom.phantom_path + '/' + phantom.basename + '_'
    filenames_last_part = '_' + str(phantom.num_cols) + 'x' + str(phantom.num_rows) + 'x' + str(phantom.num_slices) + '.raw'
    mu_list = []
    volume_fraction_array = {}
    volume_fraction_filenames = []
    for index, material in enumerate(phantom.materials):
        
        # Calculate mu values for each material.
        mu_list.append(GetMu(material, phantom.mu_energy)[0])
        
        # Allocate an array for the volume fraction map of each material.
        volume_fraction_array.update({material: np.zeros((phantom.num_slices, phantom.num_cols, phantom.num_rows), dtype=np.float32)})

        # Assign output filenames for each material.
        volume_fraction_filenames.append(filenames_first_part + material + filenames_last_part)

        # Calculate the thresholds - mu boundaries between materials
        if material == phantom.materials[0]:
            threshold_list = [0]
        else:
            # 55% lower mu material, 45% upper mu material
            threshold_list.append(mu_list[index-1]*0.55 + mu_list[index]*0.45)

    # Allocate an array for the HU data.
    volume_fraction_array.update({'HU data': np.zeros((phantom.num_slices, phantom.num_cols, phantom.num_rows), dtype=np.float32)})

    # Assign an output filename for the HU data.
    volume_fraction_filenames.append(filenames_first_part + 'HU_data' + filenames_last_part)

    # sort the mu_list and keep the index of the new values
    indices = sorted(range(len(mu_list)), key=mu_list.__getitem__)

    # for each index calculate the threshold between the previous and current materials
    # match the order for the mu, materials, and thresholds
    mu_list1 = [mu_list[index] for index in indices]
    material_names1 = [phantom.materials[index] for index in indices]
    for index, material in enumerate(material_names1):
        if material == material_names1[0]:
            threshold_list1 = [0]
        else:
            # 55% lower mu material, 45% upper mu material
            threshold_list1.append(mu_list1[index-1]*0.55 + mu_list1[index]*0.45)

    # Initialize the material dictionary.
    materials_dict = {'material_names': material_names1,
                      'volume_fraction_filenames': volume_fraction_filenames,
                      'mu_values': mu_list1,
                      'threshold_values': threshold_list1}

    print('*')
    print('*********************************************')
    print('* Segmentation parameters:')

    # If thresholds were correctly specified in the config file, over-ride calculated thresholds.
    try: 
        if len(phantom.thresholds) == len(materials_dict['threshold_values']):
            # First convert the specified thresholds to LACs (mu).
            mu_thresholds = [(hu + 1000) * phantom.mu_water / 1000 for hu in phantom.thresholds]
            materials_dict.update({'threshold_values': mu_thresholds})
            print('* Using thresholds specified in the config file.')
    except:
       print('* Using calculated thresholds.')
       phantom.thresholds = materials_dict['threshold_values']

    print('* Materials: {}'.format(materials_dict['material_names']))
    print('* mu values (/cm): {}'.format([round(e, 2) for e in materials_dict['mu_values']]))
    print('* mu(water) (/cm): {}'.format(round(phantom.mu_water, 2)))
    print('* Thresholds (/cm): {}'.format([round(e, 2) for e in materials_dict['threshold_values']]))
    hu_thresholds = [(1000 * mu / phantom.mu_water - 1000) for mu in materials_dict['threshold_values']]
    print('* Thresholds (HU): {}'.format([round(e, 0) for e in hu_thresholds]))
    print('*********************************************')

    # Create the phantom folder if it doesn't exist.
    if not os.path.exists(phantom.phantom_path):
        os.makedirs(phantom.phantom_path)
    else:
        try:
            overwrite = phantom.overwrite
        except:
            overwrite = False

        if not overwrite:
            # The folder is there, but are there files in it that will be over-written?
            files_exist = False
            for material_index in range(0, phantom.num_materials):
                if os.path.exists(volume_fraction_filenames[material_index]):
                    files_exist = True
            
            # If files will be over-written, print a warning.
            if files_exist:
                print('*')
                print('*********************************************')
                print('* Warning!')
                print('* This folder exists:')
                print('*     {:s}'.format(os.path.abspath(phantom.phantom_path)))
                print('* These files exist:')
                for material_index in range(0, phantom.num_materials):
                    if os.path.exists(volume_fraction_filenames[material_index]):
                        print('*     {:s}'.format(os.path.abspath(volume_fraction_filenames[material_index])))
                print('* These files will be overwritten.')
                print('* Press enter to continue or Ctrl-C to quit.')
                input('*********************************************')

    return phantom, dicom_filenames, volume_fraction_array, materials_dict


def compute_volume_fraction_array(phantom, dicom_filenames, materials_dict, volume_fraction_array):

    threshold_list = materials_dict['threshold_values']
    material_names = materials_dict['material_names']
    mu_list = materials_dict['mu_values']
    print('* Calculating volume fraction maps for {} materials and {} slices...'.format(len(threshold_list), len(dicom_filenames)))

    # Generate volume fraction arrays: 
    #   Read in each DICOM file
    #   Separate into materials using thresholds calculated above
    #   Calculate volume fraction
    #   Append to arrays linked to materials
    for dcm_index, dicom_filename in enumerate(dicom_filenames):
        dicom_pathname = os.path.join(phantom.dicom_path, dicom_filename)
        dicom_data = pydicom.dcmread(dicom_pathname)             # Read DICOM file
        hu_array = dicom_data.pixel_array                        # Extract the data - in HU but with a "rescale intercept".
        hu_array = hu_array + int(dicom_data.RescaleIntercept)   # Add the "rescale intercept" to get it into HU.
        volume_fraction_array['HU data'][dcm_index] = hu_array   # Store the HU data in the volume fraction array
        mu_array = (hu_array + 1000) * phantom.mu_water / 1000   # Convert HU to LAC (mu)
        mu_array[mu_array < 0] = 0                               # Remove negative values (non-physical)
        bounds = copy.deepcopy(threshold_list)                   # Prevent the next line from appending to threshold_list
        bounds.append(1.1*mu_array.max())                        # Set the upper bound for the last material to include the highest pixel in the array
        for material_index, material in enumerate(material_names):  # For each material,
            array0 = copy.deepcopy(mu_array)                     # Start with the mu array,
            array0[array0 < bounds[material_index]] = 0          # Zero out pixels below lower threshold
            array0[array0 >= bounds[material_index+1]] = 0       # Zero out pixels above upper threshold
            array0 /= mu_list[material_index]                    # Calculate mu fraction relative to mu for this material.
            volume_fraction_array[material][dcm_index] = array0  # Store the mu fraction in the volume fraction array

    return volume_fraction_array


def write_files(phantom, materials_dict, volume_fraction_array):

    volume_fraction_filenames = materials_dict['volume_fraction_filenames']
    material_names = materials_dict['material_names']
    material_names.append('HU data')
    print('* Writing volume fraction files for {} materials and {} slices, plus the HU data...'.format(phantom.num_materials, phantom.num_slices))

    for index, volume_fraction_filename in enumerate(volume_fraction_filenames):
        print('* Writing {:s}...'.format(volume_fraction_filename))
        with open(volume_fraction_filename, 'wb') as fout:
            fout.write(volume_fraction_array[material_names[index]])
   
    # Remove entries associated with the HU data.
    material_names = materials_dict['material_names'].pop()
    volume_fraction_filenames = materials_dict['volume_fraction_filenames'].pop()

    # Generate and write the .json file
    write_json_file(phantom, materials_dict)

    return 0


def write_json_file(phantom, materials_dict):

    json_file_contents = {
        "construction_description": 'Created by DICOM_to_voxelized_phantom.py',
        "n_materials": phantom.num_materials,
        "mat_name": materials_dict['material_names'],
        "mu_values": materials_dict['mu_values'],
        "mu_thresholds": materials_dict['threshold_values'],
        "volumefractionmap_filename": materials_dict['volume_fraction_filenames'],
        "volumefractionmap_datatype": ["float"] * phantom.num_materials,
        "cols": [phantom.num_cols] * phantom.num_materials,
        "rows": [phantom.num_rows] * phantom.num_materials,
        "slices": [phantom.num_slices] * phantom.num_materials,
        "x_size": [phantom.pixel_size_x] * phantom.num_materials,
        "y_size": [phantom.pixel_size_y] * phantom.num_materials,
        "z_size": [phantom.pixel_size_z] * phantom.num_materials,
        "x_offset": [0.5 + phantom.num_cols / 2] * phantom.num_materials,
        "y_offset": [0.5 + phantom.num_rows / 2] * phantom.num_materials,
        "z_offset": [0.5 + phantom.num_slices / 2] * phantom.num_materials
    }

    print('* Writing {:s}...'.format(phantom.json_filename))
    with open(phantom.json_filename, 'w') as outfile:
        json.dump(json_file_contents, outfile, indent=4)

    return json_file_contents


def DICOM_to_voxelized_phantom(phantom):

    # Initialize.
    phantom, dicom_filenames, volume_fraction_array, materials_dict = initialize(phantom)

    # Generate the volume fraction maps.
    volume_fraction_array = compute_volume_fraction_array(phantom, dicom_filenames, materials_dict, volume_fraction_array)
    
    # Write the output files - a .json file and volume fraction files for each material.
    write_files(phantom, materials_dict, volume_fraction_array)

    try:
        # If the user has defined show_phantom as True, display the phantom.
        if phantom.show_phantom:
 
            if  phantom.num_materials <= 3:                       # With 3 or fewer materials define a 1 by x grid
                rows = 1
                cols = phantom.num_materials
            elif phantom.num_materials == 4:                      # With 4 materials define a 2x2 grid
                rows = 2
                cols = 2
            elif phantom.num_materials <= 6:                      # With 5 or 6 materials define a 2x3 grid
                rows = 2
                cols = 3
            # Need to add grids for 7-8 (2x4), 9-12 (3x4), 13-16 (4x4), and 17-25(5x5) materials. Beyond 25 is very unlikely to be useful or meaningful

            # Define a plot with rows and axes defined above, all axes are linked for zooming purposes
            fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
            tracker = []
            for plot_num, material in enumerate(materials_dict['material_names']):
                
                # Identify the subplot
                if phantom.num_materials <= 3:
                    this_axis = ax[plot_num%cols]
                else:
                    col = plot_num%cols
                    this_axis = ax[(phantom.num_materials - col) / rows, col]

                # Link this subplot to the tracker.
                tracker.append(IndexTracker(this_axis,volume_fraction_array[material]))
                
                # Define event of scrolling to change the view slice.
                fig.canvas.mpl_connect('scroll_event', tracker[plot_num].on_scroll)
                
                # Name the subplot as the material name.
                this_axis.set_title(material)
            plt.show()
    except:
        pass


if __name__ == "__main__":

    # read in config file
    config = source_cfg('C:/Users/200003237/Documents/GitHub/phantoms-voxelized/DICOM_to_voxelized/DICOM_to_voxelized_example_abdomen.cfg')

    # When integrating into CatSim the location of the phantom file needs to be passed with the call statement
    DICOM_to_voxelized_phantom(config.phantom)
