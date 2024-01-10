
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: isabellegarnreiter

This code is set up to process SMLM data by creating point-density based clusters 
If a widefield image path is specified the algorithm will use the widefield image to compute clusters, 
otherwise an approximation will be made based on a gaussian filter of the smlm data. In this case, 
the generated image will be saved within a 'data' folder under each acquisition. 

Some fine tuning is possible by defining the minimum distance between 2 clusters, and the size boundaries for a cluster.

The output is:

masks_647 = an array with the same size as the image (either the widefield or the generated gaussian approximation) 
            with all the numbered clusters.
clusters_647 = dictionary, where the keys correspond to the cluster number and the value is an array of the specified cluster mask
points_647 = dictionary, where the keys correspond to the cluster number and the value is an array of the all the points within a cluster,
             size: Nx3, where N is the number of points.

There is possibility in the future to create the same clusters in the 2nd channel using the gaussian filtering method on the 2nd channel. 
"""

import numpy as np
import pandas as pd
from glob import glob
import os
import re
import functions as fcts
from tkinter import filedialog



#target_dir =  filedialog.askdirectory()
target_dir = '/Users/isabellegarnreiter/Desktop/storm'
widefield_image = '' #input name of widefield images or keep as an empty string if there is none


date_pattern = re.compile(r'^\d') #pattern used to match the filter out foldernames that don't start with a date 

#initialise paramaters for the detection/selection of synapses. Go to given function to check default parameters.
params = fcts.get_default_params()

for folder in os.listdir(target_dir):
    folder_path = os.path.join(target_dir, folder)
    # Check if folder is a directory and if the name starts with a date
    if os.path.isdir(folder_path) and re.compile(r'^\d').match(folder):
        print(folder)

        Demix_folders = glob(folder_path + '*/CellZone*/*emix')
        for demix in Demix_folders:
            if os.path.isdir(demix):

                data_folder = os.path.join(demix, 'data')
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

                # MAKE SURE THE CHANNELS ARE ASSIGNED CORRECTLY. ON OLDER DATA CH1 -> 680nm BUT ON NEWER DATA CH2-> 647nm
                if params['647_channel'] == 'channel1':
                    channel_647 = glob(demix + '*/*w1*.csv')[0]
                    channel_680 = glob(demix + '*/*w2*.csv')[0]

                elif params['647_channel'] == 'channel2':
                    channel_647 = glob(demix + '*/*w2*.csv')[0]
                    channel_680 = glob(demix + '*/*w1*.csv')[0]

                data_in_647 = pd.read_csv(channel_647)[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
                data_in_647[:,2] +=550

                if glob(folder_path + f'*/Acquisition*/{widefield_image}'):
                    wfi = widefield_image #turn to numpy array

                else: 
                    image_size = (params['true_roi_size'][0]//params['sf'][0], params['true_roi_size'][1]//params['sf'][1], params['true_roi_size'][2]//params['sf'][2])
                    wfi = fcts.get_gaussiankde(data_in_647, params)
                    np.save(os.path.join(data_folder, 'simulated_widefield.npy'), wfi, allow_pickle=True)

                masks_647 = fcts.get_clusters(wfi, params)
                cluster_points_647 = fcts.get_points(data_in_647, masks_647, params)
                seperated_clusters_647 = fcts.seperate_clusters(masks_647)

                np.save(os.path.join(data_folder, 'masks_647.npy'), np.array(masks_647), allow_pickle=True)
                np.save(os.path.join(data_folder, 'clusters_647.npy'), np.array(seperated_clusters_647), allow_pickle=True)
                np.save(os.path.join(data_folder, 'points_647.npy'), np.array(cluster_points_647), allow_pickle=True)

                if params['filter_680'] == True:
                    data_in_680 = pd.read_csv(channel_680)[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
                    data_in_680[:,2] +=550

                    if params['use_wf_680'] == False:
                        wfi_680 = fcts.get_gaussiankde(data_in_680, params)
                        np.save(os.path.join(data_folder, 'simulated_widefiel_680.npy'), wfi_680, allow_pickle=True)

                    if params['use_wf_680'] == True:
                        wfi_680 = wfi

                    masks_680 = fcts.get_clusters(wfi_680, params)
                    cluster_points_680 = fcts.get_points(data_in_680, masks_680, params)
                    seperated_clusters_680 = fcts.seperate_clusters(masks_680)

                    np.save(os.path.join(data_folder, 'masks_680.npy'), np.array(masks_680), allow_pickle=True)
                    np.save(os.path.join(data_folder, 'clusters_680.npy'), np.array(seperated_clusters_680), allow_pickle=True)
                    np.save(os.path.join(data_folder, 'points_680.npy'), np.array(cluster_points_680), allow_pickle=True)



            

