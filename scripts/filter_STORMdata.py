import numpy as np
import pandas as pd
from glob import glob
import os
import re
import functions as fcts
from tkinter import filedialog

#initialise paramaters for the detection/selection of synapses

params = {}
params['true_roi_size'] = (49660,49660,1100) #size of the 3d Tiff in microns
params['sf']  = (68, 68, 180) #defines the level of downsampling to create the filtered image
params['kernel_size'] = (40,40,2) #kernel size of the gaussian filter
params['sigma'] = 8 #intensity of the gaussian filter
params['max_threshold_ves'] = 4 #threshold for the extraction of intensity blobs in the image
params['min_peak_dist'] = 16 #min distance between 2 peaks (in pixels) - if distance is smaller, the 2 peaks are merged
params['min_cluster_area'] = 32 #min area of a cluster (in pixels)
params['max_cluster_area'] = 32000 #max area of a cluster (in pixels)


#target_dir =  filedialog.askdirectory()
target_dir = '/Users/isabellegarnreiter/Desktop/storm'
widefield_image = ''

date_pattern = re.compile(r'^\d')
        
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

                channel1 = glob(demix + '*/*w1*.csv')[0]
                channel2 = glob(demix + '*/*w2*.csv')[0]

                data_in_647 = pd.read_csv(channel1)[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
                data_in_647[:,2] +=550

                data_in_680 = pd.read_csv(channel2)[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
                data_in_680[:,2] +=550

                if glob(folder_path + f'*/Acquisition*/{widefield_image}'):
                    wfi = widefield_image #turn to numpy array

                else: 
                    image_size = (params['true_roi_size'][0]//params['sf'][0], params['true_roi_size'][1]//params['sf'][1], params['true_roi_size'][2]//params['sf'][2])
                    wfi = fcts.get_gaussiankde(data_in_647, params)
                    np.save(os.path.join(data_folder, 'simulated_widefield.npy'), wfi, allow_pickle=True)

                clusters_647 = fcts.get_clusters(wfi, params)
                cluster_points_647 = fcts.get_points(data_in_647, clusters_647, params)
                seperated_clusters_647 = fcts.seperate_clusters(clusters_647)

                np.save(os.path.join(data_folder, 'clusters_647.npy'), np.array(seperated_clusters_647), allow_pickle=True)
                np.save(os.path.join(data_folder, 'points_647.npy'), np.array(cluster_points_647), allow_pickle=True)



            

