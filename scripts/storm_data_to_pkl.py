#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:02:29 2023

@author: isabellegarnreiter
"""
import numpy as np
import pandas as pd
from glob import glob
from src import functions as fcts
from scipy.spatial import KDTree
from src import multivariateripleyk as rk
import ripleyk

#initialise paramaters for the detection/selection of synapses

params = {}
params['true_roi_size'] = (49660,49660,1100)
params['sf']  = (68, 68, 180)
params['kernel_size'] = (50,50,2)
params['sigma'] = 10
params['max_threshold_ves'] = 6
params['min_peak_dist'] = 20
params['min_cluster_area'] = 10
params['max_cluster_area'] = 50000


target_marker = 'VAMP2680'
access = 'drive'
target_directory_comp = '/users/isabellegarnreiter/documents/vesicleSTORM/data/STORM_csv_files/'
target_directory_drive = '/users/isabellegarnreiter/desktop/'
target_file_path = '/users/isabellegarnreiter/documents/vesicleSTORM/data/vamp2_storm_data.pkl'

markers = ['SPON647', 'DEP647', 'PSD680', 'Bassoon680', 'VAMP2680', 'VGLUT647']
DIVs = ['8DIV', '10DIV']

assert target_marker in markers, "target_marker not in the list of available markers"


# Import all files (cell zones) under one experiment type, but for both dep and spon vesicle targets
# and store the data in dictionaries.

vesicles_data = {}
syn_marker_data = {}


if access == 'computer':
    target_directory = target_directory_comp
    list_of_files = np.array([file for file in glob(target_directory + f'*/*{target_marker}*')])

elif access == 'drive':
    target_directory = target_directory_drive
    list_of_files = np.array([file for file in glob(target_directory + f'*{target_marker}*/*/*emix/*w*.csv')])
    
    
try:
    list_of_files = list_of_files.reshape(list_of_files.shape[0]//2,2)
except ValueError:
    print('Value Error: one channel is missing from one of the cellzones.')

    
usable_exp = pd.read_csv('/users/isabellegarnreiter/documents/vesicleSTORM/data/STORM_binary_list.csv',encoding='latin', sep=',').to_numpy()
filename  = usable_exp[:,0]+'_'+usable_exp[:,1]
files_infos = dict(zip(filename, usable_exp[:,2:]))

for i in range(0, len(list_of_files)):
    if access == 'drive':
        new_file_name = f"{(list_of_files[i][0]).split('/')[4]}_{(list_of_files[i][0]).split('/')[5]}"
    elif access == 'computer':
        new_file_name = f"{(list_of_files[i][0]).split('/')[-1][0:-3]}"

    if new_file_name not in list(files_infos.keys()) or files_infos[new_file_name][0] == 1:
        vesicles = pd.read_csv(list_of_files[i][1])[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
        vesicles[:,2] +=550

        PSD = pd.read_csv(list_of_files[i][0])[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
        PSD[:,2] +=550

        vesicles_data[new_file_name] = vesicles
        syn_marker_data[new_file_name] = PSD

    else:
        pass

    
#create a dictionary where each file is a key and the values are a secondary dictionary containing the locations of vesicles, segregated by synapse.
vesicle_clusters = {}
synapses = {}
synapses_unseperated = {}

for key in list(vesicles_data.keys()):
    print(key)
    filtered_clusters = fcts.get_synapses(vesicles_data[key], params)
    vesicle_clusters_loc = fcts.get_points(vesicles_data[key], filtered_clusters, params)
    seperated_clusters = fcts.seperate_clusters(filtered_clusters)

    vesicle_clusters[key] = vesicle_clusters_loc
    synapses[key] = seperated_clusters
    synapses_unseperated[key] = filtered_clusters
    
print('done')

#initiate the csv file to store the information

cz = np.linspace(0,12,13).astype(int)
       
storm_data = pd.DataFrame(columns = ['FileName', 
                                     'Date', 
                                     '647nm', 
                                     '680nm', 
                                     'DIV', 
                                     'cellzone', 
                                     'ROI label', 
                                     'ROI', 
                                     'points', 
                                     'nearest_neighbor_680', 
                                     'nearest_neighbors_647' , 
                                     'volume', 
                                     'spherecity', 
                                     'univariate_ripleyk', 
                                     'multivariate_ripleyk'])

i = 0
for k1 in vesicle_clusters.keys():
    print(k1)
    for k2 in vesicle_clusters[k1].keys():
        i=i+1
        Filename=k1
        Date = k1[:6]
        marker = [x for x in markers if x in k1]
        DIV = [x for x in DIVs if x in k1][0]
        cellzone = [x for x in cz if str(x) in k1[-4:]][0]
        ROI_label = int(k2)
        ROI = synapses[k1][k2]
        points = vesicle_clusters[k1][k2]
        
        PSD_dist = KDTree(syn_marker_data[k1])
        nearest_dist_psd, nearest_ind_psd = PSD_dist.query(points, k=1) 
        
        self_dist = KDTree(points)        
        nearest_dist_ves, nearest_ind_ves = PSD_dist.query(points, k=10) 
        
        volume, sphericity = fcts.calculate_volume_sphericity(ROI, params)
        
        x = vesicle_clusters[k1][k2][:,0]*1e-3
        y = vesicle_clusters[k1][k2][:,1]*1e-3
        z = vesicle_clusters[k1][k2][:,2]*1e-3
        
        
        f = syn_marker_data[k1][:,0]*1e-3
        j = syn_marker_data[k1][:,1]*1e-3
        k = syn_marker_data[k1][:,2]*1e-3
        
        radii = list(np.linspace(0,1.5,21))
        image_vol = params['true_roi_size'][0]*params['true_roi_size'][1]*params['true_roi_size'][2]*1e-9
        
        uni_ripleyk = ripleyk.calculate_ripley(radii, volume, d1=x, d2=y, d3=z, CSR_Normalise=True)
        multi_ripleyk = rk.calculate_ripley(radii, image_vol, d1=f, d2=j, d3=k, s1=x, s2=y, s3=z, CSR_Normalise=True)
        

        storm_data.loc[i] = [Filename, 
                             Date, 
                             marker[0], 
                             marker[1], 
                             DIV, 
                             cellzone, 
                             ROI_label, 
                             ROI, 
                             points, 
                             nearest_dist_psd, 
                             nearest_dist_ves, 
                             volume, 
                             sphericity, 
                             uni_ripleyk, 
                             multi_ripleyk]

print('done')

# Add  new columns to the DataFrame

storm_data['mean_coloc'] = np.nan
storm_data['stderror_coloc'] = np.nan
storm_data['point_count'] = np.nan

#add the values to the DataFrame
for i, row in storm_data.iterrows():
    storm_data.loc[i, 'mean_coloc'] = row['nearest_neighbor_680'].mean()
    storm_data.loc[i, 'stderror_coloc'] = row['nearest_neighbor_680'].std()
    storm_data.loc[i, 'point_count'] = storm_data.loc[i,'points'].shape[0]
    


storm_data.to_pickle(target_file_path)  
