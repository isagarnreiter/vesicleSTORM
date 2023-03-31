#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:02:41 2023

@author: isabellegarnreiter
"""
import numpy as np
from numba import njit, prange
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def get_wide_field(data, image_size, kernel_size, sigma):
    
    #prep data for image
    data[:,2] =+ 485
    data = data/49660*image_size
    
    #initialise background with a buffer
    buffered_image_dim = tuple(map(lambda i, j: i + j, image_size, kernel_size))
    buffered_image = np.zeros(buffered_image_dim)

    #initialise the gaussian distribution with size being the size of the kernel and sigma defining the standard deviation of the point spread function
    kernel = np.fromfunction(lambda x, y, z : (1/(2*np.pi*sigma**2)) * np.exp((-1*((x-(kernel_size[0]-1)/2)**2+(y-(kernel_size[1]-1)/2)**2+(z-(kernel_size[2]-1)/2)**2)/(2*sigma**2))), kernel_size)
    kernel = kernel / np.max(kernel)

    #get the pixel location of the data
    point_coord = np.round(data).astype(int)
    #add gaussian point spread function at each point location
    for y, x, z in point_coord:
        buffered_image[x:x+kernel.shape[0], y:y+kernel.shape[1], z:z+kernel.shape[2]] += kernel
    
    #remove buffer from image
    image = buffered_image[kernel.shape[0]//2:-kernel.shape[0]//2, kernel.shape[1]//2:-kernel.shape[1]//2, kernel.shape[2]//2:-kernel.shape[2]//2]
    
    return image

def get_local_max_3d(img, kernelsize, std_threshold):
    # Get coordinates of local maxima 

    img2 = ndimage.maximum_filter(img, size=kernelsize)

    # Threshold the image to find locations of interest
    img_thresh = img2.mean() + img2.std() * std_threshold

    # Since we're looking for maxima find areas greater than img_thresh
    labels, num_labels = ndimage.label(img2 > img_thresh)

    # Get the positions of the maxima
    coords = ndimage.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))
    coords = np.array(coords)
    # Get the maximum value in the labels    

    return coords


def create_histogram(list_of_thresholds, ves_min_dists, saveas = "", save=False):
    plt.hist(ves_min_dists, bins = list_of_thresholds)
    plt.xlabel('Distance (nm)', fontsize = 18)
    plt.ylabel('Number of localisations', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if save == True:
        plt.savefig(f'/Users/isabellegarnreiter/Documents/vesicleSTORM/results/Graphs/{saveas}.png')
    plt.show()

def sort_dict_by_keywords(d, keywords):
    sorted_dict = {}
    for key in sorted(d.keys()):
        for kw in keywords:
            if kw in key:
                if kw in sorted_dict:
                    if type(d[key]) == float:
                        sorted_dict[kw] = np.append([sorted_dict[kw], d[key]])
                    
                    else:
                        sorted_dict[kw] = np.concatenate([sorted_dict[kw], d[key]])
                else:
                    if type(d[key]) == float:
                        sorted_dict[kw] = np.array([d[key]])
                    else:
                        sorted_dict[kw] = d[key]
    return sorted_dict

@njit(fastmath=True,parallel=True)
def calc_distance_squared_two(vec_1,vec_2):

    res=np.empty((vec_1.shape[0]),dtype=vec_1.dtype)
    for i in prange(vec_1.shape[0]):
        dist= np.sqrt((vec_1[i,0]-vec_2[:,0])**2+(vec_1[i,1]-vec_2[:,1])**2+(vec_1[i,2]-vec_2[:,2])**2)
        res[i] = np.min(dist[np.nonzero(dist)])
    return res


def analysis(list_of_files_for_analysis):
    min_dist_dict = {}
    for file_name in list_of_files_for_analysis:
        
        vesicles = pd.read_csv(file_name[0])[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
        synapse_marker = pd.read_csv(file_name[1])[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)

        distances = calc_distance_squared_two(vesicles, synapse_marker)

        new_file_name = f"{(file_name[0]).split('/')[4]}_{(file_name[0]).split('/')[5]}"

        min_dist_dict[new_file_name] = distances

    
    print("Done!")
    return min_dist_dict


def save_csv_files(target_path, list_of_files):

    for i in range(list_of_files.shape[0]):

        new_file_name_w1 = f"{(list_of_files[i,0]).split('/')[4]}_{(list_of_files[i,0]).split('/')[5]}_w1"
        new_file_name_w2 = f"{(list_of_files[i,1]).split('/')[4]}_{(list_of_files[i,0]).split('/')[5]}_w2"

        new_folder = str(i)
        new_path = os.path.join(target_path, new_folder)
        print(new_path)
        os.mkdir(new_path)

        new_path_w1 = new_path+'/'+(list_of_files[i,0]).split('/')[-1]
        new_path_w2 = new_path+'/'+(list_of_files[i,1]).split('/')[-1]

        shutil.copyfile(list_of_files[i,0], new_path_w1)
        shutil.copyfile(list_of_files[i,1], new_path_w2)

        os.rename(new_path_w1, new_path+'/'+new_file_name_w1)
        os.rename(new_path_w2, new_path+'/'+new_file_name_w2)
    
    