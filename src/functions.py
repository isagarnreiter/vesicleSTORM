#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:02:41 2023

@author: isabellegarnreiter
"""
import numpy as np
from numba import njit, prange
import pandas as pd



def sort_dict_by_keywords(d, keywords):
    sorted_dict = {}
    for key in sorted(d.keys()):
        for kw in keywords:
            if kw in key:
                if kw in sorted_dict:
                    sorted_dict[kw] = np.concatenate([sorted_dict[kw], d[key]])
                else:
                    sorted_dict[kw] = d[key]
    return sorted_dict


@njit(fastmath=True,parallel=True)
def calc_distance_squared_two(vec_1,vec_2):

    res=np.empty((vec_1.shape[0]),dtype=vec_1.dtype)
    for i in prange(vec_1.shape[0]):
        res[i] = np.min(np.sqrt((vec_1[i,0]-vec_2[:,0])**2+(vec_1[i,1]-vec_2[:,1])**2+(vec_1[i,2]-vec_2[:,2])**2))
        
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
