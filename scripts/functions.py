#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:02:41 2023

@author: isabellegarnreiter
"""
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import math
from skimage import measure
import matplotlib.pyplot as plt


def get_default_params():
    params = {}
    #gaussian filtering parameters
    params['true_roi_size'] = (49660,49660,1100) #size of the 3d Tiff in microns
    params['sf']  = (68, 68, 180) #scaling factor: defines the size of a pixel in microns in the downsampled image.
    params['kernel_size'] = (40,40,2) #kernel size of the gaussian filter
    params['sigma'] = 10 #intensity of the gaussian filter
    params['max_threshold_ves'] = 4 #threshold for the extraction of intensity blobs in the image

    #fine tuning parameters
    params['min_peak_dist'] = 16 #min distance between 2 peaks (in pixels) - if distance is smaller, the 2 peaks are merged
    params['min_cluster_area'] = 32 #min area of a cluster (in pixels)
    params['max_cluster_area'] = 32000 #max area of a cluster (in pixels)
    params['647_channel'] = 'channel1' #either channel1 or channel2
    params['filter_680'] = True #if true: will apply same density based clustering algorithm to data in the 680 channel and extract points
    params['data_for_wf'] = "both" #which data to use to create the simulated widefield image. 'Both'takes in all channels, '647' only points in the 647 channel or '680' for points in the 680 channel
    return params


def map_to_im(data, og_dimensions, new_dimensions):
    """
    this function maps the location of points onto an image of lower dimension
    
    data: shape: (N,3) , contains all the points located within the ROI
    og_dimensions: original dimension of the image dtype=tuple
    new_dimensions: new dimension of the image, dtype=tuple
    """
    
    data_t = data.copy()
    data_t = data_t/og_dimensions*new_dimensions
    
    image = np.zeros(new_dimensions)
    
    #get the pixel location of the data
    point_coord = np.round(data_t).astype(int)
    for x, y, z in point_coord:
        image[x, y, z-1] += 1

    return image


def get_gaussiankde(data, params):
    """
    This function takes an image of dots and fits a gaussian function to each dot to create an intensity profile of the image
    
    image: 3-dimensional image containing pixel location of dots
    kernel_size: 3-dimensional size of the gaussian peak
    sigma: standard deviation of the gaussian function
    """

    og_dimensions = params['true_roi_size'] 
    sf = params['sf']
    kernel_size = params['kernel_size']
    sigma = params['sigma']
    new_dimensions = (og_dimensions[0]//sf[0], og_dimensions[1]//sf[1], og_dimensions[2]//sf[2])

    image = map_to_im(data, og_dimensions, new_dimensions)

    #normally, the processing steps after STORM romove noisy non-blinking data. If these are not removed, they increase the intensity variance of the generated gaussian approximation.
    #This next steps minimizes this effect by finding pixel-indices with an unreasonnable amount of points are removing them. 
    # unwanted_pixels = np.where(image>100)
    # image[unwanted_pixels] = 0
    
    indices = np.array(np.where(image>100))
    image[indices.T] = 0
    image_t = image.copy() 
    kx, ky, kz = kernel_size[0], kernel_size[1], kernel_size[2]
    
    #initialise background with a buffer
    image_buf = np.pad(image_t, ((kx//2,),(ky//2,), (kz//2,)))

    #initialise the gaussian distribution
    kernel = np.fromfunction(lambda x, y, z : (1/(2*np.pi*sigma**2)) * np.exp((-1*((x-kx/2)**2+(y-ky/2)**2+(z-kz/2)**2)/(2*sigma**2))), kernel_size)
    kernel = kernel / np.max(kernel)

    #add gaussian point spread function at each point location
    gaussian_image = np.zeros(image_buf.shape) 
    for x in range(0, image_t.shape[0]):
        for y in range(0, image_t.shape[1]):
            for z in range(0, image_t.shape[2]):
                if image_t[x,y,z] > 0:
                    gaussian_image[x:x+kx, y:y+ky, z:z+kz] += kernel * image_t[x,y,z]
    
    #remove buffer from image
    gaussian_image = gaussian_image[kx//2:-kx//2, ky//2:-ky//2, kz//2:-kz//2]
    return gaussian_image   


def filter_peaks(coords, min_distance):
    """Filters a list of coordinates to merge points that are too close to one another"""
    filtered_coords = [coords[0]]
    
    for i in range(1, len(coords)):
        prev_coord = filtered_coords[-1]
        curr_coord = coords[i]
        dist = math.dist(prev_coord, curr_coord)
        
        if dist < min_distance:
            # Calculate midpoint between the two points
            midpoint = ((prev_coord[0] + curr_coord[0]) / 2, (prev_coord[1] + curr_coord[1]) / 2, (prev_coord[2] + curr_coord[2]) / 2)
            # Replace previous coordinate with midpoint
            filtered_coords[-1] = np.round(midpoint).astype(int)
        else:
            # Add current coordinate to the list
            filtered_coords.append(curr_coord)
            
    return filtered_coords


def get_clusters(widefield_image, params):

    """
    This function takes a list of points defined in x,y,z coordinates and outputs a 3D image of blobs 
    in which the points are most densely packed.  
    """
    

    def filter_clusters(img, min_area, max_area):
        """Filters clusters in 3D image based on their area"""
        regions = measure.regionprops(img)
        filtered_img = np.zeros_like(img)
        for region in regions:
            if region.area >= min_area and region.area<=max_area:
                for coord in region.coords:
                    filtered_img[coord[0], coord[1], coord[2]] = region.label

        return filtered_img
    
    max_threshold_ves = params['max_threshold_ves']
    min_peak_dist = params['min_peak_dist'] 
    min_cluster_area = params['min_cluster_area']
    max_cluster_area = params['max_cluster_area']

    #calculate the intensity threshold for the large PSF images, depending on an arbitrary intensity threshold, dependent on the mean and std of each image.
    threshold = widefield_image.mean() + widefield_image.std() * max_threshold_ves
    #create a mask of the large PSF images where for the pixels above the threshold
    mask = (widefield_image > threshold) * 1

    #get the local peaks, defining the central coordinate of synapses
    peak_coords = peak_local_max(widefield_image,labels = mask)
    
    #filter the peak distances to exclude peaks which are too close to one another and replace them by the  midway point
    peak_coords = np.array(filter_peaks(peak_coords, min_peak_dist))

    #create labels for each peak
    shell = np.zeros(widefield_image.shape, dtype=bool)
    shell[tuple(peak_coords.T)] = True
    markers, _ = ndi.label(shell)

    #apply watershed to seperate clusters, based on the previously computed mask and the local peaks
    synapse_clusters = watershed(-widefield_image, markers = markers, mask=mask, watershed_line=True)

    #filtered_clusters
    filtered_clusters = filter_clusters(synapse_clusters, min_cluster_area, max_cluster_area)
    return filtered_clusters


def seperate_clusters(img):
    """
    This function takes in the image of indidual blobs (img) and outputs a dictionary 
    where each value corresponds to each blob in the smallest bounding box possible and each key
    being the blobs label.
    """
    
    bounding_boxes = {}
    labels = np.unique(img)[1:]

    # loop over each labeled blob
    for label in labels:

        # find indices of all pixels with the same label value
        indices = np.where(img == label)
        # find max/min values of indices to create bounding box
        min_x = np.min(indices[0])
        max_x = np.max(indices[0])
        min_y = np.min(indices[1])
        max_y = np.max(indices[1])
        min_z = np.min(indices[2])
        max_z = np.max(indices[2])

        # create bounding box array and add to list
        bounding_box = img[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
        bounding_boxes[label] = bounding_box
        
    return bounding_boxes

def get_points(data, filtered_clusters, seperate_clusters, params):
    """
    This function extracts points located within defined areas and stores them in a new dictionary vesicle_clusters_loc.
    
    data: location of points in x, y, z. Shape = (N,3) with N the number of points in the ROI
    filtered_clusters: 3-dimensional areas in pixels
    """
    
    sf = params['sf']
    
    # Dictionary to store point locations for each vesicle cluster
    vesicle_clusters_loc = {}
    
    for location in data:
        # Extract x, y, and z coordinates from the location
        x, y, z = location
        
        # Convert coordinates to indices based on scaling factors
        i, j, k = int(round(x / sf[0]))-1, int(round(y / sf[1])-1), int(round(z / sf[2])-1)
        
        # Check if the corresponding filtered cluster value is greater than 0
        
        index = filtered_clusters[i, j, k]
        
        # Check if the index already exists in the vesicle_clusters_loc dictionary
        if index in vesicle_clusters_loc:
            vesicle_clusters_loc[index] = np.append(vesicle_clusters_loc[index], [location], axis=0)
        else:
            vesicle_clusters_loc[index] = np.array([location])
        
        # Remove the entry with key 0 from vesicle_clusters_loc if it exists
        if 0 in list(vesicle_clusters_loc.keys()):
            vesicle_clusters_loc.pop(0)

    for key in seperate_clusters:
        if key not in vesicle_clusters_loc:
            vesicle_clusters_loc[key] = []
    
    # Return the dictionary of vesicle cluster point locations
    return vesicle_clusters_loc


# Define a function to calculate the volume and sphericity of an ROI
def calculate_volume_sphericity(roi, params):
    """
    This function calculates the volume (in µms) and spherecity of an ROI
    If the area is too small to calculate its spherecity (1 pixel) the function wil result in a spherecity of 1
    instead of resulting in an error.
    The spherecity isn't a perfect measure considering the scaling factor is quite large.
    """
    
    pixel_size = params['sf'][0]*params['sf'][1]*params['sf'][2]
    region = measure.regionprops(roi)[0]

    # Loop through each region and extract the size and sphericity
    volume = region.area * (pixel_size*1e-9)
    dn = region.equivalent_diameter_area
    maj_axis = region.axis_major_length
    if maj_axis == 0:
        sphericity = 1
    else:
        sphericity = dn/maj_axis
        
    return volume, sphericity



def simulation_batch(batch_nb, dim, size):
    '''
    NOT UPDATED
    
    Function to generate simulations of SMLM data for labelled synaptic proteins.
    
    Input:
    batch_nb = number of simulations to produce
    dim = resolution of the images - in the case of the nikon STORM microscope - 17x17x50
    size = size of the field of view
    
    The simulation initialises a random density of background noise (between 1e-10 and 1e-9) for the given field of view.
    Clusters vary in number (1 to 8), size, number of points contained (50 to 200) and location.
    
    Output:
    batch = dictionary containing arrays of point locations (x,y,z) for all the simulations generated.
    
    '''
    
    
    # Function to generate random points around a given center
    def generate_points_within_range(center, num_points, ranges):
        x = np.random.uniform(center[0]-ranges[0]/2, center[0]+ranges[0]/2, num_points)
        y = np.random.uniform(center[1]-ranges[1]/2, center[1]+ranges[1]/2, num_points)
        z = np.random.uniform(center[2]-ranges[2]/2, center[2]+ranges[2]/2, num_points)
        return np.column_stack((x, y, z))
    
    
    dxyz = np.array(dim)*np.array(size)
    
    batch = {}

    sim_params = {}
    sim_params['background point density'] = []
    sim_params['number of clusters'] = []
    sim_params['cluster ranges'] = []
    sim_params['points per cluster'] = []
    sim_params['cluster locations'] = []
    
    for n in range(0,batch_nb):
        
        # Define the total number of data points
        bckgrd_density = np.random.uniform(1e-10, 1e-9)
        bckgrd = int(bckgrd_density*dxyz[1]*dxyz[1]*dxyz[2])

        # Generate a random number of clusters between 5 and 15
        num_clusters = np.random.randint(1, 8)

        # Generate random cluster centers
        x = np.random.randint(0, dxyz[0], size=(num_clusters))
        y = np.random.randint(0, dxyz[1], size=(num_clusters))
        z = np.random.randint(0, dxyz[2], size=(num_clusters))

        centers = np.array([x,y,z]).T

        # Generate random background noise
        xb = np.random.randint(0, dxyz[0], size=(bckgrd))
        yb = np.random.randint(0, dxyz[1], size=(bckgrd))
        zb = np.random.randint(0, dxyz[2], size=(bckgrd))

        bckgrd_points = np.array([xb,yb,zb]).T

        # Generate range for the clusters
        xr = np.random.uniform(dim[0]*30, dim[0]*150, size=num_clusters)
        yr = np.random.uniform(dim[1]*30, dim[1]*100, size=num_clusters)
        zr = np.random.uniform(dim[2]*1, dim[2]*5, size=num_clusters)

        ranges = np.array([xr,yr,zr]).T

        # Generate data points for each cluster
        number_points = []
        data_points = []
        for i in range(num_clusters):
            num_points = np.random.randint(50, 200)
            cluster_points = generate_points_within_range(centers[i], num_points, ranges[i])
            number_points.append(num_points)
            data_points.append(cluster_points)

        # Combine all the data points into a single array
        data_points = np.vstack(data_points)

        # Add background noise
        data_points_with_noise = np.vstack((data_points, bckgrd_points))
        
        
        sim_params
        batch[n] = data_points_with_noise
        
        sim_params['background point density'].append(bckgrd_density)
        sim_params['number of clusters'].append(num_clusters)
        sim_params['cluster ranges'].append(ranges)
        sim_params['points per cluster'].append(number_points)
        sim_params['cluster locations'].append(centers)
        
        
    return batch, sim_params