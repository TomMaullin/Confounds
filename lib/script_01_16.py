import os
import shutil
import numpy as np
import pandas as pd

from src.nets.nets_smooth_single import nets_smooth_single
from src.nets.nets_normalise import nets_normalise
from src.nets.nets_deconfound_multiple import nets_deconfound_multiple

# siteDATA = site_ids
# subset_IDPs_i = IDPs

# As long as everything is a dataframe new_index_sortedTime is just sub_ids and
# index_sortedTime is just sub_ids_sorted

def generate_smoothed_and_pca(IDPs, confounds, nonIDPs, data_dir, out_dir, cluster_cfg):

    # Get the subject IDs
    sub_ids = IDPs.index
    
    # -------------------------------------------------------------------------
    # Estimate the block size (number of subjects we want to allow in memory at
    # a given time).
    # -------------------------------------------------------------------------
    # Developer note: The below is only a rough estimate, but is fairly robust
    # as it is a little conservative. The rule of thumb is to take the maximum
    # amount of memory (MAXMEM) divide it by the number of subjects we have,
    # divide by 64 (as each observation is float64 at most) and then divide by
    # 8 (as we may want to make several copies of whatever we load in, but we
    # rarely make more than 8). The resulting number should be roughly the
    # number of columns of a dataframe we are able to load in at a time. This
    # doesn't need to be perfect as often python can handle more - it is just
    # a precaution, and does improve efficiency substantially.
    # -------------------------------------------------------------------------
    
    # Rough estimate of maximum memory (bytes)
    MAXMEM = 2**32

    # Get the number of subjects
    n_sub = len(sub_ids)

    # Block size computation
    blksize = int(MAXMEM/n_sub/8/64)
    
    # -------------------------------------------------------------------------
    # Deconfound IDPs
    # -------------------------------------------------------------------------

    # Switch type to reduce transfer costs
    confounds = switch_type(confounds, out_type='filename')
    IDPs = switch_type(IDPs, out_type='filename')
    
    # Deconfound IDPs
    IDPs_deconf = nets_deconfound_multiple(IDPs, confounds, 'nets_svd', 
                                           blksize=blksize, coincident=False,
                                           cluster_cfg=cluster_cfg)
    
    # Switch IDPs back
    IDPs = switch_type(IDPs, out_type='pandas')

    # -------------------------------------------------------------------------
    # Preprocess time variables
    # -------------------------------------------------------------------------

    # Get day fraction (time of day)
    day_fraction = nonIDPs[:,'TOD']

    # Normalise day fraction
    conf_acq_time_linear = nets_normalise(day_fraction)
    conf_acq_time_linear = conf_acq_time_linear.fillna(0)

    # Sort acquisition times
    conf_acq_time_linear = conf_acq_time_linear.sort_values(by='TOD')

    # Get sorted indices
    sub_ids_sorted = conf_acq_time_linear.index

    # -------------------------------------------------------------------------
    # Reorder dataframes and variables
    # -------------------------------------------------------------------------

    # Sort IDPs and IDPs_deconf based on sorted sub_ids
    IDPs_sorted = IDPs.loc[sub_ids_sorted,:]
    IDPs_deconf_sorted = IDPs_deconf.loc[sub_ids_sorted,:]
    
    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)
    site_ids.index = sub_ids

    # Sort site ids
    site_ids_sorted = site_ids.loc[sub_ids_sorted,:]

    # -------------------------------------------------------------------------
    # Get the indices for observations for each site in the sorted data
    # -------------------------------------------------------------------------
    
    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)
    
    # Initialize indSite as a list to hold the indices
    inds_per_site_sorted = {}
    
    # Loop over each value in site ids
    for site_id in (unique_site_ids + 1):
    
        # Find the indices where all elements in a row of siteDATA match the current valueSite
        # Note: This assumes siteDATA and siteValues have compatible shapes or values for comparison
        indices = np.where((site_ids_sorted == site_id-1).all(axis=1))[0]
    
        # Append the found indices to the indSite list
        inds_per_site_sorted[site_id] = indices
    
    # Delete the indices
    del indices

    # Number of sites
    num_sites = len(inds_per_site_sorted)

    # -------------------------------------------------------------------------
    # Smoothing
    # -------------------------------------------------------------------------

    # Sigma value
    sigma = 0.1

    # Loop through sites
    for site_id in inds_per_site_sorted:
    
        # Get subjects for this site
        inds_site = inds_per_site_sorted[site_id]
    
        # Get the IDPs for this site
        IDPs_for_site = IDPs.iloc[inds_site,:]
    
        # Get IDPs for site as numpy array
        IDPs_for_site = IDPs_for_site.values

        # Acquisition time data for smoothing
        conf_acq_time_linear_site = conf_acq_time_linear.iloc[inds_site,:].values

        # Perform smoothing
        smoothed_IDPs_for_site = nets_smooth_single(conf_acq_time_linear_site, 
                                                    IDPs_for_site, conf_acq_time_linear_site, 
                                                    sigma, null_thresh=0.6)
        


    


    