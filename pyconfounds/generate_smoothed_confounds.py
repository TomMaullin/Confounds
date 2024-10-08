import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

from pyconfounds.logio.my_log import my_log
from pyconfounds.logio.loading import ascii_loading_bar

from pyconfounds.preproc.switch_type import switch_type

from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF

from pyconfounds.nets.nets_svd import nets_svd
from pyconfounds.nets.nets_normalise import nets_normalise
from pyconfounds.nets.nets_load_match import nets_load_match
from pyconfounds.nets.nets_smooth_multiple import nets_smooth_multiple
from pyconfounds.nets.nets_deconfound_multiple import nets_deconfound_multiple

# =============================================================================
#
# This function takes in the confounds and IDPs, smooths the confounds
# temporarily according to scan date or time of day, performs PCA on the 
# smoothed terms and retains the components that explain 99% of the variance in
# the IDPs. The resulting components are added to the confounds matrix and 
# returned.
#
# This script was previously named script_01_16, reflecting the original matlab
# code.
#
# -----------------------------------------------------------------------------
# 
# It takes the following inputs:
#
#  - IDPs (string or MemoryMappedDF): The memory mapped IDPs.
#  - confounds (string or MemoryMappedDF): The memory mapped confounds.
#  - nonIDPs (string or MemoryMappedDF): The memory mapped nonIDPs.
#  - data_dir (string): The directory containing the input data.
#  - out_dir (string): The output directory for results to be saved to.
#  - cluster_cfg (dict): Cluster configuration dictionary containing the type
#                        of cluster we want to run (e.g. 'local', 'slurm', 
#                        'sge',... etc) and the number of nodes we want to run 
#                        on (e.g. '12').
#  - logfile (string): A html filename for the logs to be print to. If None, no
#                      logs are output.
#  - MAXMEM (int): Maximum amount of memory (in bits) that the code is allowed
#                  to work with. If MAXMEM is none (default) the code assumes
#                  the SPM default of 2^32.
#
# -----------------------------------------------------------------------------
# 
# It returns:
#
#  - IDPs_deconf (MemoryMappedDF): The IDPs deconfounded by the (original) 
#                                  confound matrix.
#  - confounds_with_smooth (MemoryMappedDF): The confound matrix with smoothed
#                                            terms added.
#
# =============================================================================
def generate_smoothed_confounds(IDPs, confounds, nonIDPs, data_dir, out_dir, 
                                cluster_cfg, logfile=None, MAXMEM=None):

    # Update log
    my_log(str(datetime.now()) +': Stage 7: Generating Smoothed Terms.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Loading and preprocessing...', mode='a', filename=logfile)
    
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
    if MAXMEM is None:
        MAXMEM = 2**32

    # Get the number of subjects
    n_sub = len(sub_ids)

    # Block size computation
    blksize = int(MAXMEM/n_sub/8/64)

    # Number of time points per block, no 8 is included here as
    # we only ever construct the relevant matrix once in 
    # nets_smooth_single (this is controlling the size of
    # the xeval*xdata matrix)
    blksize_time = int(MAXMEM/IDPs.shape[0]/64)

    # -------------------------------------------------------------------------
    # Deconfound IDPs
    # -------------------------------------------------------------------------

    # Check we have a temporary memmap directory
    if not os.path.exists(os.path.join(out_dir, 'temp_mmap')):
        os.makedirs(os.path.join(out_dir, 'temp_mmap'))

    # We first remove the IDPs columns that are more than 50% nan
    IDPs = switch_type(IDPs, out_type='pandas', out_dir=out_dir)
    
    # Calculate the percentage of NaN values in each column
    nan_pct = IDPs.isna().mean()
    
    # Create a boolean mask for columns to keep
    cols_to_keep = nan_pct[nan_pct <= 0.5].index
    
    # Return a new DataFrame with only the columns to keep
    IDPs = IDPs[cols_to_keep]

    # Update log
    my_log(str(datetime.now()) +': Data loaded and preprocessed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Deconfounding IDPs...', mode='a', filename=logfile)
    
    # Switch type to reduce transfer costs
    confounds_fname = switch_type(confounds, out_type='filename', out_dir=out_dir)
    IDPs_fname = switch_type(IDPs, out_type='filename', out_dir=out_dir)
    
    # Deconfound IDPs
    IDPs_deconf = nets_deconfound_multiple(IDPs_fname, confounds_fname, 'nets_svd', 
                                           blksize=blksize, coincident=False,
                                           cluster_cfg=cluster_cfg, 
                                           out_dir=out_dir, logfile=logfile)

    # Read IDPs into memory (we will need the whole IDP array)
    IDPs = switch_type(IDPs, out_type='pandas', out_dir=out_dir) 

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
    site_ids = nets_load_match(os.path.join(out_dir, 'ID_SITE.txt'), sub_ids)
    site_ids.index = sub_ids

    # Sort site ids
    site_ids_sorted = site_ids.loc[sub_ids_sorted,:]

    # -------------------------------------------------------------------------
    # Get the indices for observations for each site in the sorted data
    # -------------------------------------------------------------------------
    
    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)
    
    # Initialize indSite as a list to hold the indices
    inds_per_site = {}
    inds_per_site_sorted = {}
    
    # Loop over each value in site ids
    for site_id in (unique_site_ids + 1):
    
        # Find the indices where all elements in a row of siteDATA match the current valueSite
        # Note: This assumes siteDATA and siteValues have compatible shapes or values for comparison
        indices = np.where((site_ids == site_id-1).all(axis=1))[0]
        indices_sorted = np.where((site_ids_sorted == site_id-1).all(axis=1))[0]
    
        # Append the found indices to the indSite list
        inds_per_site[site_id] = indices
        inds_per_site_sorted[site_id] = indices_sorted
    
    # Delete the indices
    del indices_sorted, indices

    # Number of sites
    num_sites = len(inds_per_site_sorted)

    # Sigma value
    sigma = 0.1
    
    # -------------------------------------------------------------------------
    # Construct smoothed confounds for observations ordered by date
    # -------------------------------------------------------------------------
    
    # Dict to store smoothed IDPs and pca results
    smoothed_IDPs_dict = {}
    principal_components_dict = {}
    esm_dict = {}
    
    # Update log
    my_log(str(datetime.now()) +': IDPs deconfounded.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Smoothing date ordered IDPs...', mode='a', filename=logfile)
    
    # Loop through sites
    for site_id in inds_per_site:

        # Update log
        my_log(str(datetime.now()) +': Smoothing site ' + str(site_id) + '.', mode='a', filename=logfile)
    
        # Get subjects for this site
        inds_site = inds_per_site[site_id]
        
        # Get the IDPs for this site
        IDPs_for_site = IDPs_deconf.iloc[inds_site,:]
        
        # Get the acquisition times for this site
        times_for_site = nonIDPs[:,'SCAN_DATE'].iloc[inds_site,:]#MARKER conf_acq_time_linear.iloc[inds_site,:]
        
        # Smooth the IDPs
        smoothed_IDPs_for_site = nets_smooth_multiple(times_for_site, IDPs_for_site, sigma,
                                                      blksize=blksize, blksize_time=blksize_time,
                                                      cluster_cfg=cluster_cfg, out_dir=out_dir,
                                                      logfile=logfile)
    
        # Compute svd of IDPs
        principal_components, esm,_ = nets_svd(smoothed_IDPs_for_site.values)
    
        # Save results
        smoothed_IDPs_dict[site_id] = smoothed_IDPs_for_site
        principal_components_dict[site_id] = principal_components
        esm_dict[site_id] = esm

    
    # Update log
    my_log(str(datetime.now()) +': Date ordered IDPs smoothed.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Computing variance explained...', mode='a', filename=logfile)
    
    # Estimating the number of temporal components by choosing a number
    # that explains at least 99% of the variance in the smoothed IDPs.
    num_temp_comp = {}
    conf_acq_date_dict = {}
     
    # Loop through sites
    for site_id in principal_components_dict:
    
        # Maximum variance explained
        max_ve = 0
    
        # Get the principal components for this site
        principal_components_site = principal_components_dict[site_id]
    
        # Get the smoothed IDPs for this site
        smoothed_IDPs_site = smoothed_IDPs_dict[site_id]
    
        # Record index
        site_index = smoothed_IDPs_site.index
        
        # Current number of principal components that we have considered
        n_current = 1
    
        # Get columns of rows with all non-nan values
        non_nan_rows = ~smoothed_IDPs_site.isna().any(axis=1)
        
        # Filter principal components and smoothed IDPs row wise
        principal_components_site = principal_components_site[non_nan_rows.values,:]
        smoothed_IDPs_site = smoothed_IDPs_site[non_nan_rows].values
        
        # Loop through principal components until we have 99% variance explained
        while max_ve < 99:
    
            # Get n_current principal components
            current_pcs = principal_components_site[:,:n_current]
                
            # Compute variance explained in smoothed_IDPs_site by current_pcs
            current_pcs_pinv = np.linalg.pinv(current_pcs)
        
            # Compute projection
            proj = current_pcs @ (current_pcs_pinv @ smoothed_IDPs_site)
                        
            # Compute variance explained
            numerator = 100 * np.sum(proj.flatten() ** 2)
            denominator = np.sum(smoothed_IDPs_site.flatten() ** 2)
            max_ve = numerator / denominator
         
            # Check if max_ve is greater than 99
            if max_ve < 99:
    
                # Increment counter
                n_current = n_current + 1
                
        # Save number of components
        num_temp_comp[site_id] = n_current

        # Save new array
        principal_components_dict[site_id] = pd.DataFrame(principal_components_site[:,:n_current],
                                                          index=site_index)
        conf_acq_date_dict[site_id] = nets_normalise(principal_components_dict[site_id]).fillna(0)
        
    # Update log
    my_log(str(datetime.now()) +': Variance explained computed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Smoothing time ordered IDPs...', mode='a', filename=logfile)
    
    # Construct column names for temporal components
    tc_colnames = []
    
    # Loop through sites constructing colnames and converting principal components
    for site_id in conf_acq_date_dict:
    
        # Site columnnames 
        tc_colnames_site = ['DATE_Site_' + str(site_id) + '__' + str(pc_id) for pc_id in range(1,num_temp_comp[site_id]+1)]
    
        # Replace header on site specific dataframes
        conf_acq_date_dict[site_id].columns = tc_colnames_site
        
        # Update running column names
        tc_colnames = tc_colnames + tc_colnames_site
    
    # Number of estimated components in total
    num_temp_comp_total = 0
    
    # Sum values over sites
    for site_id in num_temp_comp:
        num_temp_comp_total = num_temp_comp_total + num_temp_comp[site_id]

    # Reconstruct principal components confound dataframe
    conf_acq_date = pd.DataFrame(np.zeros((n_sub,num_temp_comp_total)),
                                 index = sub_ids,
                                 columns = tc_colnames)
    
    # Loop through sites constructing colnames and converting principal components
    for site_id in conf_acq_date_dict:
    
        # Add in temporal components sorted
        conf_acq_date.update(conf_acq_date_dict[site_id])
        
    # Convert the indexing back to the original order
    conf_acq_date = conf_acq_date.loc[sub_ids,:]

    # -------------------------------------------------------------------------
    # Construct smoothed confounds for observations ordered by acquisition time
    # -------------------------------------------------------------------------
    
    # Dict to store smoothed IDPs and pca results (this time sorted by acquisition time)
    smoothed_IDPs_sorted_dict = {}
    principal_components_sorted_dict = {}
    esm_sorted_dict = {}
    
    # Loop through sites
    for site_id in inds_per_site_sorted:
        
        # Update log
        my_log(str(datetime.now()) +': Smoothing site ' + str(site_id) + '.', mode='a', filename=logfile)
        
        # Get subjects for this site
        inds_site = inds_per_site_sorted[site_id]
        
        # Get the IDPs for this site
        IDPs_for_site = IDPs_deconf_sorted.iloc[inds_site,:]
        
        # Get the acquisition times for this site
        times_for_site = conf_acq_time_linear.iloc[inds_site,:]
        
        # Smooth the IDPs
        smoothed_IDPs_for_site = nets_smooth_multiple(times_for_site, IDPs_for_site, sigma,
                                                      blksize=blksize, blksize_time=blksize_time,
                                                      cluster_cfg=cluster_cfg, out_dir=out_dir,
                                                      logfile=logfile)
    
        # Compute svd of IDPs
        principal_components_sorted, esm,_ = nets_svd(smoothed_IDPs_for_site.values)
    
        # Save results
        smoothed_IDPs_sorted_dict[site_id] = smoothed_IDPs_for_site
        principal_components_sorted_dict[site_id] = principal_components_sorted
        esm_sorted_dict[site_id] = esm
    
    # Estimating the number of temporal components by choosing a number
    # that explains at least 99% of the variance in the smoothed IDPs.
    num_temp_comp_sorted = {}
    conf_acq_time_dict = {}
     
    # Update log
    my_log(str(datetime.now()) +': Time ordered IDPs smoothed.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Computing variance explained...', mode='a', filename=logfile)
    
    # Loop through sites
    for site_id in principal_components_sorted_dict:
    
        # Maximum variance explained
        max_ve = 0
    
        # Get the principal components for this site
        principal_components_site = principal_components_sorted_dict[site_id]
    
        # Get the smoothed IDPs for this site
        smoothed_IDPs_site = smoothed_IDPs_sorted_dict[site_id]
    
        # Record index
        site_index = smoothed_IDPs_site.index
        
        # Current number of principal components that we have considered
        n_current = 1
    
        # Get columns of rows with all non-nan values
        non_nan_rows = ~smoothed_IDPs_site.isna().any(axis=1)
    
        # Filter principal components and smoothed IDPs row wise
        principal_components_site = principal_components_site[non_nan_rows.values,:]
        smoothed_IDPs_site = smoothed_IDPs_site[non_nan_rows].values
        
        # Loop through principal components until we have 99% variance explained
        while max_ve < 99:
    
            # Get n_current principal components
            current_pcs = principal_components_site[:,:n_current]
                
            # Compute variance explained in smoothed_IDPs_site by current_pcs
            current_pcs_pinv = np.linalg.pinv(current_pcs)
        
            # Compute projection
            proj = current_pcs @ (current_pcs_pinv @ smoothed_IDPs_site)
            
            # Compute variance explained
            numerator = 100 * np.sum(proj.flatten() ** 2)
            denominator = np.sum(smoothed_IDPs_site.flatten() ** 2)
            max_ve = numerator / denominator
            
            # Check if max_ve is greater than 99
            if max_ve < 99:
    
                # Increment counter
                n_current = n_current + 1
    
        # Save number of components
        num_temp_comp_sorted[site_id] = n_current
    
        # Save new array
        principal_components_sorted_dict[site_id] = pd.DataFrame(principal_components_site[:,:n_current],
                                                                 index=site_index)
        conf_acq_time_dict[site_id] = nets_normalise(principal_components_sorted_dict[site_id]).fillna(0)
    
    # Construct column names for temporal components
    tc_colnames = []
    
    # Loop through sites constructing colnames and converting principal components
    for site_id in conf_acq_time_dict:
    
        # Site columnnames 
        tc_colnames_site = ['ACQT_Site_' + str(site_id) + '__' + str(pc_id) for pc_id in range(1,num_temp_comp_sorted[site_id]+1)]
    
        # Replace header on site specific dataframes
        conf_acq_time_dict[site_id].columns = tc_colnames_site
    
        # Update running column names
        tc_colnames = tc_colnames + tc_colnames_site
        
    # Number of estimated components in total
    num_temp_comp_sorted_total = 0
    
    # Sum values over sites
    for site_id in num_temp_comp_sorted:
        num_temp_comp_sorted_total = num_temp_comp_sorted_total + num_temp_comp_sorted[site_id]

    # Update log
    my_log(str(datetime.now()) +': Variance explained computed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Saving results...', mode='a', filename=logfile)
    
    # Reconstruct principal components confound dataframe
    conf_acq_time = pd.DataFrame(np.zeros((n_sub,num_temp_comp_sorted_total)),
                                 index = sub_ids_sorted,
                                 columns = tc_colnames)
    
    # Loop through sites constructing colnames and converting principal components
    for site_id in conf_acq_time_dict:
    
        # Add in temporal components sorted
        conf_acq_time.update(conf_acq_time_dict[site_id])
    
    # Convert the indexing back to the original order
    conf_acq_time = conf_acq_time.loc[sub_ids,:]

    # -------------------------------------------------------------------------
    # Output results
    # -------------------------------------------------------------------------

    # Create output memorymapped dataframe
    confounds_with_smooth = pd.concat((confounds[:,:], conf_acq_date, conf_acq_time),axis=1)
    confounds_with_smooth = MemoryMappedDF(confounds_with_smooth, directory=out_dir)

    # Add groupings (This code is over-complicated but I left it this way in case we
    # ever want to output the smoothed confounds as MemoryMappedDFs before this and give 
    # them groupings).
    groups = {**confounds.__dict__['groups']}

    # Loop through groups
    for group_name in groups:
    
        # Read in the current variable group
        current_group = groups[group_name]
    
        # Initialise empty list for this group
        updated_group = []
        
        # Loop through the variables
        for variable in current_group:
    
            # Check if its in the reduced confounds
            if (variable in confounds.columns):
    
                # Add to updated_group
                updated_group = updated_group + [variable]
    
        # If the new groups not empty save it as a group in the new memory mapped df
        if len(updated_group) > 0:
    
            # Add the updated group
            confounds_with_smooth.set_group(group_name, updated_group)
    
    # Set date and time confound groups
    confounds_with_smooth.set_group('acq date', list(conf_acq_date.columns))
    confounds_with_smooth.set_group('acq time', list(conf_acq_time.columns))

    # Save deconfounded IDPs as memory mapped df
    IDPs_deconf = switch_type(IDPs_deconf, out_type='MemoryMappedDF',out_dir=out_dir)
    
    # Update log
    my_log(str(datetime.now()) +': Results saved.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Stage 7 complete.', mode='a', filename=logfile)
    
    # Return memory mapped dataframes
    return(IDPs_deconf, confounds_with_smooth)
    
    

    