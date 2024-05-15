import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import scoreatpercentile
from dask.distributed import Client, as_completed

from logio.my_log import my_log
from logio.loading import ascii_loading_bar

from preproc.switch_type import switch_type
from preproc.filter_columns_by_site import filter_columns_by_site

from nets.nets_load_match import nets_load_match
from nets.nets_deconfound_single import nets_deconfound_single
from nets.nets_deconfound_multiple import nets_deconfound_multiple

from memmap.addBlockToMmap import addBlockToMmap
from memmap.MemoryMappedDF import MemoryMappedDF

from dasktools.connect_to_cluster import connect_to_cluster

from script_01_12_to_15 import construct_and_deconfound_ct

# ------------------------------------------------------------------------------
# This code is based on the matlab version of the confounds code which has the
# below structure:
# ------------------------------------------------------------------------------
# - script_01_09_gen_ct_conf_gpu_no_cluster.m*
#       Creates crossed confounds, regresses all other confounds out of them,
#       regresses non-crossed out of IDPs too (using parpool but no cluster),
#       saves blocks of crossed confounds to ct_q files in ws_01 folder.
#
# - script_01_09_gen_ct_conf_gpu.m
#       Identical to above?
#
# - script_01_10_gen_jobs.m
#       Lists the jobs to be run (which are submissions of script_01_12) as bash
#       scripts for the cluster.
#
# - script_01_11_gen_ct_conf.sh
#       Submits jobs using fsl_sub (removing previous files where needed)
#
# - script_01_12_gen_ct_conf.sh**
#       Bash script which calls to and runs func_01_12_gen_ct_conf.m
#
# - func_01_12_gen_ct_conf.m**
#       Computes variance explained for each crossed term.
#
# - script_01_13_gen_ct_conf_no_cluster.m*
#       Thresholds the crossed terms based on variance explained and makes plots.
#
# - script_01_13_gen_ct_conf.m
#       Same as above, but doesn't regenerate failed threshold jobs.
#
# - script_01_14_gen_ct_conf.sh
#       Concatenates thresholded results into one file.
#
# - script_01_15_gen_ct_conf.m
#       Cleaning up and reducing crossed terms
#
# *script not used.
# **script not called directly from general.m.
# ------------------------------------------------------------------------------
# The updated code has been refactored for ease and now has the below structure.
# ------------------------------------------------------------------------------
#
# - script_01_09_to_12.py
#    This file contains a single high level function which handles all cluster 
#    interaction, cleanup and file handling. This covers the functionality that
#    was previously found in:
#
#       script_01_10_gen_jobs.m, script_01_11_gen_ct_conf.sh, 
#       script_01_12_gen_ct_conf.sh, script_01_13_gen_ct_conf.m
#       script_01_14_gen_ct_conf.sh, script_01_15_gen_ct_conf.m
#
# - script 01_12_to_15.py
#    This file contains the code submitted for a single job. It covers the
#    functionality previously found in:
#
#       script_01_09_gen_ct_conf_gpu.m, func_01_12_gen_ct_conf.m
#
# ------------------------------------------------------------------------------

def generate_crossed_confounds_cluster(IDPs, confounds, nonlinear_confounds, data_dir, out_dir, cluster_cfg=None, logfile=None):

    # Update log
    my_log(str(datetime.now()) +': Stage 5: Generating Crossed Terms.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Loading and preprocessing...', mode='a', filename=logfile)
    
    # --------------------------------------------------------------------------
    # Check the confounds and nonlinear confounds are in a useful format.
    # --------------------------------------------------------------------------    
    # Convert input to memory mapped dataframes if it isn't already
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF')
    confounds = switch_type(confounds, out_type='MemoryMappedDF')
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF')

    # --------------------------------------------------------------------------
    # Concatenate and store confounds
    # --------------------------------------------------------------------------   

    # Combine the two
    confounds_full = pd.concat([confounds[:,:],nonlinear_confounds[:,:]], axis=1)
    confounds_full = MemoryMappedDF(confounds_full)

    # Add groupings
    groups = {**confounds.__dict__['groups'],**nonlinear_confounds.__dict__['groups']}
    
    # Loop through groups
    for group_name in groups:
    
        # Read in the current variable group
        current_group = groups[group_name]
    
        # Initialise empty list for this group
        updated_group = []
        
        # Loop through the variables
        for variable in current_group:
    
            # Check if its in the reduced confounds
            if (variable in nonlinear_confounds.columns) or (variable in confounds.columns):
    
                # Add to updated_group
                updated_group = updated_group + [variable]
    
        # If the new groups not empty save it as a group in the new memory mapped df
        if len(updated_group) > 0:
    
            # Add the updated group
            confounds_full.set_group(group_name, updated_group)

    # Set nonlinear confound group
    confounds_full.set_group('nonlin', list(nonlinear_confounds.columns))

    # --------------------------------------------------------------------------
    # Get the subject ids
    # --------------------------------------------------------------------------
    sub_ids = nonlinear_confounds.index
    
    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)
    
    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)
    
    # Initialize indSite as a list to hold the indices
    inds_per_site = {}
    
    # Loop over each value in site ids
    for site_id in (unique_site_ids + 1):
    
        # Find the indices where all elements in a row of siteDATA match the current valueSite
        # Note: This assumes siteDATA and siteValues have compatible shapes or values for comparison
        indices = np.where((site_ids == site_id-1).all(axis=1))[0]
    
        # Append the found indices to the indSite list
        inds_per_site[site_id] = indices
    
    # Delete the indices
    del indices
    
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

    # Update log
    my_log(str(datetime.now()) +': Data Loaded and preprocessed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Deconfounding IDPs...', mode='a', filename=logfile)
    
    # -------------------------------------------------------------------------
    # Deconfound IDPs
    # -------------------------------------------------------------------------

    # Switch type to reduce transfer costs
    confounds_full = switch_type(confounds_full, out_type='filename')
    IDPs = switch_type(IDPs, out_type='filename')
    
    # Deconfound IDPs
    IDPs = nets_deconfound_multiple(IDPs, confounds_full, 'nets_svd', 
                                    blksize=blksize, coincident=False,
                                    cluster_cfg=cluster_cfg)

    # Update log
    my_log(str(datetime.now()) +': IDPs deconfounded.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Connecting to cluster...', mode='a', filename=logfile)

    # -------------------------------------------------------------------------
    # Get the number of crossed terms (should be around 42,000)
    # -------------------------------------------------------------------------
    
    # Initialise empty dict to store headers
    columns_for_sites = {}
    
    # Number of crossed terms we will consider
    n_ct = 0
    n_ct_per_site = {}

    # Number of confound in each site
    n_conf_per_site = {}
    
    # Create a dict of site-specific column headers
    for site_index in (unique_site_ids + 1):
    
        # Get the columns for this site
        columns_for_sites[site_index] = filter_columns_by_site(confounds, 
                                                               site_index, return_df=False)
    
        # Add nonlinear columns
        columns_for_sites[site_index] = columns_for_sites[site_index] + \
                                        filter_columns_by_site(nonlinear_confounds, 
                                                               site_index, return_df=False)
        
        # Add the number of confounds for this site
        n_conf_per_site[site_index] = int(len(columns_for_sites[site_index]))
        
        # Add the number of crossed terms for this site
        n_ct_per_site[site_index] = int((len(columns_for_sites[site_index])-1)*(len(columns_for_sites[site_index]))/2)
        n_ct = n_ct + n_ct_per_site[site_index]

    
    # -------------------------------------------------------------------------
    # Get the index matrices for generating crossed terms
    # -------------------------------------------------------------------------

    # This array gives the indices for the site-specific confounds in the crossed 
    # term confound matrix. e.g. crossed_terms[:,site_idx[i]:site_idx[i+1]] are 
    # crossed terms for site i.
    site_idx = np.cumsum([n_ct_per_site[site] for site in n_ct_per_site])
    site_idx = np.insert(site_idx,0,0)

    # Initialise empty matrix
    crossed_inds = np.array(np.zeros((n_ct,3)),dtype='int16')

    # We now construct the crossed_inds matrix. This is interpreted as follows: row
    # k represents the k-th confound - it is constructed from the product of the
    # crossed_inds[k,1]^th and crossed_inds[k,2]^th terms from site number
    # crossed_inds[k,0].
    for i in range(len(site_idx)-1):
    
        # Set the site indices
        crossed_inds[site_idx[i]:site_idx[i+1],0]=i
    
        # Set the indices for the first crossed factor
        crossed_inds[site_idx[i]:site_idx[i+1],1] = np.concatenate([np.repeat(i+1, i+1) for i in range(n_conf_per_site[i+1]-1)])
        
        # Set the indices for the second crossed factor
        crossed_inds[site_idx[i]:site_idx[i+1],2] = np.concatenate([np.arange(i+1) for i in range(n_conf_per_site[i+1]-1)])

    # Save crossed inds as a memory map (The nan check is just an easy way of getting all indices).
    addBlockToMmap(os.path.join(os.getcwd(),'temp_mmap', 'crossed_inds.dat'), 
                   crossed_inds, np.where(~np.isnan(crossed_inds)),
                   crossed_inds.shape, dtype='int16')

    # --------------------------------------------------------------------------------
    # Connect to the cluster
    # --------------------------------------------------------------------------------
    cluster, client = connect_to_cluster(cluster_cfg)

    # -------------------------------------------------------------------------
    # Construct and deconfound crossed terms
    # -------------------------------------------------------------------------

    # Switch type to reduce transfer costs
    IDPs = switch_type(IDPs, out_type='filename')
    
    # Get the number of blocks we are breaking computation into
    num_blks = int(np.ceil(n_ct/blksize))
    
    # Get the indices for each block
    idx = np.arange(n_ct)
    blocks = [idx[i*blksize:min((i+1)*blksize,n_ct)] for i in range(num_blks)]

    # Scatter the data across the workers
    scattered_conf = client.scatter(confounds_full)
    scattered_IDPs = client.scatter(IDPs)
    scattered_data_dir = client.scatter(data_dir)
    scattered_crossed_inds = client.scatter(os.path.join(os.getcwd(),'temp_mmap', 'crossed_inds.dat'))
    scattered_mode = client.scatter('nets_svd')
    scattered_blksize = client.scatter(blksize)
    
    # Empty futures list
    futures = []
    
    # Update log
    my_log(str(datetime.now()) +': Connected to cluster.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Constructing and deconfounding (all) crossed terms...', mode='a', filename=logfile)
    
    # Loop through each block
    for block in blocks:

        # Submit job to the local cluster
        future_i = client.submit(construct_and_deconfound_ct,
                                 scattered_IDPs, scattered_conf, 
                                 scattered_data_dir,
                                 scattered_crossed_inds,
                                 scattered_mode, scattered_blksize, 
                                 block, pure=False)
    
        # Append to list 
        futures.append(future_i)
    
    # Completed jobs
    completed = as_completed(futures)

    # Set loading bar to zero
    my_log(ascii_loading_bar(0), mode='a', filename=logfile)
    
    # Wait for results
    j = 0
    for i in completed:
        i.result()
        j = j+1
        my_log(ascii_loading_bar(100*j/num_blks), mode='r', filename=logfile)
    
    # Delete the future objects.
    del i, completed, futures, future_i

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    
    # Update log
    my_log(str(datetime.now()) +': Computing variance explained...', mode='a', filename=logfile)
    
    # Close the cluster and client
    client.close()
    client.shutdown()

    # Delete the objects for good measure
    del client, cluster

    # Switch types back
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF')
    confounds_full = switch_type(confounds_full, out_type='MemoryMappedDF')
    
    # Read in resulting variance explained and convert to dataframe
    ve_ct = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 've_ct.npy'), dtype=np.float64,
                      shape=(IDPs.shape[1], n_ct),mode='r')[:,:]
    ve_ct = pd.DataFrame(ve_ct)
    
    # Get the average and maximum variance explained
    avg_ve = ve_ct.mean()
    max_ve = ve_ct.max()
    
    # Get percentage thresholds
    thr_for_avg = scoreatpercentile(avg_ve.dropna(), 99.9)
    thr_for_ve = max(1, scoreatpercentile(ve_ct.dropna(), 99.999))
    
    # Find indices where average is larger than threshold
    inds_for_avg = np.where(avg_ve>thr_for_avg)[0]
    inds_for_ve = np.where(ve_ct.values>thr_for_ve)
    
    # Get sorted unique ct indices
    inds_ct = np.unique(np.concatenate((inds_for_ve[1],inds_for_avg)))

    # Create empty reduced confounds dataframe
    conf_ct_reduced = pd.DataFrame(np.zeros((n_sub, len(inds_ct))), index=confounds.index, dtype=np.float64).astype('object')
    
    # Create empty mean ve
    avg_ve_reduced = pd.DataFrame(np.zeros((1, len(inds_for_avg))), dtype=np.float64).astype('object')

    # Update log
    my_log(str(datetime.now()) +': Variance explained computed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Recomputing reduced crossed terms...', mode='a', filename=logfile)
    
    # List for column names for all cts
    col_names = []
    
    # List for column names for average ve selected cts
    col_names_avg_ve = []
    
    # Counter for average ve
    avg_ve_count = 0
    
    # Loop reduced crossed inds
    for new_row, row in enumerate(inds_ct):
    
        # Get indices to construct confound
        site_no = crossed_inds[int(row),0]
        i = crossed_inds[int(row),1]
        j = crossed_inds[int(row),2]
    
        # Get the confounds for this site
        confounds_for_site = filter_columns_by_site(confounds_full, site_no+1, return_df=False)
    
        # Get confounds i and j
        conf_i = confounds_full[:,confounds_for_site[i]].values
        conf_j = confounds_full[:,confounds_for_site[j]].values
    
        # Work out column name
        col_names = col_names + [confounds_for_site[i] + '__x__' + confounds_for_site[j]]
    
        # Compute crossed term
        conf_ct_reduced.iloc[:,new_row] = conf_i*conf_j
    
        # Save details about average ves
        if row in inds_for_avg:
        
            # Work out column name
            col_names_avg_ve = col_names_avg_ve + [confounds_for_site[i] + '__x__' + confounds_for_site[j]]
    
            # Save mean variance explained
            avg_ve_reduced.iloc[0,avg_ve_count] = avg_ve[int(row)]
    
            # Update counter
            avg_ve_count = avg_ve_count + 1
    
    # Set column names
    conf_ct_reduced.columns = col_names
    avg_ve_reduced.columns = col_names_avg_ve

    # Update log
    my_log(str(datetime.now()) +': Reduced crossed terms recomputed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Deconfounding reduced crossed terms...', mode='a', filename=logfile)
    
    # Perform deconfounding
    conf_ct_reduced = nets_deconfound_single(conf_ct_reduced, confounds, col_names, 
                                             mode='nets_svd', demean=True, 
                                             dtype=np.float64, return_df=True)
    
    # Remove computational zeros
    conf_ct_reduced[conf_ct_reduced.abs()<1e-10]=0
    
    # Tables directory
    tables_dir = os.path.join(out_dir,'tables')
    
    # Check if the 'tables' directory exists
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    
    # Write avg ve results to file
    with open(os.path.join(tables_dir, 'mean_ve_ct.txt'), 'w') as f:
        for i in list(avg_ve_reduced.columns):
            f.write(f"{i} {avg_ve_reduced.loc[0,i]:.6f}\n")
    
    # Write other ve results to file
    with open(os.path.join(tables_dir, 've_ct.txt'), 'w') as f:
        for k in range(len(inds_for_ve[0])):
    
            # Get indices for ve
            IDP_ind = inds_for_ve[0][k]
            ct_ind = inds_for_ve[1][k]
            
            # Get indices to construct confound name
            site_no = crossed_inds[ct_ind,0]
            i = crossed_inds[ct_ind,1]
            j = crossed_inds[ct_ind,2]
                
            # Get the confounds for this site
            confounds_for_site = filter_columns_by_site(confounds_full, site_no+1, return_df=False)
        
            # Work out column name
            col_name = confounds_for_site[i] + '__x__' + confounds_for_site[j]
    
            # Write to file
            f.write(f"{col_name} {IDPs.columns[IDP_ind]} {ve_ct.loc[IDP_ind,ct_ind]:.6f}\n")
    
    # Output the list of crossed confound terms
    with open(os.path.join(out_dir,'tables','list_ct.txt'),'w') as f:
        for name in list(conf_ct_reduced.columns):
            f.write(name + '\n')

    # Update log
    my_log(str(datetime.now()) +': Reduced crossed terms deconfounded.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Saving results...', mode='a', filename=logfile)
    
    # Create output memorymapped dataframe
    confounds_with_ct = pd.concat((confounds_full[:,:], conf_ct_reduced),axis=1)
    confounds_with_ct = MemoryMappedDF(confounds_with_ct)

    # Add groupings (This code is over-complicated but I left it this way in case we
    # ever want to output conf_ct_reduced as a MemoryMappedDF before this and give 
    # it groupings).
    groups = {**confounds_full.__dict__['groups']}
    
    # Loop through groups
    for group_name in groups:
    
        # Read in the current variable group
        current_group = groups[group_name]
    
        # Initialise empty list for this group
        updated_group = []
        
        # Loop through the variables
        for variable in current_group:
    
            # Check if its in the reduced confounds
            if (variable in confounds_full.columns):
    
                # Add to updated_group
                updated_group = updated_group + [variable]
    
        # If the new groups not empty save it as a group in the new memory mapped df
        if len(updated_group) > 0:
    
            # Add the updated group
            confounds_with_ct.set_group(group_name, updated_group)

    # Set nonlinear confound group
    confounds_with_ct.set_group('crossed terms', list(conf_ct_reduced.columns))
    
    # Update log
    my_log(str(datetime.now()) +': Results saved.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Stage 5 complete.', mode='a', filename=logfile)
    
    # Return memory mapped dataframe and IDPs
    return(IDPs, confounds_with_ct)


    
    
        
