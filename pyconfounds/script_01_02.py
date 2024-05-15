import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

from nets.nets_normalise import nets_normalise
from nets.nets_load_match import nets_load_match
from nets.nets_inverse_normal import nets_inverse_normal
from nets.nets_deconfound_multiple import nets_deconfound_multiple

from preproc.switch_type import switch_type
from preproc.filter_columns_by_site import filter_columns_by_site

from memmap.MemoryMappedDF import MemoryMappedDF

from logio.my_log import my_log
from logio.loading import ascii_loading_bar

# =============================================================================
#
# This function deconfounded IDPs and the nonlinear confounds.
#
# -----------------------------------------------------------------------------
# 
# It takes the following inputs:
#
#  - data_dir (string): The directory containing the data.
#  - all_conf (string or MemoryMappedDF): The memory mapped confounds.
#  - IDPs (string or MemoryMappedDF): The memory mapped IDPs.
#  - cluster_cfg (dict): Cluster configuration dictionary containing the type
#                        of cluster we want to run (e.g. 'local', 'slurm', 
#                        'sge',... etc) and the number of nodes we want to run 
#                        on (e.g. '12').
#
# -----------------------------------------------------------------------------
#
# It returns:
#
#  - conf_nonlin (MemoryMappedDF): Memory mapped nonlinear confounds.
#  - IDPs_deconf (MemoryMappedDF): Memory mapped deconfounded IDPs
#
# =============================================================================
def generate_nonlin_confounds(data_dir, all_conf, IDPs, cluster_cfg, logfile=None):

    # Update log.
    my_log(str(datetime.now()) +': Stage 3: Generating nonlinear confounds.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Constructing squared and inormal terms...', mode='a', filename=logfile)
    
    # Convert input to memory mapped dataframes if it isn't already
    all_conf = switch_type(all_conf, out_type='MemoryMappedDF')
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF')

    # Confound groups we are interested in.
    conf_name = ['AGE', 'AGE_SEX', 'HEAD_SIZE',  'TE', 'STRUCT_MOTION', 
                 'DVARS', 'HEAD_MOTION', 'HEAD_MOTION_ST', 'TABLE_POS', 
                 'EDDY_QC']

    # Get all the confounds in the group
    conf_group = all_conf.get_groups(conf_name)
    
    # Get the subject ids
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
    
    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)

    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)

    # Initialize indSite as a list to hold the indices
    inds_per_site = []
    
    # Loop over each value in site ids
    for site_id in unique_site_ids:

        # Find the indices where all elements in a row of siteDATA match the current valueSite
        # Note: This assumes siteDATA and siteValues have compatible shapes or values for comparison
        indices = np.where((site_ids == site_id).all(axis=1))[0]

        # Append the found indices to the indSite list
        inds_per_site.append(indices)

    # Delete the indices
    del indices
    
    # -------------------------------------------------------------------------
    
    # Initialise empty array to store results
    conf_nonlin = pd.DataFrame(index=conf_group.index)

    # Load in confounds
    conf = all_conf[:,:]

    # Site number
    for site_index in (unique_site_ids + 1):
        
        # Subset the confounds to a specific site
        conf_group_site = filter_columns_by_site(conf_group, site_index)

        # Get indices for the current site
        site_indices = inds_per_site[site_index-1] 

        # Reduce to just the indices we're interested in
        conf_group_site = conf_group_site.iloc[site_indices, :]

        # Get all the confounds at the site
        all_conf_site = conf.iloc[site_indices, :]

        # Get index
        site_index = all_conf_site.index

        # Get conf_group_site squared
        conf_group_site_squared = nets_normalise(conf_group_site**2)
        conf_group_site_squared.columns = [f"{col}_squared" for col in conf_group_site_squared.columns]

        # Get conf_group_site inverse normalised
        conf_group_site_inormal = nets_inverse_normal(conf_group_site);
        conf_group_site_inormal.columns = [f"{col}_inormal" for col in conf_group_site_inormal.columns]

        # Get conf_group_site squared inverse normalised
        conf_group_site_squared_inormal = nets_inverse_normal(conf_group_site_squared);
        conf_group_site_squared_inormal.columns = [f"{col}_inormal" for col in conf_group_site_squared_inormal.columns]

        # Concatenate them side by side
        conf_group_site_nonlin = pd.concat([conf_group_site_squared,
                                            conf_group_site_inormal,
                                            conf_group_site_squared_inormal], axis=1)
        conf_group_site_nonlin.index = site_index

        # Catch any nans from fully empty columns (we'll drop these later)
        conf_group_site_nonlin = conf_group_site_nonlin.fillna(0)

        # -------------------------------------------------------
        # Deconfound for this site
        # -------------------------------------------------------

        # Perform deconfounding
        conf_nonlin_deconf = nets_deconfound_multiple(conf_group_site_nonlin,
                                                      all_conf_site,
                                                      mode='svd',
                                                      blksize=blksize,
                                                      logfile=logfile)
        
        # Reindex the dataframe to fill off-site values with zeros
        conf_nonlin_deconf = conf_nonlin_deconf.reindex(conf_group.index).fillna(0)

        # # Drop any columns with only 5 values or less
        # na_columns = ((~conf_group_site.isna()).sum(axis=0) >= 5)
    
        # # Columns for squared
        # na_columns_squared = na_columns.copy()
        # na_columns_squared.index = [column + '_squared' for column in na_columns_squared.index]
        
        # # Columns for inormal
        # na_columns_inormal = na_columns.copy()
        # na_columns_inormal.index = [column + '_inormal' for column in na_columns_inormal.index]
        
        # # Columns for squared inormal
        # na_columns_squared_inormal = na_columns.copy()
        # na_columns_squared_inormal.index = [column + '_squared_inormal' for column in na_columns_squared_inormal.index]
        
        # # Combine
        # na_columns = pd.concat((na_columns_squared,na_columns_inormal,na_columns_squared_inormal))
        
        # # Subset columns
        # conf_nonlin_deconf = conf_nonlin_deconf.loc[:, na_columns]
        
        # Concatenate results
        conf_nonlin = conf_nonlin.join(conf_nonlin_deconf, how='outer')
    
    # ---------------------------------------------------------
    # Reorder the columns of conf_nonlin for readability.
    # ---------------------------------------------------------
            
    # Empty array for column names
    col_names = []

    # Get column names
    for group in conf_name:

        # Get the variable names in the current group
        col_names_group = list(all_conf.get_groups(group).columns)

        # Construct the nonlinear version
        col_names_group = [col_name + '_squared' for col_name in col_names_group] + \
                          [col_name + '_inormal' for col_name in col_names_group] + \
                          [col_name + '_squared_inormal' for col_name in col_names_group]

        # Update col_names
        col_names = col_names + col_names_group

    # Remove any columns that were dropped
    col_names = [col_name for col_name in col_names if col_name in conf_nonlin.columns]

    # Reorder the columns
    conf_nonlin = conf_nonlin[[*col_names]]

    # Reindex (the rows are no longer in the correct order because we have been subsetting
    # based on site)
    conf_nonlin = conf_nonlin.reindex(sub_ids)
        
    # Update log
    my_log(str(datetime.now()) +': Squared and inormal terms constructed.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Deconfounding nonlinear terms...', mode='a', filename=logfile)
    
    # ---------------------------------------------------------
    # Create memory map for nonlinear confound output
    # ---------------------------------------------------------
    
    # Create memory mapped df for non linear confounds
    conf_nonlin = MemoryMappedDF(conf_nonlin)
    
    # Loop through groups adding names
    for group in conf_name:
        
        # Get the variable names in the group from all_conf
        var_names = all_conf.groups[group.lower()]
        
        # Loop through the variables and create the corresponding 
        # nonlinear confound names
        new_var_names = [var_name + "_squared" for var_name in var_names] + \
                        [var_name + "_inormal" for var_name in var_names] + \
                        [var_name + "_squared_inormal" for var_name in var_names]
        
        # Set the new var_names for conf_nonlin
        conf_nonlin.set_group(group + '_nonlin', new_var_names)
        
    # ---------------------------------------------------------
    # Deconfound IDPs
    # ---------------------------------------------------------

    # Switch type to reduce transfer costs
    all_conf = switch_type(all_conf, out_type='filename', fname=os.path.join(os.getcwd(),'temp_mmap','conf.dat'))
    IDPs = switch_type(IDPs, out_type='filename', fname=os.path.join(os.getcwd(),'temp_mmap','IDPs.dat'))
    
    # Deconfound IDPs
    if cluster_cfg is None:
        
        # Run nets deconfound and get output
        IDPs_deconf = nets_deconfound_multiple(IDPs, all_conf, 'nets_svd', 
                                               blksize=blksize, coincident=False, 
                                               logfile=logfile)
    
    else:
        
        # Run nets_deconfound
        IDPs_deconf = nets_deconfound_multiple(IDPs, all_conf, 'nets_svd', 
                                               cluster_cfg=cluster_cfg, blksize=blksize, 
                                               coincident=False, logfile=logfile)
    
    # Update log
    my_log(str(datetime.now()) +': Nonlinear terms deconfounded.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Saving results...', mode='a', filename=logfile)
    
    # Remove the shared version of confounds
    if os.path.exists(os.path.join(os.getcwd(),'temp_mmap','conf.dat')):  

        # Change back to memory map
        all_conf = switch_type(os.path.join(os.getcwd(),'temp_mmap','conf.dat'), 
                               out_type='MemoryMappedDF')

        # Remove files
        all_conf.cleanup()
        del all_conf

    # Remove the shared version of IDPs
    if os.path.exists(os.path.join(os.getcwd(),'temp_mmap','IDPs.dat')):  

        # Change back to memory map
        IDPs = switch_type(os.path.join(os.getcwd(),'temp_mmap','IDPs.dat'), 
                           out_type='MemoryMappedDF')

        # Remove files
        IDPs.cleanup()
        del IDPs
    
    # Create memory mapped df for deconfounded IDPs
    IDPs_deconf = MemoryMappedDF(IDPs_deconf)
    
    # Update log
    my_log(str(datetime.now()) +': Results saved.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Stage 3 complete.', mode='a', filename=logfile)
    
    # Return the result
    return(conf_nonlin, IDPs_deconf)