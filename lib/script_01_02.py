import os
import shutil
import numpy as np
import pandas as pd

from src.preproc.switch_type import switch_type

from src.nets.nets_load_match import nets_load_match
from src.nets.nets_normalise import nets_normalise
from src.nets.nets_inverse_normal import nets_inverse_normal
from src.nets.nets_deconfound import nets_deconfound

from src.preproc.filter_columns_by_site import filter_columns_by_site

from src.memmap.MemoryMappedDF import MemoryMappedDF

def generate_nonlin_confounds(data_dir, all_conf, IDPs):

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
    
    # Initialise empty array to store results
    conf_nonlin = pd.DataFrame(index=conf_group.index)

    # Site number
    for site_index in (unique_site_ids + 1):
        
        # Subset the confounds to a specific site
        conf_group_site = filter_columns_by_site(conf_group, site_index)

        # Get indices for the current site
        site_indices = inds_per_site[site_index-1] 

        # Reduce to just the indices we're interested in
        conf_group_site = conf_group_site.iloc[site_indices, :]

        # Get all the confounds at the site
        all_conf_site = all_conf[:,:].iloc[site_indices, :]

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

        # -------------------------------------------------------
        # Deconfound for this site
        # -------------------------------------------------------

        # Perform deconfounding
        conf_nonlin_deconf = nets_deconfound(conf_group_site_nonlin,
                                             all_conf_site,
                                             'svd')
        
        # Reindex the dataframe to fill off-site values with zeros
        conf_nonlin_deconf = conf_nonlin_deconf.reindex(conf_group.index).fillna(0)
        
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
        
    # ---------------------------------------------------------
    # Create memory maps for output
    # ---------------------------------------------------------
    
    # Create memory mapped df for non linear confounds
    conf_nonlin = MemoryMappedDF(conf_nonlin)
        
    # Deconfound IDPs
    IDPs_deconf = nets_deconfound(IDPs[:,:], all_conf[:,:], 'nets_svd', conf_has_nans=False)
    
    # Create memory mapped df for deconfounded IDPs
    IDPs_deconf = MemoryMappedDF(IDPs_deconf)
    
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
        
    # Return the result
    return(conf_nonlin, IDPs_deconf)