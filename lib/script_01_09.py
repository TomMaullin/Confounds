import os
import numpy as np
import pandas as pd
from src.nets.nets_load_match import nets_load_match
from src.nets.nets_deconfound import nets_deconfound
from src.memmap.MemoryMappedDF import MemoryMappedDF
from src.preproc.filter_columns_by_site import filter_columns_by_site

def gen_ct_conf(confounds, nonlinear_confounds, data_dir):
    
    # Get the subject ids
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
    
    # Initialise empty dict to store headers
    columns_for_sites = {}
    
    # Number of crossed terms we will consider
    n_ct = 0
    n_ct_per_site = {}
    
    # Create a dict of site-specific column headers
    for site_index in (unique_site_ids + 1):
    
        # Get the columns for this site
        columns_for_sites[site_index] = filter_columns_by_site(confounds, 
                                                               site_index, return_df=False)
    
        # Add nonlinear columns
        columns_for_sites[site_index] = columns_for_sites[site_index] + \
                                        filter_columns_by_site(nonlinear_confounds, 
                                                               site_index, return_df=False)
    
        # Add the number of crossed terms for this site
        n_ct_per_site[site_index] = int((len(columns_for_sites[site_index])-1)*(len(columns_for_sites[site_index]))/2)
        n_ct = n_ct + n_ct_per_site[site_index]
    
    # Get number of subjects
    n_sub = len(sub_ids)

    # Create empty dataframe for crossed terms
    conf_ct = pd.DataFrame(index=confounds.index)
    conf_ct2 = pd.DataFrame(index=confounds.index)

    # Loop through sites deconfounding each one seperately
    for site_index in (unique_site_ids + 1):

        print('Deconfounding crossed terms for site ' + str(site_index) + '...')

        # Get the number of subjects for this site
        n_sub_site_i = len(inds_per_site[site_index])
        
        # Get the non-crossed confounds for site i
        conf_site_i = filter_columns_by_site(confounds,site_index).iloc[inds_per_site[site_index],:]
        conf_nonlin_site_i = filter_columns_by_site(nonlinear_confounds,site_index).iloc[inds_per_site[site_index],:]
        
        # Combine the two
        conf_site_i = pd.concat([conf_site_i,conf_nonlin_site_i], axis=1)
        
        # We now need to initialise a memory map of size n_sub by n_ct_per_site[site_index]
        ct_site_i = pd.DataFrame(np.zeros((n_sub_site_i, n_ct_per_site[site_index])))

        # Get the site-specific columns
        site_cols = conf_site_i.columns
        
        # Current column we are adding crossed term for
        current_col = 0
        
        # List for column names
        col_names = []
        
        # Loop through generating confound terms
        for i in range(len(site_cols)):
        
            # Cross term i with term j
            for j in range(i):
        
                # Add column name
                col_names = col_names + [conf_site_i.columns[i] + '__x__' + conf_site_i.columns[j]]
        
                # Add crossed term
                ct_site_i.iloc[:,current_col] = conf_site_i.iloc[:,i]*conf_site_i.iloc[:,j]
                
                # Update current column
                current_col = current_col + 1
        
        # Update columns in df
        ct_site_i.columns = col_names

        # Run nets_deconfound
        conf_ct_site_i = nets_deconfound(ct_site_i, conf_site_i,
                                         'nets_svd', 
                                         check_nan_patterns=True)

        # Set computational zeros to actual zeros
        conf_ct_site_i[conf_ct_site_i.abs()<1e-10]=0

        # Set index
        conf_ct_site_i.index = inds_per_site[site_index]

        # Concatenate crossed terms 
        conf_ct = pd.concat((conf_ct,conf_ct_site_i),axis=1).fillna(0)
        
        print('Crossed terms for site ' + str(site_index) + ' deconfounded.')

    # Sort the columns of conf_ct
    sorted_columns = np.sort(conf_ct.columns)

    # Reorganise conf_ct as suggested
    conf_ct[[*sorted_columns]]

    # MARKER STILL TO DO DECONF MAIN

    # Convert to memory mapped dataframe
    conf_ct = MemoryMappedDF(conf_ct)

    # Return the crossed terms
    return(conf_ct)

        

