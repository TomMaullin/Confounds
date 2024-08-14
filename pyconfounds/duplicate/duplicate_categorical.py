import os
import numpy as np
import pandas as pd
from pyconfounds.nets.nets_normalise import nets_normalise
from pyconfounds.nets.nets_load_match import nets_load_match

# ==========================================================================
#
# The below function takes in categorical variables, ids and site_ids. For
# each site, it then creates a new set of confounds representing the 
# categorical confound at that site. Levels of the categorical variable with
# 8 or less observations are removed and all variables are normalised by 
# site.
#
# --------------------------------------------------------------------------
#
# The function takes in the following inputs:
#
# --------------------------------------------------------------------------
#
# - conf_name (str): Confound name, as a string, matching the filename
# - ids (npy array): Numpy array of subject IDs
# - subject_indices_by_site (list of numpy arrays): The j^th numpy array
#           contains a list of subject ids for the subjects in site j.
# - data_dir (str): Data directory.
# - preserve_nans (bool): Boolean indicating whether we should preserve the
#                         nan values in the original array (default: False)
#
# --------------------------------------------------------------------------
#
# The function returns:
#
# --------------------------------------------------------------------------
#
# - confs_out (pd Dataframe): The new confound matrix.
#
# ==========================================================================
def duplicate_categorical(conf_name, ids, subject_indices_by_site, data_dir, 
                          preserve_nans=False):
    
    # Get the names of the confounds
    names_file = os.path.join(data_dir, '..', 'NAMES_confounds',f'{conf_name}.txt')

    # Check if the names file exists, if not we must have created it
    if not os.path.exists(names_file):
        
        # In this case, the names were generated by the code recently
        # and are currently saved in the outputs directory.
        names_file = os.path.join(data_dir, f'{conf_name}.txt')
    
    # Get the names of confounds
    with open(names_file, 'r') as file:
        base_name = [line.strip() for line in file.readlines()]

    # Load in confound by name
    file_to_load = os.path.join(data_dir, f'ID_{conf_name}.txt')
    values = nets_load_match(file_to_load, ids)

    # Create an empty DataFrame
    confs_out = pd.DataFrame(index=values.index) 

    # Loop through the variables
    for k in range(values.shape[1]):  
        
        # Retain the original nan values
        original_nans = values.iloc[:,k].isna()
        
        # Loop through the sites
        for i, site_ids in enumerate(subject_indices_by_site):
            
            # Get the values for this variable at this site
            values_at_site = values.iloc[site_ids, k]
            
            # Get the value counts for this varaible at this site
            counts = values_at_site.value_counts()
            
            # Get the levels of the categorical variable that have at
            # least 8 observations
            valid_levels = counts[counts > 8].index
            
            # Get the number of unique values
            num_uniq_values = len(valid_levels)

            # If we have more than one value we need to add variables for this site
            if num_uniq_values > 1:

                # Filter the levels with too few observations
                values_at_site[~values_at_site.isin(valid_levels)] = np.nan

                # One-hot encode filtered values
                dummies = pd.get_dummies(values_at_site)

                # Ensure data type is signed integer (int8)
                dummies = dummies.astype('int8')

                # Subtract the first column of the dummies from the rest
                dummies.iloc[:, 1:] = dummies.iloc[:, 1:].sub(dummies.iloc[:,0], axis=0)

                # Remove the first column from dummies
                dummies = dummies.iloc[:, 1:]

                # Normalize 
                dummies = dummies.replace(0, np.nan)
                dummies = (dummies - np.nanmean(dummies, axis=0)) / np.nanstd(dummies, axis=0, ddof=1)
                dummies = dummies.fillna(0)

                # Create new confound names
                new_confound_names = (f"{base_name[k]}_{j+1}_Site_{i+1}" for j in np.arange(dummies.shape[1]))

                # Set names
                dummies.columns = new_confound_names
                
                # Number of output dummy vairables
                n_dummies = dummies.shape[1]

                # Merge or concatenate the new columns to the final DataFrame
                confs_out = confs_out.join(dummies, how='outer')

                # Replace the nans introduced by the join for indices representing other sites
                confs_out.iloc[:,-n_dummies:] = confs_out.iloc[:,-n_dummies:].fillna(0)
                
                # Reintroduce the original nan values
                if preserve_nans:
                    confs_out.iloc[original_nans,-n_dummies:] = np.nan
                

    # Return the final confounds
    return(confs_out)
