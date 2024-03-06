import os
import numpy as np
import pandas as pd
from src.nets.nets_normalise import nets_normalise
from src.nets.nets_load_match import nets_load_match

# ==========================================================================
#
# The below function takes in continuous variables, ids and site_ids. For
# each site, it then creates a new set of confounds representing the 
# continuous confound at that site. Outlier detection is also performed,
# details of which can be found commented in the code.
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
# - data_dir (str): Data directory
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
def duplicate_demedian_norm_by_site(conf_name, ids, subject_indices_by_site, data_dir):

    # Construct file names
    names_file =  os.path.join(data_dir, '..', 'NAMES_confounds', f'{conf_name}.txt')
    values_file =  os.path.join(data_dir, f'ID_{conf_name}.txt')

    # Get the names of confounds
    with open(names_file, 'r') as file:
        base_names = [line.strip() for line in file.readlines()]

    # Get the variable values for the IDs we have
    values = nets_load_match(values_file, ids)

    # Get the number of variables and number of sites
    num_vars = values.shape[1]
    num_sites = len(subject_indices_by_site)

    # ----------------------------------------------------------------------------------
    # Outlier detection. 
    # ----------------------------------------------------------------------------------
    # Below is a description of what is happening here, taken from "Confound modelling 
    # in UK Biobank brain imaging":
    #
    # For any given confound, we define outliers thus: First we subtract the median
    # value from all subjects’ values. We then compute the median-absolute deviation 
    # (across all subjects) and multiply this MAD by 1.48 (so that it is equal to 
    # the standard deviation if the data had been Gaussian). We then normalise all 
    # values by dividing them by this scaled MAD. Finally, we define values as 
    # outliers if their magnitude is greater than 8.
    # ----------------------------------------------------------------------------------

    # Demedian globally each column (note this automatically
    # ignores nans)
    medians = values.median()
    values = values - medians

    # Get the absolute value of the demedianed data, multiplied
    # by 1.48
    mads = 1.48 * values.abs().median()

    # Identify columns where the absolute value of mads is less than np.finfo(float).eps
    mask = mads.abs() < np.finfo(float).eps

    # Calculate standard deviation for columns where mads is too small
    std_devs = values.loc[:, mask].std()

    # Update mads to use standard deviation for these columns
    mads[mask] = std_devs

    # Standardise with the final mads
    values = values/mads

    # Create an empty DataFrame
    confs_out = pd.DataFrame()  
    
    # Save the original nans
    original_nans = values.isna()

    # For each site
    for i, subj_by_site in enumerate(subject_indices_by_site):
        
        # Get the values for this variable at this site
        values_at_site = values.iloc[subj_by_site, :]

        # Get column medians
        medians_site = values_at_site.median()

        # Set extreme values to median
        values_at_site = values_at_site.where(values_at_site <= 8, np.nan)
        values_at_site = values_at_site.where(values_at_site >= -8, np.nan)

        # Fill the na values with the appropriate medians
        values_at_site.fillna(medians_site)

        # ------------------------------------------------------------------------
        # Normalise the dataframe, ignoring zeros
        # ------------------------------------------------------------------------

        # Normalise values at site
        values_at_site = nets_normalise(values_at_site, constant=0, mode='preserve')

        # Generate new column names 
        values_at_site.columns = [f'{name}_Site_{i+1}' for name in base_names]

        # Reindex values_at_site
        values_at_site = values_at_site.reindex(values.index).fillna(0)
        
        # Merge or concatenate the new columns to the final DataFrame
        if confs_out.empty:
            
            # Initialise
            confs_out = values_at_site
            
        else:
            
            # Aligning on indices, filling missing values with zeros
            confs_out = confs_out.join(values_at_site, how='outer')
            
        # Reinsert original nans
        temporary_nan_copy = confs_out.iloc[:,-num_vars:].copy()
        temporary_nan_copy.iloc[original_nans.values] = np.nan
        confs_out.iloc[:,-num_vars:] = temporary_nan_copy
            
    # Return the confounds
    return(confs_out)