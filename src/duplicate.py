import os
import numpy as np
import pandas as pd
from src.nets.nets_load_match import nets_load_match
from src.nets.nets_inverse_normal import nets_inverse_normal
from src.nets.nets_normalise import nets_normalise

# ==========================================================================
#
# The below function takes in categorical variables, ids and site_ids. For
# each site, it then creates a new set of confounds representing the 
# categorical confound at that site. Levels of the categorical variable with
# 8 or less observations are removed and all variables are normalised by 
# site.
#
# Note: This is a deprecated version of this function, used for testing
# purposes.
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
# - final_confs (pd Dataframe): The new confound matrix.
#
# --------------------------------------------------------------------------
def duplicate_categorical_deprecated(conf_name, ids, subject_indices_by_site, data_dir):

    # Get the names of the confounds
    names_file = os.path.join(data_dir, '..', 'NAMES_confounds',f'{conf_name}.txt')
    
    # Get the names of confounds
    with open(names_file, 'r') as file:
        base_name = [line.strip() for line in file.readlines()]

    # Load in confound by name
    file_to_load = os.path.join(data_dir, f'ID_{conf_name}.txt')
    values = nets_load_match(file_to_load, ids)

    # Number of sites and number of variables
    num_sites = len(subject_indices_by_site)
    num_vars = values.shape[1]

    # Create a numpy array
    final_confs = pd.DataFrame(np.array([]).reshape(len(ids), 0))

    # Loop through variables
    for k in range(num_vars):

        # Loop through sites
        for i, subj_by_site in enumerate(subject_indices_by_site):
            
            # Get the number of new columns we are adding
            num_new_cols = 0

            # Extract the subset of values for subjects at the site for the variable of interest
            values_at_site = values.iloc[subj_by_site, k]

            # Use value_counts() to count occurrences of each unique value directly
            count_uniq_values = values_at_site.value_counts().sort_index()

            # Get the unique non-nan values for this variable and site
            uniq_values = np.array(count_uniq_values.index)

            # Remove the diff values with counts less than 8
            uniq_values_to_remove = [j for j, count in enumerate(count_uniq_values) if count <= 8]
            uniq_values = np.delete(uniq_values, uniq_values_to_remove)

            # Get the number of unique values
            num_uniq_values = len(uniq_values)

            # If we have more than one value we need to add variables for this site
            if num_uniq_values > 1:

                # Get the number of new variables we're adding
                num_new_cols = num_uniq_values - 1

                # Initialise new empty confound
                new_confound = pd.DataFrame(np.zeros((len(ids), num_uniq_values - 1)))

                # Initialise empty array for new confound names
                new_confound_names = []

                # We now loop through the unique values
                for j, uniq_value in enumerate(uniq_values):

                    # Get the indices for this level of the categorical variable
                    level_indices = values_at_site[values_at_site == uniq_value].index

                    # We are going to construct a between-category difference confound
                    # For a given level of a categorical variable, this will consist
                    # of -1 for every obervation belonging to the first level of the
                    # categorical variable, and 1 for every observation belonging
                    # to the specified level.
                    if j == 0:

                        # Set the value for the first level in all confounds to -1
                        new_confound.loc[level_indices, :] = -1

                    else:

                        # Set the value for this level in this confound to 1
                        new_confound.loc[level_indices, j - 1] = 1

                        # Get the indices of the entries we have assigned values to
                        ind_not_zero = np.where(new_confound.iloc[:, j - 1] != 0)[0]

                        # If we have more than 1 different value
                        if len(ind_not_zero) > 1:  

                            # Normalise the data
                            new_confound.iloc[ind_not_zero, j - 1] = nets_normalise(new_confound.iloc[ind_not_zero, j - 1])

                        # If all entries have the same value, it is possible that the 
                        # division by standard deviation of zero in the normalising step
                        # could create a column of all NaNs. In this case, we want to 
                        # replace all NaNs with zeros.
                        if np.all(np.isnan(new_confound.iloc[ind_not_zero, j - 1])):
                            new_confound.iloc[ind_not_zero, j - 1] = 0

                    # Add to the running count of new columns
                    if j != 0:

                        # Append names
                        new_confound_names.append(f"{base_name[k]}_{j}_Site_{i+1}")

                # Set the names on the dataframe
                new_confound.columns = new_confound_names    

                # Perform an outer merge; assumes that both DataFrames are aligned by their index
                final_confs = final_confs.merge(new_confound, 
                                                left_index=True, right_index=True, 
                                                how='outer', sort=True).fillna(0)
    
    # Return result 
    return(final_confs)


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
def duplicate_categorical(conf_name, ids, subject_indices_by_site, data_dir):
    
    # Get the names of the confounds
    names_file = os.path.join(data_dir, '..', 'NAMES_confounds',f'{conf_name}.txt')
    
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
                dummies = (dummies - np.nanmean(dummies, axis=0)) / np.nanstd(dummies, axis=0)
                dummies = dummies.fillna(0)

                # Create new confound names
                new_confound_names = (f"{base_name[k]}_{j+1}_Site_{i+1}" for j in np.arange(dummies.shape[1]))

                # Set names
                dummies.columns = new_confound_names
                
                # Number of output dummy vairables
                n_dummies = dummies.shape[1]

                # Merge or concatenate the new columns to the final DataFrame
                confs_out = confs_out.join(dummies, how='outer').fillna(0)
                
                # Retain the original nan values
                confs_out.iloc[original_nans,-n_dummies:] = np.nan
                

    # Return the final confounds
    return(confs_out)


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