import os
import warnings
import numpy as np
import pandas as pd

# ==========================================================================
#
# The below function takes in a data file and a set of ids and returns only
# those rows of the data in the file whose subject id appears in the ids. It
# assumes that the subject id appears in the first column of the file.
#
# --------------------------------------------------------------------------
#
# The function takes in the following inputs:
#
# --------------------------------------------------------------------------
#
#  - file (str): File name of data file.
#  - ids (np array): Array containing subject ids whose data we want.
#  - del_first_col (boolean): If true this will delete the original subject
#                             ID column (Optional)
#  - dtypes (dict): Dict of datatypes expected in the file. Providing this 
#                   dict may improve speed as pandas then doesn't have to 
#                   determine the datatypes manually. (Optional)
#  - col_names: list of str, default None, names to assign to the columns.
#  - use_mmap: bool, default False, whether to use memory mapping to load 
#              the file.
#
# --------------------------------------------------------------------------
#
# The function returns:
#
# --------------------------------------------------------------------------
#
#  - data_merged: numpy array, matched data with optional transformations
#                 applied.
# 
# ==========================================================================
def nets_load_match(file, ids, del_first_col=True, dtypes=None, 
                    col_names=None, use_mmap=False):

    # Determine the mode for reading the file, potentially using memmap
    mode = 'r' if use_mmap else None

    # Load data from file into a pandas df, considering memap and col names
    if col_names is not None:

        # Read in data with column names
        data = pd.read_csv(file, sep='\s+', header=None, dtype=dtypes,
                           names=col_names, memory_map=use_mmap)
        
    else:

        # Read in data without column names
        data = pd.read_csv(file, sep='\s+', header=None, dtype=dtypes, 
                           memory_map=use_mmap)

    # For merging purposes, we'll name the first column 'Subject ID'
    data.rename(columns={data.columns[0]: 'ID'}, inplace=True)

    # Convert ids to a DataFrame and ensure it's a column vector
    ids = pd.DataFrame(ids)
    if ids.shape[1] == 1:

         # Name the column for easier reference
        ids.columns = ['ID'] 

    else:

        # Always set the first column to 'Subject ID'
        ids = ids.reset_index(drop=True).rename(columns={0: 'ID'})

    # Check for duplicate IDs in the first column of data
    if data.iloc[:, 0].duplicated().any():
        warnings.warn('There are duplicate subject ids in ' + file + '. ' +
                      'The code will continue processing but something ' +
                      'may be wrong.')

    # If del_first_col is True, adjust for initial column removal
    if del_first_col:

        # Exclude the first column name if it's to be deleted
        data_columns = data.columns[1:]  

    else:

        # Use existing column names or default integer labels
        data_columns = data.columns  

    # Merge the ids DataFrame with the data DataFrame based on the ID column
    try:
        data_merged = pd.merge(ids, data, on='ID', how='left')
    except:
        print(ids, data)
        
    # If del_first_col is True, drop the first ID column after merging
    if del_first_col:

        # Get merged data
        data_merged.drop(columns='ID', inplace=True)

        # Rename columns after dropping 'ID' if col_names were
        # provided
        data_merged.columns = data_columns  
    
    return(data_merged)
