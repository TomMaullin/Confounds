import os 
import numpy as np
import pandas as pd
from pyconfounds.nantools.nan_pattern import nan_pattern

# ------------------------------------------------------------------------------
#
# The compute NaN patterns function creates a dictionary of the unique patterns
# of NaN values seen in a dataframe. This function is currently unused as the
# patterns in the biobank data were too variable to make this approach efficient.
#
# ------------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# - 'data' (pandas dataframe): The pandas dataframe we wish to identify unique
#                              NaN patterns for.
# - `thresh` (int): If thresh = c, we return only those columns with more than c
#                   non-nan values.
#
# ------------------------------------------------------------------------------
#
# It returns as output:
#
# - 'nan_patterns' (dict): The dict contains all unique patterns of nans seen in
#                          the dataframe alongside the names of the columns 
#                          which contained those nan values.
#
# ------------------------------------------------------------------------------
def create_nan_patterns(data, thresh=0):

    # Initializing a dictionary to store the patterns
    nan_patterns = {}
    
    # Iterating through each column to categorize based on NaN pattern
    for column in data:

        # Using tuple for hashability
        pattern = tuple(nan_pattern(data[column]))  

        # Only record nan patterns if they are large enough
        if np.sum(~np.array(pattern,dtype=bool)) > thresh:
    
            # Check if the nan pattern is already in the pattern dictionary
            if pattern not in nan_patterns:
    
                # If it isn't add it to the dictionary.
                nan_patterns[pattern] = {'columns': [], 'pattern': list(pattern)}
    
            # Append the pattern to the appropriate name
            nan_patterns[pattern]['columns'].append(column)

    # Converting dictionary keys to consecutive integers
    nan_patterns_consecutive = {i: v for i, (k, v) in enumerate(nan_patterns.items())}

    # Return result
    return(nan_patterns_consecutive)