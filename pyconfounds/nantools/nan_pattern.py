import os 
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
#
# Helper function to convert NaN values to 1 and non-NaN values to 0.
#
# ------------------------------------------------------------------------------
#
# This function takes as inputs:
#
#  - column (pd.Series): A column of data from a pandas dataframe.
#
# ------------------------------------------------------------------------------
#
# - list: boolean mask representing where nan values were seen, in binary form.
#
# ------------------------------------------------------------------------------
def nan_pattern(column):

    # Return the column of isnan values
    return column.isna().astype(int).tolist()