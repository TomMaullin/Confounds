import os 
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Helper function to convert NaN values to 1 and non-NaN values to 0
# ------------------------------------------------------------------------------
def nan_pattern(column):

    # Return the column of isnan values
    return column.isna().astype(int).tolist()