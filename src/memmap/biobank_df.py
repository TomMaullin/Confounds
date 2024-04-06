import os
import uuid
import atexit
import pickle
import shutil
import fnmatch
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_expr._collection import Index
from src.nantools.create_nan_patterns import create_nan_patterns



# ==================================================================================
#
# This class initialises a dataframe for uk biobank data extending the dask 
# dataframe class.
#
# Developers note: A lot of the code below seems like it could be done more
# efficiently by using class inheritance properties. I tried this first but I was
# getting deep errors from somewhere within the dask_expr library. For this reason,
# methods have just been copied over/manually inherited as and when they were
# needed.
#
# ==================================================================================
class biobank_df():
    

    # ------------------------------------------------------------------------------
    # Init function for biobank df
    # ------------------------------------------------------------------------------
    def __init__(self, df, index=None, columns=None, subset_rows=None, sep=r'\s+', dtypes=None):

        # Check if the df is a filename
        if type(df)==str:

            # Read in df
            df = dd.read_csv(df, sep=sep, header=None, dtype=dtypes)
            
        # Set the dataframe index
        if index is not None:
            df = df.set_index(index, sorted=True)

        # Set the dataframe columns
        if columns is not None:
            df.columns = columns
        
        # Check if we are including all rows
        if subset_rows is not None:
            df = df.loc[list(subset_rows)]
            
        # Set the dataframe
        self.df = df

        # Save the columns and row indices
        self.columns = df.columns
        self.index = df.index # Note this will be a dask index object which may not be computed
        self.shape = None # Note this is computed under compute_shape

        # ID groups for quick access of groups of variables
        self.groups = {}

        # Add loc indexer
        if hasattr(df, 'loc'):
            self.loc = biobank_loc(df.loc)

        # Add iloc indexer
        if hasattr(df, 'iloc'):
            self.iloc = biobank_iloc(df.iloc)
    
    # ------------------------------------------------------------------------------
    # Compute shape (developer note: this is a seperate function, rather than a
    # field as obtaining the number of rows requires a compute operation and it may 
    # not be desirable to perform this compute repeatedly throughout the code).
    # ------------------------------------------------------------------------------
    def compute_shape(self):

        # Set the index
        if type(self.index) ==  Index:
            self.index = df.index.compute()

        # Set the shape
        if self.shape is None:
            self.shape = (len(self.index),len(self.columns))
    
    # ------------------------------------------------------------------------------
    # Set column index
    # ------------------------------------------------------------------------------
    def set_columns(self,columns):

        # Save columns
        self.columns = columns

        # Modify df accordingly
        self.df.columns = columns
        
    # ------------------------------------------------------------------------------
    # Set row index
    # ------------------------------------------------------------------------------
    def set_rows(self,index):

        # Save index
        self.index = index

        # Modify df accordingly
        self.df.index = index
        
    # ------------------------------------------------------------------------------
    # Compute function
    # ------------------------------------------------------------------------------
    def compute(self):
        return(self.df.compute())

    # ------------------------------------------------------------------------------
    # Drop function (Note: inplace functionality is not currently supported)
    # ------------------------------------------------------------------------------
    def drop(self, *args, **kwargs):
        return(biobank_df(self.df.drop(*args, **kwargs)))
        
    # ------------------------------------------------------------------------------
    # Function for indexing (basically we are just replacting df indexing here).
    # ------------------------------------------------------------------------------
    def __getitem__(self, key):
        return(self.df.__getitem__(key))
    
    # ------------------------------------------------------------------------------
    # The set group function allows us to setup quick access groups of variables. 
    # ------------------------------------------------------------------------------
    #
    # This function takes the following inputs:
    # - group_name: A name for the variable group, e.g. 'age'
    # - group_vals: A list of variable names in this group, e.g. the names of all 
    #   'age'-related variables.
    #
    # ------------------------------------------------------------------------------
    def set_group(self, group_name, group_vals):
        
        # Tuple of allowed types
        allowed_types = (slice, list, np.ndarray, int, np.integer, str)

        # Check if the variable is an instance of the allowed types
        if not isinstance(group_vals, allowed_types):
            raise TypeError("group_vals must be a slice, list, np.ndarray, int, np.integer, or str.")

        # Check column headers are present. This step only really makes sense
        # if we have been given column names as a string or list of strings.
        if isinstance(group_vals, (list, str)):

            # Check if the string is in the column names
            if isinstance(group_vals, str):

                # Convert to a list
                group_vals = [group_vals]

            # Filter the column names
            group_vals = [name for name in group_vals if name in self.columns]
        
        # Save group name
        self.groups[group_name.lower()] = group_vals

    # ------------------------------------------------------------------------------
    # Check is nan
    # ------------------------------------------------------------------------------
    def isna(self, *args, **kwargs):
        return(self.df.isna(*args, **kwargs))
        
    # ------------------------------------------------------------------------------
    # Save to csv
    # ------------------------------------------------------------------------------
    def to_csv(self, *args, **kwargs):
        return(biobank_df(self.df.to_csv(*args, **kwargs)))
        
    # ------------------------------------------------------------------------------
    # The list group function allows us to quickly see the variable groups. If full
    # is True, we list the variables as well.
    # ------------------------------------------------------------------------------
    def list_groups(self, full=False):

        # Loop through groups
        for group in self.groups:

            # Print the group name
            if not full:
                print(group)

            # Print the group name and variables
            else:
                print('Group name: ', group)
                print('Variables in group: ', self.groups[group])
        
    # ------------------------------------------------------------------------------
    # The get group function allows us to quickly access groups of variables by name 
    # ------------------------------------------------------------------------------
    #
    # - group_names (list, str): The name or names of the groups we want to access
    # - row_slice: row indices for the data we want (if left blank, all data for 
    #              these variables are returned)
    #
    # ------------------------------------------------------------------------------
    def get_groups(self, group_names, row_slice=None):
        
        # Check if group names is a string and if so make it a list
        if isinstance(group_names, str):
            
            # Convert it to a list containing only that string
            group_names = [group_names]
        
        
        # Check if group names is a list and if not make it a list
        if not isinstance(group_names, list):
            
            # Convert it to a list containing only that string
            group_names = list(group_names)
        
        # Empty variable name list
        var_names = []
        
        # Loop though all group names concatenating them into a list of strings
        for group_name in group_names:
            
            # Add group_names to list
            var_names = var_names + self.groups[group_name.lower()]
            
        # If row slice is none
        if row_slice is None:
            
            # Return all rows for group variables
            return(self.loc[:,var_names])
        
        else:
            
            # Return selected rows for group variables
            return(self.loc[row_slice,var_names])
            

    # ------------------------------------------------------------------------------
    # Search columns with a unix style regular expression
    # ------------------------------------------------------------------------------
    def search_cols(self, reg_exp):

        # List to hold matches
        matched_strings = [item for item in self.columns if fnmatch.fnmatch(item, reg_exp)]

        return(self.loc[:,matched_strings])
    
    # ------------------------------------------------------------------------------
    # Function for indexing (basically we are just replacing df indexing here).
    # ------------------------------------------------------------------------------
    def __getitem__(self, key):

        rows,cols = key
        return(self.loc[rows,cols])

    # ------------------------------------------------------------------------------
    # Function for setting items
    # ------------------------------------------------------------------------------
    def __setitem__(self,*args,**kwargs):
        self.df.__setitem__(*args,**kwargs)
        
    # ------------------------------------------------------------------------------
    # Persist function
    # ------------------------------------------------------------------------------
    def persist(self):

        rows,cols = key
        return(self.df.persist())
        
    # ------------------------------------------------------------------------------
    # Addition
    # ------------------------------------------------------------------------------
    def __add__(self,*args):
        return(biobank_df(self.df.__add__(*args)))

    # ------------------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------------------
    def __sub__(self,*args):
        return(biobank_df(self.df.__sub__(*args)))
        
    # ------------------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------------------
    def __mul__(self,*args):
        return(biobank_df(self.df.__mul__(*args)))
        
    # ------------------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------------------
    def __matmul__(self,*args):
        return(biobank_df(self.df.__matmul__(*args)))
        
    # ------------------------------------------------------------------------------
    # True divide
    # ------------------------------------------------------------------------------
    def __truediv__(self,*args):
        return(biobank_df(self.df.__truediv__(*args)))

    # ------------------------------------------------------------------------------
    # Floor divide
    # ------------------------------------------------------------------------------
    def __floordiv__(self,*args):
        return(biobank_df(self.df.__floordiv__(*args)))

    # ------------------------------------------------------------------------------
    # Modular arithmetic
    # ------------------------------------------------------------------------------
    def __mod__(self,*args):
        return(biobank_df(self.df.__mod__(*args)))

    # ------------------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------------------
    def __pow__(self,*args):
        return(biobank_df(self.df.__pow__(*args)))

    # ------------------------------------------------------------------------------
    # Logical and
    # ------------------------------------------------------------------------------
    def __and__(self,*args):
        return(biobank_df(self.df.__and__(*args)))
        
    # ------------------------------------------------------------------------------
    # Logical exclusive or
    # ------------------------------------------------------------------------------
    def __xor__(self,*args):
        return(biobank_df(self.df.__xor__(*args)))
        
    # ------------------------------------------------------------------------------
    # Logical or
    # ------------------------------------------------------------------------------
    def __or__(self,*args):
        return(biobank_df(self.df.__or__(*args)))
        
    # ------------------------------------------------------------------------------
    # Absolute value
    # ------------------------------------------------------------------------------
    def abs(self,*args):
        return(biobank_df(self.abs(*args)))
        
    # ------------------------------------------------------------------------------
    # As type
    # ------------------------------------------------------------------------------
    def astype(self,*args):
        return(biobank_df(self.df.astype(*args)))
        


# -----------------------------------------------------------------------------------
# Simple class copying over the loc behaviour, but returning a biobank df
# -----------------------------------------------------------------------------------
class biobank_loc():

    def __init__(self, loc):
        self.loc = loc

    def __getitem__(self,key):
        return(biobank_df(self.loc.__getitem__(key)))


# -----------------------------------------------------------------------------------
# Simple class copying over the iloc behaviour, but returning a biobank df
# -----------------------------------------------------------------------------------
class biobank_iloc():

    def __init__(self, iloc):
        self.iloc = iloc

    def __getitem__(self,key):
        return(biobank_df(self.iloc.__getitem__(key)))
