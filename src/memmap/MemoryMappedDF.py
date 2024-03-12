import os
import uuid
import atexit
import pickle
import numpy as np
import pandas as pd
from src.nantools.create_nan_patterns import create_nan_patterns

# ==================================================================================
#
# This class initialises a dataframe stored using numpy memory maps. For each 
# datatype stored in the dataframe, a different numpy memmap is used to store it 
# behind the scenes. This should prevent large memory usage in the code.
#
# ==================================================================================
class MemoryMappedDF:
    
    # ------------------------------------------------------------------------------
    # The main init function creates a new memory mapped dataframe.
    # ------------------------------------------------------------------------------
    #
    # It takes the inputs:
    #
    #  - dataframe: The dataframe to store in memory mapped format.
    #  - directory: The directory to store the internal memory map.
    #  - get_nan_patterns: If true, the object will store a dictionary containing 
    #                      the unique patterns of nan values in the columns.
    #
    # ------------------------------------------------------------------------------
    def __init__(self, dataframe, directory="temp_mmap", get_nan_patterns=False):
        
        # Initialize dictionaries to hold memory maps, column headers, and data types
        self.memory_maps = {}
        self.directory = directory
        self.column_headers = {}
        self.data_types = {}
        self.shape = dataframe.shape
        self.type = 'Memory Mapped DataFrame'
        
        # Work out the NaN patterns for the dataframe
        if get_nan_patterns:
            self.nan_patterns = create_nan_patterns(dataframe)
        
        # ID groups for quick access of groups of variables
        self.groups = {}
        
        # Hash to store data under
        self.hash = str(uuid.uuid4())
        
        # Reduce down any dtypes that are using excessive memory
        dtypes = {}

        # Iterate through each column to determine if dtype changes are needed
        for column in dataframe.columns:
            
            # Reduce float64s to float32s
            if dataframe[column].dtype == np.float64:
                dtypes[column] = np.float32
                
            # Reduce int64s to int32s
            elif dataframe[column].dtype == np.int64:
                dtypes[column] = np.int32
                
            else:
                # For other dtypes, keep them as they are
                dtypes[column] = dataframe[column].dtype

        # Apply the dtype changes to the DataFrame
        dataframe = dataframe.astype(dtypes)
        self.dtypes = dataframe.dtypes
        
        # Store the row indices from the input dataframe
        self.index = dataframe.index
        
        # Store the original order of column headers
        self.columns = dataframe.columns
        
        # Create the directory for memory-mapped files if it doesn't exist
        if not os.path.exists(self.directory):
            
            # Make the directory
            os.makedirs(self.directory)
        
        # Group dataframe columns by their data types
        dtype_groups = dataframe.columns.to_series().groupby(dataframe.dtypes).groups
        
        # Iterate over each data type group
        for dtype, columns in dtype_groups.items():
            
            # Create a subset dataframe for the current data type
            df_subset = dataframe[columns]
            
            # Convert the subset dataframe to a numpy array
            array = df_subset.to_numpy(dtype=dtype.type)
            
            # Store column headers and data types for each dtype
            self.column_headers[dtype.name] = df_subset.columns.tolist()
            self.data_types[dtype.name] = dtype.type
            
            # Create a filename for the memory-mapped file
            filename = os.path.join(self.directory, f"{self.hash}_{dtype.name}.dat")
            
            # Create and initialize the memory-mapped file
            memmap_file = np.memmap(filename, dtype=dtype.type, mode='w+', shape=array.shape)
            memmap_file[:] = array[:]
            
            # Flush changes to ensure they are written to disk
            memmap_file.flush()
            
            # Store the memory-mapped file in the dictionary
            self.memory_maps[dtype.name] = memmap_file
            
        # Initiate cleanup method
        atexit.register(self.cleanup)
            
            
    # ------------------------------------------------------------------------------
    # The getitem function allows us to index the memory mapped dataframe.
    # ------------------------------------------------------------------------------
    #
    # It takes the inputs:
    #
    #  - key: the slice indices for accessing the dataframe. Can use slices or names
    #         for columns. 
    #
    # ------------------------------------------------------------------------------
    #
    # Example usage:
    #
    #         # Create a dataframe
    #         df = pd.DataFrame({
    #             'A': range(1, 101),
    #             'B': np.random.rand(100),
    #             'C': np.random.randint(1, 100, size=100)
    #         })
    #
    #         # Memory mapped version
    #         memory_mapped_df = MemoryMappedDF(df)
    #
    #         # Access all elements
    #         memory_mapped_df[:,:]
    #
    #         # Access data using row index and column names
    #         memory_mapped_df[1:20, ['A', 'B']]
    #
    #         # Access data using natural slicing syntax
    #         memory_mapped_df[1:20, 0:1]
    #         memory_mapped_df[1:20, 0]
    #
    #         # Accessing a single entry
    #         memory_mapped_df[10, 'A']
    #         memory_mapped_df[3, 0]
    #
    # ------------------------------------------------------------------------------
    def __getitem__(self, key):
        
        # Handle both row and column slicing
        if isinstance(key, tuple):
            
            # Rows and columns
            row_slice, col_slice = key
        
        # Default to all columns if only row_slice is provided
        else:
            
            # row slice is input, column slice is none
            row_slice = key
            col_slice = slice(None)

        # We are going to construct a results dictionary
        result = {}
        
        # Process row_slice to determine the subset of rows to access
        if isinstance(row_slice, slice):
            
            # Subset row_ids to get what we are interested in
            row_ids_subset = self.index.tolist()[row_slice]
            
        # Check if the row_slices are given as a list or array
        elif isinstance(row_slice, (list, np.ndarray)):
            
            # Subset row_ids to get what we are interested in
            row_ids_subset = [self.index.tolist()[i] for i in row_slice]
            
        # Handle single integer for row_slice
        else:  
            
            # Subset row_ids to get what we are interested in
            row_ids_subset = [self.index.tolist()[row_slice]]

        # Process col_slice to translate into actual column names
        all_columns = self.columns
        
        # If col_slice is a slice we need to use it directly to index the column names
        if isinstance(col_slice, slice):
            col_names = all_columns[col_slice]
            
        # Else if col_slice is a numpy array or list we need to use it to index
        # the column names using comprehension
        elif isinstance(col_slice, (list, np.ndarray)):
            col_names = [all_columns[i] if isinstance(i, (int, np.integer)) else i for i in col_slice]
        
        # If it is an integer there is just a single value to return
        elif isinstance(col_slice, (int, np.integer)):
            col_names = [all_columns[col_slice]]
            
        # If it is a string, we already have what we need
        elif isinstance(col_slice, str):
            col_names = [col_slice]
        
        # Otherwise error
        else:
            raise ValueError("Column selection must be properly defined.")

        # Extract data for the selected rows and columns
        for dtype, memmap_file in self.memory_maps.items():
            
            # Get the column headers for this datatype
            dtype_columns = self.column_headers[dtype]
            
            # Get the column indices for this datatype
            col_indices = [dtype_columns.index(col) for col in col_names if col in dtype_columns]
            
            # Get the selected columns for this datatype
            selected_columns = [dtype_columns[i] for i in col_indices]

            # Extract data from memory map
            if isinstance(row_slice, (int, np.integer)):
                data = memmap_file[row_slice, col_indices]
            else:
                data = memmap_file[row_slice, :][:, col_indices]

            # Ensure data has at least two dimensions
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)

            # Populate result dictionary with extracted data
            for i, col in enumerate(selected_columns):
                result[col] = data[:, i] if data.ndim > 1 else data

        # Convert the result dictionary to a DataFrame
        df = pd.DataFrame(result, index=row_ids_subset)
        
        # Reorder the DataFrame columns to match the original order
        reordering = [column for column in self.columns if column in df.columns]
        df = df[reordering]
                
        # Return result
        return(df)

    
    # ------------------------------------------------------------------------------
    # del method which calls to cleanup.
    # ------------------------------------------------------------------------------
    def __del__(self):
        
        # Cleanup the files
        self.cleanup()
        
        
    # ------------------------------------------------------------------------------
    # Cleanup method which deletes all files associated to the memory map, and the 
    # memmap folder if needed.
    # ------------------------------------------------------------------------------
    def cleanup(self):
        
        # This will be called when the memmap is about to be destroyed
        for dtype_name in self.data_types:
        
            # Get filename for datatype
            fname_dtype = os.path.join(self.directory, f"{self.hash}_{dtype_name}.dat")
            
            # Check if file exists for datatype
            if os.path.exists(fname_dtype):
            
                # Remove the file
                os.remove(fname_dtype)
                
        # Check if the output folder is empty
        if os.listdir(self.directory) == []:
            
            # Folder is empty, delete it
            os.rmdir(self.directory)
            
        
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
            return(self[:,var_names])
        
        else:
            
            # Return selected rows for group variables
            return(self[row_slice,var_names])
        
          
    # ------------------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------------------
    def save(self, fname):
        
        # Get the directory from the filename
        directory = os.path.dirname(fname)
        
        # Create the directory for memory-mapped files if it doesn't exist
        if not os.path.exists(directory):
            
            # Make the directory
            os.makedirs(directory)
            
        # Loop through self.memory_maps copying each memory map file to the new 
        # directory
        for dtype, memmap_file in self.memory_maps.items():
            filename = os.path.join(directory, f"{fname}_{dtype}.dat")
            np.memmap(filename, dtype=self.data_types[dtype], mode='w+', shape=memmap_file.shape)[:] = memmap_file[:]
        
        # Make a copy of self and change self_copy.memory_maps to the new filenames
        self_copy = self.__class__(pd.DataFrame())
        self_copy.memory_maps = {dtype: os.path.join(directory, f"{fname}_{dtype}.dat") for dtype in self.memory_maps}
        self_copy.column_headers = self.column_headers
        self_copy.data_types = self.data_types
        self_copy.shape = self.shape
        self_copy.index = self.index
        self_copy.columns = self.columns
        self_copy.dtypes = self.dtypes
        
        # Save copy
        with open(fname, 'wb') as f:
            pickle.dump(self_copy, f)
