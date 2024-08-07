import os
import uuid
import atexit
import pickle
import shutil
import fnmatch
import numpy as np
import pandas as pd
from pyconfounds.nantools.create_nan_patterns import create_nan_patterns

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
    #  - dataframe: The dataframe to store in memory mapped format. At present the 
    #               dataframes may include columns of any numeric type, and string
    #               columns,constructed via something like:
    #
    #                            df[columns] = df[columns].astype("string")
    #
    #               String nans are stored as b'<na>' and converted back on read.
    #
    #  - directory: The directory to store the internal memory map (by default if None
    #               data will be saved to a directory named "temp_memmap" in the current
    #               directory).
    #  - get_nan_patterns: If true, the object will store a dictionary containing 
    #                      the unique patterns of nan values in the columns (This is
    #                      currently unused/deprecated).
    #
    # ------------------------------------------------------------------------------
    def __init__(self, dataframe, directory=None, get_nan_patterns=False):
        
        # Initialize dictionaries to hold memory maps, column headers, and data types
        self.memory_maps = {}
        self.column_headers = {}
        self.data_types = {}
        self.shape = dataframe.shape
        self.type = 'Memory Mapped DataFrame'
        self.mode = 'r+'

        # Check if we have an input directory
        if directory is None:
            directory = os.path.join(os.getcwd(),"temp_mmap")

        # Save the directory
        self.directory = directory

        # We assume by default that we don't want this memory map to stick around 
        # in our files if we delete it from memory.
        self.persist = False
        
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
            
            # Save dtype
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
        
        # Group unique data types
        data_types = dataframe.dtypes.unique()
        
        # Iterate over each data type group
        for dtype in data_types:

            # Get the corresponding columns
            columns = dataframe.dtypes[dataframe.dtypes == dtype].index
            
            # Create a subset dataframe for the current data type
            df_subset = dataframe[columns]

            # If we have a string datatype, numpy memmaps require us to specify
            # the dtype in terms of the length of the string
            if dtype == 'string[python]':
            
                # Get the maximum length of a string
                str_len = max([df_subset[col].str.len().max() for col in df_subset.columns])
            
                # Output datatype 
                dtype_out = "S" + str(str_len)
            
            else:
            
                # Output datatype
                dtype_out = dtype.type
            
            # Convert the subset dataframe to a numpy array
            array = df_subset.to_numpy(dtype=dtype_out).T # MARKER ADDED .T HERE
            
            # Store column headers and data types for each dtype
            self.column_headers[dtype.name] = df_subset.columns.tolist()
            self.data_types[dtype.name] = dtype_out
            
            # Create a filename for the memory-mapped file
            filename = os.path.join(self.directory, f"{self.hash}_{dtype.name}.dat")
            
            # Create and initialize the memory-mapped file
            memmap_file = np.memmap(filename, dtype=dtype_out, mode='w+', shape=array.shape)
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
    # And returns:
    #
    #  - pd.DataFrame: The data from the requested indexes.
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
            if isinstance(col_indices, (int, np.integer)):# MARKER if isinstance(row_slice, (int, np.integer)):
                data = memmap_file[col_indices, row_slice].T# MARKER data = memmap_file[row_slice, col_indices]
            else:
                data = memmap_file[col_indices, :][:, row_slice].T# MARKER data = memmap_file[row_slice, :][:, col_indices]
            
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

        # Check if we have any string columns
        if 'string' in self.column_headers:

            # Get the string columns
            dtype_columns = self.column_headers['string']

            # Intersect this list with the columns we have
            dtype_columns = set(dtype_columns).intersection(set(df.columns))
            dtype_columns = list(dtype_columns)

            # If we have any string columns in the dataframe itself,
            # we need to convert
            if len(dtype_columns)>0:

                # Convert columns back to string
                df[dtype_columns]=df[dtype_columns].astype('string')

                # Replace string NAs with NAN values.
                df = df.replace('<NA>',pd.NA)
                
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

        # If we have decided not to let the data persist in memory
        if not self.persist:
            
            # This will be called when the memmap is about to be destroyed
            for dtype_name in self.data_types:
            
                # Get filename for datatype
                fname_dtype = os.path.join(self.directory, f"{self.hash}_{dtype_name}.dat")
                
                # Check if file exists for datatype
                if os.path.exists(fname_dtype):
                
                    # Remove the file
                    os.remove(fname_dtype)

            # Check if self.directory even exists
            if os.path.exists(self.directory):
                
                # Check if the output folder is empty
                if os.listdir(self.directory) == []:
                    
                    # Folder is empty, delete it
                    os.rmdir(self.directory)

    
    # ------------------------------------------------------------------------------
    # Running mmap.persist() sets the persist variable to true, which ensures that
    # the underlying memory map files are not deleted when the memory map is.
    # ------------------------------------------------------------------------------
    def persist(self):
        self.persist = True

    
    # ------------------------------------------------------------------------------
    # Running mmap.unpersist() sets the persist variable to false, which means that
    # the underlying memory map files will be deleted when the memory map is.
    # ------------------------------------------------------------------------------
    def unpersist(self):
        self.persist = False

    
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
    # The list group function allows us to quickly see the variable groups. 
    # ------------------------------------------------------------------------------
    #
    # It takes as inputs:
    # - full (boolean): If full is True, we list the variables as well.
    #
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
    # It takes as inputs:
    #
    # - group_names (list, str): The name or names of the groups we want to access
    # - row_slice: row indices for the data we want (if left blank, all data for 
    #              these variables is returned)
    #
    # ------------------------------------------------------------------------------
    #
    # It returns:
    # - pd.Dataframe: A pandas dataframe containing the variables in the listed
    #                 groups for the given row indices.
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
    # Search columns with a unix style regular expression
    # ------------------------------------------------------------------------------
    #
    # This function takes as inputs:
    # - reg_exp (str): A regular expression string.
    #
    # ------------------------------------------------------------------------------
    # 
    # It returns as outputs:
    # - pd.Dataframe: A dataframe whose column names match the given regular 
    #                 expression.
    #
    # ------------------------------------------------------------------------------
    def search_cols(self, reg_exp):

        # List to hold matches
        matched_strings = [item for item in self.columns if fnmatch.fnmatch(item, reg_exp)]

        return(self[:,matched_strings])

    
    # ------------------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------------------
    #
    # This function takes as inputs:
    # - fname (str, optional): Filename to output to, if set to None (default), a 
    #                          filename with a random hash is generated.
    # - overwrite (boolean, optional): If true, we overwrite self with the copy, and
    #                                  delete all files associated with self. 
    #
    # ------------------------------------------------------------------------------
    # 
    # It returns as outputs:
    # - fname (str): The filename data was saved to (useful when saving to a random
    #                hash).
    #
    # ------------------------------------------------------------------------------
    def save(self, fname=None, overwrite=False):

        # If the filename is not none
        if fname is not None:

            # Seperate the filename and extension
            file, ext = os.path.splitext(os.path.basename(fname))

            # Default .dat file extension
            if ext == '':
                ext = '.dat'
            
            # Get the directory from the filename
            directory = os.path.dirname(fname)
            
            # Create the directory for memory-mapped files if it doesn't exist
            if not os.path.exists(directory):
                
                # Make the directory
                os.makedirs(directory)
                
            # Loop through self.memory_maps copying each memory map file to the new 
            # directory
            for dtype, memmap_file in self.memory_maps.items():
                
                # Create new filename
                filename = os.path.join(directory, f"{file}_{dtype}{ext}")
                        
                # Copy memmap file to new filename
                shutil.copy(memmap_file.filename, filename)

            # List the memory map filenames
            mmap_fnames = {dtype: os.path.join(directory, f"{file}_{dtype}{ext}") for dtype in self.memory_maps}

        # Otherwise, just save metadata without copying everything
        else:

            # Get the filename we will save the metadata to
            fname = os.path.join(self.directory, f"{self.hash}.npz")

            # Directory is unchanged in this case
            directory = self.directory
            
            # Save the memory map filenames
            mmap_fnames = {dtype: self.memory_maps[dtype].filename for dtype in self.memory_maps}
                
        # Make a copy of self and change self_copy.memory_maps to the new filenames
        self_copy = self.__class__(pd.DataFrame(),directory=self.directory)
        self_copy.memory_maps = mmap_fnames
        self_copy.column_headers = self.column_headers
        self_copy.data_types = self.data_types
        self_copy.directory = directory
        self_copy.shape = self.shape
        self_copy.index = self.index
        self_copy.columns = self.columns
        self_copy.groups = self.groups
        self_copy.dtypes = self.dtypes
        self_copy.type = self.type

        # Check if file is in use
        fileLocked = True
        while fileLocked:
            try:
                # Create lock file, so other jobs know we are writing to this file
                f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
                fileLocked = False
            except FileExistsError:
                fileLocked = True
        
        # Save copy
        with open(fname, 'wb') as file:
            pickle.dump(self_copy, file)

        # Release the file lock
        os.remove(fname + ".lock")
        os.close(f)

        # If we're overwriting, we delete the original version, and replace with the copy
        if overwrite:
            
            # Delete original self
            self.persist=False
            self.cleanup()

        # Return the filename
        return(fname)

    
    # ------------------------------------------------------------------------------
    # Save to csv
    # ------------------------------------------------------------------------------
    #
    # This function takes as inputs:
    # - fname (str): Filename to output to.
    #
    # ------------------------------------------------------------------------------
    def to_csv(self, fname):

        # Output to CSV, including index and column names
        self[:,:].to_csv(fname, index=True)

