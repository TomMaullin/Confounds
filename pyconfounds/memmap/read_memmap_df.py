import os
import uuid
import shutil
import pickle
import numpy as np
from memmap.MemoryMappedDF import MemoryMappedDF

# ------------------------------------------------------------------------------
#
# Function to read a MemoryMappedDF instance from a file.
#
# ------------------------------------------------------------------------------
#
# This function takes as inputs:
#
#  - filename (str): The name of the file containing the memory mapped df.
#  - mode (str): The mode to open the memmap files (e.g. 'r', 'a', etc)
#  - make_copy (boolean): If true, the function will make a new copy of the 
#                         memory map on disk. Otherwise it will work with the 
#                         existing files.
#
# ------------------------------------------------------------------------------
#
# It then returns:
#
#  - MemoryMappedDF: The memory mapped dataframe object read in from the file.
#
# ------------------------------------------------------------------------------
def read_memmap_df(filename, mode='r',make_copy=False):
    
    # Read in self_copy and create a new instance
    with open(filename, 'rb') as f:
        unpickler = MyUnpickler(f)
        self_copy = unpickler.load()
    
    # Create a new hash
    self_copy.hash = str(uuid.uuid4())

    # Save the mode
    self_copy.mode = mode

    # Assume we don't want the copied files to persist/remain after deletion
    # if we make a copy
    if make_copy:
        self_copy.persist = False

    # Otherwise let's not delete the user's files
    else:
        self_copy.persist = True

    # If we are making a copy read the whole memory map in and copy it
    if make_copy:
        
        # Check output directory exists
        if not os.path.exists(self_copy.directory):
    
            # Make the directory
            os.makedirs(self_copy.directory)
        
        # We now make a new copy of the memory map files behind the scenes so we 
        # don't delete the original files on close
        for dtype, memmap_fname in self_copy.memory_maps.items():
            
            # Create new memory map filename using new hash
            filename = os.path.join(self_copy.directory, f"{self_copy.hash}_{dtype}.dat")
            
            # Copy memmap file to new filename
            shutil.copy(memmap_fname, filename)
            
            # Get number of elements in original memmap
            memmap_numel = np.memmap(filename, 
                                     dtype=self_copy.data_types[dtype], 
                                     mode='c').shape[0]
            
            # Get number of columns in original memmap
            memmap_ncol = len(self_copy.column_headers[dtype])
            
            # Get memmap shape (note: we have transposed as it makes column
            # access quicker to save as (columns,rows))
            memmap_shape = (memmap_ncol, memmap_numel//memmap_ncol)
            
            # Create new memory map
            self_copy.memory_maps[dtype]= np.memmap(filename, 
                                                    dtype=self_copy.data_types[dtype], 
                                                    mode=mode, 
                                                    shape=memmap_shape)

    # Otherwise just open the existing file
    else:
    
        # We now make a new copy of the memory map files behind the scenes so we 
        # don't delete the original files on close
        for dtype, memmap_fname in self_copy.memory_maps.items():

            # Get number of elements in original memmap
            memmap_numel = np.memmap(memmap_fname, 
                                     dtype=self_copy.data_types[dtype], 
                                     mode='c').shape[0]
            
            # Get number of columns in original memmap
            memmap_ncol = len(self_copy.column_headers[dtype])
            
            # Get memmap shape (note: we have transposed as it makes column
            # access quicker to save as (columns,rows))
            memmap_shape = (memmap_ncol, memmap_numel//memmap_ncol)
            
            # Create new memory map
            self_copy.memory_maps[dtype]= np.memmap(memmap_fname, 
                                                    dtype=self_copy.data_types[dtype], 
                                                    mode=mode, 
                                                    shape=memmap_shape)
            
    # Return the new instance
    return(self_copy)


# ------------------------------------------------------------------------------
#
# Unpickler class used to load in pickled object if class definition isn't where
# expected.
#
# ------------------------------------------------------------------------------
#
class MyUnpickler(pickle.Unpickler):
    
    # --------------------------------------------------------------------------
    # Look for the memory mapped df class
    # --------------------------------------------------------------------------
    # Inputs:
    #  - Module: The class module
    #  - name: The datatype to look for.
    # --------------------------------------------------------------------------
    # Outputs:
    #  - The class of the module.
    # --------------------------------------------------------------------------
    def find_class(self, module, name):
        
        # If the name of the object is MemoryMappedDF return the class we loaded
        if name == 'MemoryMappedDF':
            return MemoryMappedDF
        
        # Else return the standard class
        else:
            return super().find_class(module, name)
