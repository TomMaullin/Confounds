import os
import uuid
import shutil
import pickle
import numpy as np
from src.memmap.MemoryMappedDF import MemoryMappedDF

# ------------------------------------------------------------------------------
# Function to read a MemoryMappedDF instance from a file
# ------------------------------------------------------------------------------
def read_memmap_df(filename):
    
    # Read in self_copy and create a new instance
    with open(filename, 'rb') as f:
        unpickler = MyUnpickler(f)
        self_copy = unpickler.load()
    
    # Create a new hash
    self_copy.hash = str(uuid.uuid4())
    
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
        
        # Get memmap shape
        memmap_shape = (memmap_numel//memmap_ncol, memmap_ncol)
        
        # Create new memory map
        self_copy.memory_maps[dtype]= np.memmap(filename, 
                                                dtype=self_copy.data_types[dtype], 
                                                mode='r+', 
                                                shape=memmap_shape)
    
    # Return the new instance
    return(self_copy)


# Unpickler class used to load in pickled object if class definition isn't where
# expected.
class MyUnpickler(pickle.Unpickler):
    
    # Look for the memory mapped df class
    def find_class(self, module, name):
        
        # If the name of the object is MemoryMappedDF return the class we loaded
        if name == 'MemoryMappedDF':
            return MemoryMappedDF
        
        # Else return the standard class
        else:
            return super().find_class(module, name)
