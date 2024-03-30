import os
import numpy as np

# ============================================================================
#
# The below function adds a block of values to a pre-existing memory map file
# or creates a new memory map of specified dimensions if not.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `fname`: An absolute path to the mmap file.
# - `block`: The block of values to write to the mmap.
# - `blockInds`: The indices representing the entries `block` should be 
#                written to in the mmap. 
# - `dim`: Dimensions of data in file
# - `dtype` (optional): Data type of output, by default float32
#
# ============================================================================
def addBlockToMmap(fname, block, blockInds, dim, dtype=np.float64):
    
    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Double check if the dimensions are a tuple or a numpy array
    if isinstance(dim, np.ndarray):
        
        # Flatten the array and convert to a tuple
        dim = tuple(np.ndarray.flatten(dim)) 
        
    elif isinstance(dim, list):
        
        # Convert to tuple for consistency
        dim = tuple(dim)
    
    # Load the file if it exists already
    if os.path.isfile(fname):

        # Load the existing memory map
        memmap = np.memmap(fname, dtype=dtype, mode='r+', shape=dim)
        
    else:
            
        # Create a new memory-mapped file with correct dimensions
        memmap = np.memmap(fname, dtype=dtype, mode='w+', shape=dim)
        
    # Get the number of indices we have
    inds_size = max(ind.size for ind in blockInds)
            
    # We need data for all values
    if inds_size!=block.size:

        # Raise an error
        raise ValueError('This code expected data of size ' + 
                         str(inds_size) +
                         '. However, you passed it data of size ' + 
                         str(block.size) + '.')

    # Add to memmap
    memmap[blockInds] = block.reshape(memmap[blockInds].shape)
    
    # Flush the result
    memmap.flush()

    # Sync changes and close the file
    del memmap

    # Release the file lock
    os.remove(fname + ".lock")
    os.close(f)