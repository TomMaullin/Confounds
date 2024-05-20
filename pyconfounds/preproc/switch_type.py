import numpy
import pandas
from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF
from pyconfounds.memmap.read_memmap_df import read_memmap_df

# ----------------------------------------------------------------------------
#
# The below function takes in an object that is either a mmap df filename,
# memmapped df, or a numpy array and returns either the memory mapped df or
# a numpy array.
#
# ----------------------------------------------------------------------------
#
# It takes as inputs:
#
# - x (numpy array, pd dataframe, filname for saved MemoryMappedDF or 
#   MemoryMappedDF): The data to be converted.
# - out_type (str): The output type. Options are 'filename' for MemoryMappedDF
#                   saved to file, 'MemoryMappedDF' for a memory mapped data-
#                   frame object, 'numpy' for a numpy array or 'pandas' for
#                   a pandas dataframe.
# - out_dir (string): The output directory for MemoryMappedDFs to be stored in.
#                     If set to None (default) then dfs will be saved in the 
#                     current directory.
# - fname (str): Filename to save the object to (only used when out_type=
#                'filename').
#
# ----------------------------------------------------------------------------
#
# It returns:
#
# - x (numpy array, pd dataframe, filname for saved MemoryMappedDF or 
#   MemoryMappedDF): The converted data.
#
# ----------------------------------------------------------------------------

def switch_type(x,out_type='numpy',out_dir=None,fname=None):

    # Check if the input is a string/file name
    if type(x) == str:

        # If it is a filename and type is not "filename", read in the data as
        # a memorymapped df 
        if out_type != 'filename':

            # Read in as a memory mapped dataframe.
            x = read_memmap_df(x, mode='r')

    # If x is a memory mapped dataframe read but we want a numpy array, read
    # convert the values
    if type(x) == MemoryMappedDF:

        # If we want a numpy array, read all values in
        if out_type == 'numpy':

            # Convert Mmap Df to np array.
            x = x[:,:].values
    
        # If we want a pandas df, read all values in
        elif out_type == 'pandas':

            # Convert Mmap Df to pandas df
            x = x[:,:]

        # If we just want a filename, save the memory map temporarily to filename
        elif out_type == 'filename':
                
            # Save Mmap dataframe
            x = x.save(fname)

    # If x is a pandas dataframe.
    if type(x) == pandas.core.frame.DataFrame:

        # If the output type is a numpy array save it as such
        if out_type == 'numpy':

            # Convert x
            x = x.values

        # If we want to convert to a memmory mapped dataframe.
        elif out_type != 'pandas':

            # Convert to memory mapped df
            x = MemoryMappedDF(x,directory=out_dir)
            
            # If we want to convert to a file.
            if out_type == 'filename':
                
                # Save to file
                x = x.save(fname)

    # If x is a pandas dataframe.
    if  type(x)==numpy.ndarray:

        # If the output type is a numpy array save it as such
        if out_type == 'pandas':

            # Convert x
            x = pandas.DataFrame(x)

        # Otherwise we want to convert to a memmory mapped dataframe.
        elif out_type != 'numpy':

            # Raise error
            raise ValueError('Conversion from np array to memory mapped dataframe is ' + \
                             'not supported by this function. Please use the MemoryMappedDF' + \
                             ' constructor instead.')

    # Return the result
    return(x)
        

    