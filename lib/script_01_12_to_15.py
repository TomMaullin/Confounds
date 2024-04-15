import os
import numpy as np
import pandas as pd

from src.nets.nets_load_match import nets_load_match
from src.nets.nets_deconfound_multiple import nets_deconfound_multiple

from src.memmap.MemoryMappedDF import MemoryMappedDF

from src.preproc.switch_type import switch_type
from src.preproc.filter_columns_by_site import filter_columns_by_site


def construct_and_deconfound_ct(IDPs, confounds, crossed_inds, mode, blksize, block):

    # Switch type to save transfer costs (we need all of confounds in memory)
    confounds = switch_type(confounds, out_type='pandas')
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF')
    
    # Get the subject ids
    sub_ids = confounds.index

    # Number of subjects
    n_sub = len(sub_ids)
    
    # Read in crossed indices
    crossed_inds = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 'crossed_inds.dat'), shape=(n_ct,3),dtype='int16') 
    crossed_inds = crossed_inds[block,:]

    # Number of crossed terms (in this block)
    n_ct_block = crossed_inds.shape[0]

    # Initialise empty crossed terms
    conf_ct = pd.DataFrame(np.zeros(n_sub, n_ct_block), index=confounds.index)

    # List for column names
    col_names = []

    # Loop through elements of the block constructing conf_ct
    for row in n_ct_block:

        # Get indices to construct confound
        site_no = crossed_inds[row,0]
        i = crossed_inds[row,1]
        j = crossed_inds[row,2]

        # Get the confounds for this site
        confounds_for_site = filter_columns_by_site(confounds_full, site_no+1, return_df=False)

        # Get confounds i and j
        conf_i = confounds[:,confounds_for_site[i]].values
        conf_j = confounds[:,confounds_for_site[j]].values

        # Work out column name
        col_names = col_names + [confounds_for_site[i] + '__x__' + confounds_for_site[j]]

        # Compute crossed term
        conf_ct.iloc[:,row] = conf_i*conf_j

    # Set conf_ct columns
    conf_ct.columns = col_names

    # Get filename to save this block to
    out_fname = os.path.join(os.getcwd(), 'temp_mmap',
                             'ct_q_' + str(int(block[0]/blksize)) + '.dat')
    
    # Run nets_deconfound
    nets_deconfound_multiple(conf_ct, conf, mode='nets_svd', demean=True, dtype='float64', 
                             blksize=blksize, coincident=True, return_result=False, 
                             out_fname=out_fname)
    
    # # Read in conf_ct to get variance explained
    # conf_ct = switch_type(out_fname, out_type='MemoryMappedDF')

    # # Loop through columns
    # for column in conf_ct.columns:

    #     # Read in column
        

    

        
    # Return the crossed terms
    # return(conf_ct)

        

