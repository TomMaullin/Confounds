import os
import time
import numpy as np
import pandas as pd

from nets.nets_load_match import nets_load_match
from nets.nets_deconfound_single import nets_deconfound_single

from memmap.addBlockToMmap import addBlockToMmap
from memmap.MemoryMappedDF import MemoryMappedDF

from preproc.switch_type import switch_type
from preproc.filter_columns_by_site import filter_columns_by_site

from script_01_05 import func_01_05_gen_nonlin_conf


def construct_and_deconfound_ct(IDPs, confounds, data_dir, crossed_inds, mode, blksize, block):

    # Switch type to save transfer costs (we need all of confounds in memory)
    confounds = switch_type(confounds, out_type='pandas')
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF')
    
    # Get the subject ids
    sub_ids = confounds.index

    # Number of subjects
    n_sub = len(sub_ids)
        
    # Number of IDPs
    n_IDPs = IDPs.shape[1]
    
    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)
    
    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)
    
    # Number of crossed terms we will consider
    n_ct = 0
    
    # Create a dict of site-specific column headers
    for site_index in (unique_site_ids + 1):
    
        # Get the columns for this site
        columns_for_site = filter_columns_by_site(confounds, site_index, return_df=False)
        
        # Add the number of crossed terms for this site
        n_ct = n_ct + int((len(columns_for_site)-1)*(len(columns_for_site))/2)
    
    # Read in crossed indices
    crossed_inds = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 'crossed_inds.dat'), shape=(n_ct,3),dtype='int16') 
    crossed_inds = crossed_inds[block,:]

    # Number of crossed terms (in this block)
    n_ct_block = crossed_inds.shape[0]

    # Initialise empty crossed terms
    conf_ct = pd.DataFrame(np.zeros((n_sub, n_ct_block)), index=confounds.index)

    # List for column names
    col_names = []

    # Loop through elements of the block constructing conf_ct
    for row in range(n_ct_block):

        # Get indices to construct confound
        site_no = crossed_inds[row,0]
        i = crossed_inds[row,1]
        j = crossed_inds[row,2]

        # Get the confounds for this site
        confounds_for_site = filter_columns_by_site(confounds, site_no+1, return_df=False)

        # Get confounds i and j
        conf_i = confounds[confounds_for_site[i]].values
        conf_j = confounds[confounds_for_site[j]].values

        # Work out column name
        col_names = col_names + [confounds_for_site[i] + '__x__' + confounds_for_site[j]]

        # Compute crossed term
        conf_ct.iloc[:,row] = conf_i*conf_j

    # Set conf_ct columns
    conf_ct.columns = col_names

    # Perform deconfounding # MARKER NEEDS TO BE DONE SITEWISE
    conf_ct = nets_deconfound_single(conf_ct, confounds, col_names, 
                                     mode='nets_svd', demean=True, 
                                     dtype=np.float64, return_df=True)
    
    # Get the number of blocks we are breaking computation into
    num_blks_IDPs = int(np.ceil(IDPs.shape[1]/blksize))
    
    # Get the indices for each block of IDPs
    idx = np.arange(IDPs.shape[1])
    blocks_IDPs = [idx[i*blksize:min((i+1)*blksize,IDPs.shape[1])] for i in range(num_blks_IDPs)]
    
    # Get variance explained and p values
    for block_IDP in blocks_IDPs:
        
        # Perform ve and p thresholding
        ve, p = func_01_05_gen_nonlin_conf(data_dir, block_IDP, 
                                           conf_ct, IDPs, method=4,
                                           return_df=True)
        
        # Indices for where to add to memmap
        indices = np.ix_(block_IDP,block)
        
        # Add p values to memory map
        addBlockToMmap(os.path.join(os.getcwd(),'temp_mmap','p_ct.npy'),
                       p, indices,(n_IDPs, n_ct),dtype=np.float64)
        
        # Add ve values to memory map
        addBlockToMmap(os.path.join(os.getcwd(),'temp_mmap','ve_ct.npy'),
                       ve, indices,(n_IDPs, n_ct),dtype=np.float64)
