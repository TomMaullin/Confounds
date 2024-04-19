import os
import time
import numpy as np
import pandas as pd

from src.nets.nets_load_match import nets_load_match
from src.nets.nets_deconfound_single import nets_deconfound_single

from src.memmap.addBlockToMmap import addBlockToMmap
from src.memmap.MemoryMappedDF import MemoryMappedDF

from src.preproc.switch_type import switch_type
from src.preproc.filter_columns_by_site import filter_columns_by_site

from lib.script_01_05 import func_01_05_gen_nonlin_conf


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

    # Perform deconfounding
    conf_ct = nets_deconfound_single(conf_ct, confounds, col_names, 
                                     mode='nets_svd', demean=True, 
                                     dtype=np.float64, return_df=True)

    with open(os.path.join(os.getcwd(),'tmp.txt'),mode="a") as f_tmp:
        print('conf_ct shape ', conf_ct.shape, confounds.shape, file=f_tmp)
        
    with open(os.path.join(os.getcwd(),'tmp.txt'),mode="a") as f_tmp:
        print('IDPs shape ', IDPs.shape, file=f_tmp)
        
    # Get variance explained and p values
    for IDP_index in range(IDPs.shape[1]):

        with open(os.path.join(os.getcwd(),'tmp.txt'),mode="a") as f_tmp:
            print('IDP index ', IDP_index, file=f_tmp)
            
        t1 = time.time()
        # Perform ve and p thresholding
        ve, p = func_01_05_gen_nonlin_conf(data_dir, IDP_index, 
                                           conf_ct, IDPs, return_df=True)
        
        t2 = time.time()

        with open(os.path.join(os.getcwd(),'tmp.txt'),mode="a") as f_tmp:
            print('MARKER6 ', t1-t2, file=f_tmp)

        t1 = time.time()
        # Indices for where to add to memmap
        indices = np.ix_([IDP_index],block)
        
        # Add p values to memory map
        addBlockToMmap(os.path.join(os.getcwd(),'temp_mmap','p_ct.npy'),
                       p, indices,(n_IDPs, n_ct),dtype=np.float64)
        
        # Add ve values to memory map
        addBlockToMmap(os.path.join(os.getcwd(),'temp_mmap','ve_ct.npy'),
                       p, indices,(n_IDPs, n_ct),dtype=np.float64)
        
        t2 = time.time()

        with open(os.path.join(os.getcwd(),'tmp.txt'),mode="a") as f_tmp:
            print('MARKER7 ', t1-t2, file=f_tmp)
        

    

        
    # Return the crossed terms
    # return(conf_ct)

        

