import os
import numpy as np
from scipy.stats import scoreatpercentile
from src.preproc.switch_type import switch_type

# -------------------------------------------------------------------------------
# Script structure:
# -------------------------------------------------------------------------------
#
# In the matlab version of this code, script_01_06 thresholded the variance
# explained values and output those which remained after thresholding to the 
# 'tables/UVE_nonlin.txt' file. Then the bash script script_01_07 sorted and
# concatenated these tables.
#
# In the Python version, this is all done in script 01_07 below.
# -------------------------------------------------------------------------------

# =============================================================================
#
# This function takes the below inputs:
# - ve (filename or MemoryMappedDF): The variance explained memory map.
#
# -----------------------------------------------------------------------------
#
# It then thresholds the variance explained and saves a number of files with 
# the results.
#
# =============================================================================
def threshold_ve(ve, out_dir):
    
    # Convert input to memory mapped dataframes if it isn't already
    ve = switch_type(ve, out_type='MemoryMappedDF')
    
    # Get the average and maximum variance explained
    avg_ve = ve[:,:].mean()
    max_ve = ve[:,:].max()
    
    # Get percentage thresholds
    thr_for_avg = scoreatpercentile(avg_ve, 95)
    thr_for_ve = max(0.75, scoreatpercentile(ve[:,:].fillna(0).values.flatten(), 99.9))

    # Find indices where average is larger than threshold
    inds_for_avg = np.where(avg_ve>thr_for_avg)[0]
    inds_for_ve = np.where(ve[:,:].values>thr_for_ve)
    
    # Tables directory
    tables_dir = os.path.join(out_dir,'tables')
    
    # Check if the 'tables' directory exists
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    
    # Write avg ve results to file
    with open(os.path.join(tables_dir, 'mean_ve_nonlin.txt'), 'w') as f:
        for i in inds_for_avg:
            f.write(f"{list(ve.columns)[i]} {avg_ve.iloc[i]:.6f}\n")
    
    # Write other ve results to file
    with open(os.path.join(tables_dir, 've_nonlin.txt'), 'w') as f:
        for k in range(len(inds_for_ve[0])):
            
            # Get indices
            i = inds_for_ve[0][k]
            j = inds_for_ve[1][k]
    
            # Write to file
            f.write(f"{list(ve.columns)[j]} {list(ve.index)[i]} {ve[i,j].values[0,0]:.6f}\n")

    # Get list of nonlinear confounds we've found
    nonlin_list = [list(ve.columns)[i] for i in inds_for_ve[1]] + [list(ve.columns)[i] for i in inds_for_avg]
    
    # Add back in _inormals - I'm not sure why this is being done, but it was done
    # in script 01_07 in the original
    nonlin_list = nonlin_list + [conf.replace('_squared_inormal','') + '_inormal' for conf in nonlin_list if '_squared_inormal' in conf]
    
    # Filter to unique values
    nonlin_list = np.unique(nonlin_list)
    
    # Sort the list
    nonlin_list = np.sort(nonlin_list)
    
    # Output the list of nonlinear confounds
    with open(os.path.join(out_dir,'tables','list_nonlin.txt'),'w') as f:
        for name in nonlin_list:
            f.write(name + '\n')

