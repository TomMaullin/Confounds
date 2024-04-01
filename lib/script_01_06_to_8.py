import os
import numpy as np
from scipy.stats import scoreatpercentile
from src.preproc.switch_type import switch_type
from src.memmap.MemoryMappedDF import MemoryMappedDF

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
# - nonlinear_confounds (filename or MemoryMappedDF): The non-linear confounds
#                                                     memory map.
#
# -----------------------------------------------------------------------------
#
# It then thresholds the variance explained and saves a number of files with 
# the results. It returns:
# - nonlinear_confounds_reduced (MemoryMappedDF): The nonlinear confounds which
#                                                 survived thresholding.
#
# =============================================================================
def threshold_ve(ve, nonlinear_confounds, out_dir):
    
    # Convert input to memory mapped dataframes if it isn't already
    ve = switch_type(ve, out_type='MemoryMappedDF')
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF')
    
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

    # Get the reduced nonlinear confounds
    nonlinear_confounds_reduced = nonlinear_confounds[:,nonlin_list]

    # Memory map them
    nonlinear_confounds_reduced = MemoryMappedDF(nonlinear_confounds_reduced)

    # Add groupings
    groups = nonlinear_confounds.__dict__['groups']
    
    # Loop through groups
    for group_name in groups:
    
        # Read in the current variable group
        current_group = groups[group_name]
    
        # Initialise empty list for this group
        updated_group = []
        
        # Loop through the variables
        for variable in current_group:
    
            # Check if its in the reduced confounds
            if variable in nonlinear_confounds_reduced.columns:
    
                # Add to updated_group
                updated_group = updated_group + [variable]
    
        # If the new groups not empty save it as a group in the new memory mapped df
        if len(updated_group) > 0:
    
            # Add the updated group
            nonlinear_confounds_reduced.set_group(group_name, updated_group)

    # Return result
    return(nonlinear_confounds_reduced)
    
