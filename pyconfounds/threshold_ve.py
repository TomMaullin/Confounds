import os
import numpy as np
from datetime import datetime

from pyconfounds.logio.my_log import my_log
from pyconfounds.logio.loading import ascii_loading_bar

from pyconfounds.preproc.switch_type import switch_type
from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF

from pyconfounds.nets.nets_percentile import nets_percentile

# -------------------------------------------------------------------------------
# Script structure:
# -------------------------------------------------------------------------------
#
# This script was previously named script_01_06_to_08, reflecting the original
# matlab code. In the matlab version of this code, script_01_06 thresholded the 
# variance explained values and output those which remained after thresholding to
# the 'tables/UVE_nonlin.txt' file. Then the bash script script_01_07 sorted and
# concatenated these tables.
#
# In the Python version, this is all done in the function below.
# -------------------------------------------------------------------------------

# =============================================================================
#
# This function takes the below inputs:
# - ve (filename or MemoryMappedDF): The variance explained memory map.
# - nonlinear_confounds (filename or MemoryMappedDF): The non-linear confounds
#                                                     memory map.
# - out_dir (string): The output directory for results to be saved to.
# - logfile (string): A html filename for the logs to be print to. If None, 
#                     no logs are output.
#
# -----------------------------------------------------------------------------
#
# It then thresholds the variance explained and saves the nonlinear confounds 
# which survive the thresholding process. It returns:
# - nonlinear_confounds_reduced (MemoryMappedDF): The nonlinear confounds which
#                                                 survived thresholding.
#
# =============================================================================
def threshold_ve(ve, nonlinear_confounds, out_dir, logfile=None):
    
    # Update log
    my_log(str(datetime.now()) +': Stage 5: Thresholding variance explained.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Loading and thresholding...', mode='a', filename=logfile)
    
    # Check we have a temporary memmap directory
    if not os.path.exists(os.path.join(out_dir, 'temp_mmap')):
        os.makedirs(os.path.join(out_dir, 'temp_mmap'))

    # Convert input to memory mapped dataframes if it isn't already
    ve = switch_type(ve, out_type='MemoryMappedDF',out_dir=out_dir)
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF',out_dir=out_dir)
    
    # Get the average and maximum variance explained
    avg_ve = ve[:,:].mean()
    max_ve = ve[:,:].max()
    
    # Get threshold for column mean variance explained
    thr_for_avg = nets_percentile(avg_ve, 95)

    # Get threshold for elementwise variance explained
    flattened_ve = ve[:,:].values.flatten()
    flattened_ve = flattened_ve[~np.isnan(flattened_ve)]
    thr_for_ve = max(0.75, nets_percentile(flattened_ve, 99.9))

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
    nonlinear_confounds_reduced = MemoryMappedDF(nonlinear_confounds_reduced, directory=out_dir)

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

    # Update log
    my_log(str(datetime.now()) +': Data thresholded.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Stage 5: Complete.', mode='a', filename=logfile)
    
    # Return result
    return(nonlinear_confounds_reduced)
    
