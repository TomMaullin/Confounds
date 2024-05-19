import os
import time 
import shutil
import argparse
import numpy as np
import pandas as pd

from script_01_00 import generate_initial_variables
from script_01_01 import generate_raw_confounds
from script_01_02 import generate_nonlin_confounds
from script_01_03_to_4 import get_p_vals_and_ve
from script_01_06_to_8 import threshold_ve
from script_01_09_to_12 import generate_crossed_confounds_cluster
from script_01_16 import generate_smoothed_confounds

from memmap.MemoryMappedDF import MemoryMappedDF
from memmap.read_memmap_df import read_memmap_df 

def _main(argv=None):
    
    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Create the parser and add argument
    parser = argparse.ArgumentParser(description="PyConfounds cluster script")
    parser.add_argument('inputs_yml', type=str, nargs='?', default=os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            'pyconfounds_config.yml'), 
                        help='Path to inputs yaml file')

    # Parse the arguments
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    # If the argument is just a filename without a directory, 
    # prepend the current working directory
    if os.path.dirname(args.inputs_yml) == '':
        args.inputs_yml = os.path.join(os.getcwd(), args.inputs_yml)
    inputs_yml = args.inputs_yml

    # Change to absolute path if necessary
    inputs_yml = os.path.abspath(inputs_yml)
    
    # Load the inputs yaml file
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)
        
    # --------------------------------------------------------------------------------
    # Read directories
    # --------------------------------------------------------------------------------
    
    # Output directory (will eventually be equal to data_dir)
    out_dir = inputs['outdir']
    
    # Make directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Data directory
    data_dir = os.path.join(out_dir, 'data')

    # Make directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Log file
    if 'logfile' in inputs:
        logfile = inputs['logfile']
    else:
        logfile = os.path.join(os.getcwd(), 'log.html')
    
    # Remove previous log file
    if os.path.exists(logfile):
        os.remove(logfile)
        
    # --------------------------------------------------------------------------------
    # Stage 1: Generate initial variables
    # --------------------------------------------------------------------------------

    # Run stage 1
    IDPs, nonIDPs, misc = generate_initial_variables(data_dir, out_dir, logfile=logfile)
    
    # Files we can reconstruct memory mapped dataframes from
    IDPs_fname = os.path.join(os.getcwd(),'saved_memmaps','IDPs.npz')
    nonIDPs_fname = os.path.join(os.getcwd(),'saved_memmaps','nonIDPs.npz')
    misc_fname = os.path.join(os.getcwd(),'saved_memmaps','misc.npz')

    # Save the results
    IDPs.save(IDPs_fname)
    nonIDPs.save(nonIDPs_fname)
    misc.save(misc_fname)

if __name__ == "__main__":
    _main(sys.argv[1:])