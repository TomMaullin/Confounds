import os
import shutil
import numpy as np
import pandas as pd

from src.nets.nets_load_match import nets_load_match
from src.nets.nets_inverse_normal import nets_inverse_normal

from src.preproc.datenum import datenum
from src.preproc.days_in_year import days_in_year

from src.memmap.MemoryMappedDF import MemoryMappedDF


def generate_initial_variables(data_dir, out_dir):
    
    
    # ----------------------------------------------------------------------------------
    # Read subject IDs in and exclude those that have been dropped
    # ----------------------------------------------------------------------------------

    # Load the subject ids
    sub_ids = np.loadtxt(os.path.join(data_dir, 'subj.txt'), dtype=int)

    # Load the subject ids to be excluded
    sub_ids_to_exclude = np.loadtxt(os.path.join(data_dir, 'excluded_subjects.txt'), dtype=int)

    # Find the common elements between the two arrays
    common_ids = np.intersect1d(sub_ids, sub_ids_to_exclude)

    # Remove the common elements from sub_ids
    sub_ids = np.setdiff1d(sub_ids, common_ids)

    # Clean workspace
    del sub_ids_to_exclude, common_ids
    
    
    # ----------------------------------------------------------------------------------
    # Read all IDPs
    # ----------------------------------------------------------------------------------

    # dtypes for IDPs
    dtypes = {i: 'float32' for i in range(893)}
    dtypes[0] = 'int32'
    dtypes[892] = 'int32'
    
    # Load the IDPs
    all_IDPs = nets_load_match(os.path.join(data_dir, 'IDPs.txt'), sub_ids, dtypes=dtypes)

    # Find subjects with T1
    t1_subs = ~all_IDPs.iloc[:, 16].isna()

    # Remove non-T1 subs from IDPs and subject ids
    all_IDPs = all_IDPs.loc[t1_subs]
    sub_ids = sub_ids[t1_subs]

    # Get the number of rows in ALL_IDs
    n = len(sub_ids)

    # Clean workspace
    del t1_subs
    
    
    # ----------------------------------------------------------------------------------
    # Read fmrib info and compute time stamps
    # ----------------------------------------------------------------------------------

    # dtypes for fmrib info
    dtypes = {i: 'float32' for i in range(6)}
    dtypes[0] = 'int32'
    dtypes[1] = 'int32'
    dtypes[4] = 'int32'
    
    # Read in info 
    fmrib_info = nets_load_match(os.path.join(data_dir, 'ID_initial_workspace.txt'), sub_ids, dtypes=dtypes)

    # Get the acquisition time
    time_stamp_h = np.floor(fmrib_info.iloc[:, 1] / 10000)
    time_stamp_m = np.floor((fmrib_info.iloc[:, 1] - time_stamp_h * 10000) / 100)
    time_stamp_s = fmrib_info.iloc[:, 1] - time_stamp_h * 10000 - time_stamp_m * 100

    # Convert time of day to "decimal" hours
    fmrib_info.iloc[:, 1] = time_stamp_h + time_stamp_m / 60 + time_stamp_s / 3600

    # Get the fraction of the day when the subject was acquired
    day_fraction = (fmrib_info.iloc[:, 1] - 7) / 13

    # Get acquisition date
    time_stamp_y = np.floor(fmrib_info.iloc[:, 0] / 10000)
    time_stamp_m = np.floor((fmrib_info.iloc[:, 0] - time_stamp_y * 10000) / 100)
    time_stamp_d = fmrib_info.iloc[:, 0] - time_stamp_y * 10000 - time_stamp_m * 100

    # Convert scan date to "decimal" years
    dates = [datenum(int(y), int(m), int(d)) for y, m, d in zip(time_stamp_y, time_stamp_m, time_stamp_d)]
    days_since_year_start = np.array([(date - datenum(int(y), 1, 1)) for date, y in zip(dates, time_stamp_y)])

    # Output decimal years (note we need change datatype for compatibility
    fmrib_info = fmrib_info.astype({1: 'float32'})
    fmrib_info.iloc[:, 0] = time_stamp_y + days_since_year_start / days_in_year(time_stamp_y)

    # Calculate the discrete and continuous scan date (that is scan date given to the nearest day vs
    # to the nearest second)
    scan_date = fmrib_info.iloc[:, 0]
    scan_date_cont = time_stamp_y + (days_since_year_start + day_fraction) / days_in_year(time_stamp_y)

    # Clean workspace
    del time_stamp_h, time_stamp_m, time_stamp_s, time_stamp_d, time_stamp_y
    del dates, days_since_year_start
    
    
    # ----------------------------------------------------------------------------------
    # Read resting state IDPs
    # ----------------------------------------------------------------------------------

    # dtypes for noise 25
    dtypes = {i: 'float32' for i in range(22)}
    dtypes[0] = 'int32'
    
    # dtypes for noise 100
    dtypes = {i: 'float32' for i in range(56)}
    dtypes[0] = 'int32'
    
    # Read in node amplitudes
    node_amps_25 = nets_load_match(os.path.join(data_dir, 'rfMRI_d25_NodeAmplitudes_v1.txt'), sub_ids, dtypes=dtypes)
    node_amps_100 = nets_load_match(os.path.join(data_dir, 'rfMRI_d100_NodeAmplitudes_v1.txt'), sub_ids, dtypes=dtypes)

    # dtypes for noise 100
    dtypes = {i: 'float32' for i in range(211)}
    dtypes[0] = 'int32'
    
    # Read in partial correlation network IDPs
    net_25 = nets_load_match(os.path.join(data_dir, 'rfMRI_d25_partialcorr_v1.txt'), sub_ids, dtypes=dtypes)
    net_100 = nets_load_match(os.path.join(data_dir, 'rfMRI_d100_partialcorr_v1.txt'), sub_ids, dtypes=dtypes)
    
    
    # ----------------------------------------------------------------------------------
    # Read in FS IDPs
    # ----------------------------------------------------------------------------------
    # dtypes for FS
    dtypes = {i: 'float32' for i in range(1274)}
    dtypes[0] = 'int32'
    
    FS = nets_load_match(os.path.join(data_dir, 'FS_IDPs.txt'), sub_ids, dtypes=dtypes)
    FS_use_T2 = FS.iloc[:, 0]  # Get the first column
    FS = FS.iloc[:, 1:]  # Get the rest of the columns except the first

    
    # ----------------------------------------------------------------------------------
    # Read in ASL IDPs
    # ----------------------------------------------------------------------------------
    # dtypes for ASL
    dtypes = {i: 'float32' for i in range(51)}
    dtypes[0] = 'int32'
    
    ASL = nets_load_match(os.path.join(data_dir, 'ASL_IDPs.txt'), sub_ids, dtypes=dtypes)
    
    # ----------------------------------------------------------------------------------
    # Read in QSM IDPs
    # ----------------------------------------------------------------------------------
    # dtypes for QSM
    dtypes = {i: 'float32' for i in range(19)}
    dtypes[0] = 'int32'
    
    QSM = nets_load_match(os.path.join(data_dir, 'QSM_IDPs.txt'), sub_ids, dtypes=dtypes)

    
    # ----------------------------------------------------------------------------------
    # Read in WMH 
    # ----------------------------------------------------------------------------------
    # dtypes for QSM
    dtypes = {i: 'float32' for i in range(3)}
    dtypes[0] = 'int32'
    
    WMH = nets_load_match(os.path.join(data_dir, 'ID_WMH.txt'), sub_ids, dtypes=dtypes)

    
    # ----------------------------------------------------------------------------------
    # Read in IDP names
    # ----------------------------------------------------------------------------------

    # Assuming the file is space or tab delimited. 
    df = pd.read_csv(os.path.join(data_dir, "IDPinfo.txt"), sep='\t', usecols=[0], header=0)

    # Remove the first name as this is the subject index column, which we will delete
    IDP_names = df.values
    IDP_names = list(IDP_names.reshape(np.prod(IDP_names.shape)))

    # Clean up
    del df

    
    # ----------------------------------------------------------------------------------
    # Add IDP names for resting state ICA
    # ----------------------------------------------------------------------------------

    # Append rfMRI amplitudes (ICA25 nodes)
    IDP_names.extend([f'rfMRI amplitudes (ICA25 node {i})' for i in range(1, node_amps_25.shape[1] + 1)])

    # Append rfMRI amplitudes (ICA100 nodes)
    IDP_names.extend([f'rfMRI amplitudes (ICA100 node {i})' for i in range(1, node_amps_100.shape[1] + 1)])

    # Append rfMRI connectivity (ICA25 edges)
    IDP_names.extend([f'rfMRI connectivity (ICA25 edge {i})' for i in range(1, net_25.shape[1] + 1)])

    # Append rfMRI connectivity (ICA100 edges)
    IDP_names.extend([f'rfMRI connectivity (ICA100 edge {i})' for i in range(1, net_100.shape[1] + 1)])
    
    
    # ----------------------------------------------------------------------------------
    # Add IDP names for for FS, ASL, QSM and QC
    # ----------------------------------------------------------------------------------

    # Read FS_names.txt
    with open(f"{data_dir}/FS_names.txt", 'r') as file:
        FS_names = file.read().splitlines()

    # Exclude the first name
    FS_names = FS_names[1:]  

    # Read ASL_names.txt
    with open(f"{data_dir}/ASL_names.txt", 'r') as file:
        ASL_names = file.read().splitlines()

    # Read QSM_names.txt
    with open(f"{data_dir}/QSM_names.txt", 'r') as file:
        QSM_names = file.read().splitlines()

    # Add the 2 IDP names from WMH
    WMH_names =['IDP_T2_FLAIR_BIANCA_periventWMH_volume',
                'IDP_T2_FLAIR_BIANCA_deepWMH_volume']

    # Add to running IDP names
    IDP_names = IDP_names + FS_names + ASL_names + QSM_names + WMH_names
    
    # QC IDPs and names
    QC_IDPs = all_IDPs.iloc[:, 0:16]
    QC_IDPs_names = IDP_names[0:16]

    
    # ----------------------------------------------------------------------------------
    # IDPs to exclude vs include
    # ----------------------------------------------------------------------------------
    ind_IDPs_to_exclude = list(range(1, 18)) + list(range(888, 893))
    ind_IDPs_to_include = np.setdiff1d(np.arange(all_IDPs.shape[1]), np.array(ind_IDPs_to_exclude) - 1)

    # Select the specified columns from all_IDPs
    subset_all_IDPs = all_IDPs.iloc[:, ind_IDPs_to_include]
    
    # Concatenate all the DataFrames/Series horizontally
    subset_IDPs = pd.concat([
        subset_all_IDPs.reset_index(drop=True), 
        node_amps_25.reset_index(drop=True),
        node_amps_100.reset_index(drop=True),
        net_25.reset_index(drop=True),
        net_100.reset_index(drop=True),
        FS.reset_index(drop=True),
        ASL.reset_index(drop=True),
        QSM.reset_index(drop=True),
        WMH.reset_index(drop=True)
    ], axis=1)
    
    # Update the included indices to include the node_amps_25, node_amps_100, etc.
    ind_names_to_include = np.hstack((ind_IDPs_to_include + 1, range(893, len(IDP_names) + 1))) - 1  

    # Construct IDP_names by subsetting
    IDP_names = [IDP_names[i] for i in ind_names_to_include]

    # Add names to all_IDPs
    subset_IDPs.columns = IDP_names
    
    
    # ----------------------------------------------------------------------------------
    # Outlier detection. 
    # ----------------------------------------------------------------------------------
    # Below is a description of what is happening here, taken from "Confound modelling 
    # in UK Biobank brain imaging":
    #
    # For any given confound, we define outliers thus: First we subtract the median
    # value from all subjects’ values. We then compute the median-absolutedeviation 
    # (across all subjects) and multiply this MAD by 1.48 (so that it is equal to 
    # the standard deviation if the data had been Gaussian). We then normalise all 
    # values by dividing them by this scaled MAD. Finally, we define values as 
    # outliers if their magnitude is greater than 8.
    # ----------------------------------------------------------------------------------
    
    # Subtract the median, ignoring NaNs
    subset_IDPs_m = subset_IDPs - np.nanmedian(subset_IDPs, axis=0)

    # Calculate the median absolute deviation, again ignoring NaNs
    medabs = np.nanmedian(np.abs(subset_IDPs_m), axis=0)

    # np.finfo(float).eps is machine epsilon for float64
    eps = np.finfo(float).eps

    # Get a mask for the absolute median
    low_medabs_mask = medabs < eps

    # Standardise the non-zero medians
    if np.any(low_medabs_mask):
        medabs[low_medabs_mask] = np.nanstd(subset_IDPs_m.iloc[:, low_medabs_mask], axis=0) / 1.48

    # Divide by medabs
    subset_IDPs_m = subset_IDPs_m / medabs

    # Set values with absolute value greater than 5 to NaN
    subset_IDPs_m[np.abs(subset_IDPs_m) > 5] = np.nan
    
    
    # ----------------------------------------------------------------------------------
    # Quartile Normalisation of IDPS
    # ----------------------------------------------------------------------------------
    IDPs = nets_inverse_normal(subset_IDPs_m)
    
    
    # ----------------------------------------------------------------------------------
    # Non-IDPS
    # ----------------------------------------------------------------------------------
    # Read general names
    with open(os.path.join(data_dir, 'OTHER_GENERAL_names.txt'), 'r') as file:
        gen_names = [line.strip() for line in file]

    # Datatypes for this file (general variables have a mix so best specify)
    dtypes = {0: 'int32', 1: 'float32', 2: 'float32', 3: 'float32', 4: 'object', 5: 'object', 6: 'object', 7: 'object'}

    # General variables
    gen_vars = nets_load_match(os.path.join(data_dir, 'ID_OTHER_GENERAL.txt'), sub_ids, dtypes=dtypes)
    
    
    # ----------------------------------------------------------------------------------
    # Work out scan dates
    # ----------------------------------------------------------------------------------
    # Replace NaN in scan_date with the maximum value plus 0.1
    scan_date = scan_date.values
    scan_date[np.isnan(scan_date)] = np.nanmax(scan_date) + 0.1
    scan_date = pd.Series(scan_date)

    # Extract sex, year of birth (yob) and month of birth (mob) from GEN_vars
    sex = gen_vars.iloc[:, 0]
    yob = gen_vars.iloc[:, 1]
    mob = gen_vars.iloc[:, 2]

    # Calculate birth_date as year plus the adjusted month value divided by 12
    birth_date = yob + (mob - 0.5) / 12

    # Calculate age by subtracting birth_date from scan_date
    age = scan_date - birth_date
    
    
    # ----------------------------------------------------------------------------------
    # Write out age, sex and head-size
    # ----------------------------------------------------------------------------------

    # Create a Pandas DataFrame to organize the data
    nonIDPs = pd.DataFrame({
        'ID': sub_ids,
        'AGE': age,
        'SEX': sex,
        'HEADSIZE': np.where(np.isnan(all_IDPs.iloc[:, 17]), np.nan, all_IDPs.iloc[:, 16]),
        'TOD': day_fraction,
        'FST2': FS_use_T2
    })

    # Create the data directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Create a mapping between column names and file names
    column_file_mapping = {'AGE': 'ID_AGE.txt',
                           'SEX': 'ID_SEX.txt',
                           'HEADSIZE': 'ID_HEADSIZE.txt',
                           'TOD': 'ID_TOD.txt',
                           'FST2': 'ID_FST2.txt'}

    # Save each dataframe column to a separate text file
    for col_name in column_file_mapping.keys():

        # Get the filepath
        file_path = os.path.join(out_dir, column_file_mapping[col_name])

        # Remove previous file if needed
        if os.path.exists(file_path):
            os.remove(file_path)

        # Save column
        nonIDPs[['ID', col_name]].to_csv(file_path, sep=' ', index=False, header=False, na_rep='NaN')
        
    
    # ----------------------------------------------------------------------------------
    # Read in Eddy currents and tablepos (currently unused)
    # ----------------------------------------------------------------------------------

    # Define the directory paths
    workspaces_dir = os.path.join(out_dir, 'workspaces', 'ws_00')
    figs_dir = os.path.join(out_dir, 'figs', 'EDDYQC')

    # Create directories if they don't exist
    os.makedirs(workspaces_dir, exist_ok=True)

    # Datatypes for ed
    dtypes = {0: 'int32', 1: 'float32', 2: 'float32', 3: 'float32', 4: 'object'}
    ed = nets_load_match(os.path.join(data_dir, 'ID_EDDYQC.txt'), sub_ids, dtypes=dtypes)
    
    # Datatypes for ta
    dtypes = {0: 'int32', 1: 'float32', 2: 'float32', 3: 'float32'}
    ta = nets_load_match(os.path.join(data_dir, 'ID_TABLEPOS.txt'), sub_ids, dtypes=dtypes)
    
    
    # ----------------------------------------------------------------------------------
    # Sort data
    # ----------------------------------------------------------------------------------

    # Sort scan_date and get the sorted indices
    index_sorted_date = np.argsort(scan_date)
    sorted_date = scan_date.iloc[index_sorted_date]
    
    # Sort IDPs
    IDPs = IDPs.iloc[index_sorted_date,:]
    nonIDPs = nonIDPs.iloc[index_sorted_date,:]
    
    # Reorder subject ids based on sorted indices
    sub_ids = sub_ids[index_sorted_date]
    
    # Set row indices on dataframes
    IDPs.index = sub_ids
    nonIDPs.index = sub_ids
    
    # ----------------------------------------------------------------------------------
    # Miscellaneous variables (this houses any variables that I'm unsure are used)
    # ----------------------------------------------------------------------------------
    
    # Get the general variables that we haven't saved elsewhere
    misc = gen_vars.iloc[index_sorted_date, 1:]
    
    # Read general names
    with open(f"{data_dir}/OTHER_GENERAL_names.txt", 'r') as file:
        gen_names = file.read().splitlines()
        
    # Set column names
    misc.columns = gen_names[1:]
    misc.index = sub_ids
    
    # ----------------------------------------------------------------------------------
    # Output memmaps
    # ----------------------------------------------------------------------------------

    # Return IDPs dataframe
    IDPs = MemoryMappedDF(IDPs)
    
    # Return non-IDPs dataframe
    nonIDPs = MemoryMappedDF(nonIDPs)
    
    # Return miscellaneous dataframe
    misc = MemoryMappedDF(misc)
    
    # Return IDPs and nonIDPs
    return(IDPs, nonIDPs, misc)