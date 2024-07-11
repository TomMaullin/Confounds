# PyConfounds

This repository contains the code for UK Biobank Deconfounding python code. For best performance, this code should be run on a high performance computing cluster.

## Requirements
The pyconfounds code requires Python version 3.9.2 or higher to run. To use the pyconfounds code, please pip install like so:

```
git clone https://github.com/TomMaullin/Confounds.git
cd Confounds
pip install .
```

You must also set up your `dask-jobqueue` configuration file, which is likely located at `~/.config/dask/jobqueue.yaml`. This will require you to provide some details about your HPC system. See [here](https://jobqueue.dask.org/en/latest/configuration-setup.html#managing-configuration-files) for further detail. For instance, if you are using rescomp your `jobqueue.yaml` file may look something like this:

```
jobqueue:
  slurm:
    name: dask-worker

    # Dask worker options
    cores: 1                 # Total number of cores per job
    memory: "100GB"                # Total amount of memory per job
    processes: 1                # Number of Python processes per job

    interface: ib0             # Network interface to use like eth0 or ib0
    death-timeout: 60           # Number of seconds to wait if a worker can not find a scheduler
    local-directory: "/path/of/your/choosing/"       # Location of fast local storage like /scratch or $TMPDIR
    log-directory: "/path/of/your/choosing/"
    silence_logs: True

    # SLURM resource manager options
    shebang: "#!/usr/bin/bash"
    queue: short
    project: null
    walltime: '01:00:00'
    job-cpu: null
    job-mem: null
    log-directory: null

    # Scheduler options
    scheduler-options: {'dashboard_address': ':46405'}
```


## Usage
To run `pyconfounds` first specify your settings in `config.yml` and then run using the below guidelines. Below is a complete list of possible inputs to this file.

#### Mandatory fields
The following fields are mandatory:

 - `cluster_cfg`: The settings for cluster configuration. This must contain two fields:
   - `cluster_type`: The type of cluster you wish to run your analysis using (e.g. `sge`, `slurm`, `local`, etc).
   - `num_nodes`: The number of nodes (threads on a local cluster) which you wish to request for computation (e.g. 12).
 - `datadir`: A directory containing the input data in the matlab format.
 - `outdir`: The output directory for results to be saved to.
 
#### Optional fields

The following fields are optional:

 - `MAXMEM`: This is the maximum amount of memory (in bits) that the code is allowed to work with. How this should be set depends on your machine capabilities; the default value however matches the SPM default of 2^32 (note this must be in python notation i.e. `2**32`).
 - `logfile`:  A filepath for a html log file (e.g. `/path/to/folder/log.html`) which will show you the progress of the code.
 - `output_dtype`: File type for output. Either `MemoryMappedDF` or `csv` (default: `csv`).

#### Examples

Below is an example of the `config.yml` file.

```
cluster_cfg:
  cluster_type: sge
  num_nodes: 100
datadir: /path/to/data/directory/
outdir: /path/to/output/directory/
logfile: /path/to/log.html
MAXMEM: 2**32
```


### Running the Analysis


PyConfounds can be run from the terminal as follows:

```
pyconfounds <name_of_your_yaml_file>.yml
```

You can watch your analysis progress either by checking the `logfile` (see above). To do so, run the following command:

```
cd /path/to/log/html/
python -m http.server <remote port>
```
where `<remote port>` is the port you want to host the file on (e.g. `8701`). In a seperate terminal, you must then tunnel into your HPC as follows:

```
ssh -L <local port>:localhost:<remote port> username@hpc_address
```

where the local port is the port you want to view on your local machine and the remote port is port hosting the html log file. You should now be able to access the HTML log file in browser by opening `http://localhost:<local port>/<your log file>.html`. When parallelized computation is being performed using dask, the dask console address is displayed in the `log.html` file and can be accessed by porting in a similar manner to that described above.

### Output

Output from pyconfounds can be given in one of two ways; as a `csv` file, or as a collection of `MemoryMappedDF` `npz` files. A list of files output are as follows:


| Filename  | Description  |
|---|---|
| `IDPs.csv(.npz)` | The initial IDPs. |
| `nonIDPs.csv(.npz)` | Variables that are used but neither IDPs, nor confounds. |
| `misc.csv(.npz)` | Miscellanous variables kept for posterity. |
| `confounds.csv(.npz)` | The initial confounds. |
| `nonconfounds.csv(.npz)` | The nonlinear confounds. |
| `IDPs_deconf.csv(.npz)` | The IDPs after deconfounding with inital confounds. |
| `p.csv(.npz)` | P-values for nonlinear confounds variance explained. |
| `ve.csv(.npz)` | Variance explained for nonlinear confounds. |
| `nonlinear_confounds_reduced.csv(.npz)` | Reduced nonlinear confounds. |
| `IDPs_deconf_ct.csv(.npz)` |  The IDPs after deconfounding with nonlinear confounds. |
| `confounds_with_ct.csv(.npz)` | The confounds with nonlinear and crossed terms. |
| `IDPs_deconf_smooth.csv(.npz)` | The IDPs after deconfounding with nonlinear and crossed confounds. |
| `confounds_with_smooth.csv(.npz)` | The confounds with nonlinear, crossed and smoothed terms.  |

If data are output using the MemoryMappedDF format they may be read into python as follows:

```
# Import the MemoryMappedDF class
from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF

# Read in MemoryMappedDF
memory_mapped_df = read_memmap_df(<filename for dataframe>)
```

The memory mapped dataframe object can be indexed and manipulated in a Jupyter notebook in a number of ways. Here is some example usage:

```
Example usage:

# Create a dataframe
df = pd.DataFrame({
            'A': range(1, 101),
            'B': np.random.rand(100),
            'C': np.random.randint(1, 100, size=100)
        })

# Memory mapped version
memory_mapped_df = MemoryMappedDF(df)

# Access all elements
memory_mapped_df[:,:]

# Access data using row index and column names
memory_mapped_df[1:20, ['A', 'B']]

# Access data using natural slicing syntax
memory_mapped_df[1:20, 0:1]
memory_mapped_df[1:20, 0]

# Accessing a single entry
memory_mapped_df[10, 'A']
memory_mapped_df[3, 0]
```

The `MemoryMappedDF` has the advantage that it can store metadata such as groupings of variables. You can list variable groupings in the `MemoryMappedDF` object as follows:

```
memory_mapped_df.list_groups()
```

And retrieve groups of variables using:

```
memory_mapped_df.get_group(<group name>)
```

You can also search the columns using regular expressions:
```
memory_mapped_df.search_cols("Age*Site_*")
```

If you wish to convert the `memory_mapped_df` to a csv you can do so using the following command:

```
memory_mapped_df.to_csv(filename_you_wish_to_save_to)
```

**Please note:** At present, the `MemoryMappedDF` class is saved across several files, so if you move files next to the main `npz` file, you may find the `MemoryMappedDF` can no longer be opened. Also, at present, the locations of files in the `MemoryMappedDF` objects are hard coded, so it is not recommended to move them around. 

### Structure of the repository

This repository has the following structure.

 - `pyconfounds`: Contains the main Python package code.
   - `generate_initial_variables.py`: Reads in and generates a dataframe of IDPs.
   - `generate_raw_confounds.py`: Reads in and generates a dataframe of confounds.
   - `generate_nonlin_confounds.py`: Generates nonlinear confounds.
   - `get_p_vals_and_ve_cluster.py`: Parallelized computation of p-values and ve for nonlinear confounds.
   - `get_p_vals_and_ve.py`: Serial computation of p-values and ve for nonlinear confounds.
   - `threshold_ve.py`: Thresholds variance explained.
   - `generate_crossed_confounds_cluster.py`: Generates the crossed confounds via parallelised computation.
   - `construct_and_deconfound_ct.py`: Constructs and deconfounds a block of crossed terms in serial.
   - `generate_smoothed_confounds.py`: Generates smoothed confounds, by date and time.
   - `dask_tools`: This folder contains the helper functions used to interface with dask.
     - `connect_to_cluster.py`: Handles cluster connection for specified cluster_cfgs.
   - `duplicate`: Contains code for constructing site level (duplicated) confounds.
     - `duplicate_categorical.py`: Constructs categorical site-level confounds.
     - `duplicate_demedian_norm_by_site.py`: Constructs and preprocesses continuous site-level confounds.
   - `logio`: Functions for generating the log file.
     - `loading.py`: Ascii loading bar function.
     - `my_log.py`: Function which makes and adds to the html log file.
   - `memmap`: Functions and classes for memory mapping.
     - `MemoryMappedDF.py`: Class definition and methods for the Memory Mapped Dataframe.
     - `addBlockToMmap.py`: Helper function to add chunks of data to numpy memmaps.
     - `read_memmap_df.py`: Reads in MemoryMappedDF objects.
   - `nantools`: Functions for categorising patterns of nan values in the data (mostly deprecated).
     - `all_non_nan_inds.py`: Give the indices of columns of all nan value in a dataframe.
     - `create_nan_patterns.py`: Saves unique nan patterns from a dataframe to a list.
     - `format_constant_cols.py`: Helper function to decide what to do in the face of columns of only constant and/or nan values.
     - `nan_pattern.py`: Returns the pattern of NaN values in a column of data in a useful format.
   - `nets`: This folder contains statistical methods, most of which are based on counterparts in the FSLnets matlab package.
     - `nets_deconfound_multiple.py`: Parallelised deconfounding of IDPs. 
     - `nets_deconfound_single.py`: Serial instance of decofounding IDPs.
     - `nets_demean.py`: Demeans data columnwise.
     - `nets_inverse_normal.py`: Performs inverse normal transformation of data columnwise.
     - `nets_load_match.py`: Loads data and matches subject index.
     - `nets_normalise.py`: Normalises data columnwise, ignoring NaNs.
     - `nets_pearson.py`: Computes Pearson correlation.
     - `nets_percentile.py`: Calculates the percentile of a numpy array, matching matlab conventions.
     - `nets_rank_transform.py`: Performs inverse normal transform.
     - `nets_smooth_multiple.py`: Parallelised smoothing of IDPs.
     - `nets_smooth_single.py`: Serial instance of smoothing IDPs.
     - `nets_svd.py`: Performs singular value decomposition of data.
     - `nets_unique.py`: Adapted numpy unique code which also returns the permutation from argsort.
   - `preproc`: Miscellaneous preprocessing functions.
     - `datenum.py`: Gives a numerical representation of a date in (day, week, month) format.
     - `days_in_year.py`: Gets the number of days in each of a list of years.
     - `filter_columns_by_site.py`: Filters the columns of a dataframe to get site-specific confounds.
     - `switch_type.py`: Switches between numpy, pandas, memory mapped dataframe and filenae datatypes.
     - `print_call_stack`: Helper function for debugging python errors.
 - `confounds.py`: Main module for running deconfounding code.
 - `config.yml`: Empty yaml configuration file.
 - `.gitignore`: Specifies files and directories for Git to ignore.
 - `LICENSE`: Contains the licensing information for the project.
 - `README.md`: Provides an overview, installation instructions, and usage guidelines.
 - `pyproject.toml`: Contains project configuration, dependencies, and build instructions.
 - `requirements.txt`: Lists Python package dependencies required for the project.
