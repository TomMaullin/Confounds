# PyConfounds

This repository contains the code for UK Biobank Deconfounding python code. For best performance, this code should be run on a high performance computing cluster.

## Requirements
To use the pyconfounds code, please pip install like so:

```
git clone git@github.com:TomMaullin/Confounds.git
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

where the local port is the port you want to view on your local machine and the remote port is port hosting the html log file. You should now be able to access the HTML log file in browser by opening `http://localhost:<local port>/<your log file>.html`.

### Structure of the repository


