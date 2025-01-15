#!/bin/bash
### Job Name
#PBS -N roadcross_simulation
### required runtime
#PBS -l walltime=24:00:00
### queue for submission
#PBS -q cpu_b

### Merge output and error files
#PBS -j oe

### Request 150 GB of memory and 16 CPU core on 1 compute node
#PBS -l select=1:mem=200G:ncpus=16

# For UTF8 encoding
export LANG=cs_CZ.UTF-8

# Load modules
module load cuda/11.7
module load python/3.10.9



### start job in the directory it was submitted from
cd $PBS_O_WORKDIR

### Create Log Directory
LOG_DIR="$PBS_O_WORKDIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PBS_JOBID}_${PBS_JOBNAME}.log"
exec > "$LOG_FILE" 2>&1


# activate the Python virtual environment
source /mnt/lustre/helios-home/prusemic/myenv/bin/activate

### run the application
python /mnt/lustre/helios-home/prusemic/AISee/SSI/config_parameters.py --n_jobs 16
