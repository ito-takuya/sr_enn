#!/bin/bash

## Amarel (NM3) cluster (at Rutgers University-Newark) batch script
##
## This script runs a python command
##


#!/bin/bash

#SBATCH --partition=nm3
#SBATCH --job-name=sr-fcshuffle
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=10
#SBATCH --mem=50000
#SBATCH --time=02:00:00
module purge
module load intel/19.0.3 
echo "Get a random number from the system:"
export sysrand=$(echo $RANDOM)
echo $sysrand
python -W ignore -c 'import runENNs as run; run.runSRModelShuffleFC(nproc=10,computeFC=False)'
sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID

