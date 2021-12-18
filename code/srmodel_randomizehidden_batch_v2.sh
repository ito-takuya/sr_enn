#!/bin/bash

## Amarel (NM3) cluster (at Rutgers University-Newark) batch script
##
## This script runs a python command
##


#!/bin/bash

#SBATCH --partition=nm3
#SBATCH --job-name=srran
#SBATCH --array=201-400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100000
#SBATCH --time=0-02:00:00
module purge
module load intel/19.0.3 
#echo "Get a random number from the system:"
#export sysrand=$(echo $RANDOM)
echo $SLURM_ARRAY_TASK_ID
echo "Creating scratch fc dir: /scratch/ti61/SRActFlow/fc/$SLURM_ARRAY_TASK_ID/"
#python -W ignore -c "import runENNs as run; run.runRandomizedHiddenLayerModel_v2(nproc=20,computeFC=True,scratchfcdir='/scratch/ti61/SRActFlow/fc/$SLURM_ARRAY_TASK_ID/',iteration=$SLURM_ARRAY_TASK_ID)"
python runENNs.py --nproc 20 --computeFC --model 'randomhidden' --scratchfcdir '/scratch/ti61/SRActFlow/fc/$SLURM_ARRAY_TASK_ID/' --iteration $SLURM_ARRAY_TASK_ID
rm -rf /scratch/ti61/SRActFlow/fc/$SLURM_ARRAY_TASK_ID/
sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID

