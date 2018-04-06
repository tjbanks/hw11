#!/bin/bash
#-------------------------------------------------------------------------------
## resources
#SBATCH -p gpu3  # partition (which set of nodes to run on)
#SBATCH -N1  # nodes
#SBATCH -n20  # tasks (cores)
#SBATCH --mem=10G  # total RAM
#SBATCH -t 0-01:00  # time (days-hours:minutes)
#SBATCH --qos=normal  # qos level
#SBATCH --exclusive  # reserve entire node
#
## labels and outputs
#SBATCH -J hello_cuda  # job name - shows up in sacct and squeue
#SBATCH -o results_cuda-%j.out  # filename for the output from this job (%j = job#)
#SBATCH -A general  # investor account
#
## notifications
#SBATCH --mail-user=tbg28@mail.missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#
##-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

# load modules then display what we have
module load cuda/cuda-7.5
module list

# Serial operations - only runs on the first core
echo "### Starting at: $(date)"
echo "First core reporting from node:"
hostname

echo "Currently working in directory:"
pwd

echo "Files in this folder:"
ls -l

# Execute the hello_cuda script:
./lfp_classify.py

echo "### Ending at: $(date) ###"
