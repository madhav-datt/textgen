#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 2-00:00              # Runtime in D-HH:MM
#SBATCH -p serial_requeue       # Partition to submit to
#SBATCH --mem=10000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o hostname_%j.out      # File to which STDOUT will be written
#SBATCH -e hostname_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andrewzhou@college.harvard.edu # Email to which notifications will be sent
 
hostname
module load python
source activate mypython
echo "test"
srun -n $SLURM_NTASKS --accel-bind=g python src/train-simple.py