#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 1-00:00              # Runtime in D-HH:MM
#SBATCH -p serial_requeue       # Partition to submit to
#SBATCH --mem=1000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o theano-test.out      # File to which STDOUT will be written
#SBATCH -e theano-test.err      # File to which STDERR will be written
#SBATCH --mail-type=END         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andrewzhou@college.harvard.edu # Email to which notifications will be sent
 
hostname
echo "test"