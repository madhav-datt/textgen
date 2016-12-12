#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 5-00:00              # Runtime in D-HH:MM
#SBATCH -p serial_requeue       # Partition to submit to
#SBATCH --mem=20000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --constraint=cuda-7.5
#SBATCH -o train.out      # File to which STDOUT will be written
#SBATCH -e train.err      # File to which STDERR will be written
 
hostname
# Load python and CUDA and CUDNN for Theano
module load python
module load cuda/7.5-fasrc01
module load cudnn/7.0-fasrc01

# Set the parameters for the script. 
# training/]filename].txt is the corpus
filename="4-mod"
# Weights will be saved to weights/[weightfile][epoch#].hdf5
weightfile="sherlock-4-1l-512n-"
# Number of hidden layers
layers="1"
# Number of nodes in each layer. For 1 layer, just a number like 512 is fine. For 2 or more
# layers, separate with periods: 512.256 for 512, then 256 nodes, for instance.
nodes="512"
# Number of epochs to run the script for.
epochs="1000"

# Active the mypython environment, which the user must have defined and installed the necessary
# packages in
source activate mypython
KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1 python src/train-generic.py -t $filename -w $weightfile -l $layers -n $nodes -e $epochs