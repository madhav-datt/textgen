#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 5-00:00              # Runtime in D-HH:MM
#SBATCH -p serial_requeue       # Partition to submit to
#SBATCH --mem=20000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --constraint=cuda-7.5
#SBATCH -o generate.out      # File to which STDOUT will be written
#SBATCH -e generate.err      # File to which STDERR will be written

hostname
# Load python and CUDA and CUDNN for Theano
module load python
module load cuda/7.5-fasrc01
module load cudnn/7.0-fasrc01

# Set the parameters for the script. 
# training/filename].txt is the corpus
filename="4-mod"
# The base name for the weight file; weights/[weightfile][num].hdf5 will be loaded for
# each num in the for statement below. The result for num epochs will be saved in 
# output/output-[weightfile][num].txt
weightfile="sherlock-4-1l-512n-"
# Number of hidden layers
layers="1"
# Number of nodes in each layer. For 1 layer, just a number like 512 is fine. For 2 or more
# layers, separate with periods: 512.256 for 512, then 256 nodes, for instance.
nodes="512"
# Number of characters to generate
numchars="1000"

source activate mypython

# The list of epoch numbers for weightfiles to generate results for
for num in 10 # 25 50 100 153 203 251 304 356 406 458 503 551 602 653 704 755 800 870 905 950 991
do
  thefile="$weightfile$num"
  output="output-$thefile"
  KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1 python src/generate-generic.py -t $filename -w $thefile -l $layers -n $nodes -o $output
done
