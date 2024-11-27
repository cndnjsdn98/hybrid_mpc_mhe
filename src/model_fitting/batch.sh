#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#-SBATCH --gres=gpu:1
#-SBATCH --partition=gpu
#SBATCH --mem=45G
#SBATCH --output=results/node_%j_stdout.txt
#SBATCH --error=results/node_%j_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=NODE
#SBATCH --mail-user=w.choo@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/NODE-MPC-MHE/model_fitting/

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

module load Python
source $HOME/node/bin/activate

export PYTHONPATH=/home/cs504306/NODE-MPC-MHE/:$PYTHONPATH

python train_nn.py
