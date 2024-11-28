#!/bin/bash
#
#-SBATCH --gres=gpu:1
#-SBATCH --partition=gpu
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=45
#SBATCH --mem=45G
#SBATCH --output=results/node_%j_stdout.txt
#SBATCH --error=results/node_%j_stderr.txt
#SBATCH --time=20:00:00
#SBATCH --job-name=NODE
#SBATCH --mail-user=w.choo@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/NODE-MPC-MHE/src/model_fitting/

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

# module load Python
module load Python/3.10.8-GCCcore-12.2.0
source /home/cs504306/node/bin/activate

export PYTHONPATH=/home/cs504306/NODE-MPC-MHE/:$PYTHONPATH

python ./train_nn.py
