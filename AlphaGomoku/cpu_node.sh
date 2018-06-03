#!/bin/bash
 
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -w node1                                                                                
#SBATCH -o slurm.out
#SBATCH -e slurm.err

python alpha_gomoku.py --train