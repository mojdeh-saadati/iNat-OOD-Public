#!/bin/bash


#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 16 processor core(s) per node 
#SBATCH --job-name="copying iNat"
#SBATCH --mail-user=msaadati@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#  --output # job standard output file (%j replaced by job id)
#SBATCH --error="training.e" # job standard error file (%j replaced by job id)

rsync -a /work/baskarg/iNaturalist/iNat_2526_13mil/ /work/baskarg/Mojdeh/AlphaNet/Data/iNat_2526_13mil