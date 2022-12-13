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

#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/trainOOD/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/prelogits/ --model regnet32

