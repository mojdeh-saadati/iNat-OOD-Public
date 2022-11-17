#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8  # 16 processor core(s) per node 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="calculate accuracies for resnet and vgg"
#SBATCH --mail-user=msaadati@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm-%j.out # job standard output file (%j replaced by job id)
#SBATCH --error="training.e" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source activate iNatSoftware

srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/EnergyBasedModel_MaximumSoftmaxProbability.py  --indist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50 --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_49.pth --checkpoint model_49  --MSP-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/ --EBM-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/ --model regnet32
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/EnergyBasedModel_MaximumSoftmaxProbability.py  --indist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50 --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/OODInsect/  --model-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_49.pth --checkpoint model_49  --MSP-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/ --EBM-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/ --model regnet32
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/EnergyBasedModel_MaximumSoftmaxProbability.py  --indist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50 --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/imagenet_2012_5000-NoInsects/  --model-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_49.pth --checkpoint model_49  --MSP-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/ --EBM-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/ --model regnet32
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/EnergyBasedModel_MaximumSoftmaxProbability.py  --indist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50 --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/Correct_and_NoMask_facemask_recog_datasets/  --model-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_49.pth --checkpoint model_49  --MSP-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/ --EBM-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/ --model regnet32

