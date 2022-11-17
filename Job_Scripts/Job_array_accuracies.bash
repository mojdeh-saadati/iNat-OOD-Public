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


srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/ModelAccuracyReport.py --test-data /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/val/ --result-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/accuracies.txt --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_0.pth --model VGG --checkpoints model_0
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/ModelAccuracyReport.py --test-data /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/val/ --result-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/accuracies.txt --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_1.pth --model VGG --checkpoints model_1
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/ModelAccuracyReport.py --test-data /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/val/ --result-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/accuracies.txt --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_10.pth --model VGG --checkpoints model_10
srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/ModelAccuracyReport.py --test-data /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/val/ --result-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/accuracies.txt --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_49.pth --model VGG --checkpoints model_49
