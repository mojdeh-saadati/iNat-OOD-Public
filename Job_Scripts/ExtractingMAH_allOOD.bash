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

#srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/Mahalanobis_distance1.py --outDistValid-embeds  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49Correct_and_NoMask_facemask_recog_datasets.pt --inDistValid-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistValid_embeds.pt --inDistTrain-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_embeds.pt --inDistTrain-labels /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_labels.pt --MAH-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD --checkpoints model_49
#srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/Mahalanobis_distance1.py --outDistValid-embeds  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49imagenet_2012.pt --inDistValid-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistValid_embeds.pt --inDistTrain-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_embeds.pt --inDistTrain-labels /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_labels.pt --MAH-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD --checkpoints model_49
#srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/Mahalanobis_distance1.py --outDistValid-embeds  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49noninsecta2526valid_embeds.pt --inDistValid-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistValid_embeds.pt --inDistTrain-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_embeds.pt --inDistTrain-labels /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_labels.pt --MAH-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD --checkpoints model_49
#srun python /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/Mahalanobis_distance1.py --outDistValid-embeds  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49OODInsect.pt --inDistValid-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistValid_embeds.pt --inDistTrain-embeds /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_embeds.pt --inDistTrain-labels /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits_allOOD/model_49inDistTrain_labels.pt --MAH-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD --checkpoints model_49
