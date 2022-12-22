#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-gpu=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH --account=baskargroup-a100gpu
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

#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_0.pth --checkpoints  model_0  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_1.pth --checkpoints  model_1  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_2.pth --checkpoints  model_2  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_3.pth --checkpoints  model_3  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_4.pth --checkpoints  model_4  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_5.pth --checkpoints  model_5  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_6.pth --checkpoints  model_6  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_7.pth --checkpoints  model_7  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_8.pth --checkpoints  model_8  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_9.pth --checkpoints  model_9  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_10.pth --checkpoints  model_10  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_20.pth --checkpoints  model_20  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_30.pth --checkpoints  model_30  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_40.pth --checkpoints  model_40  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/prelogits/ --model regnet32

#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_0.pth --checkpoints  model_0  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_1.pth --checkpoints  model_1  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_2.pth --checkpoints  model_2  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_3.pth --checkpoints  model_3  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_4.pth --checkpoints  model_4  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_5.pth --checkpoints  model_5  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_6.pth --checkpoints  model_6  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_7.pth --checkpoints  model_7  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_8.pth --checkpoints  model_8  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_9.pth --checkpoints  model_9  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_10.pth --checkpoints  model_10  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_20.pth --checkpoints  model_20  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_30.pth --checkpoints  model_30  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/checkpoints/model_40.pth --checkpoints  model_40  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet59-142/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/prelogits/ --model resnet50

#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_0.pth --checkpoints  model_0  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_1.pth --checkpoints  model_1  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_2.pth --checkpoints  model_2  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_3.pth --checkpoints  model_3  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_4.pth --checkpoints  model_4  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_5.pth --checkpoints  model_5  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_6.pth --checkpoints  model_6  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_7.pth --checkpoints  model_7  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_8.pth --checkpoints  model_8  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_9.pth --checkpoints  model_9  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_10.pth --checkpoints  model_10  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogitVGG.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid50/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/prelogits/ --model VGG

# balanced vs unbalanced ood metrices
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/trainOOD/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/trainOOD/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/prelogits/ --model regnet32
#srun python  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods/extract_prelogit.py  --indist-train-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/trainOOD/   --indist-test-path  /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/ --outdist-test-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/noninsecta-2526c-valid/  --model-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/checkpoints/model_49.pth --checkpoints  model_49  --logits-path /work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/prelogits/ --model regnet32
