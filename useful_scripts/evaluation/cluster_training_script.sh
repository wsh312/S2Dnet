#!/bin/bash
#SBATCH --mail-user=wu@iam.unibe.ch
#SBATCH --mail-type=end
#SBATCH --job-name="101CP"
#SBATCH --nodes=1
#SBATCH --time=23:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1080ti:1


module add gcc/5.3.0
srun python train.py --dataroot ../huge_uni_render_rnn --logroot ./logs/job101CP --name job_submit_101C_re1_pixel --model cycle_gan --no_dropout --loadSize 512 --fineSize 512 --patchSize 256 --which_model_netG unet_512_Re1 --which_model_netD patch_512_256_multi_new --lambda_A 10 --lambda_B 10 --use_vgg --lambda_GD 0  --use_SGD --norm pixel --continue_train
