#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=1-90
#SBATCH -J epoch
#SBATCH -o epoch.%J.out
#SBATCH -e epoch.%J.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

# load the module
module load pytorch/1.2.0-cuda10.0-cudnn7.6-py3.7

cluster_values=( 16 )

frac_values=( 0.1 )
bs_values=( 100 )
rho_values=( -0.2 -0.1 0 0.1 0.2  )
lb_values=( 20 )
le_values=( 1 )
close_values=( 3 )
period_values=( 3 )
hidden_values=( 64 )
phi_values=( 0.01 0.1 1.0 )

lr_values=( 1e-2 )
w_lr_values=( 1e-1 )

out_dim_values=( 1 )

type_values=( 'sms' 'call' 'net' )
file_values=( 'milano.h5' 'trento.h5')

trial=${SLURM_ARRAY_TASK_ID}
cluster=${cluster_values[$(( trial % ${#cluster_values[@]} ))]}
trial=$(( trial / ${#cluster_values[@]} ))
frac=${frac_values[$(( trial % ${#frac_values[@]} ))]}
trial=$(( trial / ${#frac_values[@]} ))
bs=${bs_values[$(( trial % ${#bs_values[@]} ))]}
trial=$(( trial / ${#bs_values[@]} ))
lr=${lr_values[$(( trial % ${#lr_values[@]} ))]}
trial=$(( trial / ${#lr_values[@]} ))
w_lr=${w_lr_values[$(( trial % ${#w_lr_values[@]} ))]}
trial=$(( trial / ${#w_lr_values[@]} ))

lb=${lb_values[$(( trial % ${#lb_values[@]} ))]}
trial=$(( trial / ${#lb_values[@]} ))
le=${le_values[$(( trial % ${#le_values[@]} ))]}
trial=$(( trial / ${#le_values[@]} ))

close=${close_values[$(( trial % ${#close_values[@]} ))]}
trial=$(( trial / ${#close_values[@]} ))
period=${period_values[$(( trial % ${#period_values[@]} ))]}
trial=$(( trial / ${#period_values[@]} ))


hidden=${hidden_values[$(( trial % ${#hidden_values[@]} ))]}
trial=$(( trial / ${#hidden_values[@]} ))
phi=${phi_values[$(( trial % ${#phi_values[@]} ))]}
trial=$(( trial / ${#phi_values[@]} ))

out=${out_dim_values[$(( trial % ${#out_dim_values[@]} ))]}
trial=$(( trial / ${#out_dim_values[@]} ))


rho=${rho_values[$(( trial % ${#rho_values[@]} ))]}
trial=$(( trial / ${#rho_values[@]} ))
type=${type_values[$(( trial % ${#type_values[@]} ))]}
trial=$(( trial / ${#type_values[@]} ))
file=${file_values[$(( trial % ${#file_values[@]} ))]}

python fed_dual_att.py --file ${file} --type ${type} --bs ${bs} --frac ${frac} --cluster ${cluster} --rho ${rho} --lr ${lr} --w_lr ${w_lr} --close_size ${close} --period_size ${period} --hidden_dim ${hidden} --phi ${phi} --out_dim ${out} --opt 'sgd'