#!/bin/bash

min_seed=0
max_seed=9
model_name=${1}
grad_iter=${2}
grad_stride=${3}
use_loss_1=${4}
use_loss_2=${5}
proj_loss_reg=${6}
mem_size=${7}
extra_opt=${8}
extra=""
extra_arg=""


#if [[ -n "$9" ]]; then
#  mem_bs=${9}
#else
#  mem_bs=50
#fi
mem_bs=50

if [[ -n "$9" ]]; then
  min_seed=${9}
fi
if [[ -n "${10}" ]]; then
  max_seed=${10}
fi

seed=$min_seed

if [[ -n "${11}" ]]; then
  extra_args=${11}
fi

if [[ ${extra_opt} == "mir" ]]
then
  extra="EXTERNAL.REPLAY.MEM_BS=${mem_bs} EXTERNAL.REPLAY.MIR_K=10 EXTERNAL.OCL.MIR=1"
fi
if [[ ${extra_opt} == "mirl" ]]
then
  extra="EXTERNAL.REPLAY.MEM_BS=${mem_bs} EXTERNAL.REPLAY.MIR_K=10 EXTERNAL.OCL.MIR=1 EXTERNAL.OCL.EDIT_LEAST=1"
fi
if [[ ${extra_opt} == "mirr" ]]
then
  extra="EXTERNAL.REPLAY.MEM_BS=${mem_bs} EXTERNAL.REPLAY.MIR_K=10 EXTERNAL.OCL.MIR=1 EXTERNAL.OCL.EDIT_RANDOM=1"
fi
if [[ ${extra_opt} == "supp_proj" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=1"
fi
if [[ ${extra_opt} == "supp_reg" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=2"
fi
if [[ ${extra_opt} == "supp_proj_meta" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=3"
fi
if [[ ${extra_opt} == "supp_reg_meta" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=4"
fi

config_file="configs/memevolve/ver_permute_approx.yaml"
if [[ ${extra_opt} == "hal" ]]
then
  config_file="configs/baselines/hal_permuted_mnist.yaml"
fi


while(( $seed<= $max_seed  ))
do
  python train.py --name ${1} --config configs/memevolve/ver_rotate_approx.yaml --seed ${seed} --cfg EXTERNAL.OCL.GRAD_ITER=${grad_iter} EXTERNAL.OCL.GRAD_STRIDE=${grad_stride} EXTERNAL.OCL.USE_LOSS_1=${use_loss_1} EXTERNAL.OCL.USE_LOSS_2=${use_loss_2} EXTERNAL.OCL.PROJ_LOSS_REG=${proj_loss_reg} EXTERNAL.REPLAY.MEM_LIMIT=${mem_size} ${extra} ${extra_args}
  let "seed++"
done