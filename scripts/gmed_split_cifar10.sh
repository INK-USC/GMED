#!/bin/bash

min_seed=0
max_seed=4
seed=$min_seed
model_name=${1}
grad_iter=${2}
grad_stride=${3}
use_loss_1=${4}
use_loss_2=${5}
proj_loss_reg=${6}
mem_size=${7}
extra_opt=${8}
extra=""
extra_args=""

if [[ -n "$9" ]]; then
  min_seed=${9}
fi
if [[ -n "${10}" ]]; then
  max_seed=${10}
fi
if [[ -n "${11}" ]]; then
  extra_args=${11}
fi
#if [[ -n "$9" ]]; then
#  mem_bs=${9}
#else
#  mem_bs=25
#fi
mem_bs=50

seed=$min_seed

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
if [[ ${extra_opt} == "rrw" ]]
then
  extra="EXTERNAL.OCL.REPLACE_REWEIGHT=1"
fi
if [[ ${extra_opt} == "rrwr" ]]
then
  extra="EXTERNAL.OCL.REPLACE_REWEIGHT=2"
fi
if [[ ${extra_opt} == "supp_proj" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=1"
fi
if [[ ${extra_opt} == "supp_reg" ]]
then
  extra="EXTERNAL.OCL.REG_SUPPORTIVE=2"
fi
if [[ ${extra_opt} == 'relu' ]]
then
  extra="EXTERNAL.OCL.USE_RELU=1"
fi

config_file="configs/memevolve/verx_cifar.yaml"
if [[ ${extra_opt} == "hal" ]]
then
  config_file="configs/baselines/hal_split_cifar.yaml"
fi

while(( $seed<= $max_seed  ))
do
  python train.py --name ${1} --config ${config_file} --seed ${seed} --cfg EXTERNAL.OCL.GRAD_ITER=${grad_iter} EXTERNAL.OCL.GRAD_STRIDE=${grad_stride} EXTERNAL.OCL.USE_LOSS_1=${use_loss_1} EXTERNAL.OCL.USE_LOSS_2=${use_loss_2} EXTERNAL.OCL.PROJ_LOSS_REG=${proj_loss_reg} EXTERNAL.REPLAY.MEM_LIMIT=${mem_size} ${extra} ${extra_args}
  let "seed++"
done