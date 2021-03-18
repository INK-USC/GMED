# Gradient based Memory Editing for Task-Free continual Learning

Code for paper "Gradient based Memory Editing for Task-Free continual Learning", accepted at 4th Lifelong Learning Workshop@ICML 2020. 

Paper on [Arxiv](https://arxiv.org/abs/2006.15294)

```
@article{Jin2020GradientBM,
  title={Gradient Based Memory Editing for Task-Free Continual Learning},
  author={Xisen Jin and Junyi Du and X. Ren},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.15294}
}
```

## Requirements

```
conda create -n gmed
conda activate gmed
pip install -r requirements.txt
```

## Running experiments
Running experiments on MNIST

```
export stride=0.1 # editing stride (alpha)
export reg=0.01 # reg strength (beta)
export dataset="split_mnist"
export mem=500
export start_seed=100
export stop_seed=109
export method="-" # change to "mirr" for MIR+GMED

./scripts/gme_tune_${dataset}.sh  ${dataset}_iter1_stride${stride}_1_0_2${reg}_m${mem} 1 ${stride} 1 0 2 ${mem} ${method} ${start_seed} ${stop_seed} "OUTPUT_DIR=runs EXTERNAL.OCL.REG_STRENGTH=${reg}"
```

Similary, experiments on other datasets can be run by changing the name of the dataset variable above.
