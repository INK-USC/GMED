# GMED-anonymous-submission

## Links for downloadable datasets:
- MNIST dataset

    http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    
- CIFAR-10 and CIFAR-100 datsets

   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz


- mini-ImageNet

   https://data.deepai.org/miniimagenet.zip

   The partition of the datasets into tasks will be performed each time you train the model.

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
  
