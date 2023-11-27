# surrogate-mtl
Source code for the paper "Exploring Multi-Task Learning for Explainability" presented at ECAI's [XI-ML 2023 workshop](https://www.imageclef.org/2022/medical/caption](http://www.cslab.cc/xi-ml-2023/)http://www.cslab.cc/xi-ml-2023/).

## Dependencies

- TensorFlow 2/Keras
- TensorFlow Addons
- scikit-learn
- NumPy
- Pandas
- tqdm
- lime (https://github.com/marcotcr/lime/tree/master)

## Datasets 

Experiments were conducted using some datasets from UCI such as [Adult](https://archive.ics.uci.edu/dataset/2/adult), [AutoMPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) and [Red Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) as well as [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) and [Titanic](https://www.openml.org/search?type=data&sort=runs&id=40945) datasets.

## Description

| File | Content description |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [models/tree.py](models/tree.py)    | Implementation of SBDT model in `tf.keras` with all details as stated in the [paper](https://arxiv.org/pdf/1711.09784.pdf). Some parts such as loss regularization term calculation are done in pure TensorFlow. The rest is encapsulated into keras custom layers. |
|            | Due to lack of keras' flexibility, SBDT model is not save-able using keras' serialization methods so `tf.Saver` is used instead. This also means that keras callback for [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) won't work with this implementation (unless the model is re-written to avoid using `tf.Tensor` objects as keras `Layer` arguments). |
|            | Due to use of [moving averages](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage) in calculation of penalty terms, custom two-step initialization of model parameters is required and model training (evaluation of `tree.loss` tensorflow graph op) is `batch_size`-dependent.  This also means, that `batch_size % train_data_size == 0` must hold, otherwise shape mismatch will be encountered at the end of training epoch (keras will feed the remainder as a smaller minibatch). |
| [models/convnet.py](models/convnet.py) | Implementation of basic convolutional NN model in `tf.keras` as given by keras [MNIST-CNN](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) basic example. |
| [models/utils.py](models/utils.py)   | Utility functions for re-setting tensorflow session and visualizing model parameters. |
| [makegif.sh](makegif.sh) | Converts directory of images into animation and labels frames based on folder name and file names. See [mnist.ipynb](./mnist.ipynb) for exemplary usage. |
| [data_files](data_files/)    | Saved model checkpoints (for easier reproducibility) and illustrative images / animations.

## Usage


