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
| [models.py](models.py)    | Implementation of STL baselines and MTL-based model classes in `tf.keras`. |
| [trainer.py](trainer.py) | Implementation of a `Trainer` class used to train the models. |
| [evaluator.py](evaluator.py)   | Implementation of an `Evaluator` class used to evaluate models using several metrics. |
| [experiment.py](experiment.py) | Class containing the logic for conducting the experiments. |
| [main.py](main.py) | Class that parses command-line arguments and calls `Experiment` from `experiment.py`. |
| [data_files](data_files/)    | Data files for the [Adult](https://archive.ics.uci.edu/dataset/2/adult) dataset.

## Usage


