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
- If you want to create plots you will also need
    - pickle
    - matplotlib
    - paretoset (https://github.com/tommyod/paretoset)

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

```
usage: main.py [-h] [-dataset {adult,wine,titanic,autompg}]
               [-stl_epochs STL_EPOCHS] [-mtl_epochs MTL_EPOCHS] [-regression]
               [-runs RUNS] [-es_patience ES_PATIENCE]
               [-pl_patience PL_PATIENCE] [-verbose] [-tune_arch]
               [-show_full_scores] [-save_plots]

optional arguments:
  -h, --help            show this help message and exit
  -dataset {adult,wine,titanic,autompg}
                        Name of dataset.
  -stl_epochs STL_EPOCHS
                        Number of STL training epochs.
  -mtl_epochs MTL_EPOCHS
                        Number of MTL training epochs.
  -regression           Whether the task is regression or (binary)
                        classification.
  -runs RUNS            Number of runs.
  -es_patience ES_PATIENCE
                        Early Stopping patience.
  -pl_patience PL_PATIENCE
                        Reduce learning rate on plateau patience.
  -verbose              Print training process.
  -tune_arch            Tune the MLP architecture.
  -show_full_scores     Prints a Pandas DataFrame with multiple scores in the
                        MTL setting.
  -save_plots           Whether to save plots of Accuracy/MSE-Fidelity.
```

For example: 
```
python main.py -show_full_scores -runs=1 -dataset='adult'
```


