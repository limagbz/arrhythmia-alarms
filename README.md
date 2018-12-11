Detecting false arrhythmia alarms in the ICU using Convolutional Neural Networks
==============================

The objective of this work is to detect false arrhythmia alarms using convolutional neural networks as proposed in [_Reducing False Arrhythmia Alarms in the ICU: the PhysioNet/Computing in Cardiology Challenge 2015_](https://physionet.org/challenge/2015/). This project was the code from my undergraduate thesis to finish my computer engineering degree. 

Project Organization
------------

    ├── LICENSE
    ├── README.md                  <- This file
    ├── reports
    |   └── figures                <- Figures used in the final report
    |   └── results                <- Tables with results of the 3-step grid-search training
    │
    ├── requirements.txt           <- pip requirements
    ├── requirements-conda.txt     <- conda requirements
    │
    ├── src
    │   ├── data
    │   │   ├── make_dataset.py     <- code to download, create and prepare database
    │   │   ├── plot_dataset.py     <- auxiliar functions to create the plots used as input
    │   │   ├── prepare_dataset.py  <- auxiliar code to prepare dataset
    │   │   └── resample_dataset.py <- random oversampling code
    │   │
    │   ├── features
    │   │   └── build_features.py   <- bottleneck features generation
    │   │
    │   ├── models
    │   │   ├── metrics_callback.py <- class for using as callback on keras models
    │   │   ├── models.py           <- models used in training
    │   │   └── train_model.py      <- code for model training
    │   └── experiment.py           <- experiment run
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
