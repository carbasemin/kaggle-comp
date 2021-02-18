Titanic and House Prices (and a couple more to come).

[titanic model](https://github.com/carbasemin/kaggle_noob/blob/main/titanic/titanic.py) - Random forest with hyperparameter optimization. Preprocessing is done with functions and pipelines, which is the better practice.

hp
  - a most basic [random forest](https://github.com/carbasemin/kaggle_noob/blob/main/hp/model_basicRF.py); overfits, scores ~0.14 on the competition. Not even top %50.
  - [linear regression](https://github.com/carbasemin/kaggle_noob/blob/main/hp/model_lr.py) with features that have correlation 0.5 or above with the target; underfits, scores ~0.30 on the competition. Even worse than no-brain RF.
  - [degree 2 polynomial regression](https://github.com/carbasemin/kaggle_noob/blob/main/hp/model_polyR.py) with the same base features as the linear regression, still underfits, scores ~0.26. Degree 3 and 4 better fits the training data but fails on the test, on kaggle that is. 
