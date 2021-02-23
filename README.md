Titanic and House Prices (and a couple more to come).

[titanic model](https://github.com/carbasemin/kaggle_noob/blob/main/titanic/titanic.py) - Random forest with hyperparameter optimization.

hp
  - [Linear regression](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_lr.py) with features that have correlation 0.5 or above with the target; underfits, scores ~0.30 on the competition.
  - [Ridge](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_ridge.py) scored ~0.14. Lasso and ElasticNet perform slightly worse so I'm going with Ridge (not even %50).
  - a most basic [random forest](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_basicRF.py). Overfits, scores ~0.14 on the competition. Worse than Ridge.
  - [Degree 2 poly-Lasso](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_polyR.py) with Lasso. Scores ~0.12.
  - [degree2 Lasso-poly-Lasso](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_poly-lasso.py); got the relevant categorical and numerical features seperately with Lasso, i.e., used the regularization as a feature engineer, then fit a degree 2 polynomial regression with Lasso again. Scores ~0.12, best one yet (top %10).