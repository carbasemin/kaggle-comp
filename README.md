Titanic, House Prices and Digit Recognizer (and a couple more to come).

titanic - [RF with hyperparameter optimiziation](https://github.com/carbasemin/kaggle_noob/blob/main/titanic/titanic.py), top %14. Not good, nor bad, just used it to learn the basics of kaggle and all.

hp
  - [Linear regression](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_lr.py) with features that have correlation 0.5 or above with the target; underfits, scores ~0.30 on the competition.
  - [Ridge](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_ridge.py) scored ~0.14. Lasso and ElasticNet perform slightly worse so I'm going with Ridge (not even %50).
  - a most basic [Random Forest](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_basicRF.py). Overfits, scores ~0.14 on the competition. Worse than Ridge.
  - [Degree 2 poly-Lasso](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_polyR.py) with Lasso. Scores ~0.12.
  - [degree2 Lasso-poly-Lasso](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/model_poly-lasso.py) - got the relevant categorical and numerical features seperately with Lasso, i.e., used the regularization as a feature engineer, then fit a degree 2 polynomial regression with Lasso again. Scores ~0.12, best one yet (top %10).
  - [Bagging Lasso](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/bagging_stuff.py) - Not particulary different then lasso. Increasing n_estimators doesn't do much --maybe I should've *really* increased it. 
  - [RF with PCA](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/PCA_stuff.py) - Altough 4 principal components explain 0.99 of the variance, it does significantly worse than basicRF that uses all the features.
  - [Gradient Boosting Regressor (GBR)](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/boosting_stuff.py) scores ~0.13 with hyperparameter optimization. Second best model.
  - [Ensemble of Lasso-poly-Lasso and GBR](https://github.com/carbasemin/kaggle_noob/blob/main/hp/models/ensemble.py) - 0.8 from Lasso-poly-Lasso and 0.2 from GBR. A touch better than L-p-L. Still top %10 but increased the ranking about 10 steps.

digit_recognizer
 - [base](https://github.com/carbasemin/kaggle_noob/blob/main/digit_recognizer/models/base.py) bunch of models with PCA, seeing how they do.
 - K-NN looked promising, so I optimized it; [model](https://github.com/carbasemin/kaggle_noob/blob/main/digit_recognizer/models/K-NN.py) - PCA with 0.90 of the variance, K-NN with 3 neighbours, scores ~0.97, %71. It's quite bad. Maybe I dropped too many features, didn't optimize the PCA part, maybe that's the most K-NN can do?