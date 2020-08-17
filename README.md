Информация
----------
Решение соревнования по предсказанию победителя матча в Dota 2:  
https://www.kaggle.com/c/mlcourse-dota2-win-prediction/  
Соревнование проходило в рамках mlcourse.ai.

Как запустить
-------------
1. Copy DotAI folder and solution_0.86399.py, solution_0.86378.py files
2. Open solution_0.86399.py, solution_0.86378.py files and specify:
- PATH_IN – folder with initial jsonl files
- PATH_OUT – folder where file with predictions and generated csv's (around 400 MB) will be
3. Run solution_0.86399.py, solution_0.86378.py files

Notes
-------------
All preprocessing and feature generation will be done by DotAI module.
Feature selection and prediction will be done by logistic regression with l1-regularization on full and time-sliced (0-180 seconds) data.