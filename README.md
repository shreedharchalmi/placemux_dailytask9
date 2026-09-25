\# Task 9: Hyperparameter Tuning



This repository implements systematic, leak-free hyperparameter optimization using Stratified 5-Fold Cross-Validation and GridSearchCV, validating real performance gains on an untouched held-out test set.



\## Optimized Hyperparameters

\* Estimator: `LogisticRegression`

\* Tuned parameters: `C=0.1`, `penalty='l2'`, `solver='liblinear'`, `class\_weight='balanced'`

\* Cross-validation scheme: `StratifiedKFold(n\_splits=5, shuffle=True, random\_state=42)`



\## Test Set Verification Lift

\* \*\*Malignant Recall (Class 0)\*\*: Improved from \*\*90.62%\*\* to \*\*96.88%\*\* (+6.26%).

\* \*\*False Negatives\*\*: Reduced from \*\*3 missed cancers\*\* down to \*\*1\*\*.

\* \*\*Test Accuracy\*\*: Improved from \*\*96.51%\*\* to \*\*97.67%\*\*.



\## Execution

```bash

git clone \[https://github.com/shreedharchalmi/placemux\_dailytask9.git](https://github.com/shreedharchalmi/placemux\_dailytask9.git)

cd placemux\_dailytask9

pip install scikit-learn pandas numpy joblib

python task\_9\_hyperparameter\_tuning.py

