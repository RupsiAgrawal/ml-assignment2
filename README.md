# T20 Innings Tempo Classifier

## a. Problem Statement
Given the match statistics of one T20 cricket innings (boundaries, dot balls, wickets, dismissal types, extras and so on), classify the innings as **high tempo (1)** or **low tempo (0)**. An innings is high tempo when its run rate (runs per legal ball) is at or above the median run rate of all innings (2.4364 runs per legal ball). This is a **binary classification** problem. Five classifiers are trained and compared, and an interactive Streamlit app lets a user upload test data and evaluate any saved model.

## b. Dataset Description
- **Source:** `deliveries.csv`, a ball-by-ball cricket (IPL-style) dataset. Source: Kaggle, "IPL Complete Dataset 2008-2025 (Enhanced Edition)" - https://www.kaggle.com/datasets/meruvakodandasuraj/ipl-complete-dataset-2008-2025-enhanced-edition
- **Raw file:** 134,190 deliveries, 1,158 matches, 17 teams, 18 columns.
- **Why aggregated:** a gradient-boosting probe on the raw ball-level data could not predict ball outcomes (AUC about 0.50 for wicket, boundary, dot ball and extra), so the file was aggregated to **one row per innings** with `prepare_data.py`.
- **Modelling dataset (`innings_dataset.csv`):** 2,290 instances (innings), **24 input features**, 1 target (`high_tempo`), plus `match_id`, which is used only to keep both innings of a match on the same side of the train/test split.
- **Target:** `high_tempo` = 1 if innings run rate >= median, else 0. Classes are balanced (1,145 vs 1,145).
- **Features:** innings number, batting team, bowling team, legal balls, wickets, fours, sixes, boundary %, dot-ball %, wides, no-balls, byes, leg-byes, extras %, caught, bowled, run-outs, stumped, lbw, wickets per over, powerplay balls, death-over balls, distinct bowlers, distinct batters.
- **Not given to the models:** total runs, run rate, and the counts of singles and twos, so the label is not a direct formula of the inputs.
- **Preprocessing:** numeric features scaled with `StandardScaler`, the two team columns one-hot encoded; both steps are inside each saved pipeline, so the app accepts raw rows. No missing values.
- **Split:** 80/20 grouped by match (`GroupShuffleSplit`, `random_state=42`) giving 1,829 training and 461 test innings. `test_data.csv` is that test split.
- **Caveats:** the innings in this file are partial (average about 56 legal balls out of 120), so `legal_balls` varies a lot. The task is descriptive (what an innings looked like), not a forecast made before the innings starts.

## c. GitHub Repository Link
`<https://github.com/YOUR-USERNAME/YOUR-REPO>`

Live Streamlit app: `<https://YOUR-APP.streamlit.app>`

Repository contents: `app.py`, `train.py`, `prepare_data.py`, `requirements.txt`, `README.md`, `test_data.csv`, `innings_dataset.csv`, `deliveries.csv`, `model/` (five saved pipelines, label encoder, metrics table).

## d. Models Used

Metrics on the held-out test set (461 innings, from matches not seen in training):

<!-- METRICS_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9371 | 0.9857 | 0.9383 | 0.9342 | 0.9363 | 0.8742 |
| Decision Tree | 0.8221 | 0.8222 | 0.8174 | 0.8246 | 0.8210 | 0.6443 |
| KNN | 0.7961 | 0.8689 | 0.8073 | 0.7719 | 0.7892 | 0.5925 |
| Naive Bayes | 0.7267 | 0.8071 | 0.6861 | 0.8246 | 0.7490 | 0.4638 |
| Support Vector Machine | 0.9371 | 0.9802 | 0.9462 | 0.9254 | 0.9357 | 0.8743 |
<!-- METRICS_END -->

Precision, recall and F1 are for the positive class (high tempo = 1).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC (0.9857) and joint-best accuracy and MCC. Precision (0.938) and recall (0.934) are balanced, with only 14 false positives and 15 false negatives. The label comes from a run rate, which is close to a linear mix of boundary %, dot-ball % and extras %, so a linear model on scaled features fits it very well. |
| Decision Tree | Accuracy 0.822 and MCC 0.644, clearly below the linear models. The unpruned tree outputs almost hard 0/1 probabilities, which is why its AUC (0.822) is about the same as its accuracy. Axis-aligned splits approximate a smooth linear boundary poorly, and the tree overfits the training innings. |
| KNN | Accuracy 0.796 and a respectable AUC of 0.869, but the lowest recall of all five models (0.772, so 52 high-tempo innings were missed). With 24 features plus one-hot team columns, many weakly relevant counts (stumped, lbw, distinct bowlers) dilute the distance, so neighbours are less informative. |
| Naive Bayes | Weakest model (accuracy 0.727, MCC 0.464). It over-predicts high tempo: recall is 0.825 but precision only 0.686, with 86 false positives. Many features are strongly correlated (fours and sixes with boundary %, wides and no-balls with extras %, wickets with dismissal counts), which breaks the independence assumption and double-counts evidence; count features are also not Gaussian. |
| Support Vector Machine | Virtually tied with Logistic Regression: same accuracy (0.9371), MCC 0.8743 versus 0.8742, and the highest precision (0.946). Its AUC (0.980) is slightly lower. The RBF kernel adds almost nothing because the class boundary is already nearly linear. |
| **Overall winner for this dataset?** | **Logistic Regression.** It has the best AUC (0.9857) and matches the SVM on accuracy and MCC; the 0.0001 MCC gap in the SVM's favour is negligible on 461 test rows. It is also simpler, faster, and interpretable. SVM is a close second. |

## Environment used for the reported results
<!-- ENV_START -->
- Python 3.13.17
- scikit-learn 1.9.1
- pandas 3.0.5
- numpy 2.5.3
- joblib 1.6.0
- matplotlib 3.11.2
- seaborn 0.13.2
- streamlit (not installed here)
<!-- ENV_END -->

## How to run
```
pip install -r requirements.txt
python prepare_data.py --src deliveries.csv --out innings_dataset.csv
python train.py --data innings_dataset.csv --target high_tempo --group match_id
streamlit run app.py
```
Upload `test_data.csv` in the app, choose a model from the dropdown, and view the metrics, confusion matrix, ROC curve and classification report. The app loads the saved models and never retrains on the uploaded file.
