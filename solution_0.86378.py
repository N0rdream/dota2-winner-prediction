import os
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from DotAI.composers import get_dfs, get_dfs_t
from DotAI.helpers import save_submit
from DotAI.extractors import extract_targets, extract_features, extract_t_features
from DotAI.processors import get_final_df

# path to folder with jsonl files and to folder with final solution (might be the same)
PATH_IN = 'initial data'
PATH_OUT = 'team_1'
RAW_TRAIN_DATA = 'train_matches.jsonl'
RAW_TEST_DATA = 'test_matches.jsonl'

# files generated during processing jsonl files
PROCESSED_TARGETS = 'dota_targets.csv'
PROCESSED_TRAIN = 'dota_train.csv'
PROCESSED_TEST = 'dota_test.csv'
FINAL_DATASET = 'dota_df.csv'
PROCESSED_TRAIN_TIME_SLICED = 'train_data_0-180.csv'
PROCESSED_TEST_TIME_SLICED = 'test_data_0-180.csv'

# extract useful data from json files
extract_targets(PATH_IN, PATH_OUT, RAW_TRAIN_DATA, PROCESSED_TARGETS)
extract_features(PATH_IN, PATH_OUT, RAW_TRAIN_DATA, PROCESSED_TRAIN)
extract_features(PATH_IN, PATH_OUT, RAW_TEST_DATA, PROCESSED_TEST)
extract_t_features(PATH_IN, PATH_OUT, RAW_TRAIN_DATA, PROCESSED_TRAIN_TIME_SLICED, 0, 180)
extract_t_features(PATH_IN, PATH_OUT, RAW_TEST_DATA, PROCESSED_TEST_TIME_SLICED, 0, 180)
get_final_df(PATH_OUT, PROCESSED_TARGETS, PROCESSED_TRAIN, PROCESSED_TEST, FINAL_DATASET)

# create train/test dataframes from all data
initial_drop_cols = [
    'is_train', 
    'game_time', 
    'lobby_type', 
    'game_mode', 
]
fdf = pd.read_csv(os.path.join(PATH_OUT, FINAL_DATASET), index_col='match_id_hash')
X_train = fdf[fdf.is_train == 1].drop(columns=initial_drop_cols)
y_train = fdf[fdf.is_train == 1].target
X_test = fdf[fdf.is_train == 0].drop(columns=initial_drop_cols)
X_train_united_full, _ = get_dfs(X_train, vect_min_inv=10)
X_train_sparse_full = csr_matrix(X_train_united_full)

# fit first logreg for feature selection
logit_full = LogisticRegression(
    C=0.5, 
    random_state=17, 
    solver='liblinear',
    penalty='l1',
)
logit_full.fit(X_train_sparse_full, y_train)
fi_train_full = pd.DataFrame(
    sorted(zip(logit_full.coef_[0].tolist(), X_train_united_full.columns), reverse=True),
    columns=['Value','Feature']
)

# columns with |coefficient| <= 0.042
COLS_TO_DROP = fi_train_full[np.abs(fi_train_full.Value) <= 0.042].Feature

# create train/test dataframes and drop columns from COLS_TO_DROP
X_train_united, X_test_united = get_dfs(X_train, X_test, vect_min_inv=10)
X_train_united.drop(columns=COLS_TO_DROP, inplace=True)
X_test_united.drop(columns=COLS_TO_DROP, inplace=True)
X_train_sparse, X_test_sparse = csr_matrix(X_train_united), csr_matrix(X_test_united)

# fit second logreg + some averaging on random_state
y_pred_total = np.zeros(X_test.shape[0])
for rs in [1, 43, 432, 1777, 5555]:
    logit_test = LogisticRegression(
        C=1.5, 
        random_state=rs, 
        solver='liblinear',
        penalty='l1',
    )
    logit_test.fit(X_train_sparse, y_train);
    y_pred_test = logit_test.predict_proba(X_test_sparse)[:, 1]
    y_pred_total += y_pred_test

# create dataframe from timeseries data: from 0 to 180 seconds
dfts_train = pd.read_csv(os.path.join(PATH_OUT, PROCESSED_TRAIN_TIME_SLICED), index_col='match_id_hash')
dfts_test = pd.read_csv(os.path.join(PATH_OUT, PROCESSED_TEST_TIME_SLICED), index_col='match_id_hash')
dfts = pd.concat([dfts_train, dfts_test])
for label in [col for col in dfts.columns if 'item' in col]:
    dfts[label] = dfts[label].astype(str)

# create train/test dataframes
X_train_t = dfts[dfts.is_train == 1]
X_test_t = dfts[dfts.is_train == 0]
y_train_t = dfts[dfts.is_train == 1].target
X_train_united_t, X_test_united_t = get_dfs_t(X_train_t, X_test_t, vect_min_inv=10)
X_train_sparse_t, X_test_sparse_t = csr_matrix(X_train_united_t), csr_matrix(X_test_united_t)

# predict probabilities on timeseries data
logit_t = LogisticRegression(
    C=0.5, 
    random_state=17, 
    solver='liblinear',
    penalty='l1',
)

logit_t.fit(X_train_sparse_t, y_train_t);
y_time = logit_t.predict_proba(X_test_sparse_t)[:, 1]

# blending predictions on full data with timeseries predictions:
# if game_time <= 180 seconds - average predictions 50/50,
# else - take predictions on full data without averaging
df_result = fdf[fdf.is_train == 0][['game_time']]
df_result['y_base'] = y_pred_total / 5
df_result['y_time'] = y_time

def apply_average(row):
    if row.game_time < 180:
        return (row.y_time + row.y_base) / 2
    return row.y_base

df_result['total'] = df_result.apply(apply_average, axis=1)

# submit
save_submit(PATH_OUT, 'team_1_solution_0.86378', df_result.index, df_result['total'].values)