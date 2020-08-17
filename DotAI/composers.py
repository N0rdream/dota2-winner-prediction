import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')


HEROES_LABELS_RAD = ['r1_hero_name', 'r2_hero_name', 'r3_hero_name', 'r4_hero_name', 'r5_hero_name']
HEROES_LABELS_DIR = ['d1_hero_name', 'd2_hero_name', 'd3_hero_name', 'd4_hero_name', 'd5_hero_name']
HEROES_LABELS = HEROES_LABELS_RAD + HEROES_LABELS_DIR

ITEMS_LABELS_RAD = ['r1_items', 'r2_items', 'r3_items', 'r4_items', 'r5_items']
ITEMS_LABELS_DIR = ['d1_items', 'd2_items', 'd3_items', 'd4_items', 'd5_items']
ITEMS_LABELS = ITEMS_LABELS_RAD + ITEMS_LABELS_DIR


def get_inventory(X_train, X_test=None, vect_min_inv=0):
    
    VECT_INVENTORY = TfidfVectorizer(
        ngram_range=(1, 1), 
        max_features=1000,
        min_df=vect_min_inv,
    )
    
    VECT_INVENTORY.fit(('-'.join(inv) for inv in X_train[['rad_inventory', 'dir_inventory']].values))
    
    cols = ['inventory_' + i for i in VECT_INVENTORY.get_feature_names()]
    
    df_rad_train = VECT_INVENTORY.transform(X_train['rad_inventory'].values)
    df_dir_train = VECT_INVENTORY.transform(X_train['dir_inventory'].values)
    df_train = df_rad_train - df_dir_train
    df_train = pd.DataFrame(df_train.todense(), columns=cols, index=X_train.index)
    
    if X_test is not None:
        df_rad_test = VECT_INVENTORY.transform(X_test['rad_inventory'].values)
        df_dir_test = VECT_INVENTORY.transform(X_test['dir_inventory'].values)
        df_test = df_rad_test - df_dir_test 
        df_test = pd.DataFrame(df_test.todense(), columns=cols, index=X_test.index)
        
        return df_train, df_test
    
    return df_train, None
    
    
def get_heroes(X_train, X_test=None):
    
    VECT_HEROES = CountVectorizer(
        ngram_range=(1, 1), 
        max_features=150, 
        binary=True,
    )

    VECT_HEROES.fit((';'.join(heroes) for heroes in X_train[HEROES_LABELS].values))
    
    cols = VECT_HEROES.get_feature_names()
    
    df_rad_train = VECT_HEROES.transform(
        (';'.join(sorted(heroes)) for heroes in X_train[HEROES_LABELS_RAD].values))
    df_dir_train = VECT_HEROES.transform(
        (';'.join(sorted(heroes)) for heroes in X_train[HEROES_LABELS_DIR].values))
    df_train = df_rad_train - df_dir_train
    df_train = pd.DataFrame(df_train.todense(), columns=cols, index=X_train.index)
    
    if X_test is not None: 
        
        df_rad_test = VECT_HEROES.transform(
            (';'.join(sorted(heroes)) for heroes in X_test[HEROES_LABELS_RAD].values))
        df_dir_test = VECT_HEROES.transform(
            (';'.join(sorted(heroes)) for heroes in X_test[HEROES_LABELS_DIR].values))
        df_test= df_rad_test - df_dir_test
        df_test = pd.DataFrame(df_test.todense(), columns=cols, index=X_test.index)
        
        return df_train, df_test
    
    return df_train, None      


def get_main_df(X_train, X_test=None):
    cols = ['target', 'rad_inventory', 'dir_inventory'] + HEROES_LABELS
    
    if X_test is not None:
        return X_train.drop(columns=cols), X_test.drop(columns=cols)
    
    return X_train.drop(columns=cols), None


def get_main_scaled(X_train, X_test=None):
    scaler = StandardScaler()
    df_train = scaler.fit_transform(X_train)
    df_train = pd.DataFrame(df_train, columns=X_train.columns, index=X_train.index)
    if X_test is not None:
        df_test = scaler.transform(X_test)
        df_test = pd.DataFrame(df_test, columns=X_test.columns, index=X_test.index)
        return df_train, df_test
    return df_train, None


def get_dfs(X_train, X_test=None, vect_min_inv=10):
    
    df_heroes_train, df_heroes_test = get_heroes(X_train, X_test)
    df_inv_train, df_inv_test = get_inventory(X_train, X_test, vect_min_inv)
    df_main_train, df_main_test = get_main_df(X_train, X_test)
    
    df_main_train_scaled, df_main_test_scaled = get_main_scaled(df_main_train, df_main_test)

    X_train_united = pd.concat([
        df_main_train_scaled,
        df_heroes_train,
        df_inv_train,
    ], axis=1)

    if X_test is not None:
        X_test_united = pd.concat([
            df_main_test_scaled,
            df_heroes_test,
            df_inv_test,
        ], axis=1)

        return X_train_united, X_test_united
    
    return X_train_united, None


def get_inventory_t(X_train, X_test=None, vect_min_inv=10):
    
    VECT_INVENTORY = TfidfVectorizer(
        ngram_range=(1, 1), 
        max_features=1000,
        min_df=vect_min_inv,
    )
    
    VECT_INVENTORY.fit((';'.join(heroes) for heroes in X_train[ITEMS_LABELS].values))
    
    cols = ['inventory_' + i for i in VECT_INVENTORY.get_feature_names()]
    
    df_rad_train = VECT_INVENTORY.transform(('-'.join(heroes) for heroes in X_train[ITEMS_LABELS_RAD].values))
    df_dir_train = VECT_INVENTORY.transform(('-'.join(heroes) for heroes in X_train[ITEMS_LABELS_DIR].values))
    df_train = df_rad_train - df_dir_train
    df_train = pd.DataFrame(df_train.todense(), columns=cols, index=X_train.index)
    
    if X_test is not None:
        df_rad_test = VECT_INVENTORY.transform(('-'.join(heroes) for heroes in X_test[ITEMS_LABELS_RAD].values))
        df_dir_test = VECT_INVENTORY.transform(('-'.join(heroes) for heroes in X_test[ITEMS_LABELS_DIR].values))
        df_test = df_rad_test - df_dir_test 
        df_test = pd.DataFrame(df_test.todense(), columns=cols, index=X_test.index)
        
        return df_train, df_test
    
    return df_train, None
    
  
def get_main_df_t(X_train, X_test=None):
    cols = [
        'target',
        'is_train', 
        'game_time', 
        'first_tower',
    ] + HEROES_LABELS + ITEMS_LABELS
    
    if X_test is not None:
        return X_train.drop(columns=cols), X_test.drop(columns=cols)
    
    return X_train.drop(columns=cols), None


def get_dfs_t(X_train, X_test=None, vect_min_inv=0):
    
    df_heroes_train, df_heroes_test = get_heroes(X_train, X_test)
    df_inv_train, df_inv_test = get_inventory_t(X_train, X_test, vect_min_inv)
    df_main_train, df_main_test = get_main_df_t(X_train, X_test)

    df_main_train_scaled, df_main_test_scaled = get_main_scaled(df_main_train, df_main_test)

    X_train_united = pd.concat([
        df_main_train_scaled,
        df_heroes_train,
        df_inv_train,
    ], axis=1)

    if X_test is not None:
        X_test_united = pd.concat([
            df_main_test_scaled,
            df_heroes_test,
            df_inv_test,
        ], axis=1)

        return X_train_united, X_test_united
    
    return X_train_united, None