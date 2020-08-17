import pandas as pd
import numpy as np
import os


HEROES_LABELS_RAD = ['r1_hero_name', 'r2_hero_name', 'r3_hero_name', 'r4_hero_name', 'r5_hero_name']
HEROES_LABELS_DIR = ['d1_hero_name', 'd2_hero_name', 'd3_hero_name', 'd4_hero_name', 'd5_hero_name']
HEROES_LABELS = HEROES_LABELS_RAD + HEROES_LABELS_DIR

PLAYERS_LABELS_RAD = ['r1_', 'r2_', 'r3_', 'r4_', 'r5_']
PLAYERS_LABELS_DIR = ['d1_', 'd2_', 'd3_', 'd4_', 'd5_']
PLAYERS_LABELS = PLAYERS_LABELS_RAD + PLAYERS_LABELS_DIR


def get_rad_labels(feature):
    return [pl + feature for pl in PLAYERS_LABELS_RAD]


def get_dir_labels(feature):
    return [pl + feature for pl in PLAYERS_LABELS_DIR]


def get_united_df(path, file_targets, file_train, file_test):
    df_targets = pd.read_csv(os.path.join(path, file_targets))
    df_train = pd.read_csv(os.path.join(path, file_train))
    df_test = pd.read_csv(os.path.join(path, file_test))
    df_train['is_train'] = 1
    df_train['target'] = df_targets.radiant_win.apply(lambda t: 1 if t else -1)
    df_test['is_train'] = 0
    df_test['target'] = np.nan
    df = pd.concat([df_train, df_test]).set_index('match_id_hash')
    df['game_time'] = df[['r1_life_state_0', 'r1_life_state_1', 'r1_life_state_2']].sum(axis=1) / 2
    players_features_base_drop = ['hero_id', 'player_slot', 'obs_placed', 'account_id_hash']
    players_drop_cols = [pl + c for pl in PLAYERS_LABELS for c in players_features_base_drop]
    return df.drop(columns=players_drop_cols)


def get_final_df(path, file_targets, file_train, file_test, file_out):
    df = get_united_df(path, file_targets, file_train, file_test)
    # runes, pings, max_hero_hit
    fdf = df[['target', 'is_train', 'game_mode', 'lobby_type']].copy()
    fdf[HEROES_LABELS] = df[HEROES_LABELS]
    fdf['rad_inventory'] = df['rad_inventory']
    fdf['dir_inventory'] = df['dir_inventory']
    fdf['game_time'] = df.game_time
    fdf['towers_kills_last'] = df.towers_kills_last
    fdf['teamfights_all'] = df.teamfights_all

    ### Coordinates
    for pr in PLAYERS_LABELS_RAD:
        fdf[pr + 'dist'] = np.sqrt((186 - df[pr + 'x'])**2 + (186 - df[pr + 'y'])**2)

    for pr in PLAYERS_LABELS_DIR:
        fdf[pr + 'dist'] = np.sqrt((66 - df[pr + 'x'])**2 + (66 - df[pr + 'y'])**2)

    fdf['rad_sum_dist'] = np.sqrt(fdf[[pl + 'dist' for pl in PLAYERS_LABELS_RAD]].sum(axis=1))
    fdf['dir_sum_dist'] = np.sqrt(fdf[[pl + 'dist' for pl in PLAYERS_LABELS_DIR]].sum(axis=1))
    fdf['sub_sum_dist'] = fdf['rad_sum_dist'] - fdf['dir_sum_dist']
    fdf.drop(columns=['rad_sum_dist', 'dir_sum_dist'], inplace=True)
    fdf.drop(columns=[pl + 'dist' for pl in PLAYERS_LABELS_RAD], inplace=True)
    fdf.drop(columns=[pl + 'dist' for pl in PLAYERS_LABELS_DIR], inplace=True)
    fdf['rad_barracks'] = df['rad_barracks']
    fdf['dir_barracks'] = df['dir_barracks']
    fdf['dff_barracks'] = fdf['rad_barracks'] - fdf['dir_barracks']
    fdf.drop(columns=['rad_barracks', 'dir_barracks'], inplace=True)

    def add_difference(feature, scaler):
        if scaler == 'as_is':
            fdf[f'rad_sum_{feature}'] = df[get_rad_labels(feature)].sum(axis=1)
            fdf[f'dir_sum_{feature}'] = df[get_dir_labels(feature)].sum(axis=1)
        if scaler == 'sqrt':
            fdf[f'rad_sum_{feature}'] = np.sqrt(np.abs(df[get_rad_labels(feature)].sum(axis=1)))
            fdf[f'dir_sum_{feature}'] = np.sqrt(np.abs(df[get_dir_labels(feature)].sum(axis=1)))    
        if scaler == 'cbrt':
            fdf[f'rad_sum_{feature}'] = np.cbrt(np.abs(df[get_rad_labels(feature)].sum(axis=1)))
            fdf[f'dir_sum_{feature}'] = np.cbrt(np.abs(df[get_dir_labels(feature)].sum(axis=1)))
        if scaler == 'log1p':
            fdf[f'rad_sum_{feature}'] = np.log1p(np.abs(df[get_rad_labels(feature)].sum(axis=1)))
            fdf[f'dir_sum_{feature}'] = np.log1p(np.abs(df[get_dir_labels(feature)].sum(axis=1))) 
        fdf[f'sub_sum_{feature}'] = fdf[f'rad_sum_{feature}'] - fdf[f'dir_sum_{feature}']
        fdf.drop(columns=[f'rad_sum_{feature}', f'dir_sum_{feature}'], inplace=True)

    #add_difference('max_hero_hit', 'sqrt')
    add_difference('life_state_1', 'sqrt')
    #add_difference('life_state_2', 'sqrt')
    #add_difference('ping', 'sqrt')
    add_difference('gold', 'sqrt')
    add_difference('gold_reason_hero', 'log1p')
    add_difference('gold_reason_other', 'log1p')
    add_difference('gold_reason_death', 'sqrt')
    add_difference('gold_reason_buyback', 'sqrt')
    #add_difference('gold_reason_abandon', 'as_is')
    #add_difference('gold_reason_sell', 'as_is')
    add_difference('gold_reason_structure', 'sqrt')
    add_difference('gold_reason_creep', 'cbrt')
    add_difference('healing', 'sqrt')
    add_difference('actions_sum', 'log1p')
    #add_difference('actions_len', 'as_is')
    add_difference('damage_taken_hero', 'cbrt')
    add_difference('damage_taken_host', 'log1p')
    add_difference('damage_taken_neut', 'log1p')
    add_difference('damage_ally', 'log1p')
    add_difference('damage_neut', 'log1p')
    #add_difference('assists', 'as_is')
    add_difference('creeps_stacked', 'sqrt')
    #add_difference('deaths', 'as_is')
    add_difference('denies', 'sqrt')
    add_difference('health', 'sqrt')
    add_difference('max_health', 'log1p')
    #add_difference('kills', 'as_is')
    add_difference('kill_streak', 'as_is')
    add_difference('level', 'cbrt')
    #add_difference('lh', 'as_is')
    add_difference('life_state_0', 'sqrt')
    add_difference('max_mana', 'sqrt')
    add_difference('multi_kill', 'as_is')
    add_difference('nearby_creep_death_count', 'log1p')
    add_difference('observers_placed', 'sqrt')
    add_difference('xp', 'sqrt')
    add_difference('randomed', 'sqrt')
    add_difference('rune_pickups', 'log1p')
    add_difference('sen_placed', 'sqrt')
    add_difference('stuns', 'sqrt')
    #add_difference('towers_killed', 'as_is')
    #add_difference('damage_tower_1', 'as_is')
    #add_difference('damage_tower_2', 'as_is')
    #add_difference('damage_tower_3', 'as_is')

    features_to_add = {
        #'ping',
        #'max_hero_hit',
        'gold_reason_other', 
        'gold_reason_death', 
        'gold_reason_buyback',
        'gold_reason_abandon',
        'gold_reason_sell',
        'gold_reason_structure',
        'gold_reason_hero',
        'gold_reason_creep',
        'gold_reason_roshan',
        'gold_reason_courier', 
        'damage_taken_hero', 
        'damage_taken_host', 
        'damage_taken_neut',
        'damage_hero', 
        'damage_creep', 
        'damage_neut', 
        'damage_ally',
        'damage_tower_1',
        'damage_tower_2',
        'damage_tower_3',
        #'damage_tower_4',
        'actions_sum',
        'actions_len',
        'healing',
        'assists',
        #'camps_stacked',
        'creeps_stacked',
        'deaths',
        'denies',
        #'firstblood_claimed',
        'gold',
        'health',
        #'hero_name',
        #'kill_streak',
        'kills',
        'level',
        'lh',
        'life_state_0',
        'life_state_1',
        #'life_state_2',
        #'rune_type_0',
        #'rune_type_1',
        #'rune_type_2',
        #'rune_type_3',
        #'rune_type_4',
        #'rune_type_5',
        #'rune_type_6',
        'max_health',
        'max_mana',
        #'multi_kill',
        'nearby_creep_death_count',
        #'observers_placed',
        #'pred_vict',
        'randomed',
        'roshans_killed',
        'rune_pickups',
        'sen_placed',
        'stuns',
        'teamfight_participation',
        'towers_killed',
        #'x',
        'xp',
        #'y',
    }

    for f in features_to_add:
        fdf['rad_sum_' + f] = df[[pl + f for pl in PLAYERS_LABELS_RAD]].sum(axis=1)
        fdf['dir_sum_' + f] = df[[pl + f for pl in PLAYERS_LABELS_DIR]].sum(axis=1)
        fdf['rad_std_' + f] = df[[pl + f for pl in PLAYERS_LABELS_RAD]].std(axis=1)
        fdf['dir_std_' + f] = df[[pl + f for pl in PLAYERS_LABELS_DIR]].std(axis=1)
        fdf['sub_sum_rpt_' + f] = (fdf['rad_sum_' + f] - fdf['dir_sum_' + f]) / (fdf['game_time'] // 60 + 1)
        fdf['sub_sum_lin_' + f] = fdf['rad_sum_' + f] - fdf['dir_sum_' + f]
        fdf['sub_std_rpt_' + f] = (fdf['rad_std_' + f] - fdf['dir_std_' + f]) / (fdf['game_time'] // 60 + 1)
        fdf['sub_std_lin_' + f] = fdf['rad_std_' + f] - fdf['dir_std_' + f]
        fdf.drop(columns=['rad_sum_' + f, 'dir_sum_' + f], inplace=True)
        fdf.drop(columns=['rad_std_' + f, 'dir_std_' + f], inplace=True)
        
    fdf.to_csv(os.path.join(path, file_out))