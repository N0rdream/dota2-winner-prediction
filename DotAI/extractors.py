import pandas as pd
import numpy as np
import json
import csv
import os


PLAYERS_LABELS = ['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5']

PLAYERS_BASE = [
    'player_slot',
    'hero_id',
    'hero_name',
    'account_id_hash',
    'obs_placed',
    'sen_placed',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'towers_killed',
    'roshans_killed',
    'observers_placed',
    'stuns',
    'randomed',
    'pred_vict',
    'gold',
    'lh',
    'xp',
    'x',
    'y',
    'health',
    'max_health',
    'max_mana',
    'level',
    'kills',
    'deaths',
    'assists',
    'denies',
    'nearby_creep_death_count',
]

base_damage_features = [
    'damage_hero', 
    'damage_creep', 
    'damage_neut', 
    'damage_ally',
    'damage_tower_1',
    'damage_tower_2',
    'damage_tower_3',
    'damage_tower_4',
]

base_damage_taken_features = [
    'damage_taken_hero', 
    'damage_taken_host', 
    'damage_taken_neut'
]

base_gold_reasons_features = [
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
]

base_killed_towers = [
    'killed_towers_1', 
    'killed_towers_2', 
    'killed_towers_3',
    'killed_towers_4',
]
        

FEATURE_LABELS = [
    'game_time', 'match_id_hash', 'game_mode', 'lobby_type',
    *[l + '_' + f for l in PLAYERS_LABELS for f in PLAYERS_BASE],
    *[f'{p}_life_state_{n}' for p in PLAYERS_LABELS for n in ['0', '1', '2']],
    *[f'{p}_kill_streak' for p in PLAYERS_LABELS],
    *[f'{p}_multi_kill' for p in PLAYERS_LABELS],
    *[f'{p}_{f}' for p in PLAYERS_LABELS for f in ['actions_sum', 'actions_len']],
    *[f'{p}_healing' for p in PLAYERS_LABELS],
    *[f'{p}_{f}' for p in PLAYERS_LABELS for f in base_damage_features],
    *[f'{p}_{f}' for p in PLAYERS_LABELS for f in base_damage_taken_features],
    *[f'{p}_{f}' for p in PLAYERS_LABELS for f in base_gold_reasons_features],
    'rad_inventory', 'dir_inventory',
    'rad_barracks', 'dir_barracks',
    *[f'{p}_ping' for p in PLAYERS_LABELS],
    *[f'{p}_xp_reason_{f}' for p in PLAYERS_LABELS for f in ['0', '1', '2', '3']],
    *[f'{p}_rune_type_{f}' for p in PLAYERS_LABELS for f in ['0', '1', '2', '3', '4', '5', '6']],
    *[f'{p}_{f}' for p in PLAYERS_LABELS for f in base_killed_towers],
    'towers_kills_last',
    'teamfights_all',  
]

T_FEATURES = [
    'match_id_hash',
    'target',
    'game_time',
    'first_tower',
    'sub_gold',
    'sub_xp',
    'sub_lh',
    'sub_denies',
    'r1_hero_name', 'r2_hero_name', 'r3_hero_name', 'r4_hero_name', 'r5_hero_name',
    'd1_hero_name', 'd2_hero_name', 'd3_hero_name', 'd4_hero_name', 'd5_hero_name',
    'is_train',
    'r1_items', 'r2_items', 'r3_items', 'r4_items', 'r5_items',
    'd1_items', 'd2_items', 'd3_items', 'd4_items', 'd5_items',
]


def extract_targets(path_in, path_out, file_in, file_out):
    with open(
        os.path.join(path_in, file_in)
    ) as fi, open(
        os.path.join(path_out, file_out), 'w', newline='\n'
    ) as fo:
        writer = csv.writer(fo)
        writer.writerow([
            'duration',
            'radiant_win',
            'next_roshan_team',
        ]) 
        lines = fi.readlines()
        for line in lines:
            data = json.loads(line, encoding='utf-8')
            writer.writerow([
                data['targets']['duration'],
                data['targets']['radiant_win'],
                data['targets']['next_roshan_team'],
            ])


def get_ids(row):
    return row['game_time'], row['match_id_hash'], row['game_mode'], row['lobby_type']


def get_base_features(row):
    features = []
    for player in row['players']:
        for feature in PLAYERS_BASE:
            features.append(player[feature])
    return features


def get_life_states(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('life_state'):
                features.extend([
                    player[feature].get('0', 0),
                    player[feature].get('1', 0),
                    player[feature].get('2', 0),
                ])
    return features


def get_kill_streaks(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('kill_streaks'):
                features.append(len(player[feature]))
    return features


def get_multi_kills(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('multi_kills'):
                features.append(len(player[feature]))
    return features


def get_actions(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('actions'):
                features.append(sum(v for k, v in player[feature].items()))
                features.append(len(player[feature]))
    return features


def get_healing(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('healing'):
                features.append(sum(v for k, v in player[feature].items()))
    return features


def get_damage(row):
    features = []
    for i, player in enumerate(row['players']):
        for feature in player.keys():
            if feature.endswith('damage') and i < 5:
                features.append(sum(v for k, v in player[feature].items() if 'hero' in k))
                features.append(sum(v for k, v in player[feature].items() if 'badguys' in k and 'tower' not in k))
                features.append(sum(v for k, v in player[feature].items() if 'neutral' in k))
                features.append(sum(v for k, v in player[feature].items() if 'goodguys' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower1' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower2' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower3' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower4' in k))
            if feature.endswith('damage') and i >= 5:
                features.append(sum(v for k, v in player[feature].items() if 'hero' in k))
                features.append(sum(v for k, v in player[feature].items() if 'goodguys' in k and 'tower' not in k))
                features.append(sum(v for k, v in player[feature].items() if 'neutral' in k))
                features.append(sum(v for k, v in player[feature].items() if 'badguys' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower1' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower2' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower3' in k))
                features.append(sum(v for k, v in player[feature].items() if 'tower4' in k))
    return features


def get_damage_taken(row):
    features = []
    for i, player in enumerate(row['players']):
        for feature in player.keys():
            if feature.endswith('damage_taken') and i < 5:
                features.append(sum(v for k, v in player[feature].items() if 'hero' in k))
                features.append(sum(v for k, v in player[feature].items() if 'badguys' in k))
                features.append(sum(v for k, v in player[feature].items() if 'neutral' in k))
            if feature.endswith('damage_taken') and i >= 5:
                features.append(sum(v for k, v in player[feature].items() if 'hero' in k))
                features.append(sum(v for k, v in player[feature].items() if 'goodguys' in k))
                features.append(sum(v for k, v in player[feature].items() if 'neutral' in k))
    return features


def get_gold_reasons(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('gold_reasons'):
                features.append(player[feature].get('0', 0))
                features.append(player[feature].get('1', 0))
                features.append(player[feature].get('2', 0))
                features.append(player[feature].get('5', 0))
                features.append(player[feature].get('6', 0))
                features.append(player[feature].get('11', 0))
                features.append(player[feature].get('12', 0))
                features.append(player[feature].get('13', 0))
                features.append(player[feature].get('14', 0))
                features.append(player[feature].get('15', 0))
    return features


def get_inventory(row):
    rad_items = []
    dir_items = []
    for i, player in enumerate(row['players']):
        for feature in player.keys():
            if feature.endswith('hero_inventory') and i < 5:
                for item in player[feature]:
                    rad_items.append(item['id'].split('item_')[-1])
            if feature.endswith('hero_inventory') and i >= 5:
                for item in player[feature]:
                    dir_items.append(item['id'].split('item_')[-1])                
    return ';'.join(rad_items), ';'.join(dir_items)


def get_barracks(row):              
    barracks_by_rad = 0
    barracks_by_dir = 0
    for event in row['objectives']:
        if event['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
            if event['key'] in ['1', '2', '4', '8', '16', '32']:
                barracks_by_rad += 1
            if event['key'] in ['64', '128', '256', '512', '1024', '2048']:
                barracks_by_dir += 1             
    return barracks_by_rad, barracks_by_dir


def get_pings(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('pings'):
                features.append(player[feature].get('0', 0)) 
    return features


def get_xp_reasons(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('xp_reasons'):
                features.append(player[feature].get('0', 0))
                features.append(player[feature].get('1', 0))
                features.append(player[feature].get('2', 0))
                features.append(player[feature].get('3', 0)) 
    return features


def get_runes(row):
    features = []
    for player in row['players']:
        for feature in player.keys():
            if feature.endswith('runes'):
                features.append(player[feature].get('0', 0))
                features.append(player[feature].get('1', 0))
                features.append(player[feature].get('2', 0))
                features.append(player[feature].get('3', 0))
                features.append(player[feature].get('4', 0))
                features.append(player[feature].get('5', 0))
                features.append(player[feature].get('6', 0)) 
    return features


def get_killed_towers(row):
    features = []
    for i, player in enumerate(row['players']):
        for feature in player.keys():
            if feature == 'killed' and i < 5:
                features.append(sum(1 for k, v in player[feature].items() if 'tower1' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower2' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower3' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower4' in k))
            if feature == 'killed' and i >= 5:
                features.append(sum(1 for k, v in player[feature].items() if 'tower1' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower2' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower3' in k))
                features.append(sum(1 for k, v in player[feature].items() if 'tower4' in k))
    return features


def get_towers_kills_last(row):              
    t_0 = 0
    t_n = 0
    acc = 0
    for event in row['objectives']:
        if event['type'] == 'CHAT_MESSAGE_TOWER_KILL':
            if event['team'] == 2:
                if acc >= 0:
                    acc += 1
                else:
                    acc = 1
            if event['team'] == 3:
                if acc <= 0:
                    acc -= 1
                else:
                    acc = -1
        if event['type'] == 'CHAT_MESSAGE_TOWER_DENY':
            if event['player_slot'] > 100:
                if acc >= 0:
                    acc += 1
                else:
                    acc = 1
            if event['player_slot'] < 100:
                if acc <= 0:
                    acc -= 1
                else:
                    acc = -1          
    return acc


def get_teamfights_all(row):              
    acc_rad = 0
    acc_dir = 0
    for fight in row['teamfights']:
        gains_rad = sum(p['xp_delta'] + p['gold_delta'] for p in fight['players'][:5])
        kills_rad = sum(len(p['killed']) for p in fight['players'][:5])
        gains_dir = sum(p['xp_delta'] + p['gold_delta'] for p in fight['players'][5:10])
        kills_dir = sum(len(p['killed']) for p in fight['players'][5:10])
        if kills_rad > kills_dir:
            acc_rad += 1
        if kills_rad < kills_dir:
            acc_dir += 1
        if kills_rad == kills_dir:
            if gains_rad > gains_dir:
                acc_rad += 1
            if gains_rad < gains_dir:
                acc_dir += 1          
    return acc_rad - acc_dir


def extract_features(path_in, path_out, file_in, file_out):
    with open(
        os.path.join(path_in, file_in)
    ) as fi, open(
        os.path.join(path_out, file_out), 'w', newline='\n'
    ) as fo:
        writer = csv.writer(fo)
        writer.writerow(FEATURE_LABELS)
        lines = fi.readlines()
        for line in lines:
            row = json.loads(line, encoding='utf-8')
            writer.writerow([
                *get_ids(row),
                *get_base_features(row),
                *get_life_states(row),
                *get_kill_streaks(row),
                *get_multi_kills(row),
                *get_actions(row),
                *get_healing(row),
                *get_damage(row),
                *get_damage_taken(row),
                *get_gold_reasons(row),
                *get_inventory(row),
                *get_barracks(row),
                *get_pings(row),
                *get_xp_reasons(row),
                *get_runes(row),
                *get_killed_towers(row),
                get_towers_kills_last(row),
                get_teamfights_all(row),
            ])


def get_t_items(row, time_slice_right):
    items = []
    for p in row['players']:
        items.append(';'.join([i['key'] for i in p['purchase_log'] if i['time'] <= time_slice_right]))
    return items


def get_t_first_tower(row, time_slice_right):
    if len(row['objectives']) == 0:
        return 0
    towers_rad = 0
    towers_dir = 0
    for event in row['objectives']:
        if event['time'] <= time_slice_right:
            if event['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if event['team'] == 2:
                    towers_rad += 1
                if event['team'] == 3:
                    towers_dir += 1
            if event['type'] == 'CHAT_MESSAGE_TOWER_DENY':
                if event['player_slot'] > 100:
                    towers_rad += 1
                if event['player_slot'] < 100:
                    towers_dir += 1
    return towers_rad - towers_dir


def get_t_features(row, time_slice_left, time_slice_right):
    game_time = int(sum(v for k, v in row['players'][0]['life_state'].items()) / 2)
    if time_slice_left < game_time <= time_slice_right:
        gd_rad = sum(p['gold'] for p in row['players'][:5])
        xp_rad = sum(p['xp'] for p in row['players'][:5])
        lh_rad = sum(p['lh'] for p in row['players'][:5])
        dn_rad = sum(p['denies'] for p in row['players'][:5])
        gd_dir = sum(p['gold'] for p in row['players'][5:10])
        xp_dir = sum(p['xp'] for p in row['players'][5:10])
        lh_dir = sum(p['lh'] for p in row['players'][5:10])
        dn_dir = sum(p['denies'] for p in row['players'][5:10]) 
    if game_time > time_slice_right:
        time_index = row['players'][0]['times'].index(time_slice_right)
        gd_rad = sum(p['gold_t'][time_index] for p in row['players'][:5])
        xp_rad = sum(p['xp_t'][time_index] for p in row['players'][:5])
        lh_rad = sum(p['lh_t'][time_index] for p in row['players'][:5])
        dn_rad = sum(p['dn_t'][time_index] for p in row['players'][:5])
        gd_dir = sum(p['gold_t'][time_index] for p in row['players'][5:10])
        xp_dir = sum(p['xp_t'][time_index] for p in row['players'][5:10])
        lh_dir = sum(p['lh_t'][time_index] for p in row['players'][5:10])
        dn_dir = sum(p['dn_t'][time_index] for p in row['players'][5:10])
    return [gd_rad - gd_dir, xp_rad - xp_dir, lh_rad - lh_dir, dn_rad - dn_dir]


def extract_t_features(path_in, path_out, file_in, file_out, time_slice_left, time_slice_right):
    with open(
        os.path.join(path_in, file_in)
    ) as fi, open(
        os.path.join(path_out, file_out), 'w', newline='\n'
    ) as fo:
        writer = csv.writer(fo)
        writer.writerow(T_FEATURES)
        for line in fi.readlines():
            row = json.loads(line, encoding='utf-8')
            match_id_hash = row['match_id_hash']
            if 'test' in file_in:
                target = -999
                is_train = 0
            if 'train' in file_in:
                target = row['targets']['radiant_win']
                is_train = 1
            first_tower = get_t_first_tower(row, time_slice_right)
            game_time_ext = int(sum(v for k, v in row['players'][0]['life_state'].items()) / 2)
            items = get_t_items(row, time_slice_right)
            if game_time_ext <= time_slice_left:
                continue
            game_time = game_time_ext if game_time_ext <= time_slice_right else time_slice_right     
            t_features = get_t_features(row, time_slice_left, time_slice_right)
            heroes = [p['hero_name'] for p in row['players']]
            writer.writerow([
                match_id_hash,
                target,
                game_time,
                first_tower,
                *t_features,
                *heroes,
                is_train,
                *items,
            ])