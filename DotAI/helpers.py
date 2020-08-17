import pandas as pd
import os


def save_submit(path_out, file_out, indices, y_pred):
    y_df = pd.DataFrame()
    y_df['match_id_hash'] = indices
    y_df['radiant_win_prob'] = y_pred
    y_df = y_df.set_index('match_id_hash')
    y_df.to_csv(os.path.join(path_out, file_out))