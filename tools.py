import pandas as pd
import os
import datetime
from sklearn.metrics import mean_absolute_percentage_error

def log_into_csv(results_df, log_file_name='demand_runs.csv', name=None, pred_len=1):
    log_file = f'results/{log_file_name}'

    # create sample file with fencepost entry if it doesn't exist
    if not os.path.exists(log_file):
        df = pd.DataFrame({
            'timestamp': datetime.datetime.now(),
            'name': 'sample',
            'model': 'ChronosX',
            'feature_type': 'MS',
            'seq_len': 512,
            'pred_len': 12,
            'lr': 0.01,
            'bsz': 16,
            'score_type': 'mape',
            'score': 1.23,
        }, index=[0])
        df.to_csv(log_file)

    curr_run = pd.DataFrame({
        'timestamp': datetime.datetime.now(),
        'name': name,
        'model': 'ChronosX',
        'feature_type': 'MS',
        'seq_len': 512,
        'pred_len': pred_len,
        'lr': 0.001,
        'bsz': 16,
        'score_type': 'mape',
        'score': mean_absolute_percentage_error(results_df['true'], results_df['pred'])
    }, index=[0])

    df = pd.read_csv(log_file, index_col=0)
    assert len(df.columns) == len(curr_run.columns)

    df = pd.concat([df, curr_run]).reset_index(drop=True)
    df.to_csv(log_file)
