from pathlib import Path
import polars as pl
import numpy as np

cwd = Path.cwd()

def get_rv():
    '''
    calculate RV for [5m, 30m, 1d, 1w] and save them in data/processed
    '''
    intervals = ['5m', '30m', '1d', '1w']

    df = pl.read_csv('data/raw/CSI300.csv', schema_overrides={'':pl.Datetime})
    df = df.rename({'':'timestamp'})

    # group by 5m
    df = df.group_by_dynamic('timestamp', every='5m', closed='right', label='right').agg([
    pl.col('open').first(),
    pl.col('high').max(),
    pl.col('low').min(),
    pl.col('close').last(),
    pl.col('volume').sum()
    ])

    # calculate RV for 5m
    df = df.with_columns((pl.col('close')/pl.col('close').shift(1)).log().abs().alias('rv').fill_null(0))

    # calculate RV for different intervals
    for interval in intervals:
        if interval == '1w':
            start = 'monday'
        else:
            start = 'window'

        processed_df = df.group_by_dynamic('timestamp', every=interval, closed='right', label='right', start_by=start).agg([
        pl.col('open').first(),
        pl.col('high').max(),
        pl.col('low').min(),
        pl.col('close').last(),
        pl.col('volume').sum(),
        (pl.col('rv').pow(2).sum() / pl.len()).sqrt()
        ])

        # save as csv
        processed_df.write_csv(f'data/processed/CSI300_{interval}.csv')