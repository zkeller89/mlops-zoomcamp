#!/usr/bin/env python
# coding: utf-8

import sys
import os.path
import pickle
import pandas as pd

def get_yellow_trip_link(year, month):
    return(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

def get_yellow_trip_df(year, month):
    file = f'yellow_tripdata_{year:04d}-{month:02d}.parquet'
    fpath = os.path.join('trip-data', file)

    if os.path.isfile(fpath):
        df = pd.read_parquet(fpath)
    else:
        url = get_yellow_trip_link(year, month)
        df = pd.read_parquet(url)
        df.to_parquet(fpath)

    return df

class DataProcessor:

    categorical = ['PULocationID', 'DOLocationID']
    data_directory = 'trip-data'

    def __init__(self, df, year, month):
        self.df = df
        self.year = year
        self.month = month

    def read_data(self):
        return(self.df)

    def prepare_data(self):
        df = self.read_data()
        df['ride_id'] = f'{self.year:04d}/{self.month:02d}_' + df.index.astype('str')
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
        df[self.categorical] = df[self.categorical].fillna(-1).astype('int').astype('str')

        return(df)

def main(year, month):

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = get_yellow_trip_df(year, month)
    dp = DataProcessor(df, year, month)
    df = dp.prepare_data()

    categorical = dp.categorical
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    return 0

if __name__ == '__main__':
    sys.exit(main(2022, 2))
