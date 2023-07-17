#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-zack/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(year, month):
    fpath = get_input_path(year, month)
    df = pd.read_parquet(fpath)

    return(df)

def prepare_data(df, year, month):
    categorical = ['PULocationID', 'DOLocationID']

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return(df)

def save_data(df, year, month):
    fpath = get_output_path(year, month)
    options = {
        'client_kwargs': {
            'endpoint_url': 'S3_ENDPOINT_URL'
        }
    }
    df.to_parquet(fpath, storage_options=options)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    df = read_data(year, month)
    df = prepare_data(df, year, month)

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    options = {
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566'
        }
    }
    output_file = get_output_path(year, month)
    # df_result.to_parquet(output_file, engine='pyarrow', index=False)
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        index=False,
        storage_options=options
    )

    return 0

if __name__ == '__main__':
    sys.exit(main(2022, 2))
