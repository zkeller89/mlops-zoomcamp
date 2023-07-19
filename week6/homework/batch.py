#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd

def get_input_path(year, month, default=True):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    if default:
        input_pattern = default_input_pattern
    else:
        input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)

    return input_pattern.format(year=year, month=month)

def get_output_path(year, month, default=True):
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    if default:
        output_pattern = default_output_pattern
    else:
        output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)

    return output_pattern.format(year=year, month=month)

def read_data(fpath, options=None):
    print(fpath)
    df = pd.read_parquet(fpath, storage_options=options)

    return(df)

def prepare_data(df, year, month):
    categorical = ['PULocationID', 'DOLocationID']

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return(df)

def get_predictions(df):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    return(y_pred)

def save_data(df, fpath, default=True):
    endpoint = os.getenv("S3_ENDPOINT_URL")

    options=None
    if endpoint and not default:
        options = {
            'client_kwargs': {
                'endpoint_url': endpoint
            }
        }

    df.to_parquet(
        fpath,
        engine='pyarrow',
        index=False,
        storage_options=options
    )

def main(year, month):
    input_path = get_input_path(year, month, default=True)
    df = read_data(input_path)
    df = prepare_data(df, year, month)

    y_pred = get_predictions(df)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_path = get_output_path(year, month, default=False)
    save_data(df_result, output_path, False)

    return 0

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    sys.exit(main(year, month))
