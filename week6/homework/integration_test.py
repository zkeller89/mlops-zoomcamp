import os

from datetime import datetime
import pandas as pd
from pandas import Timestamp

from batch import (
    get_input_path, get_predictions, get_output_path,
    read_data, prepare_data
)

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = get_input_path(2021, 1, False)
output_file = get_output_path(2021, 1, False)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2022 1')

result = {
    'PULocationID': {0: '-1', 1: '1', 2: '1'},
    'DOLocationID': {0: '-1', 1: '-1', 2: '2'},
    'tpep_pickup_datetime': {
        0: Timestamp('2022-01-01 01:02:00'),
        1: Timestamp('2022-01-01 01:02:00'),
        2: Timestamp('2022-01-01 02:02:00')
    },
    'tpep_dropoff_datetime': {
        0: Timestamp('2022-01-01 01:10:00'),
        1: Timestamp('2022-01-01 01:10:00'),
        2: Timestamp('2022-01-01 02:03:00')
    },
    'ride_id': {0: '2022/01_0', 1: '2022/01_1', 2: '2022/01_2'},
    'duration': {0: 8.0, 1: 8.0, 2: 1.0}
}
res_df = pd.DataFrame(result)

df_post = prepare_data(df_input, 2022, 1)
y_pred = get_predictions(df_post)
print('predicted mean duration:', y_pred.mean())

assert res_df.equals(df_post)