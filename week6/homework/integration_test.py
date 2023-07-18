import os

from datetime import datetime
import pandas as pd

from batch import get_input_path, get_output_path

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


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

input_file = get_input_path(2021, 1)
output_file = get_output_path(2021, 1)

print(input_file)
print(options)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2021 1')
df_actual = pd.read_parquet(output_file, storage_options=options)
assert abs(df_actual['predicted_duration'].sum() - 69.28) < 0.1
