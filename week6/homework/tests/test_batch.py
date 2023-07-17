import pandas as pd
from pandas import Timestamp
from datetime import datetime

from batch import prepare_data

def test_prepare_data():
    def dt(hour, minute, second=0):
        return datetime(2022, 1, 1, hour, minute, second)

    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df = prepare_data(df, 2022, 1)

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

    assert res_df.equals(df)

