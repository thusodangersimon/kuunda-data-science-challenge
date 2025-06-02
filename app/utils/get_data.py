import os
from pathlib import Path

import boto3
import pandas as pd

# Get data from envrion varibles set when container is spun up
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def get_all_data(bucket_name, outpath: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=""):
        for obj in page.get("Contents", []):
            s3.download_file(bucket_name, obj["Key"], Path(outpath).joinpath(obj["Key"]))


def load_training_data(filepath: str) -> pd.DataFrame:
    # concat into dataframe
    df = [pd.read_csv(p) for p in Path(filepath).iterdir() if p.is_file()]
    # append all files
    df = pd.concat(df)
    # make date datetimes
    df['date_add'] = pd.to_datetime(df['date_add'])
    return df
