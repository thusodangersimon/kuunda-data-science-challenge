import os
import boto3

def download_from_s3(bucket_name: str, s3_key: str, local_path: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} from {bucket_name} to {local_path}")

if __name__ == "__main__":
    # Example usage
    bucket = "kuunda-datascience-challenge"
    key = "path/in/s3/yourfile.csv"
    dest = "yourfile.csv"
    download_from_s3(bucket, key, dest)