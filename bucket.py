# Imports the Google Cloud client library
from google.cloud import storage

def gcs_upload_image(filename):
    storage_client = storage.Client()
    bucket_name = "learning-helper-2025-451212.appspot.com"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
    print("Image uploaded")