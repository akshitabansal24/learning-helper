# Imports the Google Cloud client library
from google.cloud import storage
from io import BytesIO
from flask import send_file

def gcs_upload_image(filename):
    storage_client = storage.Client()
    bucket_name = "learning-helper-2025-451212.appspot.com"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
    blob.make_public()
    print("Image uploaded")


def download_from_gcs(filename):
    storage_client = storage.Client()
    bucket_name = "learning-helper-2025-451212.appspot.com"
    # file_path = f"/tmp/{filename}" 
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    print(blob)
    image_bytes = blob.download_as_bytes()

    return send_file(BytesIO(image_bytes), mimetype='image/jpeg')

def get_gcs_url(filename):
    """Generate a public URL for an image stored in GCS."""
    bucket_name = "learning-helper-2025-451212.appspot.com"
    return f"https://storage.googleapis.com/{bucket_name}/{filename}"