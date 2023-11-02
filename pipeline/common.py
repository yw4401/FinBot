from google.cloud import storage
from google.cloud import bigquery
import config


class StarFunc:

    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        result = self.func(*args)
        return result


def upload_blob(bucket_name, source_file_name, destination_blob_name, generation=0):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(project=config.GCP_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = generation

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)


def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client(project=config.GCP_PROJECT)
    blob = storage.Blob.from_string(source_blob_name, storage_client)
    blob.download_to_filename(destination_file_name)


class BigquerySession:
    """ContextManager wrapping a bigquerySession."""

    def __init__(self, bqclient: bigquery.Client, location="us-central1") -> None:
        """Construct instance."""
        self.client = bqclient
        self.session_id = None
        self.location = location

    def __enter__(self):
        """Initiate a Bigquery session and return the session_id."""
        job = self.client.query(
            "SELECT 1;",  # a query can't fail
            job_config=bigquery.QueryJobConfig(create_session=True),
            location=self.location
        )
        self.session_id = job.session_info.session_id
        job.result()  # wait job completion
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Abort the opened session."""
        if self.session_id:
            # abort the session in any case to have a clean state at the end
            # (sometimes in case of script failure, the table is locked in
            # the session)
            job = self.client.query(
                "CALL BQ.ABORT_SESSION();",
                job_config=bigquery.QueryJobConfig(
                    create_session=False,
                    connection_properties=[
                        bigquery.query.ConnectionProperty(
                            key="session_id", value=self.session_id
                        )
                    ],
                ),
                location=self.location
            )
            job.result()

    def begin_transaction(self):
        job = self.client.query(
            "BEGIN TRANSACTION;",
            job_config=bigquery.QueryJobConfig(
                create_session=False,
                connection_properties=[
                    bigquery.query.ConnectionProperty(
                        key="session_id", value=self.session_id
                    )
                ],
            ),
            location=self.location
        )
        job.result()

    def commit(self):
        job = self.client.query(
            "COMMIT TRANSACTION;",
            job_config=bigquery.QueryJobConfig(
                create_session=False,
                connection_properties=[
                    bigquery.query.ConnectionProperty(
                        key="session_id", value=self.session_id
                    )
                ],
            ),
            location=self.location
        )
        print(list(job.result()))


