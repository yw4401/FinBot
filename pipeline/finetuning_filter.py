from finetuning import *


with bq.Client(project=config.GCP_PROJECT) as client:
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILTERED}"
    df = get_full_data(client)
    df = df.drop_duplicates(subset=["body"])
    df = filter_summary(df, jobs=7)
    df.to_parquet(target_url, index=False)
    df = rewrite_summary(df, jobs=7)
    df.to_parquet(target_url, index=False)
    df = headline_summary(df, jobs=7)
    df.to_parquet(target_url, index=False)
    df = augment_summary(df, jobs=7)
    df.to_parquet(target_url, index=False)
