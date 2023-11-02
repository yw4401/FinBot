from finetuning import *


with bq.Client(project=config.GCP_PROJECT) as client:
    df = get_full_data(client)
    df = df.drop_duplicates(subset=["body"]).sample(30)
    df = filter_summary(df, jobs=7)
    df = rewrite_summary(df, jobs=7)
    df = headline_summary(df, jobs=7)
    df = augment_summary(df, jobs=7)

    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILTERED}"
    df.to_parquet(target_url, index=False)
