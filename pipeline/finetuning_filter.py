from finetuning import *


async def main():
    with bq.Client(project=config.GCP_PROJECT) as client:
        target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILTERED}"
        df = get_full_data(client)
        df = df.drop_duplicates(subset=["body"])
        df = await filter_summary(df)
        df.to_parquet(target_url, index=False)
        df = await rewrite_summary(df)
        df.to_parquet(target_url, index=False)
        df = await headline_summary(df)
        df.to_parquet(target_url, index=False)
        df = await augment_summary(df)
        df.to_parquet(target_url, index=False)
        return df


if __name__ == "__main__":
    asyncio.run(main())
