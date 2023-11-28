from finetuning import *


async def main():
    sample = 5000
    src_df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILTERED}")
    src_df = src_df.rename(columns={"body": "context"}).drop(["question", "reverse_question"], axis=1)
    src_df = src_df.sample(sample, random_state=93)
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_ARTICLES}"
    src_df = await augment_articles_qa(src_df)
    print("\n".join(src_df.answerable.head()))
    print("\n".join(src_df.answer.head()))
    print("\n".join(src_df.unanswerable.head()))
    src_df.to_parquet(target_url, index=False)


if __name__ == "__main__":
    asyncio.run(main())
