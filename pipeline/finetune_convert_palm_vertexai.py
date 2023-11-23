from finetuning import *
from build_vector_index_3 import create_splitter

if __name__ == "__main__":
    src_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_PATTERN}"
    src_url = src_url.format(split="train")
    df = pd.read_parquet(src_url)
    print(df.columns)
    train, test = train_test_split(df, test_size=1000)
    splitter = create_splitter()
    train_aug = inject_noise(train, splitter, context_col="context", result_col="answer",
                             impossible_resp=config.QA_IMPOSSIBLE_RESP,
                             chunk_processor=lambda row, chunks: list(chunks))
    test_aug = inject_noise(test, splitter, context_col="context", result_col="answer",
                            impossible_resp=config.QA_IMPOSSIBLE_RESP,
                            chunk_processor=lambda row, chunks: list(chunks))
    print(train_aug.head())
    print("Uploading article snapshots")
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_PATTERN}"
    train_aug.to_parquet(target_url.format(split="train"), index=False)
    test_aug.to_parquet(target_url.format(split="test"), index=False)
