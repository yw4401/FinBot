from finetuning import *
from pipeline.build_vector_index import create_splitter

if __name__ == "__main__":
    df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_COREF}")
    df = inject_noise(df, create_splitter())
    print("Uploading article snapshots")
    train, test = get_data_sets_df(df, test_instances=10)
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILE_PATTERN}"
    train.to_parquet(target_url.format(split="train"), index=False)
    test.to_parquet(target_url.format(split="test"), index=False)
