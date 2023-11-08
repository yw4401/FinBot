from finetuning import *
from build_vector_index_3 import create_splitter


def print_summaries(df):
    print("\n\n")
    for s in df.summary.sample(5):
        print(s)


if __name__ == "__main__":
    df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_COREF}")
    print(df.columns)
    train, test = get_data_sets_df(df, test_instances=1000)
    splitter = create_splitter()
    train_aug = inject_noise(train, splitter)
    test_aug = inject_noise(test, splitter)
    print_summaries(train_aug)
    print("Uploading article snapshots")
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILE_PATTERN}"
    train_aug.to_parquet(target_url.format(split="train"), index=False)
    test_aug.to_parquet(target_url.format(split="test"), index=False)
