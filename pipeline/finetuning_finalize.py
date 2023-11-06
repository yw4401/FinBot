from finetuning import *
from build_vector_index_3 import create_splitter


def print_summaries(df):
    print("\n\n")
    for s in df.summary.sample(5):
        print(s)


if __name__ == "__main__":
    df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_COREF}")
    print(df.columns)
    df = inject_noise(df, create_splitter())
    print_summaries(df)
    df = fix_summary_tagline(df)
    print_summaries(df)
    print("Uploading article snapshots")
    train, test = get_data_sets_df(df, test_instances=1000)
    print_summaries(train)
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILE_PATTERN}"
    train.to_parquet(target_url.format(split="train"), index=False)
    test.to_parquet(target_url.format(split="test"), index=False)
