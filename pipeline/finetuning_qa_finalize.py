from build_vector_index_3 import create_splitter
from finetuning import *

if __name__ == "__main__":
    df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_COREF}").sample(10000,
                                                                                                      random_state=93)
    articles = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_ARTICLES_COREF}")
    print(df.columns)
    print(articles.columns)
    print(articles.answerable.head())
    print(articles.answer.head())
    articles = articles[["context", "answerable", "unanswerable", "answer", "published"]]
    article_train, article_test = get_data_sets_df(articles, test_instances=200, context_col="context",
                                                   resp_col="answer", answerable_col="answerable",
                                                   unanswerable_col="unanswerable",
                                                   impossible_resp=config.QA_IMPOSSIBLE_RESP)
    train, test = train_test_split(df, test_size=800)
    splitter = create_splitter()
    train_aug = inject_noise(train, splitter, context_col="context", result_col="answer",
                             impossible_resp=config.QA_IMPOSSIBLE_RESP,
                             chunk_processor=lambda row, chunks: list(chunks), pure_noise=0.1)
    test_aug = inject_noise(test, splitter, context_col="context", result_col="answer",
                            impossible_resp=config.QA_IMPOSSIBLE_RESP,
                            chunk_processor=lambda row, chunks: list(chunks), pure_noise=0.1)
    article_train_aug = inject_noise(article_train, splitter, context_col="context", result_col="answer",
                                     impossible_resp=config.QA_IMPOSSIBLE_RESP, pure_noise=0.1)
    article_test_aug = inject_noise(article_test, splitter, context_col="context", result_col="answer",
                                    impossible_resp=config.QA_IMPOSSIBLE_RESP, pure_noise=0.1)
    article_train_aug = article_train_aug[["context", "question", "answer"]]
    article_test_aug = article_test_aug[["context", "question", "answer"]]
    train_aug = train_aug[["context", "question", "answer"]]
    test_aug = test_aug[["context", "question", "answer"]]
    final_train = pd.concat([train_aug, article_train_aug])
    final_test = pd.concat([test_aug, article_test_aug])
    print(final_train.head())
    print("Uploading article snapshots")
    target_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_PATTERN}"
    final_train.to_parquet(target_url.format(split="train"), index=False)
    final_test.to_parquet(target_url.format(split="test"), index=False)
