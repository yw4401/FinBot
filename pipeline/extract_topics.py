import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import config

if __name__ == "__main__":
    embedding_model = SentenceTransformer(config.TOPIC_EMBEDDING)

    umap_model = UMAP(n_neighbors=config.TOPIC_UMAP_NEIGHBORS, n_components=config.TOPIC_UMAP_COMPONENTS,
                      min_dist=config.TOPIC_UMAP_MIN_DIST, metric=config.TOPIC_UMAP_METRIC)

    hdbscan_model = HDBSCAN(min_cluster_size=config.TOPIC_HDBSCAN_MIN_SIZE,
                            metric=config.TOPIC_HDBSCAN_METRIC,
                            cluster_selection_method='eom',
                            prediction_data=True)

    vectorizer_model = CountVectorizer(stop_words="english")

    ctfidf_model = ClassTfidfTransformer()

    # All steps together
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )

    year = 2023
    month = 4

    src_url = "gs://{bucket}/{file}".format(bucket=config.ARTICLE_CONVERT_SUBSAMPLE_TARGET,
                                            file=config.ARTICLE_CONVERT_SUBSAMPLE_FILE)
    src_df = pd.read_parquet(src_url.format(year=year, month=month))
    topics, prob = topic_model.fit_transform(src_df.body)
    src_df["topic"] = topics
    src_df["probability"] = prob
    target_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                               file=config.TOPIC_SUBSAMPLE_FILE)
    src_df.to_parquet(target_url.format(year=year, month=month), index=False)
