from topic_sum import create_topic_summarizer
import config

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

if __name__ == "__main__":
    year = 2023
    month = 4

    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                            file=config.TOPIC_SUBSAMPLE_FILE)
    topic_df = pd.read_parquet(src_url.format(year=year, month=month))
    topics = list(set(topic_df.topic.unique()) - {-1})

    summarizer = create_topic_summarizer(config.TOPIC_SUM_KIND, **config.TOPIC_SUM_MODEL_PARAMS)
    target_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUM_TARGET,
                                               file=config.TOPIC_SUM_TARGET_FILE)
    try:
        topic_sum = pd.read_parquet(target_url.format(year=year, month=month))
    except:
        topic_sum = pd.DataFrame({
            "topics": topics,
            "summary": ["" for _ in topics]
        })

    works = set(topic_sum.loc[topic_sum.summary.str.len() == 0]["topics"].to_list())
    with tqdm(total=len(works)) as progress:
        for i, w in enumerate(works):
            topic = topic_sum.loc[topic_sum.topics == w].iloc[0]["topics"]
            summary = summarizer(topic_df, topic)
            topic_sum.loc[topic_sum.topics == w, "summary"] = summary
            progress.update(1)
            if i % 10 == 0:
                topic_sum.to_parquet(target_url.format(year=year, month=month),
                                     index=False)
    topic_sum.to_parquet(target_url.format(year=year, month=month), index=False)
