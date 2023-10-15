import pandas as pd
import numpy as np
import random
import config
import spacy
from tqdm import tqdm

spacy.require_gpu()
tqdm.pandas()


def strip_reuter_intro(text):
    text_chunks = text.split(" - ")
    return " - ".join(text_chunks[1:]).strip()


def create_extractor(sentence_splitter, ner_recog):

    def extractor(text):
        sentences = [s.text for s in sentence_splitter(text).sents]
        ner_results = ner_recog.pipe(sentences)
        entities = []
        for doc in ner_results:
            for ent in doc.ents:
                entities.append((ent.label_, ent.text))
        return entities

    return extractor


if __name__ == "__main__":
    year = 2023
    month = 4

    src_url = "gs://{bucket}/{file}".format(bucket=config.ARTICLE_COREF_TARGET_BUCKET,
                                            file=config.ARTICLE_COREF_FILE_PATTERN)
    src_url = src_url.format(year=year, month=month)
    coref_data = pd.read_parquet(src_url)

    sentence_splitter = spacy.load(config.NER_SPACY_MOD, enable=["senter"], config={"nlp": {"disabled": []}})
    ner_recog = spacy.load(config.NER_SPACY_MOD, enable=["ner"])
    extractor = create_extractor(sentence_splitter, ner_recog)
    coref_data["body"] = coref_data.apply(
        lambda row: strip_reuter_intro(row["body"] if row["source"] == "reuters" else row["body"]), axis=1)
    coref_data["entities"] = coref_data.body.progress_apply(extractor)

    target_url = "gs://{bucket}/{file}".format(bucket=config.NER_TARGET_BUCKET, file=config.NER_TARGET_PATTERN)
    target_url = target_url.format(year=year, month=month)
    coref_data.to_parquet(target_url, index=False)
