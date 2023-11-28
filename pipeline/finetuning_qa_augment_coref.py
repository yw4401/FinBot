import math
import re

import pandas as pd
import spacy
import torch
from allennlp.predictors.predictor import Predictor

import config
from coref_resolve import add_coreference_resolution


if __name__ == "__main__":
    src_url = f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_ARTICLES}"
    final_df = pd.read_parquet(src_url)
    print("Final Size: " + str(final_df.shape[0]))

    predictor = Predictor.from_path(config.ARTICLE_COREF_MOD_URL, cuda_device=torch.cuda.current_device())
    nlp = spacy.load(config.ARTICLE_COREF_SPACY_MOD)
    coref_df = add_coreference_resolution(final_df, predictor=predictor, nlp=nlp, src_col="context").copy()
    coref_df["context"] = coref_df.coref
    coref_df = coref_df.drop("coref", axis=1)
    coref_df.to_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_ARTICLES_COREF}", index=False)
