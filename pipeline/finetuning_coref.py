import pandas as pd
import spacy
import torch
from allennlp.predictors.predictor import Predictor

import config
from coref_resolve import add_coreference_resolution

if __name__ == "__main__":
    predictor = Predictor.from_path(config.ARTICLE_COREF_MOD_URL, cuda_device=torch.cuda.current_device())
    nlp = spacy.load(config.ARTICLE_COREF_SPACY_MOD)

    src_df = pd.read_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_FILTERED}")
    coref_df = add_coreference_resolution(src_df, predictor=predictor, nlp=nlp).copy()
    coref_df["body"] = coref_df.coref
    coref_df = coref_df.drop("coref", axis=1)
    coref_df.to_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_COREF}", index=False)

