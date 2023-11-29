# Importing necessary libraries and modules
import gc # Garbage collection for memory management
import re # Regular expression operations
from datetime import datetime # Date and time handling
from typing import List

import pandas as pd
import spacy
import torch
from allennlp.predictors.predictor import Predictor
from spacy.tokens import Doc
from spacy.tokens import Span
from tqdm import tqdm

import config


def get_span_noun_indices(doc: Doc, cluster: List[List[int]]):
    spans = [doc[span[0]:span[1] + 1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
                         if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices


def get_cluster_head_idx(doc, cluster):
    noun_indices = get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0


def print_clusters(doc, clusters):
    def get_span_words(span, allen_document):
        return ' '.join(allen_document[span[0]:span[1] + 1])

    allen_document, clusters = [t.text for t in doc], clusters
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            print(get_span_words(cluster_head, allen_document) + ' - ', end='')
            print('[', end='')
            for i, span in enumerate(cluster):
                print(get_span_words(span, allen_document) + ("; " if i + 1 < len(cluster) else ""), end='')
            print(']')


def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def original_replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]

        for coref in cluster[1:]:
            core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end + 1]
    return head_span, [head_start, head_end]


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def window_sentences(sentences, idx, pre=5, sep="\n\n"):
    start_idx = max(0, idx - pre)
    context = " ".join(sentences[start_idx:idx])
    context = re.sub(r"\s+", " ", context)
    result = context + " " + sep + " " + sentences[idx]
    return result


def coref_text_whole(article, predictor, nlp):
    article = article.strip()
    if len(article) == 0:
        return ""
    clusters = predictor.predict(article)['clusters']
    doc = nlp(article)
    coref_article = improved_replace_corefs(doc, clusters)
    return coref_article


def coref_text_parts(sentences, predictor, nlp):
    sentences = list(sentences)

    for i in range(len(sentences)):
        shard = window_sentences(sentences, i)
        clusters = predictor.predict(shard)['clusters']
        doc = nlp(shard)
        coref_shard = improved_replace_corefs(doc, clusters)
        replacement_parts = coref_shard.split("\n\n")
        if len(replacement_parts) > 2:
            raise ValueError("Incorrect number of parts: " + str(len(replacement_parts)))
        replacement = replacement_parts[1].strip()
        sentences[i] = replacement

    return sentences


def coref_text(article, predictor, nlp):
    try:
        return coref_text_whole(article, predictor, nlp)
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()
        return None


def add_coreference_resolution(df, predictor, nlp, src_col="body"):
    body = []
    with tqdm(total=len(df.index)) as progress:
        for article in df[src_col]:
            corref_body = ""
            if article and len(article.strip()) > 0:
                corref_body = coref_text(article, predictor, nlp)
            if corref_body:
                body.append(corref_body)
            else:
                body.append("ERROR")
            progress.update(1)

    df["coref"] = body
    df = df.loc[df.coref != "ERROR"]
    return df
