import sentence_transformers as st
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

TRAIN_QUESTION_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-trainq.parquet"
TRAIN_DOC_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-traindoc.parquet"
TRAIN_REL_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-trainrel.parquet"
EVAL_QUESTION_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-devq.parquet"
EVAL_DOC_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-devdoc.parquet"
EVAL_REL_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-devrel.parquet"
TEST_QUESTION_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-testq.parquet"
TEST_DOC_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-testdoc.parquet"
TEST_REL_PATH = "gs://scraped-news-article-data-null/fiqa-augmented-testrel.parquet"

BASE_MODEL = "llmrails/ember-v1"

train_q = pd.read_parquet(TRAIN_QUESTION_PATH)
train_doc = pd.read_parquet(TRAIN_DOC_PATH)
train_rel = pd.read_parquet(TRAIN_REL_PATH)

test_q = pd.read_parquet(TEST_QUESTION_PATH)
test_doc = pd.read_parquet(TEST_DOC_PATH)
test_rel = pd.read_parquet(TEST_REL_PATH)


def to_input_examples(questions, documents, relations):
    positives = pd.merge(left=questions, right=relations, on="query_id")
    positives = pd.merge(left=positives, right=documents, on="doc_id")
    for _, row in positives.iterrows():
        yield st.InputExample(texts=[row["query_text"], row["doc_text"]], label=1)


def to_retrieval_evaluator(questions, documents, relations, **kwargs):
    q_dict = {}
    doc_dict = {}
    rel_dict = {}
    for _, row in questions.iterrows():
        q_dict[str(row["query_id"])] = row["query_text"]
    for _, row in documents.iterrows():
        doc_dict[str(row["doc_id"])] = row["doc_text"]
    relations = relations.copy()
    relations["doc_id_list"] = relations.doc_id.apply(lambda x: [str(x)])
    relations_grouped = relations[["query_id", "doc_id_list"]].groupby("query_id").sum().reset_index()
    for _, row in relations_grouped.iterrows():
        rel_dict[str(row["query_id"])] = set(row["doc_id_list"])
    return InformationRetrievalEvaluator(queries=q_dict, corpus=doc_dict, relevant_docs=rel_dict, **kwargs)


train_example = [e for e in to_input_examples(train_q, train_doc, train_rel)]
train_dataloader = DataLoader(train_example, shuffle=True, batch_size=32)
eval_set_evaluator = to_retrieval_evaluator(test_q, test_doc, test_rel,
                                            show_progress_bar=True, ndcg_at_k=[2, 3], mrr_at_k=[2, 3],
                                            accuracy_at_k=[2, 3], precision_recall_at_k=[2, 3],
                                            map_at_k=[2, 3], main_score_function="cos_sim")
base_model = st.SentenceTransformer(model_name_or_path=BASE_MODEL)
base_model.fit(train_objectives=[(train_dataloader, MultipleNegativesRankingLoss(base_model))],
               epochs=10, warmup_steps=100, output_path="./embedding",
               evaluator=eval_set_evaluator)
