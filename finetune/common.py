import torch
from transformers import (
    LlamaTokenizer, StoppingCriteria, )

import config
from evaluate import load
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def restrict_article(text, max_token, tokenizer: LlamaTokenizer):
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    if len(tokenized) > max_token:
        return tokenizer.decode(tokenized[:max_token], skip_special_tokens=False)
    return text


def truncate_summary_example_chat(system, question, body, summary, tokenizer, max_context,
                                  buffer=20, template=None):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": format_llama_sum_user({"question": question, "body": ""})},
        {"role": "assistant", "content": format_llama_sum_resp({"summary": summary})}
    ]
    body_tokens = max_context - len(tokenizer.apply_chat_template(messages, chat_template=template)) - buffer
    return restrict_article(body, body_tokens, tokenizer)


def format_llama_sum_user(example):
    return config.LLAMA_USER_SUMMARY_TEMPLATE.format(context=example["body"], question=example["question"])


def format_llama_qa_user(example):
    return config.LLAMA_USER_QA_TEMPLATE.format(context=example["body"], question=example["question"])


def format_llama_sum_resp(example):
    return config.LLAMA_AI_SUMMARY_TEMPLATE.format(summary=example["summary"])


def format_llama_qa_resp(example):
    return config.LLAMA_AI_QA_TEMPLATE.format(response=example["response"])


def get_batch_row(examples, i):
    return {k: examples[k][i] for k in examples}


def format_llama_example(example, system, user_func, resp_func, tokenizer, template=None):
    output_texts = []
    for i in range(len(example['body'])):
        s = resp_func(get_batch_row(example, i))
        user = user_func(get_batch_row(example, i))

        text = tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": s}
        ], tokenize=False, chat_template=template)
        if text[:len(tokenizer.bos_token)] == tokenizer.bos_token:
            text = text[len(tokenizer.bos_token):]
        if text[-len(tokenizer.eos_token):] == tokenizer.eos_token:
            text = text[:-len(tokenizer.eos_token)]
        output_texts.append(text)

    return output_texts


def format_summary_example(example, tokenizer, template=None):
    return format_llama_example(example, config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                format_llama_sum_user, format_llama_sum_resp, tokenizer, template)


def format_qa_example(example, tokenizer, template=None):
    return format_llama_example(example, config.LLAMA_QA_SYSTEM_INSTRUCTION,
                                format_llama_qa_user, format_llama_qa_resp, tokenizer, template)


def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=()):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def create_summarization_metrics(tokenizer):
    metric = load("rouge")
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    stop_words = set([w.lower() for w in stopwords.words("english")])

    def clean_sentence(sentence):
        nltk_words = nltk.word_tokenize(sentence)
        lem_words = [lemmatizer.lemmatize(w) for w in nltk_words if w not in stop_words]
        return " ".join(lem_words)

    def compute_rouge_metrics(labels, predicted):
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(clean_sentence(nltk.sent_tokenize(pred))) for pred in labels]
        decoded_labels = ["\n".join(clean_sentence(nltk.sent_tokenize(label))) for label in predicted]

        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True,
                                use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

    def compute_classification_scores(labels, predicted):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, pos_label=True, average="binary")
        result = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return {k: round(v, 4) for k, v in result.items()}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_labels = []
        rouge_preds = []
        classify_labels = []
        classify_pred = []

        for l, p in zip(decoded_labels, decoded_preds):
            label_impossible = config.IMPOSSIBLE_INSTRUCTION.lower() in l.lower().strip()
            predict_impossible = config.IMPOSSIBLE_INSTRUCTION.lower() in p.lower().strip()

            if not label_impossible and not predict_impossible:
                rouge_labels.append(l.lower().strip())
                rouge_preds.append(l.lower().strip())
            classify_labels.append(label_impossible)
            classify_pred.append(predict_impossible)

        result = {}
        result.update(compute_rouge_metrics(rouge_labels, rouge_preds))
        result.update(compute_classification_scores(classify_labels, classify_pred))
        return result

    return compute_metrics, compute_rouge_metrics, compute_classification_scores
