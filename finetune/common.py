import dataclasses
import inspect
import os
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from evaluate import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers import (
    LlamaTokenizer, StoppingCriteria, )
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl.import_utils import is_peft_available
from trl.trainer import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM
)

import config


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
        if text[-len(tokenizer.eos_token):] != tokenizer.eos_token:
            text = text + tokenizer.eos_token
        output_texts.append(text)

    return output_texts


def format_llama_eval_example(example, system, user_func, resp_func, tokenizer, template=None):
    output_texts = []
    resp_texts = []
    for i in range(len(example['body'])):
        s = resp_func(get_batch_row(example, i))
        user = user_func(get_batch_row(example, i))

        text = tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ""}
        ], tokenize=False, chat_template=template)
        if text[:len(tokenizer.bos_token)] == tokenizer.bos_token:
            text = text[len(tokenizer.bos_token):]
        if text[-len(tokenizer.eos_token):] == tokenizer.eos_token:
            text = text[:-len(tokenizer.eos_token)]
        output_texts.append(text)
        resp_texts.append(s)

    return output_texts, resp_texts


def format_summary_example(example, tokenizer, template=None):
    return format_llama_example(example, config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                format_llama_sum_user, format_llama_sum_resp, tokenizer, template)


def format_summary_eval(example, tokenizer, template=None):
    return format_llama_eval_example(example, config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
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


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


class Seq2SeqSFTTrainer(Seq2SeqTrainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
            The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        input_format_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
        eval_format_func (`Optional[Callable]`):
            The formatting function to be used for creating the eval dataset. It needs to return a pair of list of strings, with the first list being the input, and the second list being the output.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automaticallty creating the Dataset. Defaults to `512`.
        infinite (`Optional[bool]`):
            Whether to use an infinite dataset or not. Defaults to `False`.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`. You can check how this is computed in the
            stack-llama example: https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset.
        dataset_num_proc (`Optional[int]`):
            The number of workers to use to tokenize the data. Only used when `packing=False`. Defaults to None.
        dataset_batch_size (`int`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Defaults to 1000.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This has been proven to drastically improve model performances for instrcution
            fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module, str] = None,
            args: Seq2SeqTrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional["PeftConfig"] = None,
            input_format_func: Callable = None,
            eval_format_func: Callable = None,
            packing: Optional[bool] = False,
            max_seq_length: Optional[int] = None,
            infinite: Optional[bool] = False,
            num_of_sequences: Optional[int] = 1024,
            chars_per_token: Optional[float] = 3.6,
            dataset_num_proc: Optional[int] = None,
            dataset_batch_size: int = 1000,
            neftune_noise_alpha: Optional[float] = None,
            model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SFTTrainer. But your model is already instantiated.")

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    _support_gc_kwargs = hasattr(
                        args, "gradient_checkpointing_kwargs"
                    ) and "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if _support_gc_kwargs:
                        preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

                    args = dataclasses.replace(args, gradient_checkpointing=False)

                model = get_peft_model(model, peft_config)

            if callbacks is None:
                callbacks = [PeftSavingCallback]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `TrainingArguments`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if not packing:
            if input_format_func is None:
                raise ValueError(
                    "You passed `packing=False` to the Seq2SeqSFTTrainer, but you didn't pass a `input_format_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                tokenizer,
                packing,
                max_seq_length,
                input_format_func,
                infinite,
                num_of_sequences,
                chars_per_token,
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_eval_dataset(
                eval_dataset,
                tokenizer,
                max_seq_length,
                args.generation_max_length,
                eval_format_func
            )

        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.args.max_steps > 0 and packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and packing:
            self.train_dataset.infinite = False

    @wraps(Seq2SeqTrainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    def _prepare_dataset(
            self,
            dataset,
            tokenizer,
            packing,
            max_seq_length,
            formatting_func,
            infinite,
            num_of_sequences,
            chars_per_token,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # check if torch dataset / dataloader and do nothing
        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer, dataset, max_seq_length, formatting_func
            )

        if formatting_func is not None:
            if tokenizer is None:
                raise ValueError(
                    "You need to pass a tokenizer when using the SFT Trainer when passing a `input_format_func`."
                )

            return ConstantLengthDataset(
                tokenizer,
                dataset,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=infinite,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
            )

        raise ValueError(
            "You need to pass a `input_format_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
        )

    def _prepare_eval_dataset(self, dataset, tokenizer, max_seq_length, max_gen_seq_length, formatting_func):
        if dataset is None:
            raise ValueError("The dataset should not be None")
        self._dataset_sanity_checked = False

        # check if torch dataset / dataloader and do nothing
        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
            return dataset

        if formatting_func is not None:
            if tokenizer is None:
                raise ValueError(
                    "You need to pass a tokenizer when using the SFT Trainer when passing a `input_format_func`."
                )

            def tokenize(examples):
                inputs, labels = formatting_func(examples)

                if not self._dataset_sanity_checked:
                    if not isinstance(inputs, list) or not isinstance(labels, list):
                        raise ValueError(
                            "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                        )
                    else:
                        self._dataset_sanity_checked = True

                input_tokenized = tokenizer(
                    inputs,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )
                labels_tokenized = tokenizer(
                    labels,
                    truncation=True,
                    padding=False,
                    max_length=max_gen_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )
                return {"input_ids": input_tokenized["input_ids"],
                        "attention_mask": input_tokenized["attention_mask"], "labels": labels_tokenized["input_ids"]}

            valid_data = dataset.map(lambda x: tokenize(x))
            return valid_data

        raise ValueError(
            "You need to pass a `input_format_func` argument to the SFTTrainer if you do not process the dataset."
        )

    def _prepare_non_packed_dataloader(
            self, tokenizer, dataset, max_seq_len, formatting_func
    ):
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                formatting_func(element),
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model


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
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False,
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

        # Replace -100 in the labels as we can't decode them.
        predictions = np.where(predictions != -100, predictions, tokenizer.bos_token_id)
        labels = np.where(labels != -100, labels, tokenizer.bos_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_labels = []
        rouge_preds = []
        classify_labels = []
        classify_pred = []

        for l, p in zip(decoded_labels, decoded_preds):
            print(f"Label Length {len(l)}, Predicted Length {len(p)}")
            print(f"Label: {l} vs Pred: {p}")
            label_impossible = config.IMPOSSIBLE_INSTRUCTION.lower() in l.lower().strip()
            predict_impossible = config.IMPOSSIBLE_INSTRUCTION.lower() in p.lower().strip()

            if not label_impossible and not predict_impossible:
                rouge_labels.append(l.lower().strip())
                rouge_preds.append(l.lower().strip())
            classify_labels.append(label_impossible)
            classify_pred.append(predict_impossible)

        result = {}
        print(f"Rouge Pred Length: {len(rouge_preds)}, Rouge Label Length: {len(rouge_labels)}")
        result.update(compute_rouge_metrics(rouge_labels, rouge_preds))
        result.update(compute_classification_scores(classify_labels, classify_pred))
        print(result)
        return result

    return compute_metrics, compute_rouge_metrics, compute_classification_scores
