import json
from typing import Dict, Any, List, Iterable

from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import AIMessage, BaseOutputParser, PromptValue
from pydantic import BaseModel, Field

try:
    import config
except ModuleNotFoundError:
    import finetune.config as config


class RatableText(BaseModel):
    context_text: str
    output: str


class RatingOutput(BaseModel):
    rating: float = Field(description="A rating between 1 and 5", ge=1, le=5)
    thought: str = Field(description="Thought process for the rating")


class ChromaRatingExampleSelector(BaseExampleSelector):

    def __init__(self, collection, k=1):
        BaseExampleSelector.__init__(self)
        self.collection = collection
        self.num_sample = k

    def add_example(self, example: Dict[str, str]) -> Any:
        raise NotImplementedError("Read only")

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        return self._convert_query_result(
            self.collection.query(query_texts=[input_variables["text"]],
                                  n_results=self.num_sample))

    def _convert_query_result(self, result) -> List[dict]:
        results = []
        for text, metadata in zip(result["documents"][0], result["metadatas"][0]):
            results.append({
                "context_text": text,
                "output": metadata["output"],
                "rating": metadata["rating"],
                "thought": metadata["thought"]
            })
        return results


class PromptAdoptingParser(BaseOutputParser):
    base_parser: BaseOutputParser
    prompt: PromptValue

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue):
        return self.base_parser.parse_with_prompt(completion, prompt_value)

    def parse(self, completion: str):
        return self.parse_with_prompt(completion, self.prompt)

    def get_format_instructions(self):
        return self.base_parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return self.base_parser.type


def create_label_examples(examples: Iterable[Dict[str, str]], user_prompt, reinforcement=config.LABEL_REINFORCEMENT):
    result = []
    human_template = HumanMessagePromptTemplate.from_template(reinforcement + user_prompt)
    ai_format = "Thought process:\n{thought}\n\nFinal Rating:\n{rating}"
    for e in examples:
        result.append(human_template.format(context=e["context_text"]))
        result.append(AIMessage(content=ai_format.format(thought=e["thought"], rating=e["rating"])))
    return result


def rate_data_raw(model: BaseChatModel, text: RatableText, system: str, user: str,
                  example_selector: BaseExampleSelector = None):
    examples = []
    if example_selector:
        examples = create_label_examples(example_selector.select_examples({"text": text.context_text}))
    system_template = SystemMessagePromptTemplate.from_template(system)
    human_template = HumanMessagePromptTemplate.from_template(user)
    prompt = ChatPromptTemplate.from_messages([system_template, *examples,
                                               human_template])

    prompt_val = prompt.format_prompt(context=text.context_text, output=text.output)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.LABEL_MAX_TOKEN:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.LABEL_MAX_TOKEN))

    raw_chain = LLMChain(llm=model, prompt=prompt, verbose=config.LABEL_VERBOSE)
    raw_text = raw_chain(inputs={"context": text.context_text, "output": text.output})["text"]
    return raw_text


def create_format_examples(examples: Iterable[Dict[str, str]]):
    result = []
    human_prompt = HumanMessagePromptTemplate.from_template(config.LABEL_FORMAT_USER)

    for e in examples:
        valid_obj = RatingOutput(rating=e["rating"], thought=e["thought"])
        result.append(human_prompt.format(raw=e["raw"]))
        result.append(AIMessage(content=json.dumps(valid_obj.model_dump())))

    return result


def format_appropriate_meal(model: BaseChatModel, raw_text: str, examples: Iterable[Dict[str, str]] = ()):
    format_prompt = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(config.LABEL_FORMAT_SYSTEM),
         *create_format_examples(examples),
         HumanMessagePromptTemplate.from_template(config.LABEL_FORMAT_USER)])
    output_parser = RetryWithErrorOutputParser.from_llm(
        parser=PydanticOutputParser(pydantic_object=RatingOutput),
        llm=model)
    try:
        format_prompt = format_prompt.partial(format_instructions=output_parser.get_format_instructions())
    except NotImplementedError:
        format_prompt = format_prompt.partial(format_instructions="")
    prompt_val = format_prompt.format_prompt(raw=raw_text)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.LABEL_MAX_TOKEN:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.LABEL_MAX_TOKEN))
    output_parser = PromptAdoptingParser(base_parser=output_parser, prompt=prompt_val)
    parse_chain = LLMChain(llm=model, prompt=format_prompt, output_parser=output_parser,
                           verbose=config.LABEL_VERBOSE)
    result = parse_chain(inputs={"raw": raw_text})["text"].dict()
    result["raw"] = raw_text
    return result


def evaluate_text(model: BaseChatModel, texts: RatableText, system: str, user: str,
                  format_selector: BaseExampleSelector = None,
                  meal_selector: BaseExampleSelector = None):
    format_examples = []
    if format_selector:
        format_examples = format_selector.select_examples(input_variables={})

    raw_text = rate_data_raw(model, texts, system, user, meal_selector)
    formatted = format_appropriate_meal(model, raw_text, format_examples)
    return formatted
