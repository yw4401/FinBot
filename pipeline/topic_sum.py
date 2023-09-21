import logging
import re
from datetime import datetime
from functools import cache

BASE_PROMPT = """A list of news article titles with the published time is given below. Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. Be specific about the dates and entities involved and try to not omit important details. Do not use vague terms such as "past few month" or "various companies".\n\n"""


def select_titles(topic_df, topic, limiter, base_prompt=BASE_PROMPT):
    """
    Create the base prompt that will be used to ask the LLM on the theme summarization of the topics from the
    title of the articles in the topics together with the published dates. The goal is to
    fit as many titles into the prompt as possible without exceeding the context window. The maximum
    fit is determined by the limiter parameter passed in, which is a function that takes the prompt and
    determine if it is too big.
    """

    topic_segment = topic_df.loc[topic_df.topic == topic][["title", "published", "probability"]].sort_values(
        by="probability", ascending=False)
    if len(topic_segment.index) == 0:
        raise KeyError("Invalid Topic: " + str(topic))

    current_idx = 0
    end_idx = len(topic_segment.index)
    prompt = base_prompt
    row_format = "Published at: %s Title: %s\n"
    while limiter(prompt) and current_idx < end_idx:
        timestamp = datetime.fromisoformat(topic_segment.iloc[current_idx]["published"])
        timestamp_str = timestamp.strftime("%m/%d/%Y")
        prompt = prompt + row_format % (timestamp_str, topic_segment.iloc[current_idx]["title"])
        current_idx = current_idx + 1
    logging.info("Summarizing themes using the prompt:\n%s" % prompt)
    return prompt


def create_koala_topic_summarizer(prompt=BASE_PROMPT, temperature=0.7, max_new_tokens=512, max_prompt_tokens=1536):
    """
    Creates a topic summarizer function that uses Koala.
    """

    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizer

    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto",
                                               cache_dir=cache_dir, max_input_size=2048)
    model = LlamaForCausalLM.from_pretrained("/home/jupyter/koala_transformer", torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)

    def generate_koala_prompt(prompt):
        system = """BEGINNING OF CONVERSATION: """
        template = system + "USER: %s GPT:"
        return template % prompt

    def generate_from_tokens(tokens):
        outputs = model.generate(**tokens,
                                 do_sample=True,
                                 top_p=1.0,
                                 num_beams=1,
                                 top_k=50,
                                 temperature=temperature,
                                 max_new_tokens=max_new_tokens)
        result = tokenizer.decode(outputs[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        return "".join(result)

    def token_count_limiter(final_prompt):
        final_prompt = generate_koala_prompt(final_prompt)
        tokens = tokenizer(final_prompt, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
        return token_count <= max_prompt_tokens

    def summarize_topics(topic_df, topic_number):
        final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt=prompt)
        final_prompt = generate_koala_prompt(final_prompt)
        tokens = tokenizer(final_prompt, return_tensors="pt").to("cuda")
        return generate_from_tokens(tokens)

    return summarize_topics


API_PROMPT = "A list of news article titles with the published time is given below. " + \
             "Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. " + \
             "Be specific about the dates and entities involved. " + \
             "Be concise in writing the summary, but try not to omit important details. " + \
             'Do not use vague terms such as "past few months", "various companies", or "the disease". Use the actual names of the entities such as companies, products, etc if possible. ' + \
             'Finally, explain how the summary can be used to answer investing related questions. ' + \
             'If a clear theme relevant to investing is not present, write "NO THEME" as the summary, and explain the reasons. ' + \
             'The format of the output should be: <SUMMARY>the summary</SUMMARY><EXPLAINATION>the explanation</EXPLAINATION>\n\n'


def create_palm2_summarizer(prompt=API_PROMPT, temperature=0.7, max_new_tokens=1024, max_prompt_tokens=2048):
    """
    Creates a topic summarizer function that uses Google AI Studio. The AI studio uses PALM2 in the backend.
    """

    from transformers import LlamaTokenizer

    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto",
                                               cache_dir=cache_dir, max_input_size=6656)
    project_id = "msca310019-capstone-f945"
    model_name = "text-bison@001"
    location = "us-central1"
    matcher = re.compile(r"\<SUMMARY\>(?P<summary>.+)\<\/SUMMARY\>")

    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)

    def predict_large_language_model_sample(
            temperature: float,
            max_decode_steps: int,
            top_p: float,
            top_k: int,
            content: str,
    ):

        response = model.predict(
            content,
            temperature=temperature,
            max_output_tokens=max_decode_steps,
            top_k=top_k,
            top_p=top_p)
        return response

    def summarize_topics(topic_df, topic_number):

        current_max_token = max_prompt_tokens

        while True:
            def token_count_limiter(prompt):
                tokens = tokenizer(prompt, return_tensors="pt")
                token_count = tokens["input_ids"].shape[1]
                return token_count <= current_max_token

            final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt=prompt)
            result = predict_large_language_model_sample(temperature=temperature, max_decode_steps=max_new_tokens,
                                                         top_p=0.8, top_k=40, content=final_prompt)
            result = str(result)
            match = matcher.search(result)
            if match:
                return match.group("summary")
            else:
                current_max_token = current_max_token - 16

    return summarize_topics


OPENAI_PROMPT = "A list of news article titles with the published time is given below. " + \
                "Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. " + \
                "Be specific about the dates and entities involved. " + \
                "Be concise in writing the summary, but try not to omit important details. " + \
                'Do not use vague terms such as "past few months", "various companies", or "the disease". Use the actual names if possible. ' + \
                'Finally, explain how the summary can be used to answer investing related questions. ' + \
                'If a clear theme relevant to investing is not present, maintain the specified format, use "SUMMARY: NO THEME" as the summary, and explain the reasons. ' + \
                'The format of the output should be:\nSUMMARY:\nthe summary\nEXPLANATION:\nthe explanation\n\n'


@cache
def read_api_key(location):
    with open(location, "r") as fp:
        return fp.read().strip()


def create_openai_summarizer(api_key, prompt=OPENAI_PROMPT, temperature=0.7, max_new_tokens=1024,
                             max_prompt_tokens=2560):
    """
    Creates a topic summarizer function that uses ChatGPT.
    """
    import openai
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )  # for exponential backoff
    import tiktoken

    openai.api_key = read_api_key(api_key)
    engine = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(engine)
    matcher = re.compile(r"SUMMARY:(?P<summary>(.|\n)+)EXPLANATION")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def token_count_limiter(prompt):
        tokens = encoding.encode(prompt)
        return len(tokens) <= max_prompt_tokens

    def summarize_topic(topic_df, topic_number):
        final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt=prompt)
        messages = [
            {'role': 'user', 'content': final_prompt},
        ]
        result = completion_with_backoff(messages=messages, model=engine, temperature=temperature,
                                         max_tokens=max_new_tokens)
        result_text = result['choices'][0]['message']['content']
        match = matcher.search(result_text)
        if match:
            return match.group("summary").strip()
        else:
            raise AssertionError("Did not generate correct response:\n" + result_text)

    return summarize_topic


def create_topic_summarizer(kind="openai", **kwargs):
    """
    This creates a summarizer function that will summarize the theme of the articles in a given topic.
    The signature of the summarizer function is summarizer: (DataFrame, Integer) -> String. It will select
    the appropriate rows from the dataframe and create a suitable summary for later querying.
    """

    if kind == "koala":
        return create_koala_topic_summarizer(**kwargs)
    if kind == "vertex-ai":
        return create_palm2_summarizer(**kwargs)
    if kind == "openai":
        return create_openai_summarizer(**kwargs)
    raise ValueError("Invalid kind: " + kind)