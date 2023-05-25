import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime
import re
from collections import namedtuple
from fractions import Fraction
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken


BASE_PROMPT = """A list of news article titles with the published time is given below. Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. Be specific about the dates and entities involved and try to not omit important details. Do not use vague terms such as "past few month" or "various companies".\n\n"""


def select_titles(topic_df, topic, limiter, base_prompt=BASE_PROMPT):
    """
    Create the base prompt that will be used to ask the LLM on the theme summarization of the topics from the 
    title of the articles in the topics together with the published dates. The goal is to 
    fit as many titles into the prompt as possible without exceeding the context window. The maximum 
    fit is determined by the limiter parameter passed in, which is a function that takes the prompt and 
    determine if it is too big.
    """
    
    topic_segment = topic_df.loc[topic_df.topic == topic][["title", "published", "probability"]].sort_values(by="probability", ascending=False)
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
    
    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto", cache_dir=cache_dir,  max_input_size=2048)
    model = LlamaForCausalLM.from_pretrained("/home/jupyter/koala_transformer", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
    
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


API_PROMPT = "A list of news article titles with the published time is given below. " +\
"Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. " +\
"Be specific about the dates and entities involved. " +\
"Be concise in writing the summary, but try not to omit important details. " +\
'Do not use vague terms such as "past few months", "various companies", or "the disease". Use the actual names of the entities such as companies, products, etc if possible. ' +\
'Finally, explain how the summary can be used to answer investing related questions. ' +\
'If a clear theme relevant to investing is not present, write "NO THEME" as the summary, and explain the reasons. ' +\
'The format of the output should be: <SUMMARY>the summary</SUMMARY><EXPLAINATION>the explanation</EXPLAINATION>\n\n'


def create_palm2_summarizer(prompt=API_PROMPT, temperature=0.7, max_new_tokens=1024, max_prompt_tokens=2048):
    """
    Creates a topic summarizer function that uses Google AI Studio. The AI studio uses PALM2 in the backend.
    """
    
    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto", cache_dir=cache_dir,  max_input_size=6656)
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
            result = predict_large_language_model_sample(temperature=temperature, max_decode_steps=max_new_tokens, top_p=0.8, top_k=40, content=final_prompt)
            result = str(result)
            match = matcher.search(result)
            if match:
                return match.group("summary")
            else:
                current_max_token = current_max_token - 16
    
    return summarize_topics


OPENAI_PROMPT = "A list of news article titles with the published time is given below. " +\
"Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. " +\
"Be specific about the dates and entities involved. " +\
"Be concise in writing the summary, but try not to omit important details. " +\
'Do not use vague terms such as "past few months", "various companies", or "the disease". Use the actual names if possible. ' +\
'Finally, explain how the summary can be used to answer investing related questions. ' +\
'If a clear theme relevant to investing is not present, maintain the specified format, use "SUMMARY: NO THEME" as the summary, and explain the reasons. ' +\
'The format of the output should be:\nSUMMARY:\nthe summary\nEXPLANATION:\nthe explanation\n\n'
    

def create_openai_summarizer(api_key, prompt=OPENAI_PROMPT, temperature=0.7, max_new_tokens=1024, max_prompt_tokens=2560):
    """
    Creates a topic summarizer function that uses ChatGPT.
    """
    
    openai.api_key = api_key
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
        messages=[
            {'role': 'user', 'content': final_prompt},
        ]
        result = completion_with_backoff(messages=messages, model=engine, temperature=temperature, max_tokens=max_new_tokens)
        result_text = result['choices'][0]['message']['content']
        match = matcher.search(result_text)
        if match:
            return match.group("summary").strip()
        else:
            raise AssertionError("Did not generate correct response:\n" + result_text)
            
    return summarize_topic
    
    
def create_topic_summarizer(kind="vertex-ai", **kwargs):
    """
    This creates a summarizer function that will summarize the theme of the articles in a given topic.
    The signature of the summarizer function is summarizer: (DataFrame, Integer) -> String. It will select 
    the appropriate rows from the dataframe and create a suitable summary for later querying.
    """
    
    if kind == "koala":
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer
        return create_koala_topic_summarizer(**kwargs)
    if kind == "vertex-ai":
        import vertexai
        from vertexai.preview.language_models import TextGenerationModel
        return create_palm2_summarizer(**kwargs)
    if kind == "openai":
        import openai
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )  # for exponential backoff
        import tiktoken
        return create_openai_summarizer(**kwargs)
    raise ValueError("Invalid kind: " + kind)
    
    
clean_summary_regex = re.compile(r"\s+")


def generate_topic_prompts(summary_df, inquiry, limiter, base_prompt):
    """
    This function generates the prompt to be used for filtering the topics. The 
    goal is again to fit as many topic summaries into the prompt determined by 
    the limiter function.
    """
    
    start_idx = 0
    end_idx = len(summary_df.index)
    
    row_format = "%d. %s\n"
    while start_idx < end_idx:
        prompt = base_prompt % inquiry
        current_idx = start_idx
        while limiter(prompt) and current_idx < end_idx:
            topic_id = summary_df.iloc[current_idx]["topics"]
            topic_summary = summary_df.iloc[current_idx]["summary"]
            topic_summary = clean_summary_regex.sub(" ", topic_summary)
            prompt = prompt + row_format % (topic_id, topic_summary)
            current_idx = current_idx + 1
        start_idx = current_idx
        logging.info("Filtering topics with prompt:\n%s" % prompt)
        yield prompt
        

RelevantTopic = namedtuple("RelevantTopic", "topic_number rating")


get_topics_regex = re.compile(r"(?<=RELEVANT TOPICS:)(\s*(\d+\(\d+\/\d+\)),?)+")
select_topics_regex = re.compile(r"(?P<topic>\d+)\((?P<rating>\d+\/\d+)\)")


def extract_topics(result_text):
    """
    A generator that extracts the filtered topic as well as their relevance.
    """
    
    topics_segments = get_topics_regex.findall(result_text)
    for t in topics_segments:
        part_segments = select_topics_regex.findall(t[0])
        for p in part_segments:
            yield(RelevantTopic(p[0], float(Fraction(p[1]))))
            
            
PALM_FILTER_PROMPT = "A numbered list of summaries of topics from news articles is give below. " +\
"Using only information in the summaries, identify the topics from the list that can be further investigated to answer the inquiry below. " +\
"In addition, explain why the topics identified are selected. " + \
'An example of how the answer should be formatted is as follow:\nRELEVANT TOPICS: 1(2/10), 5(6/10), 10(1/10)\nREASON: give the explanation of why the topics are selected.\n' +\
'In the example, the numbers 1, 5, and 10 are the selected topics from the list. (2/10) is the probability that topic 1 is relevant to the inquiry.'+ \
'\n\nInquiry: %s\nList of Topics:\n'


selection_regex = re.compile(r"")


def create_palm2_filter(prompt=PALM_FILTER_PROMPT, temperature=0.7, max_new_tokens=1024, max_prompt_tokens=1024):
    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto", cache_dir=cache_dir,  max_input_size=6656)
    project_id = "msca310019-capstone-f945"
    model_name = "text-bison@001"
    location = "us-central1"
    matcher = re.compile(r"\<SUMMARY\>(?P<summary>.+)\<\/SUMMARY\>")
    
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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
    
    def token_count_limiter(prompt):
        tokens = tokenizer(prompt, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
        return token_count <= max_prompt_tokens
    
    def filter_topics(topic_df, inquiry):
        
        for p in generate_topic_prompts(topic_df, inquiry, token_count_limiter, base_prompt=prompt):
            result = predict_large_language_model_sample(temperature=temperature, max_decode_steps=max_new_tokens, top_p=0.8, top_k=40, content=p)
            result = str(result)
            for topic in extract_topics(result):
                yield topic
    
    return filter_topics


BASE_FILTER_PROMPT = "A numbered list of summaries of topics from news articles is give below. " +\
"Without using prior knowledge and only focusing on information in the list, identify the topics from the list that can be used to directly answer the inquiry. " +\
'Do not speculate on possible connection between the summaries and the inquiry, such as stating that topic X may be related to the inquiry. Only focus on what is explicitly mentioned in the summaries. ' +\
"In addition, explain in detail why the topics identified are selected, and also provide a counter-argument of why the topics would not be related to the inquiry. " + \
'An example of how the answer should be formatted is as follow:\nREASON: topics 1, 5, 10 are selected because...\nCOUNTER: the counter argument.\nRELEVANT TOPICS: 1(2/10), 5(6/10), 10(1/10)\n' +\
'where 1, 5, and 10 are the selected topics from the list. (2/10) is the probability that topic 1 is relevant to the inquiry. The probability should be based on both the reason and the counter argument.\n\nInquiry: %s\nList of Topics:\n'



def create_openai_filter(api_key, prompt=BASE_FILTER_PROMPT, temperature=0.7, max_new_tokens=1024, max_prompt_tokens=2560):
    openai.api_key = api_key
    engine = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(engine)
    matcher = re.compile(r"SUMMARY:(?P<summary>(.|\n)+)EXPLANATION")
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)
    
    def token_count_limiter(prompt):
        tokens = encoding.encode(prompt)
        return len(tokens) <= max_prompt_tokens
    
    def filter_topics(topic_df, inquiry):
        
        for p in generate_topic_prompts(topic_df, inquiry, token_count_limiter, base_prompt=prompt):
            messages=[
                {'role': 'user', 'content': p},
            ]
            result = completion_with_backoff(messages=messages, model=engine, temperature=temperature, max_tokens=max_new_tokens)
            result_text = result['choices'][0]['message']['content']
            logging.info("Raw OpenAI filter response:\n%s" % result_text)
            for topic in extract_topics(result_text):
                yield topic
            
    return filter_topics


def create_koala_filter(prompt=BASE_FILTER_PROMPT, temperature=0.7, max_new_tokens=512, max_prompt_tokens=1536):
    cache_dir = "/home/jupyter/models"
    tokenizer = LlamaTokenizer.from_pretrained("/home/jupyter/koala_transformer", device_map="auto", cache_dir=cache_dir,  max_input_size=2048)
    model = LlamaForCausalLM.from_pretrained("/home/jupyter/koala_transformer", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
    
    def generate_koala_prompt(prompt):
        system = """BEGINNING OF CONVERSATION: """
        template = system + "USER: %s GPT:"
        return template % prompt
    
    def generate_from_tokens(tokens):
        outputs = model.generate(**tokens,
                             do_sample=True, 
                             top_p=1.0,
                             temperature=temperature,
                             max_new_tokens=max_new_tokens)
        result = tokenizer.decode(outputs[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        return "".join(result)
    
    def token_count_limiter(final_prompt):
        final_prompt = generate_koala_prompt(final_prompt)
        tokens = tokenizer(final_prompt, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
        return token_count <= max_prompt_tokens
    
    def filter_topics(topic_df, inquiry):
        
        for p in generate_topic_prompts(topic_df, inquiry, token_count_limiter, base_prompt=prompt):
            final_prompt = generate_koala_prompt(p)
            tokens = tokenizer(final_prompt, return_tensors="pt").to("cuda")
            print(generate_from_tokens(tokens))
    
    return filter_topics


def create_topic_filter(kind="openai", **kwargs):
    """
    Creates a filter function that takes a dataframe with a column 
    of topic IDs "topics" and a column of summaries "summary", 
    and returns an iterable of RelevantTopic
    """
    
    if kind == "koala":
        # Note the Koala one doesn't work rn
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer
        return create_koala_filter(**kwargs)
    if kind == "vertex-ai":
        import vertexai
        from vertexai.preview.language_models import TextGenerationModel
        return create_palm2_filter(**kwargs)
    if kind == "openai":
        return create_openai_filter(**kwargs)
    raise ValueError("Invalid kind: " + kind)
    
    
def load_faiss_topic_filter(path, embed_model):
    import faiss
    index = faiss.read_index(path)
    
    def filter_topic(query, init_df, k=50):
        vector_query = np.array(embed_model.get_query_embedding(query))
        vector_query = np.reshape(vector_query, (1, len(vector_query)))
        vector_query = vector_query.astype("float32")
        D, I = index.search(vector_query, k=k)
        logging.info("Semantic Index found topics: " + str(I))
        return init_df.loc[init_df.topics.isin(I[0])]
    
    return filter_topic