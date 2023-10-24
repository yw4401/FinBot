# AI Assisted Labeling
LABEL_REINFORCEMENT = "Good. "
LABEL_MAX_TOKEN = 1024 * 7
LABEL_VERBOSE = False
LABEL_FORMAT_USER = "Format:\n{raw}"
LABEL_FORMAT_SYSTEM = "You are an AI assistant that will format user inputs into json.\n{format_instructions}"

# AI Assisted Summary Rating
LABEL_SUMMARY_USER = "Text:\n{context}\n\nSummary:\n{output}\n\nOn a scale of 1-5, how well does the summary reflect the content of the text? " \
                     "Focus on the aspects of the text that may be relevant to a retail investor."
LABEL_SUMMARY_SYSTEM = "You are an AI assistant. You will rate the quality of summaries based on given texts. " \
                       "For each rating, you will first provide your thought process for the rating. " \
                       "Then, you will provide the final rating after providing the thought process."
LABEL_FORMAT_RAW = """Thought process:
The summary accurately captures the main points of the text. It mentions that Mighty Group Inc. is launching Mighty Law, a law firm that offers an alternative fee structure for personal injury cases. It also highlights the unique pricing strategies of Mighty Law, including covering 10% of the client's medical and case costs and capping fees at 30% of settlements. The summary mentions the transparency of Mighty Law's fee structure and the potential challenges it may face in terms of name recognition. Overall, the summary provides a concise and accurate overview of the text.

Final Rating:
4"""
LABEL_FORMAT_RATING = 4
LABEL_FORMAT_THOUGHT = "The summary accurately captures the main points of the text. It mentions that Mighty Group Inc. is launching Mighty Law, a law firm that offers an alternative fee structure for personal injury cases. It also highlights the unique pricing strategies of Mighty Law, including covering 10% of the client's medical and case costs and capping fees at 30% of settlements. The summary mentions the transparency of Mighty Law's fee structure and the potential challenges it may face in terms of name recognition. Overall, the summary provides a concise and accurate overview of the text."

# AI Assisted FIQA Augmentation
FIQA_SYSTEM = "You are a helpful AI assistant that will re-phrase given text into a formal tone found in news articles or financial reports. " \
              "Follow the conventions in the news article or reports genre. " \
              "Thus, avoid first person, and try to sound objective. " \
              "You will never add any new information to the given text, only to present existing information in a different way."
FIQA_USER = "{input_text}"
FIQA_MODEL = "gpt-3.5-turbo-16k"
FIQA_TEMPERATURE = 0

# Finetuning Parameters
LLAMA_SUMMARY_BULLET_INSTRUCTION = "Summarize the key-points from the given context. " \
                                   "The information in the summary should include, " \
                                   "but should not be limited to information that can help answer the given question."
LLAMA_SUMMARY_PARA_INSTRUCTION = "Summarize the given context. " \
                                 "The information in the summary should include, " \
                                 "but should not be limited to information that can help answer the given question."
LLAMA_SUMMARY_MAX_INPUT_TOKEN = 3840
LLAMA_Q_HEADER = "### Question"
LLAMA_C_HEADER = "### Context"
LLAMA_S_HEADER = "### Summary\n"
