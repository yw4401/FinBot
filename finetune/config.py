# AI Assisted Labeling
LABEL_REINFORCEMENT = "Good. "
LABEL_MAX_TOKEN = 3072
LABEL_VERBOSE = False
LABEL_FORMAT_USER = "Format:\n{raw}"
LABEL_FORMAT_SYSTEM = "You are an AI assistant that will format user inputs into json.\n{format_instructions}"

# AI Assisted Summary Rating
LABEL_SUMMARY_USER = "Text:\n{context}\n\nSummary:\n{output}\n\nOn a scale of 1-5, how well does the summary reflect the content of the text?" \
                     "Focus on the aspects of the text that may be relevant to a retail investor."
LABEL_SUMMARY_SYSTEM = "You are an AI assistant. You will rate the quality of summaries based on given texts. " \
                       "For each rating, you will first provide your thought process for the rating. " \
                       "Then, you will provide the final rating after providing the thought process."

