from summarizer import config

with open(config.ES_KEY_PATH, "r") as fp:
    es_key = fp.read().strip()
with open(config.ES_CLOUD_ID_PATH, "r") as fp:
    es_id = fp.read().strip()


def finbot_response(text, period):
    return {
        "qa": "Placeholder QA",
        "summaries": [{"title": "Main Point", "keypoints": ["K1", "K2", "K3"]}, 
                      {"title": "Lorem Ipsum", "keypoints": ["K1", "K2", "K3"]}]
    }

