import re
import logging
from .base_step import BaseStep
from nltk.corpus import stopwords


class CleanDataStep(BaseStep):
    def __init__(self):
        # If using NLTK's stopwords for the first time, you'll need to download them using nltk.download('stopwords')
        self.stop_words = set(stopwords.words("english"))

    def execute(self, data, dependencies):
        logging.info("Cleaning data...")

        data["train"].map(self.clean_text)
        data["test"].map(self.clean_text)
        data["val"].map(self.clean_text)

        return data

    def clean_text(self, data):
        # Convert to lowercase
        text = data['text'].lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove punctuations
        text = re.sub(r"[^\w\s]", "", text)

        # Remove stopwords
        text = " ".join([word for word in text.split() if word not in self.stop_words])

        return {"text": text, "label": data["label"]}
