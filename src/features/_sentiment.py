from os import path
import numpy as np
import pandas as pd

from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Sentiment:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def _get_score(self, review):
        # returns individual review's score
        sentiments = np.zeros(5)
        if review:
            sentiments[1:] = list(self.sia.polarity_scores(review).values()) # VADER
            sentiments[0] = TextBlob(review).sentiment.subjectivity     # nltk
        else:
            # if review is None, return nan. Ideally you'd check for Nones before calling `Sentiment`
            sentiments += np.nan
        return sentiments

    def extract(self, reviews):
        """Extract NLP Sentiment features.

        Args:
            reviews (array_like): the reviews to extract Sentiment from

        Returns:
            pd.DataFrame of the sentiment scores
        """
        sent_cols = ['subjectivity', 'neg', 'neu', 'pos', 'compound']
        sent_results = np.zeros((reviews.shape[0], len(sent_cols)))
        reviews = pd.Series(reviews)

        for i, row in enumerate(reviews.values):
            sent_results[i] = self._get_score(row)
        sents = pd.DataFrame(sent_results, columns=sent_cols, index=reviews.index)
        return sents


def main():
    print("\n\n")

    # Load toy data
    CURR_PATH = path.abspath(__file__) # Full path to current script
    ROOT_PATH = path.dirname(path.dirname(path.dirname(CURR_PATH)))
    file_path = path.join(ROOT_PATH, "data", "demo", "reviews_toydata.csv")
    df = pd.read_csv(file_path)

    # Test feature extraction
    extractor = Sentiment()
    features_df = extractor.extract(df["rvprcomments"])

    print("\n\n")
    print(features_df.head().round(2))


if __name__ == "__main__":
    main()
