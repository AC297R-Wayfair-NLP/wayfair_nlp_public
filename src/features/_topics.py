from os import path
import time
import shutil
import logging
import warnings
import numpy as np
import pandas as pd

import spacy
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords


class Topics:
    def __init__(self, n_topics = 20):
        #specify number of topics to explore
        self.n_topics = n_topics
        self.model = None
        self.corpus = None
        self.curr_path = path.abspath(__file__) # Full path to current class-definition script
        self.root_path = path.dirname(path.dirname(path.dirname(self.curr_path)))
        nltk.download('stopwords')

    #utils function for pre-processing
    def _remove_stopwords(self, texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def _strip_newline(self, series):
        return [review.replace('\n','') for review in series]

    def _sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def _lemmatization(self, texts, allowed_postags=['NOUN','ADJ','VERB','ADV']):
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        texts_out = []
        i = 0
        for sent in texts:
            if i % 50000 == 0:
                print(i)
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            i += 1
        return texts_out

    def _preprocess(self, reviews):
        """Preprocesses the reviews

        Args:
            reviews (list[str]): Review texts
        """
        reviews = pd.DataFrame(list(reviews), columns=['rvprcomments'])

        print('----Preprocessing----')
        reviews = reviews[~reviews.rvprcomments.isnull()].copy()
        self.non_empty_review = reviews.shape[0]
        print('strip_newline...')
        reviews['text'] = self._strip_newline(reviews.rvprcomments)

        print('sent to words ...')
        words = list(self._sent_to_words(reviews.rvprcomments))

        print('remove stop words...')
        words = self._remove_stopwords(words)

        print('lemmatization...')
        bigram = self._lemmatization(words)

        print('dictionary...')
        id2word = gensim.corpora.Dictionary(bigram)

        print('filter...')
        id2word.filter_extremes(no_below=50, no_above=0.35)

        print('compactify...')
        id2word.compactify()

        print('bag of word')
        corpus = [id2word.doc2bow(text) for text in bigram]

        self.corpus = corpus
        self.id2word = id2word
        self.bigram = bigram


    def train(self, reviews, save_filename):
        """Trains the language model

        1. preprocessing the reviews
        2. train language model & save it

        Args:
            reviews (pd.Series): column of reviews to fit the language model
        """
        # pre-process the data
        if self.corpus is None:
            self._preprocess(reviews)

        # Specify temporary file name to save model weights
        temp_filename = 'topic_model'

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            topic_model = gensim.models.ldamulticore.LdaMulticore(
                           corpus= self.corpus,
                           num_topics= self.n_topics,
                           id2word= self.id2word,
                           chunksize=10000,
                           workers= None, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
            topic_model.save(temp_filename)
            self.model = topic_model
        end = time.time()

        print(f'Training Completed. The modeling process took {(end-start)/60.} minutes')

        # Move saved files to desired location
        shutil.move(temp_filename, save_filename)
        shutil.move(temp_filename + '.state', save_filename + '.state')
        shutil.move(temp_filename + '.id2word', save_filename + '.id2word')
        shutil.move(temp_filename + '.expElogbeta.npy', save_filename + '.expElogbeta.npy')


    def load_model(self, model_filename=None):
        '''
        Read saved model with specified filename
        '''

        if model_filename is None:
            model_filename = path.join(self.root_path, 'models', 'topic_model')
        self.model = gensim.models.ldamulticore.LdaMulticore.load(model_filename)
        print('----Model loaded to disk----')


    def extract(self, reviews):
        '''
        pre-process the reviews if necessary
        use topic model to generate topic distribution
        return the topic score dataframe
        '''
        # returns (all) topic scores for each review
        # make a note that we chose to use 2 topic scores
        if self.model is None:
            self.load_model() # Use pre-trained model weights

        if self.corpus is None:
            self._preprocess(reviews)

        topic_vecs = []
        for i in range(self.non_empty_review):
            top_topics = self.model.get_document_topics(self.corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(self.n_topics)]
            topic_vecs.append(topic_vec)

        column_name = ['topic_{}'.format(i) for i in range(self.n_topics)]
        score_df = pd.DataFrame(topic_vecs, columns = column_name)

        print('----Topic Score Extracted----')
        return score_df


def test_topic_module():
    '''
    test function that test each method under topic modeling class
    '''
    # Load toy data
    CURR_PATH = path.abspath(__file__) # Full path to current script
    ROOT_PATH = path.dirname(path.dirname(path.dirname(CURR_PATH)))
    file_path = path.join(ROOT_PATH, "demo", "data", "reviews_toydata.csv")
    df = pd.read_csv(file_path)

    #test pre-process + train function
    model_path = path.join(ROOT_PATH, "demo", "models", "topic_model_demo")
    topic = Topics(20)
    topic.train(reviews=df['rvprcomments'], save_filename=model_path)

    #test load
    topic.load_model(model_filename=model_path)

    #test extract
    topic_modeling_feature = topic.extract(reviews=df['rvprcomments'])
    print(topic_modeling_feature.head())

if __name__ == "__main__":
    test_topic_module()
