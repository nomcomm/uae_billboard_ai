import re, string, sys, warnings, random
import pandas as pd
import numpy as np
import seaborn as sns
from textacy.text_stats import TextStats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text):
   text = text.lower()                                               # make text lowercase  
   text = re.sub(r'\[.*?\]', '', text)                               # remove text in square brackets
   text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation   
   text = re.sub(r'\w*\d\w*', '', text)                              # remove words containing numbers
   return text

def get_top_n_gram(corpus, ngram, n, stopwords):
    vec = CountVectorizer(ngram_range=(ngram,ngram), stop_words = stopwords)
    vec.fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def blue_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "dodgerblue" #hsl(228, 49%, 56%)" 

def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "salmon"  #hsl(0, 49%, 65%)" 