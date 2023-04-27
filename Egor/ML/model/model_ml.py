from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
import string
import pandas as pd 
import nltk
import pymorphy2
nltk.download('stopwords')
from nltk.corpus import stopwords
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

ascii = set(string.ascii_letters) 
pun = set(string.punctuation)
ascii.update(pun)


def clean(text):
    compiled_pattern = re.compile(r'[ЁёА-я]{1,10}[^\w\s][ЁёА-я]{1,10}')
    compiled_number = re.compile('\d{5,}')
    punct= re.compile(r'[^\w\s]')
    
    text = text.str.lower()
    text = text.apply(lambda x:re.sub('<.*?>', '', x))
    text = text.apply(lambda x: re.sub(punct, ' ', x) if re.search(compiled_pattern, x) is not None else x)
    text = text.apply(lambda x: re.sub(compiled_number, '', x) if re.search(compiled_number, x) is not None else x)
    text = text.apply(lambda x: re.sub(r'\n', '', x))
    text = text.apply(lambda x: re.sub(' +', ' ', x))
    
    return text

def final(user_input):

    user_input = pd.Series(user_input)

    #### Чистка
    text_cleaned = clean(user_input)

    #### Лемматизация
    morph = pymorphy2.MorphAnalyzer()
    lemmatized_text = []
    for text in text_cleaned:
        text = morph.parse(text)[0]
        lemmatized_text.append(''.join(text.normal_form))

    #### Токенизация
    reg_tokenizer = RegexpTokenizer('\w+')
    tokenized_text = reg_tokenizer.tokenize_sents(lemmatized_text)

    #### Удаление стоп-слов
    sw = stopwords.words('english')
    clean_tokenized_text = [] 
    for i, element in enumerate(tokenized_text):
        clean_tokenized_text.append(' '.join([word for word in element if word not in sw]))


    model = load('./linreg_model.joblib') 
    vector_fit = load('./vector_fit.joblib') 
    trunc_fit = load('./X_lsa.joblib') 
    pipeline = make_pipeline(vector_fit, trunc_fit)

    final = pipeline.transform(clean_tokenized_text)
    final = Normalizer(copy=False).fit_transform(final)

    unnorm = model.predict(final)
    norm = (unnorm + 0.665) / 2.254

    return norm