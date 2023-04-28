import streamlit as st
import torch
import time
from models.model_rnn import func
from models.model_ml import final
from models.LSTM_model import data_preprocessing, padding, preprocess_single_string, sentimentLSTM, load_model, predict_sentiment
from models.bert_func import load_bert_lr_model, prediction
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

txt_label = 'Enter review text'
txt = st.text_area(label=txt_label, height=200)
    
with st.form('button'):
    button_click = st.form_submit_button("Get result")    

col1, col2, col3, col4 = st.columns(4)

with col1: # ML
    st.write('ML model')
    if button_click:
        t = time.process_time()
        output = final(txt)
        elapsed_time = time.process_time() - t
        st.write('`Negative review`' if np.around(output, 0) == 0 else '`Positive review`')
        st.write('`Time elapsed :`', round(elapsed_time, 3))

with col2: # RNN
    st.write('RNN model')
    if button_click:
        t = time.process_time()
        output = func(txt)
        elapsed_time = time.process_time() - t
        st.write('`Negative review`' if np.around(output, 0) == 0 else '`Positive review`')
        st.write('`Time elapsed :`', round(elapsed_time, 3))

with col3: # LSTM
    st.write('LSTM model')
    if button_click:
        st.write(f'`{predict_sentiment(txt)}`')
        t = time.process_time()
        elapsed_time = time.process_time() - t
        st.write('`Time elapsed :`', round(elapsed_time, 3))
        
    
model, tokenizer, lr = load_bert_lr_model('models/bert_lr.joblib')

with col4: # BERT
    st.write('BERT')
    if button_click:
        t = time.process_time()
        st.write(f'`{prediction(txt, model, tokenizer, lr)}`')
        elapsed_time = time.process_time() - t
        st.write('`Time elapsed :`', round(elapsed_time, 3))
