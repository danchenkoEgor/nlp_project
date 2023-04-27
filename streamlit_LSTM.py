import streamlit as st
import torch
import time
# import sys
# sys.path.insert(1, '../model')
from LSTM_model import data_preprocessing, padding, preprocess_single_string, sentimentLSTM, load_model, predict_sentiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

txt_label = 'Enter review text'
txt = st.text_area(label=txt_label, height=200)
    
with st.form('button'):
    button_click = st.form_submit_button("Get result")    

col1, col2, col3, col4 = st.columns(4)

with col1: # ML
    pass

with col2: # RNN
    pass

with col3: # LSTM
    st.write('LSTM model')
    if button_click:
        st.write(f'`{predict_sentiment(txt)}`')
        t = time.process_time()
        elapsed_time = time.process_time() - t
        st.write('`Time elapsed :`', round(elapsed_time, 3))
        
    
with col3: # BERT
    pass
