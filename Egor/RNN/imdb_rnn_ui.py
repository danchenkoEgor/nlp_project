import time
from model.model_rnn import func
import streamlit as st 
import numpy as np


string_input = st.text_input("Enter review:")


if string_input is not None and len(string_input) > 1:
    t = time.process_time()
    

    output = func(string_input)
    elapsed_time = time.process_time() - t

    st.write('Negative' if np.around(output, 0) == 0 else 'Positive')
    st.write('Time elapsed :', round(elapsed_time, 3))