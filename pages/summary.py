import streamlit as st
import requests

API_TOKEN = st.secrets["API_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/cointegrated/rut5-base-absum"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def summarize(text):
    if len(text) > 0:
        response = query({"inputs":f'{text}', "parameters": {"num_beams": 5,
                          "do_sample" :False, "repetition_penalty": 10.0, "max_length": 50, "min_length": 5}})
        if type(response) == list:
            summary = response[0]['summary_text']
            st.markdown(f'Кажется, в этом тексте говорится о том, что `{summary}`')
            
        elif response['error']:
            st.markdown(':red[Что-то пошло не так, попробуйте ещё раз!]')
            
    else:
        st.markdown(f":red[Краткость - сестра таланта, но, может быть, напишете что-нибудь?]")
        

st.title('Text summarization')        
text_input = st.text_area('Введите текст:', height=250)
submit = st.button('Давайте резюмируем')
if submit:
   summarize(text_input) 