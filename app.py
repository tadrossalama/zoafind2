import openai
from fastai.learner import export
import streamlit as st
import pickle
from PIL import Image
from fastai.vision.all import *
from pathlib import Path

key = st.secrets['openai_key']
openai.api_key = key
path = Path()
learn_inf = load_learner(path/'learn3.pkl')
def assistant(x): 
    search = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You're name is Zoa. You are an AI coral reef expert specilizing in salt water aquariums"},
        {"role": "user", "content":x}
    ])
    return search['choices'][0]['message']['content']



st.title('Zoa Find :mag_right:')

tab1, tab2 = st.tabs(["Classify a Coral", "Ask Zoa a Question"])

with tab1:
    upload = st.file_uploader('Upload picture of coral here')
    if upload is not None:

        image = PILImage.create(upload)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image.')
            pred,pred_idx,probs = learn_inf.predict(image)
            
        #coral = Coral()
        #st.table(coral.get_species_name(pred))
        with col2:
            st.header(f'I am {probs[pred_idx] * 100:.02f}% sure this is a {pred} coral.')
            st.write(assistant(pred))
with tab2:
    text_input = st.text_input('Ask Zoa a question')
    if text_input is not None:
        st.write(assistant(text_input))



