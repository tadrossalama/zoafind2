from fastai.learner import export
import streamlit as st
import pickle
from PIL import Image
from fastai.vision.all import *
from pathlib import Path
import wikipedia



path = Path()
learn_inf = load_learner(path/'learn3.pkl')


st.title('Zoa Find :mag_right:')
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
        st.title(f'I am {probs[pred_idx] * 100:.02f}% sure this is a {pred} coral.')
        st.write(wikipedia.summary(f'{pred} coral'))


