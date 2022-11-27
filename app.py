from fastai.learner import export
import streamlit as st
import pickle
from PIL import Image
from fastai.vision.all import *
from pathlib import Path
import wikipedia
from getinfo import *
from pygbif import maps
import requests
import logging
import http.client
http.client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True


path = Path()
learn_inf = load_learner(path/'learn3.pkl')


st.title('Zoa Find :mag_right:')
upload = st.file_uploader('Upload picture of coral here')
if upload is not None:

    image = PILImage.create(upload)
    st.image(image, caption='Uploaded Image.')
    pred,pred_idx,probs = learn_inf.predict(image)

    st.title(f'I am {probs[pred_idx] * 100:.02f}% sure this is a {pred} coral.')
    coral = Coral()
    st.table(coral.get_species_name(pred))

    st.write(wikipedia.summary(f'{pred} coral'))


