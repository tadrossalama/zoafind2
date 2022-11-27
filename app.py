from fastai.learner import export
import streamlit as st
import pickle
from PIL import Image
from fastai.vision.all import *
from pathlib import Path
import wikipedia
from getinfo import *
from pygbif import maps
from pygbif import occurrences as occ
from pygbif import registry
from pygbif import species
import pandas as pd


class Coral:
    def __init__(self):
        pass

    def get_species_name(self, name):
        """
        Get the species name from the taxon key
        """
        name = species.name_suggest(q = name, limit=1)
        df = pd.DataFrame(name)
        return df[['kingdom', 'phylum', 'class', 'order', 'family', 'scientificName', 'canonicalName']]




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


