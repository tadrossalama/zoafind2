import openai
import streamlit as st
import pickle
from PIL import Image
from pathlib import Path
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

key = st.secrets['openai_key']
openai.api_key = key

def assistant(x): 
    search = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You're name is Zoa. You are an AI coral reef expert specilizing in salt water aquariums"},
        {"role": "user", "content":x}
    ])
    return search['choices'][0]['message']['content']

def classify_image(image):
    processor = ViTImageProcessor.from_pretrained("tdros/zoalearn2")
    model = AutoModelForImageClassification.from_pretrained("tdros/zoalearn2")
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs, indices = logits.softmax(-1).topk(3)
    probs = probs[0].tolist()
    indices = indices[0].tolist()
    top_class = model.config.id2label[indices[0]]

    predicted_classes = [(model.config.id2label[index], prob) for index, prob in zip(indices, probs)]
    return top_class,predicted_classes


st.title('Zoa Find :mag_right:')

tab1, tab2 = st.tabs(["Classify a Coral", "Ask Zoa a Question"])

with tab1:
    upload = st.file_uploader('Upload picture of coral here')
    if upload is not None:

        image = Image.open(upload)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image.')
            top_class, pred = classify_image(image)
            
        with col2:
            st.subheader(f'I am {pred[0][1] * 100:.02f}% sure this is a {top_class} coral.')
            st.write(f'my other guesses would be: {pred[1][0]} or a {pred[2][0]} coral.')
            st.write(assistant(top_class))
with tab2:
    text_input = st.text_input('Ask Zoa a question')
    if text_input is not None:
        st.write(assistant(text_input))



