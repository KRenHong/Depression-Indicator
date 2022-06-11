from hashlib import new
from imp import new_module
from pyexpat import model
from re import sub
from unittest import result
import streamlit as st
import requests
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import time
import pickle


st.markdown("## Machine Learning Depression Indicator")
st.markdown("##### WIA1006 MACHINE LEARNING")
st.markdown("###### Group: Mobile Legend")
st.markdown("---")

st.markdown(
    "<div style='text-align: justify'> Depression <b>major depressive disorder</b> is a common and serious medical illness that negatively affects how you feel, the way you think and how you act. It can lead to a variety of emotional and physical problems and can decrease your ability to function at work and at home. So, we build a text classification machine learning model which can predict depression level through text </div>", unsafe_allow_html=True
)

st.markdown('')

@st.cache(allow_output_mutation=True)
def load_model():
    new_model = tf.keras.models.load_model('model_depressed.h5')
    return new_model

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def sentence_process(sentence):
    max_length = 120
    new_sentences = tokenizer.texts_to_sequences(sentence)
    new_padded = pad_sequences(new_sentences, maxlen=max_length)
    return new_padded

with st.spinner("Loading Model. . . ."):
    new_model = load_model()

with st.form(key = 'form_1'):
    input_1 = st.text_area(label = 'Please enter one sentence to represent your feelings', height=100)
    confirm_button = st.form_submit_button(label= 'Confirm')
    arr = np.array([input_1])
    padded_sentence = sentence_process(arr)
    results = new_model.predict(padded_sentence)
    if results[0,0] > 0.75:
        results_2 = "No depression"
        advice = "you are totally fine"
        st.success("You are happy")
    elif 0.5 <= results[0,0] <= 0.75:
        results_2 = "Mild depression"
        advice = "Guided self-help: The person may follow an online course or manual with the support of a therapist. The course aims to provide tools that enable a person to make helpful changes"
        st.info("Mild depression")
    elif 0.25 <= results[0,0] <= 0.5:
        results_2 = "Moderate depression"
        advice = "Talking therapy: During a series of sessions, the individual will work with a counselor to identify the causes of depression and find ways of resolving it"
        st.warning("Moderate Depression")
    elif 0.001 <= results[0,0] <= 0.25:
        results_2 = "Severe depression"
        advice = "Counseling for depression: The person will explore why depression has occurred and look for ways to overcome it."
        st.error("Severe depression")
    else:
        results_2 =""

st.markdown("## Prediction Result")
st.markdown(f"Predictions : {results_2}")
my_bar = st.progress(0)

for percent_complete in range(int(results[0,0] * 100)):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1)

resultPercentage = "{:.2f}".format(results[0,0] * 100)
st.markdown(f"The Probability of not being depressed : {resultPercentage} %")
st.markdown("## Advice for " + f"{results_2}")
st.info(advice)









