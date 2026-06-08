# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Allow legacy models saved with time_major=False in SimpleRNN config to load on newer Keras
_orig_simple_rnn_init = SimpleRNN.__init__

def _patched_simple_rnn_init(self, *args, **kwargs):
    kwargs.pop('time_major', None)
    return _orig_simple_rnn_init(self, *args, **kwargs)

SimpleRNN.__init__ = _patched_simple_rnn_init

model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

st.set_page_config(
    page_title='IMDb Sentiment Analyzer',
    page_icon='🎬',
    layout='centered',
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b0b0b;
        color: #ffffff;
    }
    .title {
        font-size: 44px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #f5c518;
        margin-bottom: 0;
    }
    .subtitle {
        color: #ffffff;
        font-size: 18px;
        margin-top: 0;
        opacity: 0.85;
    }
    .imdb-box {
        background: #141414;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .imdb-highlight {
        color: #f5c518;
        font-weight: 700;
    }
    .imdb-button {
        background: #f5c518;
        color: #111 !important;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .imdb-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 18px;
    }
    .imdb-score {
        color: #f5c518;
        font-size: 22px;
        font-weight: 700;
    }
    .streamlit-expanderHeader {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='imdb-box'>
        <h1 class='title'>IMDb Sentiment Analyzer</h1>
        <p class='subtitle'>Paste a movie review below and get a sentiment prediction in IMDb-style fashion.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<div class='imdb-box'>
  <strong class='imdb-highlight'>How it works:</strong>
  This app uses a trained SimpleRNN model to score reviews as positive or negative. The layout and theme are inspired by IMDb with dark styling, bold accents, and a cinematic feel.
</div>
""",
    unsafe_allow_html=True,
)

user_input = st.text_area('Enter your review here', height=220)

if st.button('Classify'):
    if not user_input or not user_input.strip():
        st.warning('Please enter a movie review before classifying.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        score = float(prediction[0][0])

        st.markdown(
            """
            <div class='imdb-card'>
              <p><span class='imdb-highlight'>Result:</span> <strong>{sentiment}</strong></p>
              <p class='imdb-score'>IMDb Score: {score:.3f}</p>
              <p>Reviews above 0.5 are predicted as positive, while lower values are predicted as negative.</p>
            </div>
            """.format(sentiment=sentiment, score=score),
            unsafe_allow_html=True,
        )
else:
    st.info('Type a movie review and click Classify to see the sentiment result.')

with st.expander('Tips for best results'):
    st.markdown(
        """
        - Use complete sentences describing a film or performance.
        - Include both positive and negative phrases for a better sense of polarity.
        - This model is tuned for IMDb-style movie reviews, so plot summaries or terse comments may be less accurate.
        """
    )

