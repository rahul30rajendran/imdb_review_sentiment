# IMDb Sentiment Analysis with RNN

A Streamlit-based web application that uses a Recurrent Neural Network (RNN) to analyze sentiment from IMDb movie reviews. This project demonstrates deep learning for natural language processing with a pre-trained SimpleRNN model.

## 📋 Project Description

This application predicts the sentiment (positive or negative) of movie reviews using a trained RNN model. It provides an interactive interface where users can input their own reviews and get real-time sentiment predictions.

## ✨ Features

- **Pre-trained RNN Model**: Uses a SimpleRNN model trained on IMDb dataset
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Real-time Predictions**: Instant sentiment analysis of input reviews
- **Word Index Mapping**: Converts text to embeddings using IMDb word index
- **Sequence Padding**: Normalizes input sequences for consistent model predictions

## 📁 Project Structure

```
imdb_review_sentiment/
├── main.py                    # Streamlit application with prediction logic
├── simple_rnn_imdb.h5        # Pre-trained RNN model
└── README.md                  # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip

### Install Dependencies

```bash
pip install streamlit tensorflow numpy
```

## 📖 Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

The application will open in your browser. Enter a movie review to get sentiment predictions.

## 🧠 Model Details

- **Architecture**: SimpleRNN (Recurrent Neural Network)
- **Training Data**: IMDb movie reviews dataset
- **Input Size**: 500 words (padded sequences)
- **Output**: Binary classification (Positive/Negative sentiment)

## 📝 How It Works

1. Text input is converted to lowercase and split into words
2. Words are mapped to indices using the IMDb word index
3. The sequence is padded to 500 words
4. The pre-trained RNN model predicts sentiment probability
5. Results are displayed with confidence score

## 📦 Requirements

- TensorFlow
- Keras
- NumPy
- Streamlit

## 🎯 Example Usage

Input: *"This movie was absolutely fantastic! Great plot and amazing acting."*

Expected Output: **Positive sentiment** with high confidence score

## 📄 License

This project is for educational purposes.

URL: https://imdbreviewsentiment-tzhxfol3q4ajjemveqikso.streamlit.app/
