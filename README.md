# 🐦 Twitter Sentiment Analysis (LSTM)

An NLP project that classifies tweets about brands and games into four sentiment classes —
**Negative, Positive, Neutral, Irrelevant** — using NLTK text preprocessing and a Keras
**Embedding + LSTM** neural network. Includes an interactive **Streamlit** demo.

**🔗 Live demo:** https://tweet-sentiment-89.streamlit.app/

## 🚀 Features

- Tweet cleaning: Unicode normalisation, lower-casing, removal of URLs, mentions, HTML, digits and
  punctuation, hashtag text kept, stop-word removal, lemmatisation, duplicate-word removal
- Keras `Tokenizer` (20,000-word vocabulary) and padded sequences
- LSTM classifier: `Embedding(100) → LSTM(100) → Dense(32, relu) → Dense(4, softmax)`
- Trained model saved in `models/` and reused by the app
- **Streamlit demo**:
  - Analyze a tweet: predicted class, confidence, class probabilities, cleaned tokens
  - Batch analysis: one tweet per line or CSV upload (`text` column), distribution chart, CSV download
  - Model performance: accuracy, per-class precision/recall/F1, confusion matrix, training history
  - Dataset: label distribution and sample tweets

## 📊 Dataset & Results

`twitter.csv` – 75,680 labelled tweets (`labels`, `text`). After removing empty and duplicate rows,
70,251 tweets remain, split 80/20 (`random_state=42`).

| Metric (test set, 14,051 tweets) | Value |
|---|---|
| Accuracy | 82.8% |
| Macro F1 | 0.824 |

Training used `validation_split=0.1` and early stopping on validation loss (best weights: epoch 2).

> **Note:** the dataset contains many paraphrased copies of the same tweet. About 20% of test tweets
> have an identical cleaned text in the training set, so the test score is likely optimistic for
> completely new tweets.

## 🛠️ Tech Stack

Python · pandas · NumPy · NLTK · scikit-learn · TensorFlow / Keras · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
twitter/
├── twitter.csv              # Dataset
├── twitter.ipynb            # Original exploration / modelling notebook
├── twitter_sentiment.py     # Preprocessing, training, saving and prediction
├── app.py                   # Streamlit demo
├── models/
│   ├── sentiment_lstm.keras # Trained model
│   ├── tokenizer.json       # Fitted Keras tokenizer
│   └── metadata.json        # maxlen, labels, metrics, confusion matrix, history
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/twitter.git
cd twitter
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

The required NLTK data (`punkt_tab`, `stopwords`, `wordnet`) is downloaded automatically on first use.

## ▶️ Run

Live demo: https://tweet-sentiment-89.streamlit.app/

Run the Streamlit demo locally (uses the trained model in `models/`):

```bash
streamlit run app.py
```

Re-train the model and regenerate `models/` (about 3–5 minutes on CPU):

```bash
python twitter_sentiment.py
```

Notebook: `pip install jupyter`, run `jupyter notebook`, then open `twitter.ipynb`.

## 🔮 Future Improvements

- Group paraphrased tweets when splitting to get a stricter test estimate
- Use the entity/topic column of the original dataset as an extra input
- Compare with a TF-IDF + linear baseline and with transformer models (e.g. BERT)

## 👨‍💻 Author

Amira Farhan – Machine Learning & Computer Vision Engineer
