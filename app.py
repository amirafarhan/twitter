"""Streamlit demo for the Twitter entity-sentiment LSTM.

Run with:  streamlit run app.py   (train first with: python twitter_sentiment.py)
"""

import altair as alt
import pandas as pd
import streamlit as st

import twitter_sentiment as ts

MAX_CHARS = 1000
MAX_BATCH = 1000
COLORS = {"Negative": "#e4572e", "Positive": "#2a9d8f", "Neutral": "#8d99ae", "Irrelevant": "#e9c46a"}
COLOR_SCALE = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))
EXAMPLES = [
    "I absolutely love the new Borderlands update, the gameplay is amazing!",
    "Worst customer service ever. My order never arrived and nobody answers.",
    "Microsoft announced its quarterly results today.",
]

st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return ts.load_artifacts()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return ts.load_data()


def run_predictions(texts):
    probs, tokens = ts.predict(texts, model, tok, meta["maxlen"])
    df = pd.DataFrame(probs, columns=ts.LABELS)
    df.insert(0, "prediction", df[ts.LABELS].idxmax(axis=1))
    df.insert(1, "confidence", df[ts.LABELS].max(axis=1))
    df["known_tokens"] = [sum(1 for w in t if tok.word_index.get(w, ts.MAX_VOCAB) < ts.MAX_VOCAB) for t in tokens]
    return df, tokens


try:
    model, tok, meta = get_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.sidebar:
    st.header("🧠 Model")
    st.metric("Test accuracy", f"{meta['test_accuracy']:.1%}")
    st.caption(
        f"Embedding(100) → LSTM(100) → Dense(32) → Dense(4)  \n"
        f"Vocabulary: {meta['vocab_size']:,} words · max length: {meta['maxlen']} tokens  \n"
        f"Trained on {meta['n_train']:,} tweets, tested on {meta['n_test']:,}"
    )

st.title("🐦 Tweet Sentiment Analyzer")
st.caption("LSTM classifier trained on entity-level Twitter sentiment (Negative / Positive / Neutral / Irrelevant).")

tab_single, tab_batch, tab_model, tab_data = st.tabs(
    ["🔍 Analyze a tweet", "📄 Batch analysis", "📈 Model performance", "🗂️ Dataset"]
)

# ---------------- Single tweet ----------------
with tab_single:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex[:45] + "…", width="stretch", help=ex):
            st.session_state.tweet = ex
    text = st.text_area("Tweet text", key="tweet", height=110, max_chars=MAX_CHARS,
                        placeholder="Type or paste a tweet...")
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            res, tokens = run_predictions([text])
            row = res.iloc[0]
            if row["known_tokens"] == 0:
                st.warning("None of the words in this text are in the model's vocabulary, "
                           "so the prediction is not meaningful.")
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicted sentiment", row["prediction"], f"{row['confidence']:.1%} confidence",
                      delta_color="off")
            c1.write("**Cleaned tokens used by the model:**")
            c1.write(", ".join(f"`{t}`" for t in tokens[0]) or "_(none)_")
            probs = pd.DataFrame({"sentiment": ts.LABELS, "probability": [row[l] for l in ts.LABELS]})
            c2.altair_chart(
                alt.Chart(probs).mark_bar().encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("sentiment:N", sort=ts.LABELS, title=None),
                    color=alt.Color("sentiment:N", scale=COLOR_SCALE, legend=None),
                    tooltip=["sentiment", alt.Tooltip("probability:Q", format=".1%")],
                ).properties(title="Class probabilities", height=220),
                width="stretch",
            )

# ---------------- Batch ----------------
with tab_batch:
    st.write(f"Enter one tweet per line, or upload a CSV with a `text` column (max {MAX_BATCH} rows).")
    lines = st.text_area("Tweets (one per line)", height=150)
    upload = st.file_uploader("…or upload CSV", type="csv")
    if st.button("Analyze batch"):
        try:
            if upload is not None:
                frame = pd.read_csv(upload)
                if "text" not in frame.columns:
                    raise ValueError("CSV must contain a 'text' column.")
                texts = frame["text"].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in lines.splitlines() if l.strip()]
            if not texts:
                st.warning("No tweets to analyze.")
            else:
                if len(texts) > MAX_BATCH:
                    st.info(f"Only the first {MAX_BATCH} rows are analyzed.")
                    texts = texts[:MAX_BATCH]
                with st.spinner(f"Analyzing {len(texts)} tweets..."):
                    res, _ = run_predictions(texts)
                res.insert(0, "text", texts)
                left, right = st.columns([2, 1])
                left.dataframe(
                    res.drop(columns="known_tokens"), hide_index=True, width="stretch",
                    column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                                   for c in ["confidence", *ts.LABELS]},
                )
                counts = res["prediction"].value_counts().reindex(ts.LABELS, fill_value=0).reset_index()
                counts.columns = ["sentiment", "tweets"]
                right.altair_chart(
                    alt.Chart(counts).mark_arc(innerRadius=50).encode(
                        theta="tweets:Q", color=alt.Color("sentiment:N", scale=COLOR_SCALE),
                        tooltip=["sentiment", "tweets"],
                    ).properties(title="Sentiment distribution", height=280),
                    width="stretch",
                )
                st.download_button("Download results", res.to_csv(index=False), "sentiment_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not analyze the batch: {exc}")

# ---------------- Model performance ----------------
with tab_model:
    report = pd.DataFrame(meta["classification_report"]).T
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{meta['test_accuracy']:.1%}")
    c2.metric("Macro F1", f"{report.loc['macro avg', 'f1-score']:.3f}")
    val_loss = meta["history"]["val_loss"]
    c3.metric("Epochs run", len(val_loss), f"best weights: epoch {val_loss.index(min(val_loss)) + 1}",
              delta_color="off")
    st.dataframe(report.loc[ts.LABELS + ["macro avg", "weighted avg"]].style.format("{:.3f}"),
                 width="stretch")

    left, right = st.columns(2)
    cm = pd.DataFrame(meta["confusion_matrix"], index=ts.LABELS, columns=ts.LABELS)
    cm_long = cm.reset_index(names="actual").melt("actual", var_name="predicted", value_name="count")
    base = alt.Chart(cm_long).encode(x=alt.X("predicted:N", sort=ts.LABELS), y=alt.Y("actual:N", sort=ts.LABELS))
    left.altair_chart(
        (base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")))
         + base.mark_text().encode(text="count:Q"))
        .properties(title="Confusion matrix (test set)", height=320),
        width="stretch",
    )
    hist = pd.DataFrame(meta["history"]).reset_index(names="epoch")
    hist["epoch"] += 1
    acc = hist.melt("epoch", value_vars=["accuracy", "val_accuracy"], var_name="split", value_name="score")
    right.altair_chart(
        alt.Chart(acc).mark_line(point=True).encode(x="epoch:O", y=alt.Y("score:Q", title="accuracy", scale=alt.Scale(zero=False)),
                                                    color="split:N")
        .properties(title="Training history", height=320),
        width="stretch",
    )

# ---------------- Dataset ----------------
with tab_data:
    counts = pd.Series(meta["label_counts"]).reindex(ts.LABELS).reset_index()
    counts.columns = ["sentiment", "tweets"]
    st.altair_chart(
        alt.Chart(counts).mark_bar().encode(
            x="tweets:Q", y=alt.Y("sentiment:N", sort="-x", title=None),
            color=alt.Color("sentiment:N", scale=COLOR_SCALE, legend=None),
        ).properties(title="Label distribution (after removing empty and duplicate rows)", height=200),
        width="stretch",
    )
    label = st.selectbox("Show sample tweets for", ts.LABELS)
    data = get_dataset()
    st.dataframe(data[data["labels"] == label].sample(10, random_state=0), hide_index=True, width="stretch")
