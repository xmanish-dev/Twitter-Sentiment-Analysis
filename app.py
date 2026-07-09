from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
from preprocess import clean_text
from sentiment import predict_sentiment

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Analysis Dashboard")

with st.expander("📋 Project Information"):

    st.markdown("""
### Twitter Sentiment Analysis Dashboard

This dashboard analyzes Twitter sentiment using the Twitter Entity Sentiment Dataset.

**Technologies Used**

- Python
- Streamlit
- Pandas
- Plotly
- WordCloud
- VADER Sentiment Analysis

**Dataset Size**

73,996 Tweets
""")
# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/twitter_training.csv",
        header=None,
        names=["Tweet_ID", "Entity", "Sentiment", "Tweet"]
    )

    df = df.dropna()

    return df


def find_text_column(uploaded_df):
    for column in ["Tweet", "tweet", "text", "Text", "content", "body"]:
        if column in uploaded_df.columns:
            return column

    return None


def normalize_xquik_export(uploaded_df):
    text_column = find_text_column(uploaded_df)

    if text_column is None:
        return None

    source = uploaded_df.copy()
    tweets = source[text_column].astype(str)

    entity = (
        source["Entity"]
        if "Entity" in source.columns
        else source.get("author", "Xquik export")
    )

    return pd.DataFrame(
        {
            "Tweet_ID": source.get("Tweet_ID", source.index + 1),
            "Entity": entity,
            "Sentiment": tweets.apply(lambda tweet: predict_sentiment(clean_text(tweet))),
            "Tweet": tweets,
        }
    )


df = load_data()

st.success(f"Dataset Loaded Successfully! ({len(df)} Tweets)")

# ---------------- Sidebar ----------------
st.sidebar.header("Search")

search_type = st.sidebar.selectbox(
    "Search By",
    ["Tweet", "Entity", "Sentiment"]
)

keyword = st.sidebar.text_input("Enter Search")
xquik_file = st.sidebar.file_uploader(
    "Upload Xquik CSV export",
    type=["csv"],
    help="Use a CSV with Tweet, text, content, or body columns.",
)

if xquik_file is not None:
    uploaded_df = pd.read_csv(xquik_file)
    normalized_df = normalize_xquik_export(uploaded_df)

    if normalized_df is None:
        st.sidebar.error("CSV needs a Tweet, text, content, or body column.")
    else:
        df = normalized_df.dropna()
        st.sidebar.success(f"Loaded {len(df)} Xquik rows")

if keyword:

    if search_type == "Tweet":
        filtered_df = df[df["Tweet"].str.contains(keyword, case=False, na=False)]

    elif search_type == "Entity":
        filtered_df = df[df["Entity"].str.contains(keyword, case=False, na=False)]

    else:
        filtered_df = df[df["Sentiment"].str.contains(keyword, case=False, na=False)]

else:
    filtered_df = df

st.sidebar.divider()

st.sidebar.header("Dashboard")

st.sidebar.write("Dataset Size")

st.sidebar.success(f"{len(df)} Tweets")

st.sidebar.write("Developed using")

st.sidebar.info("Python + Streamlit")

# ---------------- Statistics ----------------
positive = len(filtered_df[filtered_df["Sentiment"] == "Positive"])
negative = len(filtered_df[filtered_df["Sentiment"] == "Negative"])
neutral = len(filtered_df[filtered_df["Sentiment"] == "Neutral"])
irrelevant = len(filtered_df[filtered_df["Sentiment"] == "Irrelevant"])

col1, col2, col3, col4 = st.columns(4)

total = len(filtered_df)

positive_percent = (positive / total * 100) if total else 0
negative_percent = (negative / total * 100) if total else 0
neutral_percent = (neutral / total * 100) if total else 0
irrelevant_percent = (irrelevant / total * 100) if total else 0

col1.metric(
    "😊 Positive",
    positive,
    f"{positive_percent:.1f}%"
)

col2.metric(
    "😐 Neutral",
    neutral,
    f"{neutral_percent:.1f}%"
)

col3.metric(
    "😡 Negative",
    negative,
    f"{negative_percent:.1f}%"
)

col4.metric(
    "❓ Irrelevant",
    irrelevant,
    f"{irrelevant_percent:.1f}%"
)

# ---------------- Charts ----------------
st.subheader("Sentiment Distribution")

count_df = filtered_df["Sentiment"].value_counts().reset_index()
count_df.columns = ["Sentiment", "Count"]

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        count_df,
        x="Sentiment",
        y="Count",
        title="Bar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.pie(
        count_df,
        values="Count",
        names="Sentiment",
        title="Pie Chart"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tweets Table ----------------
st.subheader("Tweets")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=500
)

st.divider()

st.header("🤖 Live Sentiment Prediction")

user_text = st.text_area("Type any sentence")

if st.button("Predict Sentiment"):

    cleaned = clean_text(user_text)

    result = predict_sentiment(cleaned)

    if result == "Positive":
        st.success("😊 Positive")

    elif result == "Negative":
        st.error("😡 Negative")

    else:
        st.info("😐 Neutral")

st.divider()

st.header("☁️ Word Cloud")

text = " ".join(filtered_df["Tweet"].astype(str))

if text.strip():
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
else:
    st.info("No tweet text is available for the current filters.")

st.divider()

st.header("🔥 Top 10 Most Discussed Entities")

entity_df = (
    filtered_df["Entity"]
    .value_counts()
    .head(10)
    .reset_index()
)

entity_df.columns = ["Entity", "Tweets"]

fig = px.bar(
    entity_df,
    x="Entity",
    y="Tweets",
    color="Tweets",
    title="Top 10 Entities"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.download_button(
    label="📥 Download Filtered Tweets",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_tweets.csv",
    mime="text/csv"
)

st.divider()

st.header("📰 Latest Tweets")

st.dataframe(
    filtered_df.tail(20),
    use_container_width=True
)

st.divider()

st.markdown(
    """
---
### 👨‍💻 Developed By

**Manish Joshi**

B.Tech (IT)

Twitter Sentiment Analysis Dashboard

Using Python, NLP and Streamlit
"""
)
