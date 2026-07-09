# 🐦 Twitter Sentiment Analysis Dashboard

A web-based dashboard for analyzing Twitter sentiment using Python and NLP.

## Features

- 🔍 Search tweets
- 😊 Sentiment summary
- 📊 Bar chart
- 🥧 Pie chart
- ☁️ Word cloud
- 🤖 Live sentiment prediction
- 📥 Download filtered results
- 📄 Upload Xquik CSV exports for the same dashboard views

## Technologies

- Python
- Streamlit
- Pandas
- Plotly
- WordCloud
- VADER Sentiment Analysis

## Project Structure

```
Twitter-Sentiment-Analysis/
│
├── app.py
├── preprocess.py
├── sentiment.py
├── twitter_api.py
├── config.py
├── requirements.txt
├── data/
└── README.md
```

## Installation

```bash
git clone https://github.com/xmanish-dev/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
pip install -r requirements.txt
python -m streamlit run app.py
```

## Xquik CSV Workflow

Export X post search or monitor results from Xquik as CSV, then upload the file
from the sidebar. The importer accepts `Tweet`, `text`, `content`, or `body`
columns, runs the existing VADER sentiment classifier, and keeps the same
filters, charts, word cloud, and download button.

## Author

**Manish Joshi**
