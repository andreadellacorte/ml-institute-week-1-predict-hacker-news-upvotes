# Hacker News Upvote Predictor

ml.institute project to predict Hacker News Upvotes

## Project Structure

```
project-root/
├── data/
│   ├── raw/           # Database extracts (items, users)
│   └── processed/     # Cleaned datasets, embeddings
├── notebooks/
│   └── EDA_HackerNews.ipynb      # Exploratory Data Analysis
├── src/
│   ├── training/      # Word2Vec and model training code
│   ├── utils/         # Helper functions and data processing
│   └── api/           # FastAPI service for predictions
├── Dockerfile
├── requirements.txt
└── README.md
```