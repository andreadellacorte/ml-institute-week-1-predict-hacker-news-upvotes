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

## Setup Instructions

1. **Clone the repository**

2. **Install Miniconda**

2. **Set up a Conda environment**
   ```bash
   # Create a virtual environment in the project directory
   conda create -n ml-institute-week1-project python=3.11
   
   # Activate the virtual environment
   conda activate ml-institute-week1-project
   ```

3. **Install dependencies**
   ```bash
   # Make sure your virtual environment is activated
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit the .env file with your database credentials if needed
   # The default values should work for team members
   ```

5. **Extract data from the database**
   ```bash
   python src/utils/eda.py
   ```
   This will:
   - Connect to the Hacker News database using credentials from your .env file
   - Extract 100k items (sorted by time, newest first)
   - Extract user data for authors of those items
   - Save the data to parquet files in `data/raw/`
   - Save the processed data into `data/processed/`

6. **Run the EDA notebook**
   ```bash
   # If you haven't installed Jupyter yet, install it in your virtual environment
   pip install jupyter
   
   # Start Jupyter notebook
   jupyter notebook notebooks/EDA_HackerNews.ipynb
   ```

## Project Workflow

1. **Data Collection**: Extract and save data from the Hacker News database using the data extraction script.

2. **Exploratory Data Analysis**: Analyze post-level and user-level features to understand patterns in the data.

3. **Word2Vec Training**: 
   - Pre-train Word2Vec models on Wikipedia corpus
   - Fine-tune on Hacker News titles
   
4. **Feature Engineering**:
   - Extract title embeddings
   - Create user-level features (karma, account age)
   - Add time-based features (hour, day of week)
   - Extract domain information

5. **Model Training**:
   - Combine all features
   - Train an MLP regression model
   - Evaluate performance
   
6. **Deployment**:
   - Package model and dependencies in Docker container
   - Deploy FastAPI service
   - Test predictions with sample posts

## Database Connection Details

The database connection details are stored in the `.env` file. The default values are:

- Host: 178.156.142.230
- Port: 5432
- Database: hd64m1ki
- Schema: hacker_news
- Tables: items, users

## Model Features

### Post-level Features
- Title embeddings (via Word2Vec)
- Title length and word count
- Post time (hour, day of week)
- Domain

### User-level Features
- Author karma
- Account age at post time

## Contributors

- @andreadellacorte