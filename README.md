# IPL Analytics & Match Predictor

An interactive cricket analytics and machine learning application to predict Indian Premier League (IPL) match outcomes and view player career statistics. Built using historical ball-by-ball IPL dataset from [Cricsheet](https://cricsheet.org/).

### Live Web Application
Check out the live app here: [https://ipl-win-prediction-ff9ghxjaaz2rwncvavbitl.streamlit.app/](https://ipl-win-prediction-ff9ghxjaaz2rwncvavbitl.streamlit.app/)

##  Project Structure

```text
 IPL Analytics
│
├── data/
│   ├── raw/                  # Downloaded CSV dataset from Cricsheet
│   └── processed/            # Cleaned matches and deliveries CSV files
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb         # Unzipping and cleaning data
│   ├── 02_EDA.ipynb                   # Basic Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb   # Preparing features for models
│   ├── 04_match_prediction.ipynb      # Training the match winner classifier
│   ├── 05_player_prediction.ipynb     # Preparing player performance summaries
│   └── 06_visualizations.ipynb        # Sample plotting of match attributes
│
├── models/
│   ├── match_predictor.pkl   # Saved Logistic Regression pipeline for match prediction
│   ├── player_predictor.pkl  # Processed player career statistics lookup
│   └── train_models.py       # Utility python script to train models and save pickles
│
├── app/
│   ├── streamlit_app.py      # Streamlit interactive application
│   └── requirements.txt      # Python dependencies list
│
└── README.md                 # Project instructions and description
```

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Install Dependencies
Navigate to the project root directory and run:
```bash
pip install -r app/requirements.txt
```

### 3. Data Download & Preprocessing
To automatically download and prepare the Cricsheet IPL CSV dataset:
```bash
# Download and unzip raw files
python data/download_data.py

# Clean and combine datasets
python data/clean_data.py
```
This generates the cleaned CSV files inside `data/processed/`.

### 4. Train Models
To train the Logistic Regression pipeline and compute player career stats:
```bash
python models/train_models.py
```
This saves the trained pickle models in the `models/` directory.

### 5. Launch the Streamlit App
To start the interactive web application, run:
```bash
streamlit run app/streamlit_app.py
```
A browser window will automatically open showing the prediction interface.

---

## 🔬 Model Details
- **Match Winner Predictor**: A scikit-learn `Pipeline` consisting of a `OneHotEncoder` and a `LogisticRegression` classifier. It uses the teams, venue, toss winner, and toss decision to predict the probability of each team winning.
- **Player Stats Lookup**: A computed dictionary of batting (runs, average, strike rate) and bowling (wickets, economy, average wickets per match) profiles compiled from historical IPL matches.
