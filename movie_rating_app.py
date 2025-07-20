import pandas as pd
import numpy as np
import ast
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    data = movies.merge(credits, on='title')
    data = data[['title', 'genres', 'cast', 'crew', 'runtime', 'popularity', 'budget', 'revenue', 'vote_average']]
    data.dropna(inplace=True)

    # Extract genre
    def extract_genre(x):
        try:
            genres = ast.literal_eval(x)
            return genres[0]['name'] if genres else "Unknown"
        except:
            return "Unknown"

    data['main_genre'] = data['genres'].apply(extract_genre)

    # Extract director
    def extract_director(x):
        try:
            crew = ast.literal_eval(x)
            for person in crew:
                if person['job'] == 'Director':
                    return person['name']
            return "Unknown"
        except:
            return "Unknown"

    data['director'] = data['crew'].apply(extract_director)

    # Extract lead actor
    def extract_actor(x):
        try:
            cast = ast.literal_eval(x)
            return cast[0]['name'] if cast else "Unknown"
        except:
            return "Unknown"

    data['lead_actor'] = data['cast'].apply(extract_actor)

    return data

# Preprocess and train model
@st.cache_resource
def train_model(data):
    df = pd.get_dummies(data, columns=['main_genre', 'director', 'lead_actor'], drop_first=True)
    X = df.drop(['title', 'genres', 'cast', 'crew', 'vote_average'], axis=1)
    y = df['vote_average']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, df

# Load and train
data = load_data()
model, scaler, df = train_model(data)

# Streamlit UI
st.set_page_config(page_title="Movie Rating Predictor", layout="centered")
st.title("🎬 Movie Rating Predictor")
st.markdown("Predicts a movie rating using ML trained on TMDB dataset")

movie_name = st.text_input("Enter Movie Title:")

if movie_name:
    row = df[df['title'].str.lower() == movie_name.lower()]
    if not row.empty:
        X_input = row.drop(['title', 'genres', 'cast', 'crew', 'vote_average'], axis=1)
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        actual = row['vote_average'].values[0]

        st.success(f"⭐ Actual Rating: {actual}")
        st.info(f"🤖 Predicted Rating: {round(prediction, 2)}")
    else:
        st.error("Movie not found in the dataset. Try another title.")
