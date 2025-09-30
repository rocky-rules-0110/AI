import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import time
import sys

# Load and preprocess the dataset
def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

movies_df = load_data()

# Vectorize the combined features and compute cosine similarity
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# List all unique genres
def list_genres(df):
    return sorted(set(genre.strip() for sublist in df["Genre"].dropna().str.split(", ") for genre in sublist))

genres = list_genres(movies_df)

# Recommend movies based on filters (genre, mood, rating)
def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df
    if genre:
        filtered_df = filtered_df[filtered_df["Genre"].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df["IMDB_Rating"] >= rating]
    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)  # Shuffle the rows

    recommendations = []
    for idx, row in filtered_df.iterrows():
        overview = row["Overview"]
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if mood and ((TextBlob(mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0):
            recommendations.append((row["Series_Title"], polarity))
        elif not mood:
            recommendations.append((row["Series_Title"], polarity))
        if len(recommendations) == top_n:
            break
    return recommendations if recommendations else "No suitable movie recommendations found."

# Display recommendations
def display_recommendations(recs, name):
    print(f"\nAI-Analyzed Movie Recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        print(f"{idx}. {title} (Polarity: {polarity:.2f}, {sentiment})")

# Small processing animation
def processing_animation():
    for _ in range(3):
        print(".", end="", flush=True)
        time.sleep(0.5)
    print()

# Handle AI recommendation flow
def handle_ai(name):
    print(f"\nLet's find the perfect movie for you, {name}!\n")

    print("Available genres:")
    for idx, genre in enumerate(genres, 1):
        print(f"{idx}. {genre}")
    print()

    while True:
        genre_input = input("Enter genre number or name: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print("\nInvalid input. Try again.\n")

    mood = input("How do you feel today? (Describe your mood): ").strip()
    print("\nAnalyzing your mood and finding movies...")
    processing_animation()

    recs = recommend_movies(genre=genre, mood=mood)
    if isinstance(recs, str):  # No recommendations found
        print(recs)
    else:
        display_recommendations(recs, name)

# Main program
if __name__ == "__main__":
    print("Welcome to the AI Movie Recommender 🎥🍿")
    user_name = input("Enter your name: ").strip()
    handle_ai(user_name)

    
