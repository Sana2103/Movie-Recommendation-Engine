# Movie Recommendation Engine
# Project Overview
This project is a simple movie recommendation engine built using Python, leveraging the power of the Cosine Similarity algorithm to recommend movies based on user preferences. It focuses on extracting meaningful features from a movie dataset and calculating the similarity between movies to provide personalized recommendations.
# Key Features
●	Extracts important features like keywords, cast, genres, and director to create a comprehensive representation of each movie.
●	Uses CountVectorizer for feature extraction and Cosine Similarity to calculate similarity scores between movies.
●	Recommends similar movies based on user input.
●	Efficient and lightweight, suitable for small to medium-sized datasets.
# Project Structure
The project consists of two main components:
# 1. Cosine Similarity Calculation (cosine_similarity.py)
This script demonstrates the core concept of cosine similarity, which is the backbone of this recommendation engine.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)

●	CountVectorizer: Transforms the text into a matrix of token counts.
●	Cosine Similarity: Computes the similarity scores between text vectors to determine how closely they are related.
# 2. Movie Recommender System (movie_recommender_completed.py)
This script implements the complete movie recommendation logic, including data preprocessing, feature extraction, and similarity calculation.
Steps Involved:
Step 1: Load Movie Dataset
Loads the movie dataset using Pandas.
import pandas as pd
import numpy as np

df = pd.read_csv("movie_dataset.csv", low_memory=False)

Step 2: Select Key Features
Selects relevant features like keywords, cast, genres, and director.
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')

Step 3: Combine Features
Combines the selected features into a single column to create a comprehensive feature set.
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]

df["combined_features"] = df.apply(combine_features, axis=1)

Step 4: Generate Count Matrix
Creates a count matrix from the combined features.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

Step 5: Compute Cosine Similarity
Calculates the cosine similarity matrix for all movies.
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix)

Step 6: Find Similar Movies
Identifies movies similar to the user’s selected movie (e.g., "Avatar").
movie_user_likes = "Avatar"
movie_index = df[df.title == movie_user_likes].index.values[0]
similar_movies = list(enumerate(cosine_sim[movie_index]))

Step 7: Sort and Display Results
Sorts the similar movies in descending order of similarity and displays the top 50 recommendations.
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

for i, element in enumerate(sorted_similar_movies[:50]):
    print(df.iloc[element[0]].title)

# How to Run the Project
1.	Clone the repository.
2.	Install the required Python packages:
pip install pandas numpy scikit-learn

3.	Place the movie_dataset.csv file in the project directory.
4.	Run the movie_recommender_completed.py script.
Future Improvements
●	Integrate with a user interface for better usability.
●	Add support for more advanced algorithms like collaborative filtering.
●	Implement real-time recommendations using a backend server.
# Conclusion
This project provides a straightforward approach to building a content-based recommendation engine using Python. It serves as a foundational project for learning recommendation systems and can be expanded for real-world applications.
Feel free to contribute or suggest improvements!
License
This project is open-source and available under the MIT License.

