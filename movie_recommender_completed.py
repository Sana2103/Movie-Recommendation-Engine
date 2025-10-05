import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title].index.values[0]
##################################################

## Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv", low_memory=False)

## Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']

## Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        pass

df["combined_features"] = df.apply(combine_features, axis=1)

## Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

## Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

## Step 6: Ask user for a movie title
movie_user_likes = input("🎬 Enter your favorite movie name: ")

try:
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    ## Step 7: Sort by similarity
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    ## Step 8: Print top 10 similar movies
    print(f"\nTop 10 movies similar to '{movie_user_likes}':\n")
    i = 0
    for element in sorted_similar_movies:
        print(get_title_from_index(element[0]))
        i += 1
        if i > 10:
            break
except:
    print("❌ Movie not found in dataset! Try again with a different name.")


