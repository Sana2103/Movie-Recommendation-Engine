# Movie Recommendation Engine

### Project Overview

This project is a content-based movie recommendation engine built in Python. It suggests movies to users based on their preferences by analyzing key features like genres, cast, director, and keywords.

The system uses **TF-IDF Vectorization** and **Cosine Similarity** to measure how similar movies are to one another and provides personalized recommendations based on a movie the user likes.

### Key Features

  - **Feature Engineering**: Combines key features (`genres`, `cast`, `director`, `keywords`) to create a unique profile for each movie.
  - **TF-IDF Vectorization**: Uses `TfidfVectorizer` to convert text features into a meaningful numerical format that emphasizes important words.
  - **Cosine Similarity**: Calculates a similarity score between all movies to find the most relevant recommendations.
  - **Interactive**: Prompts the user to enter a movie title and delivers a ranked list of recommendations.

-----

### How It Works

The recommendation logic follows these steps:

1.  **Load Data**: The `movie_dataset.csv` is loaded into a pandas DataFrame.
2.  **Clean and Combine Features**:
      - Key features (`keywords`, `cast`, `genres`, `director`) are selected, and any missing values are filled with empty strings.
      - These features are combined into a single string called `"combined_features"`.
3.  **Vectorize Text**:
      - `TfidfVectorizer` is applied to the `"combined_features"`. This converts the text into a matrix of TF-IDF features, which helps in identifying the most significant words for each movie.
4.  **Compute Similarity**:
      - The `cosine_similarity` function is used on the TF-IDF matrix to create a similarity score for every pair of movies.
5.  **Get Recommendations**:
      - The user is prompted to enter a movie title they like.
      - The system finds the movie in the dataset and retrieves its similarity scores against all other movies.
      - It then sorts these scores in descending order and displays the **top 10** most similar movies as recommendations.

-----

### How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sana2103/Movie-Recommendation-Engine.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Movie-Recommendation-Engine
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install pandas scikit-learn
    ```
4.  **Run the script:**
      - Make sure the `movie_dataset.csv` file is in the same directory.
      - Run the movie recommender script from your terminal:
        ```bash
        python movie_recommender.py
        ```
      - When prompted, enter a movie title and press Enter to get your recommendations.

-----

### License

This project is open-source and available under the **MIT License**.

