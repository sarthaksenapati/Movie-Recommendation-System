# Movie-Recommendation-System
Developed a movie recommendation system using collaborative and content-based filtering. Applied SVD, ALS, and cosine similarity for personalized suggestions. Preprocessed data using pandas and NumPy. Built models with scikit-learn and Surprise. Evaluated using RMSE, MAE, and precision-recall. Visualized insights with Matplotlib and Seaborn.


## Overview
This project implements a **Movie Recommendation System** using **collaborative filtering** and **content-based filtering**. It utilizes the **Surprise** library for collaborative filtering and **TF-IDF vectorization** for content-based recommendations.

## Datasets
- `movies.csv` - Contains movie details (movieId, title, genres)
- `ratings.csv` - Contains user ratings (userId, movieId, rating, timestamp)

## Installation
Ensure you have the required libraries installed:
```bash
pip install numpy pandas scikit-surprise scikit-learn matplotlib seaborn
```

## Features
- **Collaborative Filtering**: Uses **Singular Value Decomposition (SVD)** to recommend movies based on user preferences.
- **Content-Based Filtering**: Uses **TF-IDF vectorization** and **cosine similarity** to find similar movies based on genres.

## Usage
1. Load the datasets:
   ```python
   import pandas as pd
   movies = pd.read_csv('movies.csv')
   ratings = pd.read_csv('ratings.csv')
   ```

2. Train a collaborative filtering model:
   ```python
   from surprise import Dataset, Reader, SVD
   from surprise.model_selection import train_test_split
   reader = Reader(rating_scale=(0.5, 5.0))
   data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
   trainset, testset = train_test_split(data, test_size=0.2)
   model = SVD()
   model.fit(trainset)
   ```

3. Get movie recommendations for a user:
   ```python
   user_id = 1
   predictions = [model.predict(user_id, movie_id) for movie_id in movies['movieId'].unique()]
   top_movies = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
   ```

4. Find similar movies using content-based filtering:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
   cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
   ```

## Results
- The **collaborative filtering** model recommends movies based on user rating history.
- The **content-based** approach suggests movies similar in genre to a given movie.

## Future Improvements
- Incorporate hybrid models combining collaborative and content-based filtering.
- Use deep learning techniques for better recommendation accuracy.

## License
This project is open-source and available for use under the MIT License.

