import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movieData = pd.read_csv('movies.csv')

# print(movieData.columns)

# feature selection from our dataframe
selected_features = ['genres', 'keywords',
                     'tagline', 'cast', 'director', 'overview']
# print(selected_features)

# replacing all num values present in our selected features
for f in selected_features:

    movieData[f] = movieData[f].fillna(' ')

# combining all columns of the selected features
combined = movieData['genres'] + ' ' + movieData['keywords'] + ' ' + movieData['tagline'] + \
    ' ' + movieData['cast'] + ' ' + \
    movieData['director'] + ' ' + movieData['overview']


# print(combined)

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer .fit_transform(combined)

# print(feature_vectors)

'''Next Part : cosine Similarity'''

# getting the similarity score

similarity = cosine_similarity(feature_vectors)
# print(similarity.shape)

# getting the movie name from user input

movieName = input("Enter one of your favourite movie name : ")

# creating a list of all the movie names in the dataset

movietitle_list = movieData['title'].tolist()
# print(movietitle_list)

# will find the close match for our movie name given by user

close_match = difflib.get_close_matches(movieName, movietitle_list)

closest_match = close_match[0]

# finding index of movie(closest_match) with title

index = movieData[movieData.title == closest_match]['index'].values[0]
# print(index)

# getting a list of similar movies

similarity_score = list(enumerate(similarity[index]))

'''sorting the movies from high score value to low score value to get easily similar movies.'''

sorted_similarity_score = sorted(
    similarity_score, key=lambda x: x[1], reverse=True)


# next part : Printing the names of suggested movies for user based on index

i = 1

for movie in sorted_similarity_score:

    index = movie[0]
    title_from_index = movieData[movieData.index == index]['title'].values[0]

    if (i < 25):

        print(i, '.', title_from_index)
        i += 1
