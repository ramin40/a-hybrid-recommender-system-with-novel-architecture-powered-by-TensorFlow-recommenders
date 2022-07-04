
import tensorflow as tf
import numpy as np

ratings=tf.data.experimental.load('Z:/recommender project/data/movielens_ratings')

all_data=ratings.map(lambda x:{
    'age':x['raw_user_age'],
    'time':x['timestamp'],
    'user_id':x['user_id'],
    'movie_id':x['movie_id'],
    'gender':x['user_gender'],
    'occupation':x['user_occupation_text'],
    'movie_title':x['movie_title'],
    'genre':x['movie_genres'][0],
    'rating':x['user_rating']
})
user_data=ratings.map(lambda x:{
    'age':x['raw_user_age'],
    'user_id':x['user_id'],

    'time':x['timestamp'],
    'gender':x['user_gender'],
    'occupation':x['user_occupation_text'],
    'movie_title':x['movie_title'],
    'genre':x['movie_genres'][0],
    'rating':x['user_rating']
})
movie_data=ratings.map(lambda x:{
    'movie_id':x['movie_id'],
    'movie_title':x['movie_title'],
    'genre':x['movie_genres'][0],

})

movie_titles=user_data.map(lambda x:x['movie_title'])
unique_movie_titles=np.unique(np.concatenate(list(movie_titles.batch(1_000))))
user_occupation=user_data.map(lambda x :x['occupation'])
unique_user_occupation=np.unique(np.concatenate(list(user_occupation.batch(1_000))))
movie_genres=user_data.map(lambda x :x['genre'])
unique_movie_genres=np.unique(np.concatenate(list(movie_genres.batch(1_000))))
user_gender=movie_genres=user_data.map(lambda x :x['gender'])
unique_user_gender=np.unique(np.concatenate(list(user_gender.batch(1_000))))
timestamp=user_data.map(lambda x : x['time'])
min_timestamp=np.unique(np.concatenate(list(timestamp.batch(1_000)))).min()
max_timestamp=np.unique(np.concatenate(list(timestamp.batch(1_000)))).max()
time_bucket=np.linspace(min_timestamp,max_timestamp,1000)
ages=user_data.map(lambda x : x['age'])

embedding_dim=128