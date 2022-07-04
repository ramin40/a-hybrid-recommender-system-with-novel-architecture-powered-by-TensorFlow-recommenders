"""# utility functions"""
import numpy as np

def movie_feature_extractor(id_wanted,data):
  for j in data.movie_data.batch(1):
    i = 0
    if j['movie_id'].numpy()[0] == id_wanted:
      i += 1
      if i == 1:
        print('movie features: \n')
        print('title: ', j['movie_title'].numpy()[0])
        print('genre ', j['genre'].numpy()[0])
        print('movie_id ', j['movie_id'].numpy()[0], '\n\n')
        break
def user_feature_extractor(wanted_id,data):
  for j in data.user_data.batch(1):
    i=0
    if j['user_id'].numpy()[0]==wanted_id:
      i+=1
      if i==1:
        print('user features: \n')
        print('age: ',j['age'].numpy()[0] )
        print('gender ',j['gender'].numpy()[0])
        print('occupation ',j['occupation'].numpy()[0])
        print('user_id ',j['user_id'].numpy()[0],'\n\n')
        break

def item_output_cleaner(outputs):
  (unique, counts) = np.unique(outputs.numpy()[0], return_counts=True)
  ziped_file = zip(unique, counts)
  output_array = outputs.numpy()[0]
  for uniques in unique:
    to_be_deleted = np.where(output_array == uniques)[0]
    output_array = np.delete(output_array, to_be_deleted[1:])
  dictionary = dict(ziped_file)
  ys = []

  for i, x in enumerate(output_array):
    ys.append((f'{i + 1}th movie ', x))

  return ys,output_array


def n_first_recommended_movies(results, n):
  for i in results[:n]:
    movie_feature_extractor(i[1])
    print('-------------------------')


def n_last_recommended_movies(results, n):
  for i in results[-n:]:
    movie_feature_extractor(i[1])
    print('-------------------------')



def user_output_cleaner(outputs):
  (unique, counts) = np.unique(outputs.numpy()[0], return_counts=True)
  ziped_file=zip(unique,counts)
  output_array=outputs.numpy()[0]
  for uniques in unique:
    to_be_deleted=np.where(output_array==uniques)[0]
    output_array=np.delete(output_array,to_be_deleted[1:])

  dictionary=dict(ziped_file)
  ys =[]


  for i,x in enumerate(output_array):
    ys.append((f'{i+1}th person ', x))

  return ys,output_array


def input_data_creater_for_item_item(inputs,data):
  input_data=[]
  for results in inputs:
    for i in results:

      wanted_id=i
      for j in data.movie_data.batch(1):
        i=0
        # print(wanted_id,j['movie_id'].numpy()[0])
        if j['movie_id'].numpy()[0]==wanted_id:
          i+=1
          if i==1:
            input_data.append(j)
            break
  return input_data

def n_first_recommended_users(results,n):
  for i in results[:n]:
    user_feature_extractor(i[1])
    print('-------------------------')

def n_last_recommended_users(results,n):
  for i in results[-n:]:
    user_feature_extractor(i[1])
    print('-------------------------')

def data_creator_for_ranker_from_UTI(results, n ,data):
  input_data = []
  for i in results[0][:n]:
    wanted_id = i.numpy()
    for j in data.all_data.batch(1):
      i = 0

      if j['movie_id'].numpy()[0] == wanted_id:
        i += 1
        if i == 1:
          input_data.append(j)
          break
  return input_data


def data_creator_for_ranker_from_ITI(user_data,input,data):
  input_data = []
  user_data=[i for i in user_data][0]
  # input=[array(),array(),...]
  for j in input:
    for i in j:
      id_wanted = i

      for movie_data in data.movie_data.batch(1):
        m = 0
        # id wanted=b'12'
        # print(id_wanted,movie_data['movie_id'].numpy()[0])
        if movie_data['movie_id'].numpy()[0] == id_wanted:
          m += 1
          if m == 1:
            input_data.append(dict(movie_data,**user_data))
            break
  return input_data
def data_creator_for_ranker_from_UTI(results,data,n):
  input_data=[]
  for i in results[0][:n]:
    wanted_id=i.numpy()
    for j in data.all_data.batch(1):
      i=0

      if j['movie_id'].numpy()[0]==wanted_id:

        i+=1
        if i==1:
          input_data.append(j)
          break
  return input_data


def data_creator_for_UTI_from_UTU(inputs,data):
  input_data = []
  for results in inputs:
    for i in results:
      wanted_id = i
      for j in data.all_data.batch(1):
        i = 0
        if j['user_id'].numpy()[0] == wanted_id:
          i += 1
          if i == 1:
            input_data.append(j)
            break
  return input_data
