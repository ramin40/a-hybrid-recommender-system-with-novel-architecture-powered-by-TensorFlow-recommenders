

import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt

def model_creator(data,layers,phase,saving_filepath):
    assert phase == 'training' or phase == 'deployment' or phase == None, f"phase must be [training] or [deployment] but got: {phase}"

    # age normalizer
    age_normalizer=tf.keras.layers.Normalization(
        axis=None
    )
    ages=data.user_data.map(lambda x :x['age'])


    embedding_dim=data.embedding_dim
    # gender model
    gender_model=tf.keras.Sequential(
        [tf.keras.layers.IntegerLookup(vocabulary=[2])
        ,tf.keras.layers.Embedding(3,2)

        ]
    )


    # occupation model
    occupation_model=tf.keras.Sequential(
        [tf.keras.layers.StringLookup(vocabulary=data.unique_user_occupation),
         tf.keras.layers.Embedding(len(data.unique_user_occupation)+1,embedding_dim)
        ]
    )
    if phase=='training':
        age_normalizer.adapt(ages)
    """# query tower
    
    * user model
    """

    class UserModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.age_normalizer=age_normalizer
        self.gender_model=gender_model
        self.occupation_model=occupation_model


      def call(self,inputs):
        out=tf.concat(
            [tf.reshape(self.age_normalizer(inputs['age']),(-1,1)),
             tf.cast(self.gender_model(tf.cast(inputs['gender'],dtype=tf.float32)),dtype=tf.float32),
             self.occupation_model(inputs['occupation']),
            ],axis=1
        )
        return out

    class QueryTower(tf.keras.Model):
      def __init__(self,layers):
        super().__init__()
        self.user_model=UserModel()
        self.dense_model=tf.keras.Sequential()
        for layer in layers[:-1]:
          self.dense_model.add(tf.keras.layers.Dense(layer,activation='relu'))
        for layer in layers[-1:]:
          self.dense_model.add(tf.keras.layers.Dense(layer))
      def call(self,inputs):
        v=self.user_model(inputs)
        return self.dense_model(v)

    """# candidate tower
    * movie model
    """

    from typing import cast
    class MovieModel(tf.keras.Model):

        def __init__(self):
          super().__init__()
          self.age_normalizer=age_normalizer
          self.gender_model=gender_model
          self.occupation_model=occupation_model


        def call(self,inputs):

          out=tf.concat(
            [tf.reshape(self.age_normalizer(inputs['age']),(-1,1)),
             tf.cast(self.gender_model(inputs['gender']),dtype=tf.float32),
             self.occupation_model(inputs['occupation']),
            ],axis=1
        )
          return out

    """* candidate tower"""

    class CandidateTower(tf.keras.Model):
      def __init__(self,layers):
        super().__init__()
        self.user_model=UserModel()
        self.dense_model=tf.keras.Sequential()
        for layer in layers[:-1]:
          self.dense_model.add(tf.keras.layers.Dense(layer,activation='relu'))
        for layer in layers[-1:]:
          self.dense_model.add(tf.keras.layers.Dense(layer))
      def call(self,inputs):
        v=self.user_model(inputs)
        return self.dense_model(v)

    class CombinedModels(tfrs.models.Model):
      def __init__(self,layers):
        super().__init__()
        self.query_tower=QueryTower(layers)
        self.candidate_tower=CandidateTower(layers)
        self.task=tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=data.user_data.batch(128).map(self.candidate_tower)
            )
        )
      def compute_loss(self, features,training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings=self.query_tower({
            'age':features['age'],
            'gender':features['gender'],
            'occupation':features['occupation'],
        })
        candidate_embeddings=self.candidate_tower({
            'age':features['age'],
            'gender':features['gender'],
            'occupation':features['occupation'],
        })
        return self.task(query_embeddings,candidate_embeddings,compute_metrics=not training)

    """# preparing data to train model"""
    model = CombinedModels(layers)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    if phase=='training':

        tf.random.set_seed(123)
        shuffled = data.user_data.shuffle(1_000, reshuffle_each_iteration=False)

        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(10_000)
        cached_train = train.shuffle(100_000).batch(2048)
        cached_test = test.batch(4096).cache()

        num_epochs = input('isert number of epochs ')
        num_epochs=int(num_epochs)


        one_layer_history = model.fit(
            cached_train,
            validation_data=cached_test,
            validation_freq=5,
            epochs=num_epochs,
            verbose=1)
        accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
        print('========================================')
        print(f"Top-100 accuracy: {accuracy:.2f}.")
        print('\n\n')
        plt.title('loss')
        plt.plot(one_layer_history.history["total_loss"])
        plt.show()
        model.save_weights(saving_filepath)
    if phase=='deployment':
        model.load_weights(saving_filepath)
    return model


    """# making prediction"""
def get_output(model,data,num_retrive_users,input,phase,num_returne_users):
    assert phase == 'training' or phase == 'deployment' or phase == None, f"phase must be [training] or [deployment] but got: {phase}"
    import utility_functions as utilities
    top_k=num_retrive_users
    index=tfrs.layers.factorized_top_k.BruteForce(model.query_tower,k=top_k)
    index.index_from_dataset(
        tf.data.Dataset.zip((data.user_data.map(lambda x : x['user_id']).batch(100),
                             data.user_data.batch(100).map(model.candidate_tower)))
    )
    if phase=='training':
        d = [i for i in data.user_data.batch(1).take(20)]
        print(f"{top_k} users similar to user:{d[1]['user_id'].numpy()[0]} with below features \n\n")
        utilities.user_feature_extractor(d[1]['user_id'].numpy()[0],data)
        _,recommes=index(d[1])
        dict,raw_array=utilities.user_output_cleaner(recommes)
        print(dict)

    if phase=='deployment':
        recommended_users=[]
        for i in input:

            _, recommes = index(i)
            dict, raw_array = utilities.user_output_cleaner(recommes)
            recommended_users.append(raw_array[:num_returne_users])

        return recommended_users



