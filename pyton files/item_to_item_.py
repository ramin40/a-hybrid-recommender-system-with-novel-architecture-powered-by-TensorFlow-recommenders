import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
def model_creator(data,layers,phase,saving_filepath):


    assert phase == 'training' or phase == 'deployment' or phase == None, f"phase must be [training] or [deployment] but got: {phase}"
    embedding_dim=data.embedding_dim
    # genre model
    genre_model=tf.keras.Sequential(
        [tf.keras.layers.IntegerLookup(vocabulary=data.unique_movie_genres),
        tf.keras.layers.Embedding(len(data.unique_movie_genres)+1,embedding_dim)
        ]
        )


    # movie title model
    movie_model=tf.keras.Sequential(
        [tf.keras.layers.StringLookup(vocabulary=data.unique_movie_titles),
        tf.keras.layers.Embedding(len(data.unique_movie_titles)+1,embedding_dim)
        ]
            )

    class MovieModel(tf.keras.Model):
      def __init__(self):
        super().__init__()

        self.genre_model=genre_model
        self.movie_model=movie_model

      def call(self,inputs):
        out=tf.concat(
            [
            self.genre_model(inputs['genre']),
            self.movie_model(inputs['movie_title']),
            ],axis=1
        )
        return out

    # with strategy.scope():
    class QueryTower(tf.keras.Model):
      def __init__(self,layers):
        super().__init__()
        self.movie_model=MovieModel()
        self.dense_model=tf.keras.Sequential()
        for layer in layers[:-1]:
          self.dense_model.add(tf.keras.layers.Dense(layer,activation='relu'))
        for layer in layers[-1:]:
          self.dense_model.add(tf.keras.layers.Dense(layer))
      def call(self,inputs):
        v=self.movie_model(inputs)
        return self.dense_model(v)

    class CandidateTower(tf.keras.Model):
      def __init__(self,layers):
        super().__init__()
        self.movie_model=MovieModel()
        self.dense_model=tf.keras.Sequential()
        for layer in layers[:-1]:
          self.dense_model.add(tf.keras.layers.Dense(layer,activation='relu'))
        for layer in layers[-1:]:
          self.dense_model.add(tf.keras.layers.Dense(layer))
      def call(self,inputs):
        v=self.movie_model(inputs)
        return self.dense_model(v)

    class CombinedModels(tfrs.models.Model):
      def __init__(self,layers):
        super().__init__()
        self.query_tower=QueryTower(layers)
        self.candidate_tower=CandidateTower(layers)
        self.task=tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=data.movie_data.batch(128).map(self.candidate_tower)
            )
        )
      def compute_loss(self, features,training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings=self.query_tower({
            'genre':features['genre'],
            'movie_title':features['movie_title'],

        })
        candidate_embeddings=self.candidate_tower({
            'genre':features['genre'],
            'movie_title':features['movie_title'],
        })
        return self.task(query_embeddings,candidate_embeddings,compute_metrics=not training)

    """# preparing data to train model"""
    model = CombinedModels(layers)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    if phase=='training':
        tf.random.set_seed(123)
        shuffled = data.movie_data.shuffle(1_000, reshuffle_each_iteration=False)

        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(10_000)
        cached_train = train.shuffle(100_000).batch(2048)
        cached_test = test.batch(2048).cache()

        """# training """

        num_epochs = int(input('insert epochs number : '))


        one_layer_history = model.fit(
            cached_train,
            validation_data=cached_test,
            validation_freq=5,
            epochs=num_epochs,
            verbose=1)
        model.save_weights(saving_filepath)

        accuracy = one_layer_history.history["factorized_top_k/top_100_categorical_accuracy"][-1]
        print(f"Top-100 accuracy: {accuracy:.2f}.")


        plt.title('loss')
        plt.plot(one_layer_history.history["total_loss"])
        plt.show()
    if phase=='deployment':

        model.load_weights(saving_filepath)
    return model

def get_output(model, data, num_retrieve_items, input, phase, num_return_items):
    assert phase == 'training' or phase == 'deployment' or phase == None, f"phase must be [training] or [deployment] but got: {phase}"
    import utility_functions as utilities
    top_k=num_retrieve_items
    index=tfrs.layers.factorized_top_k.BruteForce(model.query_tower,k=top_k)
    index.index_from_dataset(
        tf.data.Dataset.zip((data.movie_data.map(lambda x : x['movie_id']).batch(100),
                             data.movie_data.batch(100).map(model.candidate_tower)))
    )
    if phase=='training':
        d = [i for i in data.all_data.batch(1).take(20)]
        print(f"{top_k} movies similar to movie :{d[1]['movie_id'].numpy()[0]} with below features \n\n")
        utilities.movie_feature_extractor(d[1]['movie_id'].numpy()[0],data)
        _,recommes=index(d[1])
        dict,raw_array=utilities.item_output_cleaner(recommes)
        print(dict)

    if phase=='deployment':
        recommended_items=[]
        for i in input:
            _, recommes = index(i)
            dict, raw_array = utilities.item_output_cleaner(recommes)
            recommended_items.append(raw_array[:num_return_items])

        return recommended_items