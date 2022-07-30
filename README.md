# a-hybrid-recommender-with-novel-architect-powerd-by-TensorFlow-recommender
## project discription
in this project I've designed a novel architecture of a ***hybrid recommender system*** this recommender system is powered by [TensorFlow recommenders](https://www.tensorflow.org/recommenders) librerye and consists of four maine components:
* _user-to-user model_: this model finds the most similar users to the target user.
* _user-to-item model_: this model finds the most relevent item to the target user.
* _item-to-item model_: this model find the most similar item to the target item.
* _ranker model_: this model takes the extracted items that are supposed to be relevant to the target user and ranks thoase.
I've combined all these 4 models together and built a hybrid recommender system you can see the designed architecture in the below image.  
#### recommender system architecture
![image](https://user-images.githubusercontent.com/74808396/181871669-08617051-b02b-41de-b1bc-ec5ff4f9585b.png)
## data
this project is based on the [MoviLenz](https://grouplens.org/datasets/movielens/) dataset but you can use your won dataset.
a dataset must consist of three parts:
- users.
- items that you want to recommend to users.
- users reactions to items e.g. rate, comment, like etc.
## how to use 
- `pip install -r requirements.txt`
- change the _data folder_ path in the **importing_data_and_libraries** code file to your local path.
- change the folder path for saving trained models in **trainer** code file to your local path.
- run **trainer** file to train all 4 models.
- run **deployer file** to run recommender system in inference mode.
## considerations
all rights are reserved for this architecture.
this repository was created ***just for education purposes*** and for the lack of code and resources for recommender systems and tensorflow recommender library. 
you ***can not*** use this repository for commercial purposes.
