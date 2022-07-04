import importing_data_and_libraries as data
import ranker
import item_to_item_ as ITI
import user_to_item_model as UTI
import user_to_user_ as UTU

print('all modules have been imported')
print('=============================')
print('\n\n train Ranker ')
saving_filepath_for_ranker='Z:/recommender project/saved models/ranker'
ranker_model=ranker.model_creator(data,[64],'training',saving_filepath_for_ranker)
ranker.get_output(ranker_model, data, 1000, None, 'training', 10)

print('=================================')
print('\n\n train item to item')
saving_filepath_for_ITI='Z:/recommender project/saved models/ITI'
ITI_model=ITI.model_creator(data,[64,32,16],'training',saving_filepath_for_ITI)
ITI.get_output(ITI_model, data, 1000, None, 'training', 10)
print('=================================')
print('\n\n train user to item')
saving_filepath_for_UTI='Z:/recommender project/saved models/ranker/UTI'
UTI_model=UTI.model_creator(data,[64,32,16],'training',saving_filepath_for_UTI)
UTI.get_output(UTI_model,data, 1000, None, 'training', 10)


print('=================================')
print('\n\n train user to user')
saving_filepath_for_UTU='Z:/recommender project/saved models/ranker/UTU'
UTU_model=UTU.model_creator(data,[64,32,16],'training',saving_filepath_for_UTU)
UTU.get_output(UTU_model,data,1000, None, 'training', 10)
