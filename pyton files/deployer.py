import importing_data_and_libraries as data
import ranker
import item_to_item_ as ITI
import user_to_item_model as UTI
import user_to_user_ as UTU
import utility_functions as utilities
print('=================================')
print('\n\n user to user')
saving_filepath_for_UTU='Z:/recommender project/saved models/ranker/UTU'
UTU_model=UTU.model_creator(data,[64,32,16],'deployment',saving_filepath_for_UTU)
recommender_input=data.all_data.batch(1).take(1)
UTU_output=UTU.get_output(UTU_model,data,num_retrive_users=1000,input=recommender_input,phase='deployment',num_returne_users=10)
UTI_input_from_UTU_output=utilities.data_creator_for_UTI_from_UTU(UTU_output,data)

print('=================================')
print('\n\n user to item')
saving_filepath_for_UTI='Z:/recommender project/saved models/ranker/UTI'
UTI_model=UTI.model_creator(data,[64,32,16],'deployment',saving_filepath_for_UTI)
UTI_output_from_recommender_input=UTI.get_output(UTI_model,data,num_retrieve_users=1000,input=recommender_input,phase='deployment',num_return_users=10)
# out= [array([out1,out2,....,])]
UTI_output_from_UTU_output=UTI.get_output(UTI_model,data,num_retrieve_items=1000,input=UTI_input_from_UTU_output,phase='deployment',num_return_items=10)
# out= [array([out1,out2,....,]),array([out1,out2,...,]),....]
all_retrieved_items=UTI_output_from_UTU_output+UTI_output_from_recommender_input
ITI_inputs=utilities.input_data_creater_for_item_item(all_retrieved_items[-2:-1],data=data)
print('=================================')
print('\n\n item to item')
saving_filepath_for_ITI='Z:/recommender project/saved models/ITI'
ITI_model=ITI.model_creator(data,[64,32,16],'deployment',saving_filepath_for_ITI)
ITI_output=ITI.get_output(ITI_model,data,num_retrieve_items=1000,input=ITI_inputs,phase='deployment',num_return_items=10)
total_items=all_retrieved_items+ITI_output
ranker_input=utilities.data_creator_for_ranker_from_ITI(recommender_input,total_items,data)
print('=============================')
print('\n\n Ranker ')
saving_filepath_for_ranker='Z:/recommender project/saved models/ranker'
ranker_model=ranker.model_creator(data,[64],'deployment',saving_filepath_for_ranker)
ranker.get_output(ranker_model,data,'deployment',ranker_input,recommender_input)





