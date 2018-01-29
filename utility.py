import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn import decomposition
from geopy.distance import great_circle
import random

def predicted_rating(ratings_mat, similarity, type='user'):
    '''
    ratings_mat: the matrix of user-restaurant ratings, each row corresponds to a user, each column corresponds to a restaurant
    similarity: similarity matrix for either user or item (restaurant)
    output: it fills out the user-restaurant rating matrix
    '''
    if type == 'user':
        mean_user_rating = ratings_mat.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings_mat - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings_mat.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


def select_photo(df):
    if 'food' in list(df['label']):
        out = df[df['label']=='food'].iloc[0]['photo_id']
    elif 'outside' in list(df['label']):
        out = df[df['label']=='outside'].iloc[0]['photo_id']
    elif 'inside' in list(df['label']):
        out = df[df['label']=='inside'].iloc[0]['photo_id']
    elif 'drink' in list(df['label']):
        out = df[df['label']=='drink'].iloc[0]['photo_id']
    else:
        out = df[df['label']=='menu'].iloc[0]['photo_id']
    return out


def group_restaurant_recommendation(model_choices, business_id_within_x_miles, group_id_list, quantile_threshold, select_top_x, restaurants_info_my_dataset):
    list_of_business_ids_different_models = []
    list_of_photo_ids_different_models = []
    list_of_business_names_different_models = []
    list_of_business_stars_different_models = []
    list_of_business_review_count_different_models = []
    list_of_business_address_different_models = []
    list_of_business_distance_different_models = []

    #Random Recommendation
    random_choice_business_id = random.choice(business_id_within_x_miles)
    random_choice = restaurants_info_my_dataset[restaurants_info_my_dataset.index == random_choice_business_id]

    random_choice_photo_id = list(random_choice['photo_id'])
    random_choice_name = list(random_choice['name'])
    random_choice_stars = list(random_choice['restaurant_stars'])
    random_choice_review_count = list(random_choice['review_count'])
    random_choice_address = list(random_choice['address']+','+random_choice['city']+ ','+random_choice['state'])
    random_choice_distance = list(np.round(np.array(random_choice['distances_to_my_location']),1))
    random_choice_list = [random_choice_business_id,random_choice_photo_id, random_choice_name,random_choice_stars,random_choice_review_count,random_choice_address, random_choice_distance]

    #Recommendation from model
    for i in model_choices:
        #print ('user_restaurant_cf_model_'+str(i)+'.csv')
        user_restaurant_cf = pd.read_csv('user_restaurant_cf_model_'+str(i)+'.csv')
        user_restaurant_cf = user_restaurant_cf.rename(columns = {'Unnamed: 0':'user_id'})
        user_restaurant_cf_group = user_restaurant_cf[user_restaurant_cf['user_id'].isin(group_id_list)]
        quantile_threshold=0.25

        ## Filter out those unacceptable restaurants for each user
        quantile_levels_users_in_group = list(user_restaurant_cf_group.quantile(quantile_threshold, axis=1))

        users_acceptable_business_ids = []
    
        for j in range(len(group_id_list)):
            user_restaurant_cf_group_tmp = user_restaurant_cf_group.set_index('user_id')
            # create subset of business ids for user j
            df = pd.DataFrame(user_restaurant_cf_group_tmp.iloc[j])
            user_acceptable_business_ids = df[df.columns[0]] > quantile_levels_users_in_group[j]
            user_acceptable_business_ids = list(df[user_acceptable_business_ids].index)
            users_acceptable_business_ids.append(user_acceptable_business_ids)
            commonly_acceptable_business_ids = list(set.intersection(*map(set, users_acceptable_business_ids)))
        
        #Filter out restaurants with low rating, at least 4
        tmp = restaurants_info_my_dataset[restaurants_info_my_dataset.index.isin(commonly_acceptable_business_ids)]
        commonly_acceptable_business_ids_with_good_rating = list(tmp[tmp['restaurant_stars']>=4].index)

        #Calculate group preference
        user_restaurant_cf_group_sum = user_restaurant_cf_group.set_index('user_id')[commonly_acceptable_business_ids_with_good_rating].sum(axis=0) 
        group_preference_df = pd.DataFrame(user_restaurant_cf_group_sum).reset_index()
        group_preference_df.columns = ['business_id', 'summed_preference']
    
        #Subset group preference given mile limit
        group_preference_within_x_miles = group_preference_df[group_preference_df.apply(lambda x: x['business_id'] in business_id_within_x_miles,axis=1)]
        group_preference_within_x_miles_sorted = group_preference_within_x_miles.sort_values('summed_preference',ascending=False)
       
        business_ids_for_group = list(group_preference_within_x_miles_sorted[0:select_top_x]['business_id'])
        business_info_for_group = restaurants_info_my_dataset[restaurants_info_my_dataset.index.isin(business_ids_for_group)]
        list_of_business_ids_different_models.append(business_ids_for_group)
        list_of_photo_ids_different_models.append(list(business_info_for_group['photo_id']))
        list_of_business_names_different_models.append(list(business_info_for_group['name']))
        list_of_business_stars_different_models.append(list(business_info_for_group['restaurant_stars']))
        list_of_business_review_count_different_models.append(list(business_info_for_group['review_count']))
        list_of_business_address_different_models.append(list(business_info_for_group['address']+','+business_info_for_group['city']+ ','+business_info_for_group['state']))
        list_of_business_distance_different_models.append(list(np.round(np.array(business_info_for_group['distances_to_my_location']),1)))
    return list_of_business_ids_different_models, list_of_photo_ids_different_models, list_of_business_names_different_models, list_of_business_stars_different_models, list_of_business_review_count_different_models, list_of_business_address_different_models, list_of_business_distance_different_models, random_choice_list

