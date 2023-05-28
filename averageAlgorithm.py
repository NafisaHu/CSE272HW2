import json
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from scipy import sparse
import numpy as np
import math

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error


with open('extracted_data.json', 'r') as file:
    json_data = json.load(file)

# Count unique reviewerIDs
unique_reviewer_ids = set(obj["reviewerID"] for obj in json_data)
reviewer_id_count = len(unique_reviewer_ids)

# Count unique asins
unique_asins = set(obj["asin"] for obj in json_data)
asin_count = len(unique_asins)

print("Unique reviewerIDs:", reviewer_id_count)
print("Unique asins:", asin_count)

users = []
products = []
ratings = []
iteration = 0
for obj in json_data:
    iteration += 1
    if iteration == 50000:
        break
    user = obj.get('reviewerID')
    product = obj.get('asin')
    rating = obj.get('overall')
    # Extract values for other fields as needed

    # Append the values to the respective lists
    users.append(user)
    products.append(product)
    ratings.append(rating)
    # Append values to other lists as needed

dataset_cols = ['reviewerID', 'asin', 'overall']
profile = pd.DataFrame(list(zip(users, products, ratings)), columns=dataset_cols) 
ground_truth_df = profile.copy()

train_df, test_df = train_test_split(profile, test_size=0.2)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ls = test_df['asin'].unique().toList()

predicted = []
# Create an empty dictionary to store item-wise average ratings
item_avg_ratings = {}


avg = -1

for index, row in test_df.iterrows():
    user = row['reviewerID']
    asin = row['asin']

    # Filter train_df based on 'asin' values in test_df
    filtered_rows = train_df[train_df['asin'] == asin]

    if not filtered_rows.empty:
        # Calculate the average rating from the filtered rows
        avg = filtered_rows['overall'].mean()
        print("item mean" , avg)
    
    if avg == -1:
        avg = 5

    predicted.append(avg)

print("predicted list: ", predicted)

test_df['p'] = predicted
mse = mean_squared_error(test_df['overall'], test_df['p'])
print("MSE: ", mse)
rmse = math.sqrt(mse)
print("RMSE: ", rmse)
ame = mean_absolute_error(test_df['overall'], test_df['p'])
print("AME: ", ame)

# Get unique items from the training set
train_items = train_df['asin'].unique()

recommendations = {}  # Dictionary to store recommendations for each user
iteration = 0
for user in test_df['reviewerID'].unique():
    iteration +=1
    if iteration == 5000:
        break
    # Filter test_df for the current user
    user_rows = test_df[test_df['reviewerID'] == user]
    
    # Get items the user has already purchased from the training set
    purchased_items = train_df[train_df['reviewerID'] == user]['asin'].values
    
    # Get items the user has not purchased
    unpurchased_items = [item for item in train_items if item not in purchased_items]
    
    # Predict ratings for unpurchased items
    predicted_ratings = []
    for item in unpurchased_items:
        filtered_rows = train_df[train_df['asin'] == item]
        if not filtered_rows.empty:
            avg = filtered_rows['overall'].mean()
        else:
            avg = 5
        predicted_ratings.append(avg)
    
    # Combine unpurchased items and predicted ratings into a DataFrame
    recommendations_df = pd.DataFrame({'asin': unpurchased_items, 'predicted_rating': predicted_ratings})
    
    # Sort items based on predicted ratings in descending order
    recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False)
    
    # Select top 10 items as recommendations
    top_recommendations = recommendations_df.head(10)['asin'].values
    
    # Store recommendations for the user
    recommendations[user] = top_recommendations
    print("User: ", user)
    print(recommendations[user])

# Print the recommendation list for each user
for user, recommended_items in recommendations.items():
    print("User:", user)
    print("Recommendations:", recommended_items)
    print()

# Assuming you have the ground truth data in a DataFrame called ground_truth_df
# with columns 'reviewerID' and 'asin' representing user and purchased item, respectively

# Initialize counters
true_positives = 0  # Number of recommended items that are actually purchased
false_positives = 0  # Number of recommended items that are not actually purchased
total_purchased = 0  # Total number of items purchased (ground truth)
total_recommendations = 0  # Total number of recommended items

for user, recommended_items in recommendations.items():
    # Get purchased items for the current user from the ground truth data
    purchased_items = ground_truth_df[ground_truth_df['reviewerID'] == user]['asin'].values
    
    # Update total_purchased counter
    total_purchased += len(purchased_items)
    
    # Update total_recommendations counter
    total_recommendations += len(recommended_items)
    
    # Count true positives and false positives
    for item in recommended_items:
        if item in purchased_items:
            true_positives += 1
        else:
            false_positives += 1

# Calculate precision, recall, and conversion rate
precision = true_positives / total_recommendations
recall = true_positives / total_purchased
conversion_rate = true_positives / len(recommendations)  # Assuming len(recommendations) represents the number of users

# Print the calculated metrics
print("Precision:", precision)
print("Recall:", recall)
print("Conversion Rate:", conversion_rate)


# Print the recommendation list for each user
'''
for user, recs in recommendations:
    print("User:", user)
    print("Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec}")
    print()
'''

'''
for indx, row in test_df.iterrows():

    reviewer_id = row['reviewerID']
    asin = row['asin']
    overall_rating = row['overall']

    # Check if the 'asin' already exists in the dictionary
    if asin in item_avg_ratings:
        # If yes, update the average rating by including the new rating
        item_avg_ratings[asin]['total_rating'] += overall_rating
        item_avg_ratings[asin]['count'] += 1
    else:
        # If not, initialize a new entry for the 'asin' in the dictionary
        item_avg_ratings[asin] = {
            'total_rating': overall_rating,
            'count': 1
        }

    
    print(test_df['reviewerID'])
    rel = train_df[train_df['asin'].isin([row['asin']])]
    #rel = rel['overall'].toList()

    avg = -1
    for i in rel:
        if avg == -1:
            avg = i
        else:
            avg = (avg + i)/2
    
    if avg == -1:
        avg = 5

    predicted.append(avg)

'''
'''
# Calculate the average rating for each item
for asin, ratings in item_avg_ratings.items():
    avg_rating = ratings['total_rating'] / ratings['count']
    item_avg_ratings[asin]['average_rating'] = avg_rating
'''

num_users = 3873247
num_items = 925387
