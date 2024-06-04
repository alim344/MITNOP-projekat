# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:51:19 2024

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV




# loading the data

data_dir = 'Data/'

# List to store DataFrames for each year
dfs = []

# Loop through each year from 2013 to 2023
for year in range(2013, 2024):
    # Construct the file path for the current year
    file_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
    
    # Read the data file into a DataFrame and append to the list
    df = pd.read_csv(file_path)
    dfs.append(df)
    
    all_matches = pd.concat(dfs, ignore_index=True)
    
players = pd.read_csv('Data/atp_players.csv')
rankings = pd.read_csv('Data/atp_rankings_current.csv')

#%% checking for null values


print(all_matches.isnull().sum())
print(players.isnull().sum())
print(rankings.isnull().sum())



#%%   cleaning the data players data set

num_rows = players.shape[0]
print("Number of rows:", num_rows)

#Droping th ecolumn we dont need
players.drop(columns=['wikidata_id'], inplace=True)

#Fill missing values in 'hand' with the mode where we put in the hand that is most frequent
players['hand'].fillna(players['hand'].mode()[0], inplace=True)

#Droping the players that dont have a name because they arent important to our analysis
players.dropna(subset=['name_first'], inplace=True)




print(players.isnull().sum())

#We are spliting our dataset for height analysis
players_with_height = players.dropna(subset=['height'])
players_missing_values = players[players['height'].isnull()]


print("Players with known heights:")
print(players_with_height['dob'].describe())

print("\nPlayers with missing heights:")
print(players_missing_values['dob'].describe())


#Filling in the missing date of birth with the mediana 

# Convert 'dob' to datetime format
players_with_height = players_with_height.copy()
players_with_height['dob'] = pd.to_datetime(players_with_height['dob'], format='%Y%m%d')

# Calculate the median value of the 'dob' column
dob_median = players_with_height['dob'].median()

# Fill missing values in the 'dob' column with the median
players_with_height['dob'].fillna(dob_median, inplace=True)


print(players_with_height.isnull().sum())

#%%  cleaning all_matches data set



#print(all_matches.isnull().sum())

num_rows = all_matches.shape[0]
print("Number of rows:", num_rows)

all_matches.drop(columns=['winner_seed','winner_entry','loser_seed','loser_entry'],inplace=True)
all_matches.dropna(subset=['surface','winner_age','loser_age','winner_rank','winner_rank_points','loser_rank','loser_rank_points'], inplace=True)


#filling in these values with mediana 
median_values = all_matches[['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','winner_ht','loser_ht']].median()

all_matches[['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','winner_ht','loser_ht']] = all_matches[['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','winner_ht','loser_ht']].fillna(median_values)

#filing in the loserhand with the most frequent one
all_matches['loser_hand'].fillna(all_matches['loser_hand'].mode()[0], inplace=True)

print(all_matches.isnull().sum())


#%% PREDICTING MATCHES LINEAR REGRESSION

merged_data_players = pd.merge(all_matches, players_with_height, left_on='winner_id', right_on='player_id', how='left')
merged_data_players = pd.merge(merged_data_players, players_with_height, left_on='loser_id', right_on='player_id', suffixes=('_winner', '_loser'), how='left')





# One-hot encode categorical variables
one_hot_encoder = OneHotEncoder()
surface_encoded = one_hot_encoder.fit_transform(merged_data_players[['surface']])
surface_encoded_df = pd.DataFrame(surface_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['surface']))

# Include player hands as features (assume 'R' = 1 and 'L' = 0)
merged_data_players['winner_hand_encoded'] = merged_data_players['winner_hand'].apply(lambda x: 1 if x == 'R' else 0)
merged_data_players['loser_hand_encoded'] = merged_data_players['loser_hand'].apply(lambda x: 1 if x == 'R' else 0)

# Create feature differences
merged_data_players['rank_diff'] = merged_data_players['winner_rank'] - merged_data_players['loser_rank']
merged_data_players['rank_points_diff'] = merged_data_players['winner_rank_points'] - merged_data_players['loser_rank_points']
merged_data_players['height_diff'] = merged_data_players['winner_ht'] - merged_data_players['loser_ht']
merged_data_players['age_diff'] = merged_data_players['winner_age'] - merged_data_players['loser_age']
merged_data_players['hand_diff'] = merged_data_players['winner_hand_encoded'] - merged_data_players['loser_hand_encoded']
merged_data_players['minutes'] = merged_data_players['minutes']

# Combine all features
features = ['rank_diff', 'rank_points_diff', 'height_diff', 'age_diff', 'hand_diff', 'minutes'] + list(surface_encoded_df.columns)
merged_data_players = pd.concat([merged_data_players, surface_encoded_df], axis=1)

# Target variable: 1 if winner, 0 if loser
merged_data_players['target'] = 1

# Creating a symmetric dataset with both perspectives (winner and loser)
loser_data = merged_data_players.copy()
loser_data['rank_diff'] = -loser_data['rank_diff']
loser_data['rank_points_diff'] = -loser_data['rank_points_diff']
loser_data['height_diff'] = -loser_data['height_diff']
loser_data['age_diff'] = -loser_data['age_diff']
loser_data['hand_diff'] = -loser_data['hand_diff']
loser_data['target'] = 0

# Combine winner and loser data
model_data = pd.concat([merged_data_players[features + ['target']], loser_data[features + ['target']]], ignore_index=True)

# Split the data into train and test sets
X = model_data[features]
y = model_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_reg = LogisticRegression(max_iter=10000)
logistic_reg.fit(X_train, y_train)

# Predict on test set
y_pred_prob = logistic_reg.predict_proba(X_test)[:, 1]  # Probability of class 1 (winner)
y_pred = (y_pred_prob > 0.5).astype(int)  # Thresholding probabilities at 0.5

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Model Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
