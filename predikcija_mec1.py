# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:40:10 2024

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:55:55 2024

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
from category_encoders import TargetEncoder


data_dir = 'Data/'
dfs = []

all_matches = pd.read_csv('Data/atp_matches_2023.csv')


all_matches['surface'] = all_matches['surface'].fillna('Hard')




all_matches.drop(columns=['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry', 'minutes',
                          'winner_age', 'loser_age', 'tourney_id', 'tourney_date'], inplace=True)


most_frequent_winner_hand = all_matches['winner_hand'].mode()[0]
all_matches['winner_hand'] = all_matches['winner_hand'].fillna(most_frequent_winner_hand)

most_frequent_loser_hand = all_matches['loser_hand'].mode()[0]
all_matches['loser_hand'] = all_matches['loser_hand'].fillna(most_frequent_loser_hand)


most_frequent_winner_height = all_matches['winner_ht'].mode()[0]
all_matches['winner_ht'] = all_matches['winner_ht'].fillna(most_frequent_winner_height)

most_frequent_loser_height = all_matches['loser_ht'].mode()[0]
all_matches['loser_ht'] = all_matches['loser_ht'].fillna(most_frequent_loser_height)


columns_to_fill = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                   'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

for column in columns_to_fill:
    most_frequent_value = all_matches[column].mode()[0]
    all_matches[column] = all_matches[column].fillna(most_frequent_value)


all_matches['winner_rank'] = all_matches['winner_rank'].fillna(1000)
all_matches['loser_rank'] = all_matches['loser_rank'].fillna(1000)


winner_rank_points_median = all_matches['winner_rank_points'].median()
all_matches['winner_rank_points'] = all_matches['winner_rank_points'].fillna(winner_rank_points_median)

loser_rank_points_median = all_matches['loser_rank_points'].median()
all_matches['loser_rank_points'] = all_matches['loser_rank_points'].fillna(loser_rank_points_median)




#%%


selected_features = ['tourney_name', 'surface', 'tourney_level', 'winner_name', 'loser_name', 'score',
                     'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc', 'round']

# One-hot encoding categorical features
#df_encoded = pd.get_dummies(all_matches, columns=selected_features)

#df_encoded = all_matches.drop(columns =selected_features)


df_encoded = all_matches.copy()


target_encoder = TargetEncoder()


for feature in selected_features:
    if feature in df_encoded.columns:
        df_encoded[feature] = target_encoder.fit_transform(df_encoded[feature], df_encoded['winner_id'])



#%%


X = df_encoded.drop(columns=['winner_id'])
Y = df_encoded['winner_id']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)*100)


#%%

coefficients = model.coef_[0]


feature_names = X.columns


coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})


coef_df['Coefficient'] = coef_df['Coefficient'].abs()


top_10 = coef_df.sort_values(by='Coefficient', ascending=False).head(10)


plt.figure(figsize=(10, 6))
plt.barh(top_10['Feature'], top_10['Coefficient'], color='skyblue')
plt.xlabel('Absolute Coefficient Value')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


