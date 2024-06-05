# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 00:25:41 2024

@author: MILA
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#%% CLEANING THE DATA


data_dir = 'Data/'
dfs = []


for year in range(2013, 2024):
    file_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
    df = pd.read_csv(file_path)
    if year >= 2022:
        df['weight'] = 2  
    else:
        df['weight'] = 1  
    dfs.append(df)

all_matches = pd.concat(dfs, ignore_index=True)


all_matches['surface'] = all_matches['surface'].fillna('Hard')


all_matches.drop(columns = ['winner_seed','winner_entry','loser_seed','loser_entry'],inplace = True)


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

print(all_matches.isna().sum())


#%% PREPARING THE DATA FOR THE MODEL AND MAKING THE MODEL


def generate_a_match(player1_id,player2_id,surface,tournament_name):
    
    
    player_matches = all_matches[(all_matches['winner_id'] == player1_id) | (all_matches['winner_id'] == player2_id) & (all_matches['surface'] == surface)]

    
    def assign_weight(row):
        if ((row['winner_id'] == player1_id and row['loser_id'] == player2_id) or
            (row['winner_id'] == player2_id and row['loser_id'] == player1_id)):
            return 3 if row['tourney_name'] == tournament_name else 2
        else:
            return 1
    
    player_matches = player_matches.copy()
    player_matches['weight'] = player_matches.apply(assign_weight, axis=1)

    
    
    #print(player_matches.columns)
    
    selected_features = ['tourney_name', 'surface', 'tourney_level', 'winner_name','loser_name','score',
                    'winner_hand', 'winner_ioc', 'loser_hand',
                    'loser_ioc', 'round',]
    
    

    df_encoded = pd.get_dummies(player_matches, columns=selected_features)
    df_encoded['tourney_id'] = pd.to_numeric(df_encoded['tourney_id'], errors='coerce')
    df_encoded['tourney_date'] = pd.to_numeric(df_encoded['tourney_date'], errors='coerce')
    
    #print(df_encoded.dtypes)
    
    X = df_encoded.drop(columns=['winner_id'])
    Y = (df_encoded['winner_id'] == player1_id).astype(int)
    sample_weights = player_matches['weight']
    
    model = RandomForestClassifier()
    model.fit(X, Y, sample_weight=sample_weights)
    
    predicted_winner = model.predict(X)

    return player1_id if predicted_winner[0] == 1 else player2_id




def get_player_name(player_id):
    players = pd.read_csv('Data/atp_players.csv')
    player_info = players.loc[players['player_id'] == player_id, ['name_first', 'name_last']]
    if not player_info.empty:
        first_name = player_info['name_first'].iloc[0]
        last_name = player_info['name_last'].iloc[0]
        return f"{first_name} {last_name}"
    else:
        return "Player not found"


def get_surface(tournament_name):
    players = pd.read_csv('Data/atp_matches_2023.csv')
    surface = players[players['tourney_name'] == tournament_name]['surface'].iloc[0]
    return surface



def simulate_tournament(tournament_name, player_ids):
    """
    Simulate a tournament based on the provided player IDs.
    """
    surface = get_surface(tournament_name)
    
    print(f"Simulating {tournament_name} with players {player_ids}")
    
    current_round_players = player_ids
    
    round_number = 1
    while len(current_round_players) > 1:
        next_round_players = []
        print(f"Round {round_number}:")
        
        
        for i in range(0, len(current_round_players), 2):
            player1_id = current_round_players[i]
            player2_id = current_round_players[i + 1]
            winner_id = generate_a_match(player1_id, player2_id, surface, tournament_name)
            next_round_players.append(winner_id)
            
            player_name1 = get_player_name(player1_id)
            player_name2 = get_player_name(player2_id)
            winner_name = get_player_name(winner_id)
            print(f"Match: {player_name1} vs {player_name2}, Winner: {winner_name}")
        
        current_round_players = next_round_players
        round_number += 1
        
        
        
    player_name = get_player_name(current_round_players[0])
    print(f"Winner of {tournament_name} is {player_name}")
    
#POCETAK UNOSA PARAMETARA

#UNESITE PARAMETRE OD DOLE 
#nazive turnira mozete naci od linije 229 i mozete staviti bilo koji umesto Australian OPen
tournament_name = "Australian Open"
#ponudjeni player_ids koje mozete iskoristiti su od linije 197, broj igraca mora biti deljiv sa 4
player_ids = [ 126774,206173 , 104745,100644,134770,207989,104925,106421  ]
#pokrenite program
simulate_tournament(tournament_name, player_ids)



#Neki od poznatih igraca koje mozete navesti:  - sto su igraci stariji i imaju manje meceva sa igracima to ce biti neprecizniji
    #Rafael Nadal - 104745
    #Novak Djokovic - 104925
    #Alexander Zverev - 100644
    #Carlos Alcaraz - 207989
    #Jannik Sinner -206173
    #Danill Medvedev - 106421
    #Andrey Rublev - 126094
    #Casper Rudd - 134770
    #Hubert Hurkacz - 128034
    #Stefanos Tsitsipas - 126774
    #Grigor Dimitrov - 105777
    # Andy Roddick - 104053
    # Holger Rune - 208029
    # Andy Murray - 104918
    #Stan Wawrinka - 104527
    # Ben Shelton - 210097
    #John Isner -104545
    #Janko Tipsarevic - 104386
    #Marin Čilić - 105227
    #David Ferrer - 103970
    #Viktor Troicki - 104678
    #Juan Martin del Potro -
    #Dominic Thiem - 105223
    #Kei Nishikori - 105453
    #Diego Schwartzman - 106043
    #Denis Shapovalov - 133430
    #Taylor Fritz - 126203
    #Gael Monfils - 104792
    #Fabio Fognini - 104926
    #Milos Raonic - 105683
    #Jo-Wilfried Tsonga - 104542

#Turniri koje mozete navesti:
    
    #par atp 500 turnira:
        #Rotterdam
        #Rio De Janeiro
        #Dubai
        #Acapulco
        
    #par atp 1000 - atp masters:
        #Indian Wells Masters
        #Rome Masters
        #Paris Masters
        #Madrid Masters

    #GRAND SLAMS:
        #Wimbledon
        #Roland Garros
        #Us Open
        #Australian Open
 














