import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'Data/'

dfs = []

for year in range(2013, 2024):
    file_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
    
    df = pd.read_csv(file_path)
    dfs.append(df)
    
all_matches = pd.concat(dfs, ignore_index=True)

rankings = pd.read_csv("Data/atp_rankings_current.csv")

matches = all_matches[['tourney_name', 'round', 'winner_name', 'winner_rank', 'loser_name', 'loser_rank']]


def head_to_head(player1, player2):
    
    player1_wins = all_matches[(all_matches['winner_name'] == player1) & (all_matches['loser_name'] == player2)]

    player2_wins = all_matches[(all_matches['winner_name'] == player2) & (all_matches['loser_name'] == player1)]

    player1_win_count = player1_wins.shape[0]
    player2_win_count = player2_wins.shape[0]

    player1_sets_won = player1_wins['score'].apply(lambda x: x.count('-')).sum()
    player2_sets_won = player2_wins['score'].apply(lambda x: x.count('-')).sum()

    player1_aces = player1_wins['w_ace'].sum()
    player2_aces = player2_wins['w_ace'].sum()

    print(f"{player1} vs {player2} Head-to-Head:")
    print(f"{player1} wins: {player1_win_count}")
    print(f"{player2} wins: {player2_win_count}")
    print(f"Total sets won by {player1}: {player1_sets_won}")
    print(f"Total sets won by {player2}: {player2_sets_won}")
    print(f"Total aces by {player1}: {player1_aces}")
    print(f"Total aces by {player2}: {player2_aces}")

# Primer korišćenja - ovde umesto 'Rafael Nadal' i 'Novak Djokovic' možete uneti ime i prezime bilo kog igrača sa donjeg spiska.
head_to_head('Rafael Nadal', 'Novak Djokovic')



#Neki od poznatih igraca koje mozete navesti
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