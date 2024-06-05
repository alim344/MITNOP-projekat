import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#napomena: Ovde se nalaze svi grafici koje smo napravili, u izveštaj smo dodali one najznačajnije.

data_dir = 'Data/'

dfs = []

for year in range(2013, 2024):
    file_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
    df = pd.read_csv(file_path)
    dfs.append(df)
    

all_matches = pd.concat(dfs, ignore_index=True)

rankings = pd.read_csv("Data/atp_rankings_current.csv")

matches = all_matches[['tourney_name', 'round', 'winner_name', 'winner_rank', 'loser_name', 'loser_rank']]

top_10_players = rankings.head(10)['player'].tolist()
top_10_matches = all_matches[all_matches['winner_id'].isin(top_10_players)]

# ---BROJ POBEDA TOP 10 IGRAČA NA SVAKOJ PODLOZI PROCENTUALNO ---
wins_by_surface = top_10_matches.groupby(['winner_name', 'surface']).size().reset_index(name='wins')

total_matches = top_10_matches.groupby('winner_name').size().reset_index(name='total_matches')

merged = pd.merge(wins_by_surface, total_matches, on='winner_name')

merged['win_percentage'] = (merged['wins'] / merged['total_matches']) * 100

pivot_win_percentage = merged.pivot(index='winner_name', columns='surface', values='win_percentage').fillna(0)


plt.figure(figsize=(12, 8))
sns.heatmap(pivot_win_percentage, annot=True, fmt=".2f", cmap='viridis')
plt.title('Top 10 Player Win Percentage by Surface')
plt.xlabel('Surface')
plt.ylabel('Player')
plt.show()

# Pojedinačni grafici za svaku podlogu
def plot_surface_wins(data, surface_name):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='winner_name', y='win_percentage', data=data, palette='viridis')
    plt.title(f'Win Percentage on {surface_name} Surface for Top 10 Players')
    plt.xlabel('Player')
    plt.ylabel('Win Percentage')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.show()

clay_wins_percentage = merged[merged['surface'] == 'Clay']
grass_wins_percentage = merged[merged['surface'] == 'Grass']
hard_wins_percentage = merged[merged['surface'] == 'Hard']

plot_surface_wins(clay_wins_percentage, 'Clay')
plot_surface_wins(grass_wins_percentage, 'Grass')
plot_surface_wins(hard_wins_percentage, 'Hard')



# ----BROJ POBEDA TOP 10 IGRAČA NA SVAKOJ PODLOZI (BROJČANO)----
top_10_players = rankings.head(10)['player'].tolist()


top_10_matches = all_matches[all_matches['winner_id'].isin(top_10_players)]


wins_by_surface = top_10_matches.groupby(['winner_name', 'surface']).size().reset_index(name='wins')


pivot_wins_by_surface = wins_by_surface.pivot(index='winner_name', columns='surface', values='wins').fillna(0)


plt.figure(figsize=(12, 8))
sns.heatmap(pivot_wins_by_surface, annot=True, fmt="g", cmap='viridis')
plt.title('Top 10 Player Wins by Surface')
plt.xlabel('Surface')
plt.ylabel('Player')
plt.show()


clay_wins = wins_by_surface[wins_by_surface['surface'] == 'Clay']
grass_wins = wins_by_surface[wins_by_surface['surface'] == 'Grass']
hard_wins = wins_by_surface[wins_by_surface['surface'] == 'Hard']


def plot_surface_wins(data, surface_name):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='winner_name', y='wins', data=data, palette='viridis')
    plt.title(f'Number of Wins on {surface_name} Surface for Top 10 Players')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45)
    plt.show()

plot_surface_wins(clay_wins, 'Clay')
plot_surface_wins(grass_wins, 'Grass')
plot_surface_wins(hard_wins, 'Hard')


# --- GRAFICI ZA UTICAJ DUŽINE MEČA NA POBEDU ---


top_10_matches['match_duration_hours'] = top_10_matches['minutes'] / 60


short_matches = top_10_matches[top_10_matches['match_duration_hours'] <= 3]
long_matches = top_10_matches[top_10_matches['match_duration_hours'] > 3]


total_matches = top_10_matches['winner_name'].value_counts() + top_10_matches['loser_name'].value_counts()


short_wins = short_matches.groupby('winner_name').size().reset_index(name='wins')

long_wins = long_matches.groupby('winner_name').size().reset_index(name='wins')


def plot_duration_wins(data, duration):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='winner_name', y='wins', data=data, palette='viridis')
    plt.title(f'Number of Wins in {duration} Matches for Top 10 Players')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45)
    plt.show()

plot_duration_wins(short_wins, 'Short (< 3 hours)')
plot_duration_wins(long_wins, 'Long (> 3 hours)')


# -- BROJ POBEDA PO GODINAMA ZA NOVAKA DJOKOVIĆA --

djokovic_matches = all_matches[all_matches['winner_name'] == 'Novak Djokovic']

djokovic_matches['tourney_year'] = pd.to_datetime(djokovic_matches['tourney_date'], format='%Y%m%d').dt.year
djokovic_matches['winner_age_during_match'] = djokovic_matches['winner_age'].apply(lambda age: int(age))

djokovic_wins_by_age = djokovic_matches.groupby('winner_age_during_match').size().reset_index(name='wins')


print(djokovic_wins_by_age.head())


plt.figure(figsize=(12, 8))
sns.lineplot(x='winner_age_during_match', y='wins', data=djokovic_wins_by_age, marker='o')
plt.title('Number of Wins by Age for Novak Djokovic')
plt.xlabel('Age')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()

#-- GRAF ZA UTICAJ STAROSTI NA POBEDU --


all_matches = all_matches.dropna(subset=['winner_age'])

all_matches['tourney_year'] = pd.to_datetime(all_matches['tourney_date'], format='%Y%m%d').dt.year
all_matches['winner_age_during_match'] = all_matches['winner_age'].apply(lambda age: int(age))


wins_by_age = all_matches.groupby('winner_age_during_match').size().reset_index(name='wins')

print(wins_by_age.head())


plt.figure(figsize=(12, 8))
sns.lineplot(x='winner_age_during_match', y='wins', data=wins_by_age, marker='o')
plt.title('Number of Wins by Age for All Players')
plt.xlabel('Age')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()


# -- GRAF ZA UTICAJ OSVAJANJA PRVOG SETA NA ISHOD MEČA --
if len(top_10_matches) > 0:
   
    total_wins = len(top_10_matches)
    wins_with_first_set = 0


    for index, row in top_10_matches.iterrows():
        score = row['score']
        sets = score.split()
        
       
        if sets[0][0] > sets[0][2]:  
            wins_with_first_set += 1

  
    wins_without_first_set = total_wins - wins_with_first_set

    labels = ['Wins with First Set', 'Wins without First Set']
    sizes = [wins_with_first_set, wins_without_first_set]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  

  
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Winning the First Set on Match Outcome for Top 10 Players')
    plt.axis('equal')  

    plt.show()
else:
    print("No matches found for the top 10 players.")



# -- GRAF ZA UTICAJ BROJA ASEVA NA POBEDNIKA MEČA --

if len(top_10_matches) > 0:
    # Initialize counters
    total_wins = len(top_10_matches)
    wins_with_more_aces = 0


    for index, row in top_10_matches.iterrows():
        winner_aces = row['w_ace']
        loser_aces = row['l_ace']
        if winner_aces > loser_aces:
            wins_with_more_aces += 1

  
    wins_with_fewer_aces = total_wins - wins_with_more_aces

   
    labels = ['Wins with More Aces', 'Wins with Fewer Aces']
    sizes = [wins_with_more_aces, wins_with_fewer_aces]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  
 
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Serving More Aces on Match Outcome for Top 10 Players')
    plt.axis('equal') 

    plt.show()
else:
    print("No matches found for the top 10 players.")
    
# -- UTICAJ RANGA NA POBEDNIKA MEČA --

if len(top_10_matches) > 0:
   
    total_wins = len(top_10_matches)
    wins_against_lower_ranked = 0

   
    for index, row in top_10_matches.iterrows():
        winner_rank = row['winner_rank']
        loser_rank = row['loser_rank']
        if winner_rank < loser_rank:
            wins_against_lower_ranked += 1

  
    wins_against_higher_ranked = total_wins - wins_against_lower_ranked

  
    labels = ['Wins Against Lower Ranked Players', 'Wins Against Higher Ranked Players']
    sizes = [wins_against_lower_ranked, wins_against_higher_ranked]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0) 

 
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Facing Lower Ranked Players on Match Outcome for Top 10 Players')
    plt.axis('equal')  

    plt.show()
else:
    print("No matches found for the top 10 players.")
    
# -- UTICAJ VISINE NA ISHOD POBEDNIKA MEČA--
if len(top_10_matches) > 0:
   
    total_wins = len(top_10_matches)
    tall_wins = 0

 
    for index, row in top_10_matches.iterrows():
        winner_height = row['winner_ht']
        loser_height = row['loser_ht']
        if winner_height > loser_height:
            tall_wins += 1

   
    short_wins = total_wins - tall_wins

   
    labels = ['Wins by Taller Players', 'Wins by Shorter Players']
    sizes = [tall_wins, short_wins]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0) 

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Player Height on Match Outcome for Top 10 Players')
    plt.axis('equal') 

    plt.show()
else:
    print("No matches found for the top 10 players.")


# -- UTICAJ GODINA NA ISHOD MEČA PIECHART --
if len(top_10_matches) > 0:
   
    total_wins = len(top_10_matches)
    young_wins = 0

   
    for index, row in top_10_matches.iterrows():
        winner_age = row['winner_age']
        loser_age = row['loser_age']
        if winner_age < loser_age:
            young_wins += 1

  
    old_wins = total_wins - young_wins

   
    labels = ['Wins by Younger Players', 'Wins by Older Players']
    sizes = [young_wins, old_wins]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  

   
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Player Age on Match Outcome for Top 10 Players')
    plt.axis('equal')  
    plt.show()
else:
    print("No matches found for the top 10 players.")
    

