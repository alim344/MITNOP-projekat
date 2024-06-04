import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

rankings = pd.read_csv("Data/atp_rankings_current.csv")

matches = all_matches[['tourney_name', 'round', 'winner_name', 'winner_rank', 'loser_name', 'loser_rank']]

top_10_players = rankings.head(10)['player'].tolist()

# ---BROJ POBEDA TOP 10 IGRAČA NA SVAKOJ PODLOZI PROCENTUALNO ---
top_10_matches = all_matches[all_matches['winner_id'].isin(top_10_players)]


wins_by_surface = top_10_matches.groupby(['winner_name', 'surface']).size().reset_index(name='wins')


total_matches = top_10_matches.groupby('winner_name').size().reset_index(name='total_matches')

# Spajanje podataka
merged = pd.merge(wins_by_surface, total_matches, on='winner_name')

# Izračunavanje procenta pobeda
merged['win_percentage'] = (merged['wins'] / merged['total_matches']) * 100

# Pivot tabela za vizualizaciju
pivot_win_percentage = merged.pivot(index='winner_name', columns='surface', values='win_percentage').fillna(0)

# Vizualizacija pivot tabele
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

# Kreiranje pojedinačnih grafika
clay_wins_percentage = merged[merged['surface'] == 'Clay']
grass_wins_percentage = merged[merged['surface'] == 'Grass']
hard_wins_percentage = merged[merged['surface'] == 'Hard']

plot_surface_wins(clay_wins_percentage, 'Clay')
plot_surface_wins(grass_wins_percentage, 'Grass')
plot_surface_wins(hard_wins_percentage, 'Hard')



# ----BROJ POBEDA TOP 10 IGRAČA NA SVAKOJ PODLOZI (BROJČANO)----
top_10_players = rankings.head(10)['player'].tolist()

# Filtriraj mečeve gde je pobednik jedan od top 10 igrača
top_10_matches = all_matches[all_matches['winner_id'].isin(top_10_players)]

# Prebroj pobede svakog igrača na svakoj podlozi
wins_by_surface = top_10_matches.groupby(['winner_name', 'surface']).size().reset_index(name='wins')

#velika tabela
# Kreiraj pivot tabelu za vizualizaciju
pivot_wins_by_surface = wins_by_surface.pivot(index='winner_name', columns='surface', values='wins').fillna(0)

# Kreiraj grafikon
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_wins_by_surface, annot=True, fmt="g", cmap='viridis')
plt.title('Top 10 Player Wins by Surface')
plt.xlabel('Surface')
plt.ylabel('Player')
plt.show()

#pojedinacni grafovi
clay_wins = wins_by_surface[wins_by_surface['surface'] == 'Clay']
grass_wins = wins_by_surface[wins_by_surface['surface'] == 'Grass']
hard_wins = wins_by_surface[wins_by_surface['surface'] == 'Hard']

# Funkcija za crtanje stubičastog grafikona
def plot_surface_wins(data, surface_name):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='winner_name', y='wins', data=data, palette='viridis')
    plt.title(f'Number of Wins on {surface_name} Surface for Top 10 Players')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45)
    plt.show()

# Crtanje grafikona za svaku podlogu
plot_surface_wins(clay_wins, 'Clay')
plot_surface_wins(grass_wins, 'Grass')
plot_surface_wins(hard_wins, 'Hard')


#%% grafici za igrac/duzina trajanja meca

# Convert minutes to hours for easier filtering
top_10_matches['match_duration_hours'] = top_10_matches['minutes'] / 60

# Filter matches by duration
short_matches = top_10_matches[top_10_matches['match_duration_hours'] <= 3]
long_matches = top_10_matches[top_10_matches['match_duration_hours'] > 3]

# Broj ukupnih mečeva za svakog igrača
total_matches = top_10_matches['winner_name'].value_counts() + top_10_matches['loser_name'].value_counts()

# Prebroj pobede svakog igrača za kratke mečeve
short_wins = short_matches.groupby('winner_name').size().reset_index(name='wins')

# Prebroj pobede svakog igrača za duge mečeve
long_wins = long_matches.groupby('winner_name').size().reset_index(name='wins')

# Funkcija za crtanje stubičastog grafikona
def plot_duration_wins(data, duration):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='winner_name', y='wins', data=data, palette='viridis')
    plt.title(f'Number of Wins in {duration} Matches for Top 10 Players')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45)
    plt.show()

# Crtanje grafikona za kratke i duge mečeve
plot_duration_wins(short_wins, 'Short (< 3 hours)')
plot_duration_wins(long_wins, 'Long (> 3 hours)')


#%%po godinama
# Filtriraj mečeve gde je pobednik Novak Djokovic
djokovic_matches = all_matches[all_matches['winner_name'] == 'Novak Djokovic']

# Izračunaj godine starosti Novaka Đokovića tokom meča
djokovic_matches['tourney_year'] = pd.to_datetime(djokovic_matches['tourney_date'], format='%Y%m%d').dt.year
djokovic_matches['winner_age_during_match'] = djokovic_matches['winner_age'].apply(lambda age: int(age))

# Grupiši pobede po godinama starosti
djokovic_wins_by_age = djokovic_matches.groupby('winner_age_during_match').size().reset_index(name='wins')

# Provera sadržaja DataFrame-a
print(djokovic_wins_by_age.head())

# Kreiranje grafikona
plt.figure(figsize=(12, 8))
sns.lineplot(x='winner_age_during_match', y='wins', data=djokovic_wins_by_age, marker='o')
plt.title('Number of Wins by Age for Novak Djokovic')
plt.xlabel('Age')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()

#%% broj pobeda po starosnoj grupi (godinama)

# Remove rows with NaN values in 'winner_age'
all_matches = all_matches.dropna(subset=['winner_age'])
# Izračunaj godine starosti pobednika tokom meča
all_matches['tourney_year'] = pd.to_datetime(all_matches['tourney_date'], format='%Y%m%d').dt.year
all_matches['winner_age_during_match'] = all_matches['winner_age'].apply(lambda age: int(age))

# Grupiši pobede po godinama starosti pobednika
wins_by_age = all_matches.groupby('winner_age_during_match').size().reset_index(name='wins')

# Provera sadržaja DataFrame-a
print(wins_by_age.head())

# Kreiranje grafikona
plt.figure(figsize=(12, 8))
sns.lineplot(x='winner_age_during_match', y='wins', data=wins_by_age, marker='o')
plt.title('Number of Wins by Age for All Players')
plt.xlabel('Age')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()


#%% pie chart za to koliko dobijanje prvog seta utice na pobednika
if len(top_10_matches) > 0:
    # Initialize counters
    total_wins = len(top_10_matches)
    wins_with_first_set = 0

    # Check for each match if the first set was won by the winner
    for index, row in top_10_matches.iterrows():
        # Extract set scores from the score column
        score = row['score']
        sets = score.split()
        
        # Check if the winner won the first set
        if sets[0][0] > sets[0][2]:  # Compare the number of games won in the first set
            wins_with_first_set += 1

    # Calculate wins without the first set
    wins_without_first_set = total_wins - wins_with_first_set

    # Create data for the pie chart
    labels = ['Wins with First Set', 'Wins without First Set']
    sizes = [wins_with_first_set, wins_without_first_set]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode the first slice

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Winning the First Set on Match Outcome for Top 10 Players')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
else:
    print("No matches found for the top 10 players.")



#%% uticaj aseva na pobednika

if len(top_10_matches) > 0:
    # Initialize counters
    total_wins = len(top_10_matches)
    wins_with_more_aces = 0

    # Check for each match if the winner served more aces than the loser
    for index, row in top_10_matches.iterrows():
        winner_aces = row['w_ace']
        loser_aces = row['l_ace']
        if winner_aces > loser_aces:
            wins_with_more_aces += 1

    # Calculate wins with fewer aces
    wins_with_fewer_aces = total_wins - wins_with_more_aces

    # Create data for the pie chart
    labels = ['Wins with More Aces', 'Wins with Fewer Aces']
    sizes = [wins_with_more_aces, wins_with_fewer_aces]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode the first slice

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Impact of Serving More Aces on Match Outcome for Top 10 Players')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
else:
    print("No matches found for the top 10 players.")



