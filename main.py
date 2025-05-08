import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Concatenate, Dense, Masking
from tensorflow.keras.models import Model
import pickle as pkl
from sklearn.preprocessing import StandardScaler
def importPlayers(Filename = r'C:\Users\rooki\PycharmProjects\OddModel\Players.csv'):


    return pd.read_csv(Filename, parse_dates = ['birthdate'])

def importStatistics(filename = r'C:\Users\rooki\PycharmProjects\OddModel\PlayerStatistics.csv'):
    return pd.read_csv(filename)
def processPlayers(playerDf):
    playerDf.drop(columns = ['firstName','lastName','lastAttended','country','draftYear','draftRound','draftNumber','guard','forward','center'], inplace = True)
    playerDf.drop(columns = ['birthdate','bodyWeight'], inplace = True)
    return playerDf



def processStatistics(statsDf):
    statsDf = statsDf[statsDf["gameDate"] > '2003-01-01']
    statsDf.drop(columns = ['firstName','lastName','playerteamCity','opponentteamCity','opponentteamName','gameType','gameLabel','gameSubLabel','seriesGameNumber','plusMinusPoints'], inplace = True)
    return statsDf
def handleMissingValues(df):
    df[df.columns] = df[df.columns].fillna(df[df.columns].mode().iloc[0])
    return df
playerDf = importPlayers()
playerDf = processPlayers(playerDf)
with open('playerDfArchie.pkl', 'wb') as f:
    pkl.dump(playerDf, f)

statsDf = importStatistics()
with open('statsDfArchie.pkl', 'wb') as f:
    pkl.dump(statsDf, f)

statsDf = processStatistics(statsDf)
print(statsDf.columns)
df =statsDf.merge(playerDf,on = 'personId', how = 'inner')
df = handleMissingValues(df)
df = df.sort_values('gameDate', ascending=False)
df =df.sort_values('gameDate', ascending=True)
Scaler = StandardScaler()

colsExcluded = ['personId', 'gameId', 'gameDate','playerteamName','win']
cols_to_scale = [col for col in df.columns if col not in colsExcluded]

# Initialize and apply the scaler
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
df.to_csv('tryguy.csv')
teamDict = dict()
for team in df['playerteamName'].unique():
    teamSequence = pd.DataFrame()
    count = 0
    for game in df[df['playerteamName'] == team]['gameId'].unique():
        if count >10000:
            break;
        wholeGame =df[df['gameId'] == game]
        desiredTeamDf = wholeGame[wholeGame['playerteamName']==team]
        opposingTeamDf = wholeGame[wholeGame['playerteamName']!=team]
        print(desiredTeamDf['numMinutes'].dtype)
        #sorts all players by number of minutes in game
        desiredTeamDf.sort_values('numMinutes', ascending=False, inplace=True)
        opposingTeamDf.sort_values('numMinutes', ascending=False, inplace=True)
        if not (desiredTeamDf.shape[0] < 5 or opposingTeamDf.shape[0] < 5):
            desiredTeamDf= desiredTeamDf.iloc[:3,:]
            opposingTeamDf = opposingTeamDf.iloc[:3,:]
            # Columns to keep only once
            desiredTeamDf.drop(columns=[ 'personId'], inplace=True)
            opposingTeamDf.drop(columns=['personId'], inplace=True)
            constant_cols = ['gameId', 'gameDate','playerteamName', 'win', 'home']

            # Get the constant values from the first row
            base = desiredTeamDf[constant_cols].iloc[0].to_dict()

            # Get the columns you want to flatten
            flatten_cols = [col for col in desiredTeamDf.columns if col not in constant_cols]

            flattened = desiredTeamDf[flatten_cols].T.values.flatten(order='F')
            flat_columns = [f'{col}_{i}' for i in range(len(desiredTeamDf)) for col in flatten_cols]

            # Combine
            flattened_dict = dict(zip(flat_columns, flattened))
            final = {**base, **flattened_dict}
            desiredTeamDf = pd.DataFrame([final])

            opposingTeamDf.drop(columns = ['win','home'], inplace = True)
            constant_cols = ['gameId', 'gameDate', 'playerteamName']

            # Get the constant values from the first row
            base = opposingTeamDf[constant_cols].iloc[0].to_dict()

            # Get the columns you want to flatten
            flatten_cols = [col for col in opposingTeamDf.columns if col not in constant_cols]

            flattened = opposingTeamDf[flatten_cols].T.values.flatten(order='F')
            flat_columns = [f'{col}_{i}' for i in range(len(opposingTeamDf)) for col in flatten_cols]

            # Combine
            flattened_dict = dict(zip(flat_columns, flattened))
            final = {**base, **flattened_dict}
            opposingTeamDf = pd.DataFrame([final])

            #combining all data into one row
            desiredTeamDf.drop(columns =['playerteamName'], inplace = True)
            desiredTeamDf = desiredTeamDf.add_prefix(team)
            opTeam= opposingTeamDf['playerteamName'].iloc[0]
            opposingTeamDf.drop(columns=['playerteamName'], inplace = True)
            opposingTeamDf = opposingTeamDf.add_prefix('opposingTeam')
            combined = pd.concat([desiredTeamDf, opposingTeamDf], axis=1)
            combined.loc[0,'nameOfOpponent'] =opTeam
            teamSequence = pd.concat([teamSequence, combined], axis=0, ignore_index=True)
            count += 1
            print('hi')
    teamDict[team] = teamSequence.reset_index(drop=True)

with open('teamDict.pkl', 'wb') as f:
    pkl.dump(teamDict, f)

