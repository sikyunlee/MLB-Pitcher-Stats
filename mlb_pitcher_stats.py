#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def get_data():
    #import required packages
    import pandas as pd
    import numpy as np
    
    #print('This function automatically reads the MLB Pitch by Pitch Data in 2019 Season when called.')
    #read base datasets -- 2019 MLB Pitch by Pitch Data with At Bats, Games, and Players Data
    atbats = pd.read_csv('2019_atbats.csv')
    games = pd.read_csv('2019_games.csv')
    pitches = pd.read_csv('2019_pitches.csv')
    names = pd.read_csv('player_names.csv')
    
    #Merge the dataset into an one big pitch by pitch dataset
    joined_atbats = atbats.copy(deep = True)
    joined_atbats = pd.merge(pitches, joined_atbats, how = 'left', on = 'ab_id')
    joined_atbats = pd.merge(joined_atbats, games, how = 'left', on ='g_id')
    joined_atbats = pd.merge(joined_atbats, names, how = 'left', left_on = 'pitcher_id', right_on = 'id')
    joined_atbats['full_name'] = joined_atbats.first_name + " " + joined_atbats.last_name
    
    #print('-------------------------------------------------')
    #print('When you see THIS text, the data is loaded and other functions in this package can be used.')
    #print('To get pitcher statistics, use their pitcher_id in the "player_names.csv" or try ID#: 502239, 547943, 607192 for example.')
    #print("To get a team's pitcher rankings, use the team's abbreviated city name such as 'oak' for Oakland A's or 'bos' for Boston Red Sox.")
    return joined_atbats


# In[ ]:


#get_data()


# In[ ]:


def metric_pitches_per_game_per_pitcher():
    import pandas as pd
    import numpy as np 
    
    print('This metric returns Pitches thrown by a Pitcher per Game, per Season and Avg. Pitches thrown by a Pitcher per Game over a Season')
    print('\n')
    print('Enter Pitcher ID Number and Press Enter: ')
    pitcher_id = int(input())
    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    joined_atbats = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]
    pitcher_name = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]['full_name'].unique()
    
    #Aggregate to get total pitches thrown by pitcher per game
    total_pitches_pitcher_game = joined_atbats.groupby(by = ['date','pitcher_id'], as_index=False)['batter_id'].agg([np.count_nonzero]).reset_index()
    total_pitches_pitcher_game = total_pitches_pitcher_game.sort_values(['date','count_nonzero','pitcher_id'], ascending=[True, False,True])
    total_pitches_pitcher_game = total_pitches_pitcher_game.sort_values(['date','count_nonzero'], ascending=[True, False])
    
    #Aggregate to get total pitches thrown by pitcher over season
    total_pitches_pitcher_season = total_pitches_pitcher_game.groupby(['pitcher_id'], as_index=False)['count_nonzero'].agg([np.sum]).reset_index()
    temp = total_pitches_pitcher_game.groupby(['pitcher_id'], as_index=False)['count_nonzero'].agg([np.sum]).reset_index()
    total_pitches_pitcher_season = temp.sort_values(['sum'], ascending=False)
    
    #Aggregate to get average pitches thrown by pitcher over season 
    total_games_pitcher = total_pitches_pitcher_game['date'].nunique()
    total_pitches_pitcher = total_pitches_pitcher_game['count_nonzero'].agg([np.sum])
    avg_pitches_pitcher = np.round(total_pitches_pitcher / total_games_pitcher, decimals = 0, out=None)
    
    #Return a DataFrame of Average Pitches 
    print('--------------------------------------------------------')
    print('For Pitcher ID: ', pitcher_id)
    print('Pitcher Name is ', pitcher_name)
    print('Total Pitches Thrown per Game :')
    print(pd.DataFrame(total_pitches_pitcher_game))
    print('\n')
    print('Total Pitches Thrown per Season :', total_pitches_pitcher_season['sum'])
    print('\n')
    print('Average Pitches Thrown per Game :', avg_pitches_pitcher)
    print('--------------------------------------------------------')


# In[ ]:


#metric_pitches_per_game_per_pitcher()


# In[ ]:


#plots of the pitches thrown per game for a pitcher

def plot_pitch_trend_per_pitcher():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    print('This metric returns Pitches thrown by a Pitcher per Game, per Season and Avg. Pitches thrown by a Pitcher per Game over a Season')
    print('\n')
    print('Enter Pitcher ID Number and Press Enter: ')
    pitcher_id = int(input())
    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    joined_atbats = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]
    pitcher_name = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]['full_name'].unique()
    
    #Aggregate to get total pitches thrown by pitcher per game
    total_pitches_pitcher_game = joined_atbats.groupby(by = ['date','pitcher_id'], as_index=False)['batter_id'].agg([np.count_nonzero]).reset_index()
    total_pitches_pitcher_game = total_pitches_pitcher_game.sort_values(['date','count_nonzero','pitcher_id'], ascending=[True, False,True])
    total_pitches_pitcher_game = total_pitches_pitcher_game.sort_values(['date','count_nonzero'], ascending=[True, False])
    
    #Get the Rolling Average of Pitches Thrown by pitcher per game
    total_pitches_pitcher_game['rolling_mean'] = np.round(total_pitches_pitcher_game['count_nonzero'].rolling(window=3).mean(), decimals = 0)
    
    
    #Output: Text and Visualization   
    print('--------------------------------------------------------')
    print('Visualization of Total Pitches thrown by ID:', pitcher_id, ' ', pitcher_name, ' over the Season')
    print('--------------------------------------------------------')

#     fig = px.line(total_pitches_pitcher_game, x="date", y="rolling_mean", title='Average Pitches Thrown over Last 3 Games')
#     fig = fig.add_bar(total_pitches_pitcher_game, x='date', y='count_nonzero', title='Total Pitches Thrown per Game over the Season',
#             hover_data = ['date', 'count_nonzero'], color = 'count_nonzero',
#         labels = {'count_nonzero' : 'Total Pitches Thrown', 'date' : 'Game Date'})
       
    fig = make_subplots(specs = [[{'secondary_y' : True}]])
    
    fig.add_trace(
        go.Bar(x=total_pitches_pitcher_game['date'], y=total_pitches_pitcher_game['count_nonzero'], name = 'Total Pitches over Season'),
        secondary_y = False,
    )
    
    fig.add_trace(
        go.Line(x=total_pitches_pitcher_game["date"], y=total_pitches_pitcher_game["rolling_mean"], name = 'Average Pitches in Last 3 Games'),
        secondary_y = True,
    )
    
    fig.update_layout(title_text = "<b>Pitches Thrown per Game over the Season</b>")
    fig.update_xaxes(title_text = '<b>Game Date</b>')
    fig.update_yaxes(title_text = '<b>Total Pitches Thrown Over the Season</b>', secondary_y = False)
    fig.update_yaxes(title_text = '<b>Average Pitches Thrown Over Last 3 Games</b>', secondary_y = True)

    fig.show()


# In[ ]:


#plot_pitch_trend_per_pitcher()


# In[ ]:


def plot_inning_trend_per_pitcher():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    print('This metric returns innings thrown by a Pitcher per Game, per Season and Avg. innings thrown by a Pitcher per Game over a Season')
    print('\n')
    print('Enter Pitcher ID Number and Press Enter: ')
    pitcher_id = int(input())
    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    joined_atbats = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]
    pitcher_name = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]['full_name'].unique()
    
    #Aggregate to get innings information for pitcher_id
    #Get innings as a column
    all_games_innings = joined_atbats.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index()
    #Get unique game dates that the pitcher pitched as a column
    all_games_dates = joined_atbats.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index()
    #Concatenate into one Pandas dataframe, drop the index column
    combined_innings = pd.concat([all_games_dates, all_games_innings], axis=1).drop(columns=['index'])
    combined_innings = combined_innings.rename(columns = {'unique': 'date'})
    
    #Get some performance metrics
    #Total innings pitched
    total_innings = all_games_innings.sum()
    #Average innings pitched
    avg_innings = all_games_innings.mean()
    #Get Rolling Average of last 3 games pitched innings
    combined_innings['rolling_mean'] = np.round(combined_innings['inning'].rolling(window=3).mean(), decimals = 0)
 
    
    #Output: Text and Visualization   
    print('--------------------------------------------------------')
    print('Visualization of Innings Pitched by ID:', pitcher_id, ' ', pitcher_name, ' over the Season')
    print('Pitcher ', pitcher_name, 'Pitched a Total of ', total_innings, ' and an Average of ', avg_innings, ' over the Season')
    print('--------------------------------------------------------')
       
    fig = make_subplots(specs = [[{'secondary_y' : True}]])
    
    fig.add_trace(
        go.Bar(x=combined_innings['date'], y=combined_innings['inning'], name = 'Innings Pitched over Season'),
        secondary_y = False,
    )
    
    fig.add_trace(
        go.Line(x=combined_innings["date"], y=combined_innings["rolling_mean"], name = 'Average Innings in Last 3 Games'),
        secondary_y = True,
    )
    
    fig.update_layout(title_text = "<b>Innings Thrown per Game over the Season</b>")
    fig.update_xaxes(title_text = '<b>Game Date</b>')
    fig.update_yaxes(title_text = '<b>Total Innings Thrown Over the Season</b>', secondary_y = False)
    fig.update_yaxes(title_text = '<b>Average Innings Thrown Over Last 3 Games</b>', secondary_y = True)

    fig.show()
    


# In[ ]:


#plot_inning_trend_per_pitcher()


# In[ ]:


def plot_batters_faced_per_pitcher():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np    
    
    print('This metrics return Batters faced by a Pitcher per Game, per Season and Avg. batters faced by a Pitcher per Game over a Season')
    print('\n')
    print('Enter Pitcher ID Number and Press Enter: ')
    pitcher_id = int(input())
    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    pitcher_data = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id] #test
    pitcher_name = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]['full_name'].unique()
    
    #Get aggregated table of batters faced per game
    innings_num = pitcher_data.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index() #test1
    games_num = pitcher_data.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index() #test2
    c1 = pd.concat([games_num, innings_num], axis=1).drop(columns=['index'])
    batters_num = pitcher_data.groupby(by=['g_id','inning'], as_index=True)['batter_id'].nunique().reset_index() #test3
    c2 = batters_num.groupby(by=['g_id'], as_index=True)['batter_id'].sum().reset_index() #25, 21
    combined_data = pd.merge(c1, c2,how='left', on = 'g_id')
    combined_data = combined_data.rename(columns = {'unique': 'date', 'batter_id': 'batters_faced'})
    
    #Get some performance metrics
    #Total innings pitched
    total_innings = combined_data['inning'].sum()
    #Average innings pitched
    avg_innings = combined_data['inning'].mean()
    #Get Rolling Average of last 3 games pitched innings
    combined_data['rolling_mean_innings'] = np.round(combined_data['inning'].rolling(window=3).mean(), decimals = 0)
    
    #Total batters faced
    total_batters = combined_data['batters_faced'].sum()
    #Average batters faced per game
    avg_batters = combined_data['batters_faced'].mean()
    #Get Rolling Average of last 3 games' batters faced number
    combined_data['rolling_mean_batters_faced'] = np.round(combined_data['batters_faced'].rolling(window=3).mean(), decimals = 0)
    
    #Average batters faced per inning
    avg_batters_inning = np.round(total_batters / total_innings, decimals = 0)
 
    #Output: Text and Visualization   
    print('--------------------------------------------------------')
    print('Visualization of Batters faced by ID:', pitcher_id, ' ', pitcher_name, ' over the Season')
    print('Pitcher ', pitcher_name, 'Pitched a Total of ', total_innings, ' innings and an Average of ', avg_innings, ' innings over the Season')
    print('Pitcher ', pitcher_name, 'Faced a Total of ', total_batters, ' batters and an Average of ', avg_batters, ' batters over the Season')
    print('Pitcher ', pitcher_name, 'Faces an Average of ', avg_batters_inning, ' batters per inning on a given game')
    print('--------------------------------------------------------')
       
    fig = make_subplots(specs = [[{'secondary_y' : True}]])
    
    fig.add_trace(
        go.Bar(x=combined_data['date'], y=combined_data['batters_faced'], name = 'Batters Faced over Season'),
        secondary_y = False,
    )
    
    fig.add_trace(
        go.Line(x=combined_data["date"], y=combined_data["rolling_mean_batters_faced"], name = 'Average Batters Faced in Last 3 Games'),
        secondary_y = True,
    )
    
    fig.update_layout(title_text = "<b>Batters Faced per Game over the Season</b>")
    fig.update_xaxes(title_text = '<b>Game Date</b>')
    fig.update_yaxes(title_text = '<b>Total Batters Faced Over the Season</b>', secondary_y = False)
    fig.update_yaxes(title_text = '<b>Average Batters Faced Over Last 3 Games</b>', secondary_y = True)

    fig.show()
        


# In[ ]:


#plot_batters_faced_per_pitcher()


# In[ ]:


#Comparing Innings Faced VS Batters Faced
def plot_compare_innings_batters():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np   
    
    print('This metrics return Batters faced by a Pitcher per Game, per Season and Innings thrown by a Pitcher per Game over a Season')
    print('\n')
    print('Enter Pitcher ID Number and Press Enter: ')
    pitcher_id = int(input())
    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    pitcher_data = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id] #test
    pitcher_name = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id]['full_name'].unique()
    
    #Get aggregated table of batters faced per game
    innings_num = pitcher_data.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index() #test1
    games_num = pitcher_data.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index() #test2
    c1 = pd.concat([games_num, innings_num], axis=1).drop(columns=['index'])
    batters_num = pitcher_data.groupby(by=['g_id','inning'], as_index=True)['batter_id'].nunique().reset_index() #test3
    c2 = batters_num.groupby(by=['g_id'], as_index=True)['batter_id'].sum().reset_index() #25, 21
    combined_data = pd.merge(c1, c2,how='left', on = 'g_id')
    combined_data = combined_data.rename(columns = {'unique': 'date', 'batter_id': 'batters_faced'})
    combined_data['ratio'] = np.round(combined_data['batters_faced'] / combined_data['inning'], decimals = 2)
    
    #Get some performance metrics
    #Total innings pitched
    total_innings = combined_data['inning'].sum()
    #Average innings pitched
    avg_innings = combined_data['inning'].mean()
    #Get Rolling Average of last 3 games pitched innings
    combined_data['rolling_mean_innings'] = np.round(combined_data['inning'].rolling(window=3).mean(), decimals = 0)
    
    #Total batters faced
    total_batters = combined_data['batters_faced'].sum()
    #Average batters faced per game
    avg_batters = combined_data['batters_faced'].mean()
    #Get Rolling Average of last 3 games' batters faced number
    combined_data['rolling_mean_batters_faced'] = np.round(combined_data['batters_faced'].rolling(window=3).mean(), decimals = 0)
    
    #Average batters faced per inning
    avg_batters_inning = np.round(total_batters / total_innings, decimals = 0)
 
    #Output: Text and Visualization   
    #Text
    print('--------------------------------------------------------')
    print('Visualization of Batters faced by ID:', pitcher_id, ' ', pitcher_name, ' over the Season')
    print('Pitcher ', pitcher_name, 'Pitched a Total of ', total_innings, ' innings and an Average of ', np.round(avg_innings, decimals=1), ' innings over the Season')
    print('Pitcher ', pitcher_name, 'Faced a Total of ', total_batters, ' batters and an Average of ', np.round(avg_batters, decimals=0), ' batters over the Season')
    print('Pitcher ', pitcher_name, 'Faces an Average of ', np.round(avg_batters_inning, decimals=1), ' batters per inning on a given game')
    print('--------------------------------------------------------')

    #Visualization
    plot = go.Figure(data=[go.Bar( 
        name = 'Innings Thrown', 
        x = combined_data['date'], 
        y = combined_data["inning"] 
       ), 
                           go.Bar( 
        name = 'Batters Faced', 
        x = combined_data['date'], 
        y = combined_data['batters_faced']
       ),
                           go.Line(
        name = 'Batter per Inning Ratio',
        x = combined_data['date'],
        y = combined_data['ratio']
       ),
    ]) 
    plot.add_shape(type = "line", line_color = "RebeccaPurple", line_width = 3, opacity = 1, line_dash = "dot",
                  x0=0, x1=1, xref= "paper", y0=3, y1=3, yref="y")
    plot.update_layout(title_text = "<b>Batters Faced Compared to Innings Thrown per Game over the Season</b>")
    plot.update_xaxes(title_text = '<b>Game Date</b>')
    plot.update_yaxes(title_text = '<b>Numbers Over the Season</b>')

    plot.show()


# In[ ]:


#plot_compare_innings_batters()


# In[ ]:


#Compare two pitchers in terms of batters faced and innings thrown
def plot_compare_two_pitchers():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np      
    
    print('This metrics return a comparison of two pitchers in terms of batters faced and innings thrown over a season')
    print('\n')
    print('Enter First Pitcher ID Number and Press Enter: ')
    pitcher_id1 = int(input())
    print('Enter Second Pitcher ID Number and Press Enter: ')
    pitcher_id2 = int(input())

    joined_atbats = get_data()
    
    #Search for that specific pitcher by ID
    pitcher_data1 = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id1] #test
    pitcher_data2 = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id2] #test
    pitcher_name1 = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id1]['full_name'].unique()
    pitcher_name2 = joined_atbats[joined_atbats['pitcher_id'] == pitcher_id2]['full_name'].unique()
    
    #Aggregate data by games and innings and dates
    innings_num1 = pitcher_data1.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index() #test1
    games_num1 = pitcher_data1.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index() #test2
    c1 = pd.concat([games_num1, innings_num1], axis=1).drop(columns=['index'])
    batters_num1 = pitcher_data1.groupby(by=['g_id','inning'], as_index=True)['batter_id'].nunique().reset_index() #test3
    c2 = batters_num1.groupby(by=['g_id'], as_index=True)['batter_id'].sum().reset_index() #25, 21
    combined_data1 = pd.merge(c1, c2,how='left', on = 'g_id')
    combined_data1 = combined_data1.rename(columns = {'unique': 'date', 'batter_id': 'batters_faced'})
    combined_data1['ratio'] = np.round(combined_data1['batters_faced'] / combined_data1['inning'], decimals = 2)    
    
    #Get aggregated table of batters faced per game for Pitcher1
    innings_num1 = pitcher_data1.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index() #test1
    games_num1 = pitcher_data1.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index() #test2
    c1 = pd.concat([games_num1, innings_num1], axis=1).drop(columns=['index'])
    batters_num1 = pitcher_data1.groupby(by=['g_id','inning'], as_index=True)['batter_id'].nunique().reset_index() #test3
    c2 = batters_num1.groupby(by=['g_id'], as_index=True)['batter_id'].sum().reset_index() #25, 21
    combined_data1 = pd.merge(c1, c2,how='left', on = 'g_id')
    combined_data1 = combined_data1.rename(columns = {'unique': 'date', 'batter_id': 'batters_faced'})
    combined_data1['ratio'] = np.round(combined_data1['batters_faced'] / combined_data1['inning'], decimals = 2)
    
    #Get aggregated table of batters faced per game for Pitcher2
    innings_num2 = pitcher_data2.groupby(by=['g_id'], as_index=False)['inning'].nunique().reset_index() #test1
    games_num2 = pitcher_data2.groupby(by=['g_id'], as_index=False)['date'].agg([np.unique]).reset_index() #test2
    c3 = pd.concat([games_num2, innings_num2], axis=1).drop(columns=['index'])
    batters_num2 = pitcher_data2.groupby(by=['g_id','inning'], as_index=True)['batter_id'].nunique().reset_index() #test3
    c4 = batters_num2.groupby(by=['g_id'], as_index=True)['batter_id'].sum().reset_index() #25, 21
    combined_data2 = pd.merge(c3, c4,how='left', on = 'g_id')
    combined_data2 = combined_data2.rename(columns = {'unique': 'date', 'batter_id': 'batters_faced'})
    combined_data2['ratio'] = np.round(combined_data2['batters_faced'] / combined_data2['inning'], decimals = 2)
    
    #Get some performance metrics
    #Pitcher1
    #Total innings pitched
    total_innings1 = combined_data1['inning'].sum()
    #Average innings pitched
    avg_innings1 = combined_data1['inning'].mean()
    #Get Rolling Average of last 3 games pitched innings
    combined_data1['rolling_mean_innings'] = np.round(combined_data1['inning'].rolling(window=3).mean(), decimals = 0)
    
    #Total batters faced
    total_batters1 = combined_data1['batters_faced'].sum()
    #Average batters faced per game
    avg_batters1 = combined_data1['batters_faced'].mean()
    #Get Rolling Average of last 3 games' batters faced number
    combined_data1['rolling_mean_batters_faced'] = np.round(combined_data1['batters_faced'].rolling(window=3).mean(), decimals = 0)
    
    #Average batters faced per inning
    avg_batters_inning1 = np.round(total_batters1 / total_innings1, decimals = 0)  
    
    #Get some performance metrics
    #Pitcher2
    #Total innings pitched
    total_innings2 = combined_data2['inning'].sum()
    #Average innings pitched
    avg_innings2 = combined_data2['inning'].mean()
    #Get Rolling Average of last 3 games pitched innings
    combined_data2['rolling_mean_innings'] = np.round(combined_data2['inning'].rolling(window=3).mean(), decimals = 0)
    
    #Total batters faced
    total_batters2 = combined_data2['batters_faced'].sum()
    #Average batters faced per game
    avg_batters2 = combined_data2['batters_faced'].mean()
    #Get Rolling Average of last 3 games' batters faced number
    combined_data2['rolling_mean_batters_faced'] = np.round(combined_data2['batters_faced'].rolling(window=3).mean(), decimals = 0)
    
    #Average batters faced per inning
    avg_batters_inning2 = np.round(total_batters2 / total_innings2, decimals = 0)
    
    #Output: Text and Visualization   
    #Text
    print('--------------------------------------------------------')
    print('Comparing Two Pitchers in terms of Innings Thrown and Batters Faced per Game over a Season')
    compare = pd.DataFrame(columns = ['Pitcher ID', 'Pitcher Name', 'Total Innings', 'Avg. Innings',
                                     'Total Batters Faced', 'Avg. Batters Faced'],
                          data = [[pitcher_id1, pitcher_name1, total_innings1, avg_innings1,
                                  total_batters1, avg_batters1],
                                 [pitcher_id2, pitcher_name2, total_innings2, avg_innings2,
                                 total_batters2, avg_batters2]])
    print('--------------------------------------------------------')
    
    #Visualization
    plot = go.Figure(data=[go.Bar( 
        name = 'Innings Thrown by Pitcher 1', 
        x = combined_data1['date'], 
        y = combined_data1["inning"],
        offsetgroup = 0,
       ), 
                           go.Bar( 
        name = 'Innings Thrown by Pitcher 2', 
        x = combined_data2['date'], 
        y = combined_data2['inning'],
        offsetgroup = 1,
       ),
                           go.Line(
        name = 'Batter per Inning Ratio by Pitcher 1',
        x = combined_data1['date'],
        y = combined_data1['ratio']
       ),
                            go.Line(
        name = 'Batter per Inning Ratio by Pitcher 2',
        x = combined_data2['date'],
        y = combined_data2['ratio']
       ),                    
    ]) 
    plot.add_shape(type = "line", line_color = "RebeccaPurple", line_width = 3, opacity = 1, line_dash = "dot",
                  x0=0, x1=1, xref= "paper", y0=3, y1=3, yref="y")
    plot.update_layout(title_text = "<b>Comparison of Two Pitchers in Terms of Innings Thrown and Batters Faced</b>")
    plot.update_xaxes(title_text = '<b>Game Date</b>')
    plot.update_yaxes(title_text = '<b>Numbers Over the Season</b>')

    return compare, plot.show()


# In[ ]:


#plot_compare_two_pitchers() # 502239,  547943 # 607192, 502239


# In[ ]:


def plot_top_pitchers_per_team():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np 
    
    print('This metrics return a comparison of top 5 pitchers in terms of batters faced and innings thrown over a season per team')
    print('\n')
    print('Enter a team by its abbreviation (e.g.: "oak" for Oakland Athletics and "bos" for Boston Red Sox)')
    team = input()

    joined_atbats = get_data()
    

    teams_home = joined_atbats[joined_atbats['home_team'] == team] #e.g.: "oak"
    teams_away = joined_atbats[joined_atbats['away_team'] == team] #e.g.: "oak"
    teams_comb = pd.concat([teams_home, teams_away], axis = 0)

    #Get total innings thrown by pitcher
    total_innings = teams_comb.groupby(by = ['full_name', 'g_id'], as_index = True)['inning'].nunique().reset_index().sort_values('inning', ascending=False)
    total_innings = total_innings.groupby(by = ['full_name'], as_index = True)['inning'].sum().reset_index().sort_values('inning', ascending=False)
    #total_innings.head()

    #Get total games played by pitcher
    total_games = teams_comb.groupby(by = ['full_name'], as_index = True)['g_id'].nunique().reset_index().sort_values('g_id', ascending=False)
    #total_games.head()

    #Get total batters faced by pitcher
    total_batters_faced = teams_comb.groupby(by = ['full_name'], as_index = True)['batter_id'].nunique().reset_index().sort_values('batter_id', ascending=False)
    #total_batters_faced.head()

    #Combine into one datafrme
    combined_df = pd.merge(total_innings, total_games, how='left', on='full_name')
    combined_df = pd.merge(combined_df, total_batters_faced, how='left', on='full_name')
    combined_df = combined_df.rename(columns = {'g_id': 'games_played', 'inning': 'innings_thrown', 'batter_id': 'batters_faced'})
    combined_df['avg_innings'] = np.round(combined_df['innings_thrown'] / combined_df['games_played'], decimals = 1)
    combined_df['avg_batters_faced'] = np.round(combined_df['batters_faced'] / combined_df['innings_thrown'], decimals = 1)

    #Lets plot the top 5 pitchers with the most innings, games played, and batters_faced
    #Get dataframes
    combined_df1 = combined_df.sort_values('avg_innings', ascending=False).head(5)
    combined_df2 = combined_df.sort_values('innings_thrown', ascending=False).head(5)
    combined_df3 = combined_df.sort_values('avg_batters_faced', ascending=False).head(5)

    #Visualization
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Top 5 Avg. Innings Thrown by Pitcher", 
                                        "Top 5 Total Innings Thrown by Pitcher", 
                                        "Top 5 Avg. Batters Faced per Inning by Pitcher")
    )

    fig.add_trace(
        go.Bar(x=combined_df1['full_name'], 
               y=combined_df1['avg_innings']),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=combined_df2['full_name'], 
               y=combined_df2['innings_thrown']),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=combined_df3['full_name'],
               y=combined_df3['avg_batters_faced']),
        row=1, col=3
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Pitcher", row=1, col=1)
    fig.update_xaxes(title_text="Pitcher", row=1, col=2)
    fig.update_xaxes(title_text="Pitcher", row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text="Total Innings", row=1, col=1)
    fig.update_yaxes(title_text="Avg. Innings", row=1, col=2)
    fig.update_yaxes(title_text="Avg. # of Batters", row=1, col=3)

    # Update title and height
    fig.update_layout(title_text="Top 5 Statistics by Team", width = 1200, height=700)

    fig.show()


# In[ ]:


#plot_top_pitchers_per_team()

