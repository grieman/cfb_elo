import pandas as pd
import numpy as np
import math
import glob
import os
from conference_mappings import conf_levels, conf_mapping
from sklearn.metrics import mean_squared_error
import gzip
import pickle
from classes import elo_score

def load_data(file = 'season_2025.csv'):

    df = pd.read_csv(file).drop_duplicates()
    away_matches = df.copy()
    away_matches.columns = ['Team','Opp','Score',"Opp_Score","Date"]
    home_matches = df.copy()
    home_matches.columns = ['Opp','Team','Opp_Score',"Score","Date"]

    away_matches['Home'] = 0
    home_matches['Home'] = 1
    home_matches = home_matches[away_matches.columns]

    df = pd.concat([away_matches, home_matches])

    df = df[~df.Score.isnull()]
    df.Score = df.Score.astype(int)
    df.Opp_Score = df.Opp_Score.astype(int)

    df['Day'] = pd.to_datetime(df.Date)
    df['grouping_date'] = df.Day.dt.isocalendar().week
    #df['grouping_date'] = df.Day.dt.month
    #df['grouping_date'] = df.Day.dt.year

    df['Result'] = (np.sign(df.Score - df.Opp_Score) + 1) / 2
    df['conf'] = df.Team.map(conf_mapping).fillna('D2')
    df['division'] = df.conf.map(conf_levels).fillna('None')
    df['opp_conf'] = df.Opp.map(conf_mapping).fillna('D2')
    df['opp_division'] = df.opp_conf.map(conf_levels).fillna('None')

    return df


def make_rankings(df):

    fbs_teams = set(df[df.division == "FBS"].Team)
    fcs_teams = set(df[df.division == "FCS"].Team)
    d2_teams = set(df[df.division == "None"].Team)

    ## Need to limit the input dataset HERE if we want to not consider games
    #df = df[df.Day <= '2024-12-14']
    ooc_games = df[df.conf != df.opp_conf]


    ## Conference pass though
    confbase = dict()
    c = 15
    all_confs = set(df.conf)
    opponent_lists = list()

    grouping_dates = list(set(ooc_games.grouping_date))
    for confname in all_confs:
        confbase[confname] = elo_score(confname, starting_mu=1500, starting_sigma=250)

    for group in grouping_dates:
        match_group = ooc_games[ooc_games.grouping_date == group]
        active_confs = set(match_group.conf)
        opponent_lists = list()
        for confname in active_confs:
            conf_subset = match_group[match_group.conf == confname]
            opponent_list = list()
            for _, row in conf_subset.iterrows():
                opponent_list.append(
                    (row['opp_conf'], confbase[row['opp_conf']].mu, confbase[row['opp_conf']].sigma, row['Score'], row['Opp_Score'],
                    row['Score'] - row['Opp_Score'], row['Result'], row['Home'], row['Date'])
                    )
            opponent_lists.append(opponent_list)        

        for resting_club in set(confbase.keys()).difference(active_confs):
            #print(first_of_month, resting_club)
            confbase[resting_club].rest_period(group, c)
            
        # Process each club
        for clubname, opponent_list in list(zip(active_confs, opponent_lists)):
            confbase[clubname].update(opponent_list, group)
        
    '''
    for confname in all_confs:
        confbase[confname] = elo_score(confname)
    for confname in all_confs:
        conf_subset = df[df.conf == confname]
        opponent_list = list()
        for _, row in conf_subset.iterrows():
            opponent_list.append(
                (row['opp_conf'], confbase[row['opp_conf']].mu, confbase[row['opp_conf']].sigma, row['Score'], row['Opp_Score'],
                row['Score'] - row['Opp_Score'], row['Result'], row['Home'], row['Date'])
                )
        opponent_lists.append(opponent_list)        
    for clubname, opponent_list in list(zip(all_confs, opponent_lists)):
        confbase[clubname].update_naive(opponent_list)
    '''
    conf_histories = pd.concat([x.return_history() for x in confbase.values()])
    max_date = conf_histories.Date.iloc[-1]
    conf_scores = conf_histories[conf_histories.Date == max_date]

    conf_histories = pd.concat([x.return_history() for x in confbase.values()])
    max_date = conf_histories.Date.iloc[-1]
    conf_scores = conf_histories[conf_histories.Date == max_date]
    conf_scores.sort_values('mu', ascending=False)[['Team', 'mu', 'sigma']]

    ## Naive Team pass though
    teambase = dict()
    c = 15
    all_teams = set(df.Team)
    opponent_lists = list()

    for conf in set(conf_scores.Team):
        conf_subset = df[df.conf == conf]
        conf_teams = set(conf_subset.Team)
        ## Using this raw may double weight out of conference games. 
        # Try a dampener? first pass may not have helped much
        conf_mu = conf_scores[conf_scores.Team == conf].mu.iloc[0]# + 1500) / 2
        conf_sigma = conf_scores[conf_scores.Team == conf].sigma.iloc[0]
        for teamname in conf_teams:
            teambase[teamname] = elo_score(teamname, conf_mu, (2*conf_sigma))
    '''
    for teamname in fbs_teams:
        teambase[teamname] = elo_score(teamname, 1500)
    for teamname in fcs_teams:
        teambase[teamname] = elo_score(teamname, 1250)
    for teamname in d2_teams:
        teambase[teamname] = elo_score(teamname, 1000)
        ## Can we get the average MOV across FBS v FCS throughout the year? Downshift FCS starting values by that much
        ## Maybe more fair to do it by division?
    '''
    for teamname in all_teams:
        team_subset = df[df.Team == teamname]
        opponent_list = list()
        for _, row in team_subset.iterrows():
            opponent_list.append(
                (row['Opp'], teambase[row['Opp']].mu, teambase[row['Opp']].sigma, row['Score'], row['Opp_Score'],
                row['Score'] - row['Opp_Score'], row['Result'], row['Home'], row['Date'])
                )
        opponent_lists.append(opponent_list)        
    for clubname, opponent_list in list(zip(all_teams, opponent_lists)):
        teambase[clubname].update_naive(opponent_list)
    rough_histories = pd.concat([x.return_history() for x in teambase.values()])
    rough_histories = rough_histories[rough_histories.Date == "NAIVE"]


    ## Weekly, Actual rating
    grouping_dates = list(set(df.grouping_date))
    for group in grouping_dates:
        match_group = df[df.grouping_date == group]
        active_teams = set(match_group.Team)
        opponent_lists = list()

        for teamname in active_teams:
            team_subset = match_group[match_group.Team == teamname]
            opponent_list = list()
            for _, row in team_subset.iterrows():
                opponent_list.append(
                    (row['Opp'], teambase[row['Opp']].mu, teambase[row['Opp']].sigma, row['Score'], row['Opp_Score'],
                    row['Score'] - row['Opp_Score'], row['Result'], row['Home'], row['Date'])
                    )
            opponent_lists.append(opponent_list)        

        for resting_club in set(teambase.keys()).difference(active_teams):
            #print(first_of_month, resting_club)
            teambase[resting_club].rest_period(group, c)
            
        # Process each club
        for clubname, opponent_list in list(zip(active_teams, opponent_lists)):
            teambase[clubname].update(opponent_list, group)

    with gzip.open('teambase.gz', 'wb') as compressed_file:
        pickle.dump(teambase, compressed_file)

    club_histories = pd.concat([x.return_history() for x in teambase.values()])
    #club_histories[club_histories.Date == 12].sort_values('mu', ascending=False).head(10)

    non_null_histories = club_histories[club_histories.Point_Diff.isna() == False]
    non_null_histories = non_null_histories[non_null_histories.home == 1]
    rmse = math.sqrt(mean_squared_error(non_null_histories.Prediction, non_null_histories.Point_Diff))
    #print(f"RMSE: {np.round(rmse, 3)}")

    #club_histories[club_histories.Opponent.isnull()].sort_values('mu', ascending=False).head(10)
    max_date = club_histories.Date.iloc[-1]
    last_week = club_histories[club_histories.Date == max_date].copy()

    last_week['rating'] = last_week.mu - (last_week.sigma * 2)
    last_week = last_week[['Team','rating','mu','sigma']]
    last_week = last_week.sort_values('rating', ascending=False).reset_index(drop=True)

    return last_week, rmse


def historical_rankings(file = 'season_2025.csv'):

    df = load_data(file)
    all_weeks = list(set(df.grouping_date))

    all_rankings = list()
    for index, cutweek in enumerate(all_weeks):
        print(f"Week {index}")
        hist_df = df[df.grouping_date <= cutweek].copy()
        week_rankings, rmse = make_rankings(hist_df)
        #week_top25 = week_rankings.head(25).copy()
        week_rankings['Week'] = index
        week_rankings['Rank'] = [1 +x for x in list(week_rankings.index)]

        all_rankings.append(week_rankings)
    
    ranks = pd.concat(all_rankings)

    ranks['scaled_rating'] = ranks.groupby("Week").rating.transform(lambda x: (x - x.min() + 1) / (x.max() - x.min() + 1)).round(4)

    print(f"RMSE: {np.round(rmse, 3)}")
    return ranks


import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_n(ranks, n, save = False, scaled = True):

    recent_top_n = ranks[(ranks.Week == max(ranks.Week)) & (ranks.Rank <= n)].copy()
    recent_histories = ranks[ranks.Team.isin(recent_top_n.Team)].copy()

    if scaled == True:
        sns.lineplot(data=recent_histories, x="Week", y="scaled_rating", hue='Team')
    else:
        sns.lineplot(data=recent_histories, x="Week", y="rating", hue='Team')
    if save:
        plt.savefig(f'outputs/top_{n}.png')
    else:
        plt.show()


def top25_table(ranks, scaled = True):
    top25s = ranks[(ranks.Rank <= 25) & (ranks.Week > 0)].copy()
    top25_names = top25s.pivot(index = 'Rank', columns = 'Week', values = 'Team')
    if scaled == True:
        top25_ratings = top25s.pivot(index = 'Rank', columns = 'Week', values = 'scaled_rating')
        for col in top25_ratings.columns:
            top25_ratings[col] = top25_ratings[col].apply(lambda x: f" ({x:.3f})")
    else:
        top25_ratings = top25s.pivot(index = 'Rank', columns = 'Week', values = 'rating')
        for col in top25_ratings.columns:
            top25_ratings[col] = top25_ratings[col].apply(lambda x: f" ({x:.1f})")

    top25s_out = top25_names + top25_ratings
    top25s_out.columns = ['Team'] + [f'Week {x}' for x in top25s_out.columns[1:]]

    return top25s_out

def current_ranks(ranks):
    return ranks[(ranks.Week == max(ranks.Week))]

if __name__ == '__main__':
    ranks = historical_rankings()
    top_25 = top25_table(ranks, scaled=False)
    top_25.to_csv('outputs/top_25.csv')
    print(top_25)

    plot_top_n(ranks, 5, save = True)
    plot_top_n(ranks, 10, save = True)
    plot_top_n(ranks, 25, save = True)
    #print(current_ranks(ranks))