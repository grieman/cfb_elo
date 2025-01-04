import pandas as pd
import numpy as np
import math
import glob
import os
from conference_mappings import conf_levels, conf_mapping


class elo_score:
    def __init__(self, id, starting_mu=1500, starting_sigma=250, score_factor=25, home_advantage = 3):
        self.id = id
        self.mu = starting_mu
        self.sigma = starting_sigma
        self.sigma_start = starting_sigma
        self.score_factor = score_factor
        self._Q = math.log(10) / score_factor**2
        self.home_score_adv = home_advantage * self.score_factor
        self.history = []

    def rest_period(self, date, c=15):
        self.sigma = min(math.sqrt(self.sigma ** 2 + c ** 2), self.sigma_start)
        self.history.append([date, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.round(self.mu, 2), np.round(self.sigma, 2)])
    
    def _impact(self, sigma):
        return 1 / math.sqrt(1 + 3*(self._Q**2)*(sigma**2)/(math.pi**2))

    def _expectation(self, own_mu, other_mu, g_sigma):
        return 1 / (1 + 10 ** (g_sigma * (own_mu - other_mu) / -self.score_factor**2))

    def update(self, opponent_list, groupdate):
        difference = 0
        d_sum = 0
        for (o_name, o_mu, o_sigma, score, o_score, score_diff, _, home, date) in opponent_list:
            if home == 1:
                mu = self.mu + (self.home_score_adv )
            elif home == 0:
                mu = self.mu
                o_mu = o_mu + (self.home_score_adv )
            
            impact = self._impact(o_sigma)
            expectation = self._expectation(mu, o_mu, impact)
            spread = (mu - o_mu) / self.score_factor
            point_outcome = 1 / (1 + 10 ** ((score_diff * self.score_factor) / -self.score_factor**2))

            difference += impact * (point_outcome - expectation) * 2 # we may want to increase this 2, but keeps model adapting quickly
            d_sum += expectation * (1 - expectation) * impact**2 * (self._Q**2)

            self.history.append([
                date, o_name, o_mu, o_sigma, spread,
                score, o_score, score_diff, home, np.round(mu, 2), np.round(self.sigma, 2)])

        denom = self.sigma ** -2 + d_sum
        self.mu += (self._Q/denom) * difference
        self.sigma = math.sqrt(1 / denom)
        self.history.append([groupdate, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.round(self.mu, 2), np.round(self.sigma, 2)])
    
    def update_naive(self, opponent_list):
        difference = 0
        d_sum = 0
        for (o_name, o_mu, o_sigma, score, o_score, score_diff, _, home, date) in opponent_list:
            if home == 1:
                mu = self.mu + (self.home_score_adv )
            elif home == 0:
                mu = self.mu
                o_mu = o_mu + (self.home_score_adv )
            
            impact = self._impact(o_sigma)
            expectation = self._expectation(mu, o_mu, impact)
            spread = (mu - o_mu) / self.score_factor
            point_outcome = 1 / (1 + 10 ** ((score_diff * self.score_factor) / -self.score_factor**2))

            difference += impact * (point_outcome - expectation) * 2 # we may want to increase this 2, but keeps model adapting quickly
            d_sum += expectation * (1 - expectation) * impact**2 * (self._Q**2)

        denom = self.sigma ** -2 + d_sum
        self.mu += (self._Q/denom) * difference
        #self.sigma = math.sqrt(1 / denom)
        self.history.append(['NAIVE', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.round(self.mu, 2), np.round(self.sigma, 2)])

    def return_history(self):
        output = pd.DataFrame(self.history)
        output.columns = ['Date', 'Opponent',"Opponent_mu",'Opponent_sigma','Prediction','Score', 'Opponent_Score', 'Point_Diff', 'home', 'mu','sigma']
        output['Team'] = self.id
        return output

    def predict(self, opponent_obj, home=True, n_sims=1000):
        if home == 'neutral':
            own_mu = self.mu
            own_sigma = self.sigma
            opponent_mu = opponent_obj.mu
            opponent_sigma = opponent_obj.sigma

        elif home == True:
            own_mu = self.mu + self.home_score_adv
            own_sigma = np.sqrt(self.sigma**2 + 10**2)  
            opponent_mu = opponent_obj.mu
            opponent_sigma = opponent_obj.sigma

        else:
            own_mu = self.mu
            own_sigma = self.sigma
            opponent_mu = opponent_obj.mu + self.home_score_adv
            opponent_sigma = np.sqrt(opponent_obj.sigma**2 + 10**2)  
        
        impact = self._impact(opponent_sigma)
        expectation = self._expectation(own_mu, opponent_mu, impact)
        spread = (own_mu - opponent_mu) / self.score_factor
        self_sims = np.random.normal(own_mu, own_sigma, size=n_sims)
        opp_sims = np.random.normal(opponent_mu, opponent_sigma, size=n_sims)
        expectations = [(1 / (1 + 10 ** (impact * (x - y) / -self.score_factor**2))) for (x, y) in zip(self_sims, opp_sims)]
        sim_elo_diffs = (self_sims - opp_sims)
        sim_elo_spreads = sim_elo_diffs / self.score_factor

        return expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims
    

def prepare_df(grouping='week'):
    df = pd.read_csv("season.csv").drop_duplicates()
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
    if grouping == 'week':
        df['grouping_date'] = df.Day.dt.isocalendar().week
    elif grouping == 'month':
        df['grouping_date'] = df.Day.dt.month
    else:
        df['grouping_date'] = df.Day.dt.year

    df['Result'] = (np.sign(df.Score - df.Opp_Score) + 1) / 2
    df['conf'] = df.Team.map(conf_mapping).fillna('D2')
    df['division'] = df.conf.map(conf_levels).fillna('None')
    df['opp_conf'] = df.Opp.map(conf_mapping).fillna('D2')
    df['opp_division'] = df.opp_conf.map(conf_levels).fillna('None')

    return df

def prerate_confs_weekly(df, all_confs):
    ooc_games = df[df.conf != df.opp_conf]

    confbase = dict()
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
            confbase[resting_club].rest_period(group)
            
        # Process each club
        for clubname, opponent_list in list(zip(active_confs, opponent_lists)):
            confbase[clubname].update(opponent_list, group)
        
    conf_histories = pd.concat([x.return_history() for x in confbase.values()])
    conf_histories = conf_histories[conf_histories.Score.isnull()]
    conf_scores = conf_histories[conf_histories.Date == conf_histories.Date.max()]
    return conf_scores

def prepare_confs_all(df, all_confs):

    ooc_games = df[df.conf != df.opp_conf]

    confbase = dict()
    opponent_lists = list()

    for confname in all_confs:
        confbase[confname] = elo_score(confname)
    for confname in all_confs:
        conf_subset = ooc_games[ooc_games.conf == confname]
        opponent_list = list()
        for _, row in conf_subset.iterrows():
            opponent_list.append(
                (row['opp_conf'], confbase[row['opp_conf']].mu, confbase[row['opp_conf']].sigma, row['Score'], row['Opp_Score'],
                row['Score'] - row['Opp_Score'], row['Result'], row['Home'], row['Date'])
                )
        opponent_lists.append(opponent_list)        
    for clubname, opponent_list in list(zip(all_confs, opponent_lists)):
        confbase[clubname].update(opponent_list, 'NAIVE')

    conf_histories = pd.concat([x.return_history() for x in confbase.values()])
    naive_scores = conf_histories[conf_histories.Date == "NAIVE"]
    return naive_scores

def naive_team_run(df, conf_scores, all_data):
    teambase = dict()
    all_teams = set(df.Team)
    opponent_lists = list()

    for conf in set(conf_scores.Team):
        conf_subset = all_data[all_data.conf == conf]
        conf_teams = set(conf_subset.Team)
        conf_mu = conf_scores[conf_scores.Team == conf].mu.iloc[0]
        conf_sigma = conf_scores[conf_scores.Team == conf].sigma.iloc[0]
        for teamname in conf_teams:
            teambase[teamname] = elo_score(teamname, conf_mu, (2*conf_sigma))
    
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
    
    rough_histories = pd.concat([x.return_history() for x in teambase.values() if len(x.history) > 0])
    rough_histories = rough_histories[rough_histories.Date == "NAIVE"]
    return teambase, rough_histories

def grouped_team_run(df, teambase):
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
            teambase[resting_club].rest_period(group)
            
        # Process each club
        for clubname, opponent_list in list(zip(active_teams, opponent_lists)):
            teambase[clubname].update(opponent_list, group)
        
    club_histories = pd.concat([x.return_history() for x in teambase.values() if len(x.history) > 0])

    last_week = club_histories[club_histories.Score.isnull()]
    last_week = last_week[last_week.Date != "NAIVE"]
    last_week = last_week[last_week.Date == last_week.Date.max()]
    last_week['rating'] = last_week.mu - (last_week.sigma * 2)
    last_week = last_week[['Team','rating','mu','sigma']]
    last_week = last_week.sort_values('rating', ascending=False).reset_index(drop=True)
    return teambase, last_week




def model_for_week(df):
    conf_scores = prerate_confs_weekly(df)
    naive_scores = prepare_confs_all(df)
    teambase, rough_histories = naive_team_run(df, conf_scores)
    teambase, last_week = grouped_team_run(df, teambase)


