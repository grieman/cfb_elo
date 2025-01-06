import pandas as pd
import numpy as np
import math
import glob
import os
from conference_mappings import conf_levels, conf_mapping
from sklearn.metrics import mean_squared_error


def brier_simple(y, y_preds):
    return (np.subtract(y, y_preds)**2).mean()

def brier_score(y, y_preds):
    '''
    uncertainty: Max of .5*(1-.5) = 0.25
    resolution: max of uncertainty, bigger is better
    reliability: low number wanted.
    '''
    uncertainty = y.mean() * (1-y.mean())
    #groups = pd.qcut(y_preds, 50, labels=False)
    values = y_preds.unique()
    reliabilities = list() # closeness of the forecast to true, given forecast (also calibration)
    resolutions = list() # how much the conditional probabilities, given forecasts, differ from overall average
    sub_means = list()
    for value in values:
        sub_preds = y_preds[y_preds == value]
        sub_means.append(sub_preds.mean())
        sub_ys = y[y_preds == value]
        reliabilities.append((np.subtract(sub_preds, sub_ys)**2).mean())
        resolutions.append((np.subtract(sub_ys, y.mean())**2).mean())
    
    reliability = np.mean(reliabilities)
    resolution = np.mean(resolutions)

    score = reliability - resolution + uncertainty
    # old retunables: sub_means, reliabilities, resolutions

    return score, reliability, resolution, uncertainty

def overall_accuracy(match_df, n_brier, n_rmse, cutoff_year, cutoff_month):
    match_df = match_df[(match_df.year >= cutoff_year) & (match_df.month >= cutoff_month)]

    brier, _, _, _ = brier_score(match_df.result, match_df.adv_prediction)
    rmse = math.sqrt(mean_squared_error(match_df.point_diff, match_df.adv_spread))

    brier_skill = 1 - (brier / n_brier)
    rmse_skill = 1 - (rmse / n_rmse)

    return brier_skill, rmse_skill

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



def model_for_week(df, all_confs, all_data):
    conf_scores = prerate_confs_weekly(df, all_confs)
    #naive_scores = prepare_confs_all(df)
    teambase, rough_histories = naive_team_run(df, conf_scores, all_data)
    teambase, last_week = grouped_team_run(df, teambase)
    return teambase

def eval_weekly(teambase, df, nextweek):

    score_factor = teambase[nextweek.Team.iloc[0]].score_factor
    nextweek['score_diff'] = nextweek.Score - nextweek.Opp_Score
    nextweek['outcome_expectation'] = 1 / (1 + 10 ** ((nextweek.score_diff * score_factor) / -score_factor**2))
    def predict_from_row(row):
        expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims = teambase[row['Team']].predict(teambase[row['Opp']], row['Home'])
        return expectation, spread
    nextweek['expectation'], nextweek['spread'] = zip(*nextweek.apply(predict_from_row, axis=1))
    nextweek['pred_sd'] = nextweek.apply(lambda x: np.sqrt((teambase[x['Team']].sigma ** 2) + (teambase[x['Opp']].sigma ** 2)) / score_factor, axis = 1)
    nextweek['outcome_z'] = (nextweek.score_diff - nextweek.spread) / nextweek.pred_sd

    ## Calc brier, average point error, average z_score?
    eval_set = nextweek[nextweek.Home == 1].copy()
    eval_subset = df[df.Home == 1].copy()
    eval_subset['score_diff'] = eval_subset.Score - eval_subset.Opp_Score
    eval_set['naive_scorediff'] = eval_subset.score_diff.mean()

    n_brier = brier_simple(eval_set.Result, eval_subset.Result.mean())
    brier = brier_simple(eval_set.Result, eval_set.expectation)
    brier_skill = 1 - (brier/n_brier)
    avg_z = eval_set.outcome_z.mean()
    avg_abs_z = abs(eval_set.outcome_z.mean())
    n_rmse = mean_squared_error(eval_set.score_diff, eval_set.naive_scorediff, squared=False)
    rmse = mean_squared_error(eval_set.score_diff, eval_set.spread, squared=False)
    rmse_skill = 1 - (rmse/n_rmse)

    return brier, brier_skill, rmse, rmse_skill, avg_z, avg_abs_z, nextweek

