import pandas as pd
import numpy as np
import math
import glob
import os

class elo_score:
    def __init__(self, id, starting_mu=1500, starting_sigma=100, score_factor=10, home_advantage = 3):
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

    def return_history(self):
        output = pd.DataFrame(self.history)
        output.columns = ['Date', 'Opponent',"Opponent_mu",'Opponent_sigma','Prediction','Score', 'Opponent_Score', 'Point_Diff', 'home', 'mu','sigma']
        output['Team'] = self.id
        return output

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
df['grouping_date'] = df.Day.dt.isocalendar().week
#df['grouping_date'] = df.Day.dt.month
#df['grouping_date'] = df.Day.dt.year

df['Result'] = (np.sign(df.Score - df.Opp_Score) + 1) / 2


teambase = dict()
c = 15
grouping_dates = list(set(df.grouping_date))
for group in grouping_dates:
    match_group = df[df.grouping_date == group]
    active_teams = set(match_group.Team)
    opponent_lists = list()

    for teamname in active_teams.difference(set(teambase.keys())):
        teambase[teamname] = elo_score(teamname)

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
    
club_histories = pd.concat([x.return_history() for x in teambase.values()])
#club_histories[club_histories.Date == 12].sort_values('mu', ascending=False).head(10)

#club_histories[club_histories.Opponent.isnull()].sort_values('mu', ascending=False).head(10)

last_week = club_histories[club_histories.Date == 51]

last_week['rating'] = last_week.mu - (last_week.sigma * 2)
last_week = last_week[['Team','rating','mu','sigma']]
last_week = last_week.sort_values('rating', ascending=False).reset_index(drop=True)
last_week.head(25)

## Issue - TAMU and NIU rated both as 1500s at beginning of year