from funcs import *

df = prepare_df()
grouping_dates = df.grouping_date.unique()
all_confs = set(df.conf)
all_teams = set(df.Team)

#for weeknum in grouping_dates:
weeknum = grouping_dates[0]
nextweek = grouping_dates[1]

subset = df[df.grouping_date <= weeknum]
current_fut = df[df.grouping_date == nextweek]

conf_scores = prerate_confs_weekly(subset, all_confs)
#naive_scores = prepare_confs_all(subset, all_confs)
teambase, rough_histories = naive_team_run(subset, conf_scores, df)
teambase, last_week = grouped_team_run(subset, teambase)

## Make predictions for current_fut

score_factor = teambase[current_fut.Team.iloc[0]].score_factor
current_fut['score_diff'] = current_fut.Score - current_fut.Opp_Score
current_fut['outcome_expectation'] = 1 / (1 + 10 ** ((current_fut.score_diff * score_factor) / -score_factor**2))
def predict_from_row(row):
    expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims = teambase[row['Team']].predict(teambase[row['Opp']], row['Home'])
    return expectation, spread
current_fut['expectation'], current_fut['spread'] = zip(*current_fut.apply(predict_from_row, axis=1))
current_fut['pred_sd'] = current_fut.apply(lambda x: np.sqrt((teambase[x['Team']].sigma ** 2) + (teambase[x['Opp']].sigma ** 2)) / score_factor, axis = 1)
current_fut['outcome_z'] = (current_fut.score_diff - current_fut.spread) / current_fut.pred_sd





row = current_fut.iloc[0]
team_obj = teambase[row.Team]
opp_obj = teambase[row.Opp]

expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims = team_obj.predict(opp_obj, home = row.Home)

score_diff = row.score_diff
outcome = row.Result
outcome_expectation = 1 / (1 + 10 ** ((score_diff * team_obj.score_factor) / -team_obj.score_factor**2))
pred_sd = np.sqrt((team_obj.sigma ** 2) + (opp_obj.sigma ** 2)) / team_obj.score_factor
outcome_z = (score_diff - spread) / pred_sd




