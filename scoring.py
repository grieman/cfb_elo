from funcs import *
from sklearn.metrics import mean_squared_error

df = prepare_df()
grouping_dates = df.grouping_date.unique()
all_confs = set(df.conf)
all_teams = set(df.Team)

#for weeknum in grouping_dates:
weeknum = grouping_dates[0]
nextweek = grouping_dates[1]

subset = df[df.grouping_date <= weeknum].copy()
current_fut = df[df.grouping_date == nextweek].copy()

teambase = model_for_week(subset, all_confs, df)

brier, brier_skill, rmse, rmse_skill, avg_z, avg_abs_z, predset = eval_weekly(teambase, subset, current_fut)

## Make predictions for current_fut
# Naive prediction == average margin of home team from previous weeks?




score_factor = teambase[current_fut.Team.iloc[0]].score_factor
current_fut['score_diff'] = current_fut.Score - current_fut.Opp_Score
current_fut['outcome_expectation'] = 1 / (1 + 10 ** ((current_fut.score_diff * score_factor) / -score_factor**2))
def predict_from_row(row):
    expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims = teambase[row['Team']].predict(teambase[row['Opp']], row['Home'])
    return expectation, spread
current_fut['expectation'], current_fut['spread'] = zip(*current_fut.apply(predict_from_row, axis=1))
current_fut['pred_sd'] = current_fut.apply(lambda x: np.sqrt((teambase[x['Team']].sigma ** 2) + (teambase[x['Opp']].sigma ** 2)) / score_factor, axis = 1)
current_fut['outcome_z'] = (current_fut.score_diff - current_fut.spread) / current_fut.pred_sd

## Calc brier, average point error, average z_score?
eval_set = current_fut[current_fut.Home == 1]

n_brier = brier_simple(eval_set.Result, subset.Result.mean())
brier = brier_simple(eval_set.Result, eval_set.expectation)
brier_skill = 1 - (brier/n_brier)
print(np.round(brier, 4), np.round(brier_skill, 3))

print(np.round(eval_set.outcome_z.mean(), 4), np.round(abs(eval_set.outcome_z).mean(), 4))

rmse = mean_squared_error(eval_set.score_diff, eval_set.spread, squared=False)
print(np.round(rmse, 4))





row = current_fut.iloc[0]
team_obj = teambase[row.Team]
opp_obj = teambase[row.Opp]

expectation, spread, expectations, sim_elo_spreads, self_sims, opp_sims = team_obj.predict(opp_obj, home = row.Home)

score_diff = row.score_diff
outcome = row.Result
outcome_expectation = 1 / (1 + 10 ** ((score_diff * team_obj.score_factor) / -team_obj.score_factor**2))
pred_sd = np.sqrt((team_obj.sigma ** 2) + (opp_obj.sigma ** 2)) / team_obj.score_factor
outcome_z = (score_diff - spread) / pred_sd




