import pandas as pd
import numpy as np
import math
import glob
import os
from conference_mappings import conf_levels, conf_mapping
import gzip
import pickle

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
