import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
import os
from bs4 import BeautifulSoup
import datetime
import time


chromedriver = "C:/bin/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument('log-level=3')
chrome_options.add_argument("--start-maximized")
chrome_options.add_experimental_option(
    "excludeSwitches", ["enable-logging"]
)
driver = webdriver.Chrome(service=Service("C:/bin/chromedriver.exe"), options=chrome_options)

'''
fbs_season = [f"https://www.espn.com/college-football/scoreboard/_/week/{x}/year/2024/seasontype/2/group/80" for x in range(1,17)]
fcs_season = [f"https://www.espn.com/college-football/scoreboard/_/week/{x}/year/2024/seasontype/2/group/81" for x in range(1,17)]
postseason = ["https://www.espn.com/college-football/scoreboard/_/week/1/year/2024/seasontype/3/group/80",
              "https://www.espn.com/college-football/scoreboard/_/week/999/year/2024/seasontype/3/group/80",
              "https://www.espn.com/college-football/scoreboard/_/week/1/year/2024/seasontype/3/group/81"]
url_list = fbs_season + fcs_season# + postseason
'''
fbs_season = [f"https://www.espn.com/college-football/scoreboard/_/week/{x}/year/2025/seasontype/2/group/80" for x in range(1,17)]
fcs_season = [f"https://www.espn.com/college-football/scoreboard/_/week/{x}/year/2025/seasontype/2/group/81" for x in range(1,17)]
postseason = ["https://www.espn.com/college-football/scoreboard/_/week/1/year/2025/seasontype/3/group/80",
              "https://www.espn.com/college-football/scoreboard/_/week/999/year/2025/seasontype/3/group/80",
              "https://www.espn.com/college-football/scoreboard/_/week/1/year/2025/seasontype/3/group/81"]
url_list = fbs_season + fcs_season# + postseason



def summary_from_tbl(game):
    teams = [x.text for x in game.find_all('div', {'class': 'ScoreCell__TeamName ScoreCell__TeamName--shortDisplayName db'})]
    scores = [x.text for x in game.find_all('div', {'class': "ScoreCell__Score h4 clr-gray-01 fw-heavy tar ScoreCell_Score--scoreboard pl2"})]
    return np.array(teams + scores)

all_results = []

for url in url_list:
    try:
        driver.get(url)
        week_response = driver.page_source
        week_soup = BeautifulSoup(week_response, "html.parser")
        days = week_soup.find_all('section', {'class': "Card gameModules"})
        for day in days:

            games = day.find_all('ul',{'class':"ScoreboardScoreCell__Competitors"})
            day_results = pd.DataFrame([summary_from_tbl(x) for x in games], columns = ['Away','Home','Away_Score','Home Score'])

            date = day.find("h3", {"class":"Card__Header__Title Card__Header__Title--no-theme"}).text
            day_results['Date'] = pd.to_datetime(date).strftime("%Y-%m-%d")
            all_results.append(day_results)

    except:
        pass

all_games = pd.concat(all_results)
all_games.to_csv('season_2025.csv', index=False)

driver.quit()