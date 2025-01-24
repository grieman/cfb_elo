import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import colorcet
import matplotlib.colors as mcolors
import pickle
import gzip

score_factor=25


with gzip.open('teambase.gz', 'rb') as compressed_file:
    teambase = pickle.load(compressed_file)

club_histories = pd.concat([x.return_history() for x in teambase.values()])


recent_matches = club_histories[(~club_histories.Opponent.isnull()) & (club_histories.Point_Diff > 0)].copy()
edges_wins = recent_matches.groupby(['Team','Opponent']).Point_Diff.count().reset_index()
edges_wins.columns = ['winner','loser','count']
edges_wins.winner = edges_wins.winner.str.encode('utf-8').str.decode('ascii', 'ignore')
edges_wins.loser = edges_wins.loser.str.encode('utf-8').str.decode('ascii', 'ignore') 
## Can we add a score margin as a weight?

import networkx as nx
import gravis as gv
# Creating Undirected graph            
G = nx.from_pandas_edgelist(edges_wins, source='winner', target='loser', edge_attr='count')
centrality = nx.algorithms.degree_centrality(G)
#cent_df = pd.DataFrame(centrality.items(), columns = ['team', 'cent'])
communities = nx.algorithms.community.greedy_modularity_communities(G)
nx.set_node_attributes(G, centrality, 'size')
# Assignment of node colors
colors = colorcet.glasbey[0:len(communities)]
for community, color in zip(communities, colors):
    for node in community:
        G.nodes[node]['color'] = color

# Drawing that graph in matplotlib
nx.draw(G, node_size=40)

# Drawing in gravis
fig = gv.d3(G, use_node_size_normalization=True, node_size_normalization_max=30,
      use_edge_size_normalization=True, edge_size_data_source='weight', edge_curvature=0.3, zoom_factor=0.6)

with open('network.html', 'wb+') as f:
    f.write(fig.to_html().encode('utf-8'))