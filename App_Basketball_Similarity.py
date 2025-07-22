#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
per_game = pd.read_csv('Player Per Game.csv')
per_game_25 = per_game[per_game['season'] == 2025]

# Aggregate per player
per_game_25_agg = per_game_25.groupby('player_id').agg({
    'player': 'first',
    'pts_per_game': 'mean',
    'trb_per_game': 'mean',
    'ast_per_game': 'mean',
    'fg_percent': 'mean',
    'x3p_percent': 'mean'
}).reset_index(drop=True)

# Features for similarity
features = ['pts_per_game', 'trb_per_game', 'ast_per_game', 'fg_percent', 'x3p_percent']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(per_game_25_agg[features])

# Streamlit UI
st.title('Player Similarity & Comparison Tool')

# Select two players
player_list = per_game_25_agg['player'].tolist()
player1 = st.selectbox('Select Player 1', player_list, index=0)
player2 = st.selectbox('Select Player 2', player_list, index=1)

# Get indices for players
idx1 = per_game_25_agg[per_game_25_agg['player'] == player1].index[0]
idx2 = per_game_25_agg[per_game_25_agg['player'] == player2].index[0]

# Calculate similarity
similarity_score = cosine_similarity([X_scaled[idx1]], [X_scaled[idx2]])[0][0]

st.subheader(f'Similarity between {player1} and {player2}: {similarity_score:.3f}')

# Show side-by-side stats
stats_df = per_game_25_agg.loc[[idx1, idx2], ['player'] + features].set_index('player').T
st.write('### Player Stats Comparison')
st.dataframe(stats_df.style.format("{:.2f}"))

