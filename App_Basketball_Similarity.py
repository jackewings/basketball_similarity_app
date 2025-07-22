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

# Filter for season 2025
per_game_25 = per_game[per_game['season'] == 2025]

# Aggregate stats by player_id, taking first player name and mean of stats
per_game_25_agg = per_game_25.groupby('player_id').agg({
    'player': 'first',
    'pts_per_game': 'mean',
    'trb_per_game': 'mean',
    'ast_per_game': 'mean',
    'fg_percent': 'mean',
    'x3p_percent': 'mean'
}).reset_index(drop=True)

# Features to use
features = ['pts_per_game', 'trb_per_game', 'ast_per_game', 'fg_percent', 'x3p_percent']

# Drop rows with missing feature values
per_game_25_agg = per_game_25_agg.dropna(subset=features).reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(per_game_25_agg[features])

# Streamlit app title
st.title('Player Similarity Tool')

# Player selectbox (sorted unique names)
selected_player = st.selectbox('Select a player', per_game_25_agg['player'].sort_values().unique())

# Find index of selected player
matching_rows = per_game_25_agg[per_game_25_agg['player'] == selected_player]
if matching_rows.empty:
    st.error("Selected player not found in the data.")
    st.stop()
idx = matching_rows.index[0]

# Calculate cosine similarities
similarities = cosine_similarity([X_scaled[idx]], X_scaled)[0]

# Add similarity scores to dataframe
per_game_25_agg['Similarity'] = similarities

# Get top 5 most similar players excluding the selected player
top_similar = per_game_25_agg[per_game_25_agg['player'] != selected_player] \
                .sort_values(by='Similarity', ascending=False) \
                .head(5)

# Display results
st.subheader(f'Players most similar to {selected_player}')
st.dataframe(top_similar[['player', 'Similarity']])
