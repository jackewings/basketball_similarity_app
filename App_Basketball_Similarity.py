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

pd.set_option('display.max_columns', None)

# Filter for season 2025
per_game_25 = per_game[per_game['season'] == 2025]

# Aggregate stats per player
per_game_25_agg = per_game_25.groupby('player_id').agg({
    'player': 'first',
    'pts_per_game': 'mean',
    'trb_per_game': 'mean',
    'ast_per_game': 'mean',
    'fg_percent': 'mean',
    'x3p_percent': 'mean'
})

# Features to use
features = ['pts_per_game', 'trb_per_game', 'ast_per_game', 'fg_percent', 'x3p_percent']

# Clean data: replace inf/-inf with NaN, then drop rows with NaN in features
per_game_25_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
per_game_25_agg.dropna(subset=features, inplace=True)

# Reset index after dropping rows
per_game_25_agg = per_game_25_agg.reset_index(drop=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(per_game_25_agg[features])

# Streamlit UI
st.title('Player Similarity Tool')

# Select a player
selected_player = st.selectbox('Select a player', per_game_25_agg['player'])

# Find player index safely
matching_rows = per_game_25_agg[per_game_25_agg['player'] == selected_player]
if matching_rows.empty:
    st.error("Selected player not found in the data.")
    st.stop()

idx = matching_rows.index[0]

# Calculate similarities
similarities = cosine_similarity([X_scaled[idx]], X_scaled)[0]

# Add similarity scores to DataFrame
per_game_25_agg['Similarity'] = similarities

# Show top 5 most similar players (excluding selected player)
top_similar = per_game_25_agg[per_game_25_agg['player'] != selected_player] \
              .sort_values(by='Similarity', ascending=False) \
              .head()

st.subheader(f'Players most similar to {selected_player}')
st.dataframe(top_similar[['player', 'Similarity']])

