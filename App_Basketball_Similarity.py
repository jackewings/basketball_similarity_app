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

st.title('Player Similarity Tool')

# --- Part 1: Single Player Similarity ---

selected_player = st.selectbox('Select a player to find similar players', per_game_25_agg['player'])

matching_rows = per_game_25_agg[per_game_25_agg['player'] == selected_player]
if matching_rows.empty:
    st.error("Selected player not found in the data.")
    st.stop()

idx = matching_rows.index[0]

similarities = cosine_similarity([X_scaled[idx]], X_scaled)[0]

per_game_25_agg['Similarity'] = similarities

top_similar = per_game_25_agg[per_game_25_agg['player'] != selected_player] \
              .sort_values(by='Similarity', ascending=False) \
              .head()

st.subheader(f'Players most similar to {selected_player}')
st.dataframe(top_similar[['player', 'Similarity']])

# --- Part 2: Side-by-Side Comparison for Two Players ---

st.markdown('---')  # Separator

st.header('Compare Two Players Side-by-Side')

player_options = per_game_25_agg['player'].tolist()

player1 = st.selectbox('Select first player', player_options, key='player1')
player2 = st.selectbox('Select second player', player_options, index=1, key='player2')

if player1 == player2:
    st.warning("Please select two different players to compare.")
else:
    idx1 = per_game_25_agg[per_game_25_agg['player'] == player1].index[0]
    idx2 = per_game_25_agg[per_game_25_agg['player'] == player2].index[0]

    # Compute cosine similarity between the two selected players
    similarity_score = cosine_similarity([X_scaled[idx1]], [X_scaled[idx2]])[0][0]

    # Prepare stats DataFrame for display
    stats_df = per_game_25_agg.loc[[idx1, idx2], ['player'] + features].set_index('player').T
    stats_df.columns.name = None

    st.subheader(f'Statistics Comparison and Similarity Score')
    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(stats_df.style.format({
            'pts_per_game': '{:.1f}',
            'trb_per_game': '{:.1f}',
            'ast_per_game': '{:.1f}',
            'fg_percent': '{:.3f}',
            'x3p_percent': '{:.3f}'
        }))

    with col2:
        st.markdown(f"### Similarity Score:")
        st.markdown(f"**{similarity_score:.3f}**")



