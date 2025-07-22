#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# In[13]:


per_game = pd.read_csv('Player Per Game.csv')

pd.set_option('display.max_columns', None)

per_game.head()


# In[19]:


per_game_25 = per_game[per_game['season'] == 2025]

per_game_25.info()


# In[33]:


per_game_25_agg = per_game_25.groupby('player_id').agg({
    'player': 'first',
    'pts_per_game': 'mean',
    'trb_per_game': 'mean',
    'ast_per_game': 'mean',
    'fg_percent': 'mean',
    'x3p_percent': 'mean'
})

per_game_25_agg.head()


# In[41]:


per_game_25_agg['player'].nunique()


# In[ ]:


scaler = StandardScaler()

X_scaled = scaler.fit_transform(per_game_25_agg[['pts_per_game', 'trb_per_game', 'ast_per_game', 'fg_percent', 'x3p_percent']])

st.title('Player Similarity Tool')

selected_player = st.selectbox('Select a player', per_game_25_agg['player'])

matching_rows = per_game_25_agg[per_game_25_agg['player'] == selected_player]

if matching_rows.empty:
    st.error("Selected player not found in the data.")
    st.stop()

idx = matching_rows.index[0]

similarities = cosine_similarity([X_scaled[idx]], X_scaled)[0]

per_game_25_agg['Similarity'] = similarities

top_similar = per_game_25_agg[per_game_25_agg['player'] != selected_player].sort_values(by = 'Similarity', ascending = False).head()

st.subheader(f'Players most similar to {selected_player}')
st.dataframe(top_similar[['player', 'Similarity']])

