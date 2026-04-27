import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Page config

st.set_page_config(
    page_title='Injury Risk Dashboard',
    page_icon=':soccer:'
)

# -----------------------------------------------------------------------------
# Load data

@st.cache_data
def get_data():
    DATA_FILENAME = Path(__file__).parent / 'data' / 'data.csv'
    df = pd.read_csv(DATA_FILENAME)
    return df

df = get_data()

target = 'Injury_Next_Season'

features = [
    'Previous_Injury_Count',
    'Training_Hours_Per_Week',
    'Sleep_Hours_Per_Night',
    'Stress_Level_Score',
    'BMI',
    'Warmup_Routine_Adherence',
    'Reaction_Time_ms',
    'Balance_Test_Score',
    'Agility_Score'
]

# -----------------------------------------------------------------------------
# Train model

@st.cache_resource
def train_model(df):
    df_model = df[features + [target]].dropna()

    X = df_model[features]
    y = df_model[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model(df)

# -----------------------------------------------------------------------------
# Title

'''
# Injury Risk Dashboard

Explore injury risk patterns, simulate player profiles, and track how risk evolves over time.
'''

''
''

# -----------------------------------------------------------------------------
# Filters

positions = df['Position'].unique()

selected_positions = st.multiselect(
    'Select positions',
    positions,
    default=positions
)

filtered_df = df[df['Position'].isin(selected_positions)]

''
''

# -----------------------------------------------------------------------------
# Overview metrics

st.header('Overview', divider='gray')

col1, col2, col3 = st.columns(3)

col1.metric(
    'Players',
    len(filtered_df)
)

col2.metric(
    'Avg Injury Rate',
    f"{filtered_df[target].mean():.2%}"
)

col3.metric(
    'Avg Training Hours',
    f"{filtered_df['Training_Hours_Per_Week'].mean():.1f}"
)

''
''

# -----------------------------------------------------------------------------
# Injury by position

st.subheader('Injury Rate by Position')

pivot_pos = filtered_df.groupby('Position')[target].mean()

st.bar_chart(pivot_pos)

''
''

# -----------------------------------------------------------------------------
# Previous injury impact

st.subheader('Previous Injury vs Risk')

pivot_prev = filtered_df.groupby('Previous_Injury_Count')[target].mean()

st.line_chart(pivot_prev)

''
''

# -----------------------------------------------------------------------------
# Player selection

st.header('Player Risk Scoring', divider='gray')

mode = st.radio(
    'Choose mode',
    ['Select Player', 'Simulate Player']
)

if mode == 'Select Player':
    player_index = st.selectbox('Select player', filtered_df.index)
    player = filtered_df.loc[player_index]

    input_data = player[features].values.reshape(1, -1)

else:
    inputs = {}

    for f in features:
        inputs[f] = st.slider(
            f,
            float(df[f].min()),
            float(df[f].max()),
            float(df[f].median())
        )

    input_data = np.array(list(inputs.values())).reshape(1, -1)

''
''

# -----------------------------------------------------------------------------
# Prediction

prob = model.predict_proba(scaler.transform(input_data))[0][1]

st.metric(
    label='Injury Risk',
    value=f"{prob:.2%}"
)

if prob > 0.75:
    st.error('High Risk')
elif prob > 0.5:
    st.warning('Moderate Risk')
else:
    st.success('Low Risk')

''
''

# -----------------------------------------------------------------------------
# Time simulation

st.header('Risk Over Time', divider='gray')

weeks = st.slider(
    'Weeks',
    min_value=1,
    max_value=12,
    value=6
)

sim = input_data.copy()
trend = []

for _ in range(weeks):
    sim[0][features.index('Training_Hours_Per_Week')] *= 1.02
    sim[0][features.index('Stress_Level_Score')] *= 1.01
    sim[0][features.index('Sleep_Hours_Per_Night')] *= 0.995

    p = model.predict_proba(scaler.transform(sim))[0][1]
    trend.append(p)

trend_df = pd.DataFrame({
    'Week': list(range(1, weeks + 1)),
    'Risk': trend
})

st.line_chart(trend_df, x='Week', y='Risk')

''
''

# -----------------------------------------------------------------------------
# Recommendations

st.header('Recommendations', divider='gray')

training = input_data[0][features.index('Training_Hours_Per_Week')]
sleep = input_data[0][features.index('Sleep_Hours_Per_Night')]
stress = input_data[0][features.index('Stress_Level_Score')]
injuries = input_data[0][features.index('Previous_Injury_Count')]
warmup = input_data[0][features.index('Warmup_Routine_Adherence')]

recs = []

if training > df['Training_Hours_Per_Week'].quantile(0.75):
    recs.append('Reduce training load by approximately 15 percent')

if sleep < df['Sleep_Hours_Per_Night'].quantile(0.4):
    recs.append('Increase sleep by 1 to 2 hours per night')

if stress > df['Stress_Level_Score'].quantile(0.7):
    recs.append('Reduce stress and introduce recovery sessions')

if injuries >= 2:
    recs.append('High injury recurrence risk, monitor closely')

if warmup == 0:
    recs.append('Introduce structured warmup routine')

if recs:
    for r in recs:
        st.warning(r)
else:
    st.success('No major adjustments needed')
