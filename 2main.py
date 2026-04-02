# noise_analysis_tool.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
st.set_page_config(page_title="Noise Analysis Tool", page_icon="🔊", layout="wide")
if 'noise_data' not in st.session_state:
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='H')
    np.random.seed(42)
    data = []
    for date in dates:
        hour = date.hour
        # Simulate realistic noise patterns
        if 7 <= hour <= 9: base = 65 + np.random.normal(0, 5)
        elif 17 <= hour <= 19: base = 70 + np.random.normal(0, 6)
        elif 22 <= hour or hour <= 6: base = 45 + np.random.normal(0, 3)
        else: base = 55 + np.random.normal(0, 4)
        location = np.random.choice(['Urban', 'Industrial', 'Residential', 'Commercial'])
        location_factor = {'Industrial': 15, 'Urban': 10, 'Commercial': 8, 'Residential': -8}
        noise = max(30, min(120, base + location_factor[location]))
        data.append({'timestamp': date, 'noise_level': noise, 'location': location, 'hour': hour})
    st.session_state.noise_data = pd.DataFrame(data)
st.sidebar.title("🔊 Noise Analysis Tool")
location = st.sidebar.multiselect("Location", st.session_state.noise_data['location'].unique(), 
                                   default=st.session_state.noise_data['location'].unique())
hours = st.sidebar.slider("Prediction Hours", 6, 48, 24)
df = st.session_state.noise_data[st.session_state.noise_data['location'].isin(location)]
st.title("📊 Noise Analysis Dashboard")
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Noise", f"{df['noise_level'].mean():.1f} dB")
col2.metric("Peak Noise", f"{df['noise_level'].max():.1f} dB")
col3.metric("Min Noise", f"{df['noise_level'].min():.1f} dB")
col4.metric("Std Deviation", f"{df['noise_level'].std():.1f} dB")
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🤖 Predictions", "🔍 Anomalies", "📊 Patterns"])
with tab1:
    st.subheader("Noise Levels Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for loc in df['location'].unique():
        loc_data = df[df['location'] == loc]
        ax.plot(loc_data['timestamp'], loc_data['noise_level'], label=loc, alpha=0.7)
    ax.axhline(y=70, color='r', linestyle='--', label='Warning (70dB)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Noise Level (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.subheader("Hourly Noise Heatmap")
    pivot = df.pivot_table(values='noise_level', index=df['timestamp'].dt.hour, 
                           columns=df['timestamp'].dt.dayofweek, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_yticks(range(24))
    ax.set_ylabel('Hour')
    ax.set_title('Noise Heatmap (dB)')
    plt.colorbar(im)
    st.pyplot(fig)
with tab2:
    st.subheader("Noise Level Predictions")
    
    if st.button("Generate Predictions"):
        with st.spinner("Training model..."):
            # Prepare features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            for lag in [1, 6, 12, 24]:
                df[f'lag_{lag}'] = df['noise_level'].shift(lag)
            
            df_clean = df.dropna()
            features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'lag_1', 'lag_6', 'lag_12', 'lag_24']
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(df_clean[features], df_clean['noise_level'])
            last = df.iloc[-24:].copy()
            predictions = []
            for i in range(hours):
                X_pred = last.iloc[-1:][features]
                pred = rf.predict(X_pred)[0]
                predictions.append(pred)
                new_row = last.iloc[-1:].copy()
                new_row['noise_level'] = pred
                new_row['timestamp'] += timedelta(hours=1)
                new_row['hour'] = new_row['timestamp'].dt.hour
                new_row['hour_sin'] = np.sin(2 * np.pi * new_row['hour'] / 24)
                new_row['hour_cos'] = np.cos(2 * np.pi * new_row['hour'] / 24)
                last = pd.concat([last, new_row])
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['timestamp'].tail(100), df['noise_level'].tail(100), 'b-', label='Historical', alpha=0.7)
            future_times = [df['timestamp'].max() + timedelta(hours=i+1) for i in range(hours)]
            ax.plot(future_times, predictions, 'r--', label='Predictions', linewidth=2)
            ax.axhline(y=70, color='orange', linestyle=':', label='Warning')
            ax.set_xlabel('Time')
            ax.set_ylabel('Noise Level (dB)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.success(f"Average predicted noise: {np.mean(predictions):.1f} dB | Peak: {np.max(predictions):.1f} dB")
with tab3:
    st.subheader("Noise Pattern Clustering")
    hourly_avg = df.groupby('hour')['noise_level'].mean().reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    hourly_avg['cluster'] = kmeans.fit_predict(hourly_avg[['noise_level']])
    cluster_names = {0: '🟢 Quiet', 1: '🟡 Moderate', 2: '🔴 Noisy'}
    hourly_avg['pattern'] = hourly_avg['cluster'].map(cluster_names)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'🟢 Quiet': 'green', '🟡 Moderate': 'orange', '🔴 Noisy': 'red'}
    for pattern in hourly_avg['pattern'].unique():
        data = hourly_avg[hourly_avg['pattern'] == pattern]
        ax.scatter(data['hour'], data['noise_level'], c=colors[pattern], label=pattern, s=100, alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Noise Level (dB)')
    ax.set_title('Hourly Noise Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.subheader("Pattern Analysis")
    for pattern in hourly_avg['pattern'].unique():
        data = hourly_avg[hourly_avg['pattern'] == pattern]
        hours_list = data['hour'].tolist()
        avg_noise = data['noise_level'].mean()
        with st.expander(f"{pattern} (Avg: {avg_noise:.1f} dB)"):
            st.write(f"**Hours:** {', '.join(map(str, hours_list))}")
            if pattern == '🔴 Noisy':
                st.warning("💡 Consider noise cancellation during these hours")
            elif pattern == '🟢 Quiet':
                st.success("💡 Ideal for focused work/study")
st.markdown("---")
st.caption("Noise Analysis Tool | Powered by Streamlit, Pandas, Matplotlib & Scikit-learn")