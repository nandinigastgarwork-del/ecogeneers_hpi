import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import datetime

# Page config
st.set_page_config(
    page_title="HPI Smart Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-safe {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .status-moderate {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .status-high {
        background: linear-gradient(135deg, #ff5722 0%, #d32f2f 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌊 HPI Smart Predictor Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 *AI-Powered Heavy Metal Pollution Assessment & Real-time Risk Analysis*")

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["📊 Data Upload & ML Prediction", "🧮 Direct HPI Calculator", "📈 Risk Analysis Dashboard"]
    )
    st.markdown("---")
    with st.expander("ℹ️ About HPI"):
        st.write("""
        **Heavy Metal Pollution Index (HPI)** evaluates the overall quality of water 
        based on heavy metal concentrations.
        
        **Classification:**
        - 🟢 **Safe**: HPI < 15
        - 🟡 **Moderate**: 15 ≤ HPI < 30
        - 🟠 **High**: 30 ≤ HPI < 75
        - 🔴 **Critical**: HPI ≥ 75
        """)
    # Live stats with mock data
    st.markdown("### 📊 Live Stats")
    st.metric("🌍 Global Samples", "15,847", "↑ 234")
    st.metric("⚠️ High-Risk Areas", "1,249", "↓ 12")
    st.metric("🤖 ML Accuracy", "94.2%", "↑ 2.1%")

# ------- Centralized file uploader ---------
uploaded_file = st.file_uploader(
    "Upload your Heavy Metal Dataset CSV",
    type=["csv"],
    help="Upload your CSV file containing metal concentrations and optional HPI column."
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # OPTIONAL: Rename columns if needed here,
    # uncomment and modify if your CSV has different column names.
    # Example:
    # if 'lattitude ' in df.columns:
    #     df.rename(columns={'lattitude ': 'latitude'}, inplace=True)

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Define standards and weights for HPI calculation
    standards = {'As': 10, 'Pb': 5, 'Cd': 1, 'Cr': 5, 'Hg': 1}
    weights = {'As': 5, 'Pb': 4, 'Cd': 3, 'Cr': 4, 'Hg': 5}

    def calculate_hpi(row):
        numerator, denominator = 0, 0
        for metal in standards:
            Ci = row.get(metal, 0)
            Si = standards[metal]
            Wi = weights[metal]
            Qi = abs(Ci - Si) / Si * 100 if Si != 0 else 0
            numerator += Wi * Qi
            denominator += Wi
        return numerator / denominator if denominator != 0 else 0

    # Calculate HPI if not given or recalc every time
    df['HPI'] = df.apply(calculate_hpi, axis=1)
else:
    df = None

# Main content based on selected mode
if analysis_mode == "📊 Data Upload & ML Prediction":

    if df is None:
        st.info("Please upload a CSV file above to start analysis.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Dataset overview
            st.markdown("### 📊 Dataset Overview")
            col1_disp, col2_disp, col3_disp, col4_disp = st.columns(4)
            with col1_disp:
                st.markdown(f'<div class="metric-container"><h3>📈 Total Samples</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
            with col2_disp:
                avg_hpi = df['HPI'].mean() if 'HPI' in df.columns else 0
                st.markdown(f'<div class="metric-container"><h3>📊 Avg HPI</h3><h2>{avg_hpi:.1f}</h2></div>', unsafe_allow_html=True)
            with col3_disp:
                metals_count = len([col for col in df.columns if col not in ['Location', 'HPI', 'latitude', 'longitude']])
                st.markdown(f'<div class="metric-container"><h3>⚗️ Metals</h3><h2>{metals_count}</h2></div>', unsafe_allow_html=True)
            with col4_disp:
                high_risk = len(df[df['HPI'] > 30]) if 'HPI' in df.columns else 0
                st.markdown(f'<div class="metric-container"><h3>⚠️ High Risk</h3><h2>{high_risk}</h2></div>', unsafe_allow_html=True)

            # Data preview table
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            if 'HPI' in df.columns:
                st.markdown("### 🤖 Machine Learning Analysis")
                X = df.select_dtypes(include=[np.number]).drop(['HPI', 'latitude', 'longitude'], axis=1, errors='ignore')

                y = df['HPI']

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)

                y_pred = model.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                col_perf, col_pred = st.columns(2)
                with col_perf:
                    st.markdown("#### 📊 Model Performance")
                    perf_df = pd.DataFrame({
                        'Metric': ['R² Score', 'RMSE', 'Accuracy'],
                        'Value': [f'{r2:.3f}', f'{rmse:.2f}', f'{r2*100:.1f}%']
                    })
                    st.table(perf_df)

                    feature_imp = pd.DataFrame({
                        'Metal': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)

                    fig_imp = px.bar(
                        feature_imp,
                        x='Importance',
                        y='Metal',
                        orientation='h',
                        title="🎯 Feature Importance",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                with col_pred:
                    fig_pred = px.scatter(
                        x=y,
                        y=y_pred,
                        title="🎯 Actual vs Predicted HPI",
                        labels={'x': 'Actual HPI', 'y': 'Predicted HPI'},
                        trendline="ols"
                    )
                    fig_pred.add_shape(
                        type="line", line=dict(dash="dash"),
                        x0=y.min(), y0=y.min(),
                        x1=y.max(), y1=y.max()
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    fig_dist = px.histogram(
                        df,
                        x='HPI',
                        nbins=20,
                        title="📊 HPI Distribution",
                        color_discrete_sequence=['#667eea']
                    )
                    fig_dist.add_vline(x=15, line_dash="dash", line_color="green", annotation_text="Safe")
                    fig_dist.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Moderate")
                    fig_dist.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="High Risk")
                    st.plotly_chart(fig_dist, use_container_width=True)

                # Interactive prediction
                st.markdown("### 🧪 Interactive Prediction")
                st.markdown("*Adjust metal concentrations to see real-time HPI predictions*")
                pred_cols = st.columns(len(X.columns))
                user_input = {}
                for i, col in enumerate(X.columns):
                    with pred_cols[i]:
                        user_input[col] = st.number_input(
                            f"{col} (mg/L)",
                            min_value=0.0,
                            value=float(df[col].mean()),
                            step=0.1,
                            key=f"pred_{col}"
                        )
                if st.button("🔮 Predict HPI", type="primary"):
                    input_df = pd.DataFrame([user_input])
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]

                    if prediction < 15:
                        status = "🟢 SAFE"
                        status_class = "status-safe"
                    elif prediction < 30:
                        status = "🟡 MODERATE"
                        status_class = "status-moderate"
                    elif prediction < 75:
                        status = "🟠 HIGH RISK"
                        status_class = "status-high"
                    else:
                        status = "🔴 CRITICAL"
                        status_class = "status-high"

                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(f'<div class="{status_class}"><h2>Predicted HPI: {prediction:.2f}</h2><h3>{status}</h3></div>', unsafe_allow_html=True)
                    with col_res2:
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=list(user_input.values()),
                            theta=list(user_input.keys()),
                            fill='toself',
                            name='Input Values'
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, max(user_input.values())*1.2])
                            ),
                            title="🎯 Metal Concentration Profile"
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

elif analysis_mode == "🧮 Direct HPI Calculator":
    st.markdown("### 🧮 Direct HPI Formula Calculator")
    st.markdown("*Calculate HPI using the standard formula without machine learning*")

    st.latex(r'''
    HPI = \frac{\sum_{i=1}^{n} W_i \times Q_i}{\sum_{i=1}^{n} W_i}
    ''')

    st.latex(r'''
    Q_i = \frac{|C_i - S_i|}{S_i} \times 100
    ''')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚗️ Metal Concentrations")
        metals = ['As', 'Pb', 'Cd', 'Cr', 'Hg']
        concentrations = {}
        for metal in metals:
            concentrations[metal] = st.number_input(
                f"{metal} Concentration (mg/L)",
                min_value=0.0,
                value=5.0,
                step=0.1
            )
    with col2:
        st.markdown("#### 📏 Standards & Weights")
        standards = {'As': 10, 'Pb': 5, 'Cd': 1, 'Cr': 5, 'Hg': 1}
        weights = {'As': 5, 'Pb': 4, 'Cd': 3, 'Cr': 4, 'Hg': 5}
        for metal in metals:
            col_std, col_wt = st.columns(2)
            with col_std:
                standards[metal] = st.number_input(f"{metal} Standard", value=float(standards[metal]), step=0.1)
            with col_wt:
                weights[metal] = st.number_input(f"{metal} Weight", value=weights[metal], step=1)

    if st.button("📊 Calculate HPI", type="primary"):
        numerator = 0
        denominator = 0
        calculation_details = []
        for metal in metals:
            Ci = concentrations[metal]
            Si = standards[metal]
            Wi = weights[metal]
            Qi = abs(Ci - Si) / Si * 100
            numerator += Wi * Qi
            denominator += Wi
            calculation_details.append({
                'Metal': metal,
                'Concentration': Ci,
                'Standard': Si,
                'Weight': Wi,
                'Qi': round(Qi, 2),
                'Wi × Qi': round(Wi * Qi, 2)
            })
        hpi_value = numerator / denominator

        col1, col2 = st.columns(2)
        with col1:
            if hpi_value < 15:
                status = "🟢 SAFE"
                status_class = "status-safe"
            elif hpi_value < 30:
                status = "🟡 MODERATE"
                status_class = "status-moderate"
            elif hpi_value < 75:
                status = "🟠 HIGH RISK"
                status_class = "status-high"
            else:
                status = "🔴 CRITICAL"
                status_class = "status-high"
            st.markdown(f'<div class="{status_class}"><h2>HPI Value: {hpi_value:.2f}</h2><h3>{status}</h3></div>', unsafe_allow_html=True)
        with col2:
            df_calc = pd.DataFrame(calculation_details)
            st.markdown("#### 📊 Calculation Breakdown")
            st.dataframe(df_calc, use_container_width=True)

            fig_metals = px.bar(
                df_calc,
                x='Metal',
                y='Wi × Qi',
                title="🔍 Individual Metal Contributions to HPI",
                color='Wi × Qi',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_metals, use_container_width=True)

else:  # Risk Analysis Dashboard
    st.markdown("### 📈 Risk Analysis Dashboard")
    st.markdown("*Comprehensive risk assessment and monitoring overview*")

    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    hpi_trend = np.random.normal(35, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5
    trend_df = pd.DataFrame({'Date': dates, 'HPI': hpi_trend})

    col1, col2 = st.columns(2)
    with col1:
        fig_trend = px.line(
            trend_df,
            x='Date',
            y='HPI',
            title="📈 HPI Trend Analysis (2024)",
            color_discrete_sequence=['#667eea']
        )
        fig_trend.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Safe Level")
        fig_trend.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
        fig_trend.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="High Risk")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        risk_data = pd.DataFrame({
            'Risk Level': ['Safe', 'Moderate', 'High', 'Critical'],
            'Count': [45, 28, 15, 12],
            'Color': ['#4CAF50', '#FF9800', '#FF5722', '#D32F2F']
        })

        fig_risk = px.pie(
            risk_data,
            values='Count',
            names='Risk Level',
            title="🎯 Risk Level Distribution",
            color_discrete_sequence=['#4CAF50', '#FF9800', '#FF5722', '#D32F2F']
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Geographic risk map
    st.markdown("#### 🗺️ Geographic Risk Assessment")

    if uploaded_file is not None and df is not None:
        map_data = df.copy()
        # If your CSV has 'lattitude ', rename it to 'latitude' before using
        if 'lattitude ' in map_data.columns:
            map_data.rename(columns={'lattitude ': 'latitude'}, inplace=True)
        map_data['latitude'] = pd.to_numeric(map_data['latitude'], errors='coerce')
        map_data['longitude'] = pd.to_numeric(map_data['longitude'], errors='coerce')
        map_data = map_data.dropna(subset=['latitude', 'longitude'])
        map_data['HPI'] = map_data['HPI'].clip(lower=0)

        fig_map = px.scatter_mapbox(
            map_data,
            lat='latitude',
            lon='longitude',
            color='HPI',
            size='HPI',
            hover_data=['Location', 'HPI'] if 'Location' in map_data.columns else None,
            color_continuous_scale='reds',
            mapbox_style='open-street-map',
            title="📍 HPI Risk Hotspots",
            zoom=5
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Alert system (Optional example, customize accordingly)
        st.markdown("#### 🚨 Live Alert System")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="status-high">
            <h4>🚨 Critical Alert</h4>
            <p>Site_23: HPI = 89.2</p>
            <p>Immediate action required</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="status-moderate">
            <h4>⚠️ Warning</h4>
            <p>Site_15: HPI = 42.1</p>
            <p>Monitor closely</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="status-safe">
            <h4>✅ Normal</h4>
            <p>Site_08: HPI = 12.5</p>
            <p>Within safe limits</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📊 Dashboard Stats")
    st.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col2:
    st.markdown("### 🔗 Quick Links")
    st.write("• [WHO Guidelines](https://who.int)")
    st.write("• [EPA Standards](https://epa.gov)")
with col3:
    st.markdown("### 🚀 Built for SIH 2025")
    st.write("**EcoGeneers Team**")
    st.write("*Smart Environmental Monitoring*")
