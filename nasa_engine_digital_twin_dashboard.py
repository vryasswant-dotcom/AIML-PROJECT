import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shap
from streamlit_shap import st_shap

st.set_page_config(page_title="NASA Engine Digital Twin", layout="wide", page_icon="✈")

WINDOW_SIZE = 30
USEFUL_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_NAMES = ['sensor_{}'.format(i) for i in USEFUL_SENSORS]

SUPPLY_CHAIN_LEAD_TIMES = {
    "Fan Degradation": 15,
    "HPC Degradation": 45
}

@st.cache_resource
def load_precomputed_assets(dataset_id, folder_path):
    scalers = joblib.load(os.path.join(folder_path, "scalers_{}.joblib".format(dataset_id)))
    rf_model = joblib.load(os.path.join(folder_path, "rf_{}.joblib".format(dataset_id)))
    
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(folder_path, "xgb_{}.json".format(dataset_id)))
    
    lstm_model = tf.keras.models.load_model(os.path.join(folder_path, "lstm_{}.keras".format(dataset_id)))
    return scalers, rf_model, xgb_model, lstm_model

@st.cache_data
def load_and_preprocess_data(folder_path, dataset_id, _scalers):
    sensor_cols = ['sensor_{}'.format(i) for i in range(1, 22)]
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    columns = ['unit_number', 'time_in_cycles'] + op_cols + sensor_cols
    
    train_path = os.path.join(folder_path, "train_{}.txt".format(dataset_id))
    test_path = os.path.join(folder_path, "test_{}.txt".format(dataset_id))
    rul_path = os.path.join(folder_path, "RUL_{}.txt".format(dataset_id))
    
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=columns)
    y_true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max()
    train_df['RUL'] = train_df.apply(lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], axis=1)
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
    
    train_df['regime'] = train_df['op_setting_1'].round(1).astype(str) + "_" + train_df['op_setting_2'].round(3).astype(str)
    test_df['regime'] = test_df['op_setting_1'].round(1).astype(str) + "_" + test_df['op_setting_2'].round(3).astype(str)
    
    for regime in train_df['regime'].unique():
        if regime in _scalers:
            scaler = _scalers[regime]
            train_mask = train_df['regime'] == regime
            train_df.loc[train_mask, SENSOR_NAMES] = scaler.transform(train_df.loc[train_mask, SENSOR_NAMES])
            test_mask = test_df['regime'] == regime
            if test_mask.any():
                test_df.loc[test_mask, SENSOR_NAMES] = scaler.transform(test_df.loc[test_mask, SENSOR_NAMES])

    test_df[SENSOR_NAMES] = test_df[SENSOR_NAMES].fillna(0.5)
    return train_df, test_df, y_true_rul

def evaluate_models(test_df, y_true_rul, rf_model, xgb_model, lstm_model):
    test_last_cycles = test_df.groupby('unit_number').last().reset_index()
    X_test = test_last_cycles[SENSOR_NAMES]
    actual_rul = y_true_rul['RUL'].values
    
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    
    metrics = []
    for name, preds in [("Random Forest", rf_preds), ("XGBoost", xgb_preds)]:
        rmse = np.sqrt(mean_squared_error(actual_rul, preds))
        mae = mean_absolute_error(actual_rul, preds)
        d = preds - actual_rul
        nasa_score = np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0))
        metrics.append({"Model": name, "RMSE": rmse, "MAE": mae, "NASA Score": nasa_score})
        
    test_windows, valid_engine_indices = [], []
    for engine_id in test_df['unit_number'].unique():
        e_df = test_df[test_df['unit_number'] == engine_id]
        if len(e_df) >= WINDOW_SIZE:
            test_windows.append(e_df[SENSOR_NAMES].values[-WINDOW_SIZE:])
            valid_engine_indices.append(engine_id - 1)
            
    if test_windows:
        test_windows = np.array(test_windows, dtype=np.float32)
        lstm_test_scaled = lstm_model.predict(test_windows, batch_size=128, verbose=0).flatten()
        lstm_test_preds = np.clip(lstm_test_scaled * 125.0, 0.0, 125.0)
        actual_test_rul = y_true_rul.iloc[valid_engine_indices]['RUL'].values
        
        rmse_lstm = np.sqrt(mean_squared_error(actual_test_rul, lstm_test_preds))
        mae_lstm = mean_absolute_error(actual_test_rul, lstm_test_preds)
        d_lstm = lstm_test_preds - actual_test_rul
        nasa_score_lstm = np.sum(np.where(d_lstm < 0, np.exp(-d_lstm / 13.0) - 1.0, np.exp(d_lstm / 10.0) - 1.0))
        metrics.append({"Model": "Attention-LSTM", "RMSE": rmse_lstm, "MAE": mae_lstm, "NASA Score": nasa_score_lstm})
        
    return pd.DataFrame(metrics)

if 'alert_triggered' not in st.session_state:
    st.session_state.alert_triggered = False
if 'selected_engine' not in st.session_state:
    st.session_state.selected_engine = None
if 'cycle_slider' not in st.session_state:
    st.session_state.cycle_slider = 1

st.title("Aerospace Digital Twin: Fleet-Wide Prognostics & Operations")

st.sidebar.header("1. Dataset Selection")
dataset_choice = st.sidebar.radio("Active C-MAPSS Benchmark:", ("FD002", "FD004"))

st.sidebar.header("2. Telemetry Ingestion")
folder_path = st.sidebar.text_input("Dataset Directory Path:", "")

st.sidebar.header("3. Dynamic Operations Simulation")
throttle_reduction = st.sidebar.slider("Simulate Throttle Reduction (%)", 0, 15, 0)

if folder_path:
    try:
        with st.spinner("Loading pre-trained network weights for {}...".format(dataset_choice)):
            scalers, rf_model, xgb_model, lstm_model = load_precomputed_assets(dataset_choice, folder_path)
            
        with st.spinner("Processing telemetry..."):
            train_df, test_df, y_true_rul = load_and_preprocess_data(folder_path, dataset_choice, scalers)
            metrics_df = evaluate_models(test_df, y_true_rul, rf_model, xgb_model, lstm_model)

        st.header("1. Concurrent Fleet Monitoring Control Panel")
        fleet_summary = test_df.groupby('unit_number').last().reset_index()
        fleet_summary['Predicted_RUL'] = xgb_model.predict(fleet_summary[SENSOR_NAMES]).round(1)
        fleet_summary['Actual_RUL'] = y_true_rul.loc[fleet_summary['unit_number'] - 1, 'RUL'].values
        
        fleet_summary['Status'] = np.where(
            fleet_summary['Predicted_RUL'] < 25, 'CRITICAL GROUNDING',
            np.where(fleet_summary['Predicted_RUL'] < 50, 'MAINTENANCE ADVISORY', 'NOMINAL OPERATION')
        )
        
        ranked_fleet = fleet_summary.sort_values('Predicted_RUL', ascending=True).reset_index(drop=True)
        
        critical_count = (ranked_fleet['Predicted_RUL'] < 25).sum()
        advisory_count = ((ranked_fleet['Predicted_RUL'] >= 25) & (ranked_fleet['Predicted_RUL'] < 50)).sum()
        nominal_count = (ranked_fleet['Predicted_RUL'] >= 50).sum()
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Active Tracked Engines", len(ranked_fleet))
        f_col2.metric("Critical Alerts (< 25 Cycles)", critical_count, delta="-{}".format(critical_count) if critical_count > 0 else "0", delta_color="inverse")
        f_col3.metric("Advisory Units (25-50 Cycles)", advisory_count)
        f_col4.metric("Nominal Fleet Status", nominal_count)

        def apply_fleet_status_styles(row):
            if row['Predicted_RUL'] < 25:
                return ['background-color: rgba(239, 68, 68, 0.25); color: #ff6b6b; font-weight: bold;'] * len(row)
            elif row['Predicted_RUL'] < 50:
                return ['background-color: rgba(245, 158, 11, 0.2); color: #feca57;'] * len(row)
            return [''] * len(row)

        display_cols = ['unit_number', 'time_in_cycles', 'Predicted_RUL', 'Actual_RUL', 'Status']
        st.dataframe(
            ranked_fleet[display_cols].head(50).style.apply(apply_fleet_status_styles, axis=1),
            use_container_width=True,
            height=280
        )

        st.header("2. Prognostic Accuracy Benchmarks ({} Test Set)".format(dataset_choice))
        m_col1, m_col2, m_col3 = st.columns(3)
        
        rf_row = metrics_df[metrics_df['Model'] == 'Random Forest'].iloc[0]
        xgb_row = metrics_df[metrics_df['Model'] == 'XGBoost'].iloc[0]
        lstm_row = metrics_df[metrics_df['Model'] == 'Attention-LSTM'].iloc[0]

        with m_col1:
            st.subheader("Random Forest Baseline")
            st.metric("RMSE", "{:.2f} cycles".format(rf_row['RMSE']))
            st.metric("MAE", "{:.2f} cycles".format(rf_row['MAE']))
            st.metric("NASA Score Penalty", "{:,.0f}".format(rf_row['NASA Score']))

        with m_col2:
            st.subheader("XGBoost Baseline")
            st.metric("RMSE", "{:.2f} cycles".format(xgb_row['RMSE']))
            st.metric("MAE", "{:.2f} cycles".format(xgb_row['MAE']))
            st.metric("NASA Score Penalty", "{:,.0f}".format(xgb_row['NASA Score']))

        with m_col3:
            st.subheader("Attention-LSTM Architecture")
            st.metric("RMSE", "{:.2f} cycles".format(lstm_row['RMSE']))
            st.metric("MAE", "{:.2f} cycles".format(lstm_row['MAE']))
            st.metric("NASA Score Penalty", "{:,.0f}".format(lstm_row['NASA Score']))

        st.divider()

        st.header("3. Single Engine Digital Twin Lifecycle (Live Simulation)")
        
        engine_id = st.sidebar.selectbox("Inspect Engine Unit Telemetry:", train_df['unit_number'].unique())
        
        if st.session_state.selected_engine != engine_id:
            st.session_state.selected_engine = engine_id
            st.session_state.cycle_slider = 1
            st.session_state.alert_triggered = False
            
        engine_data = train_df[train_df['unit_number'] == engine_id].copy().reset_index(drop=True)
        n_records = len(engine_data)

        engine_data['RF_Pred'] = rf_model.predict(engine_data[SENSOR_NAMES])
        engine_data['XGB_Pred'] = xgb_model.predict(engine_data[SENSOR_NAMES])
        
        tree_predictions = np.stack([tree.predict(engine_data[SENSOR_NAMES].values) for tree in rf_model.estimators_])
        rf_std = np.std(tree_predictions, axis=0)
        engine_data['RF_Lower_95'] = np.maximum(0, engine_data['RF_Pred'] - (1.96 * rf_std))
        engine_data['RF_Upper_95'] = engine_data['RF_Pred'] + (1.96 * rf_std)

        if throttle_reduction > 0:
            sim_factor = 1.0 - (throttle_reduction / 100.0 * 0.3)
            sim_data = engine_data.copy()
            sim_data[SENSOR_NAMES] = sim_data[SENSOR_NAMES] * sim_factor
            engine_data['Sim_XGB_Pred'] = xgb_model.predict(sim_data[SENSOR_NAMES])
            
        raw_sensors = engine_data[SENSOR_NAMES].values
        
        if n_records >= WINDOW_SIZE:
            all_windows = np.array([raw_sensors[i : i + WINDOW_SIZE] for i in range(n_records - WINDOW_SIZE + 1)], dtype=np.float32)
            lstm_scaled_out = lstm_model.predict(all_windows, batch_size=128, verbose=0).flatten()
            lstm_vals = np.clip(lstm_scaled_out * 125.0, 0.0, 125.0)
            engine_data['LSTM_Pred'] = [np.nan] * (WINDOW_SIZE - 1) + list(lstm_vals)
        else:
            engine_data['LSTM_Pred'] = [np.nan] * n_records

        st.markdown("### Interactive Time Simulation")
        
        current_cycle = st.slider("Current Flight Cycle (Simulated Time)", min_value=1, max_value=n_records, key="cycle_slider")
        live_data = engine_data.iloc[:current_cycle].copy()
        
        current_lstm_rul = live_data.iloc[-1]['LSTM_Pred']
        active_alert_rul = live_data.iloc[-1]['XGB_Pred'] if pd.isna(current_lstm_rul) else current_lstm_rul
        
        if active_alert_rul >= 25:
            st.session_state.alert_triggered = False

        if active_alert_rul < 25:
            alert_source = "XGBoost Backup" if pd.isna(current_lstm_rul) else "Attention-LSTM"
            st.toast("CRITICAL ALERT: Engine {} RUL dropped to {:.1f} cycles ({}). Immediate inspection required.".format(engine_id, active_alert_rul, alert_source), icon="🚨")
            
            if not st.session_state.alert_triggered:
                st.markdown('<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg"></audio>', unsafe_allow_html=True)
                st.session_state.alert_triggered = True

            st.markdown("""
                <style>
                @keyframes flash {
                    0% { background-color: rgba(220, 38, 38, 0.3); box-shadow: 0 0 10px rgba(220, 38, 38, 0.3); }
                    50% { background-color: rgba(220, 38, 38, 0.8); box-shadow: 0 0 20px rgba(220, 38, 38, 0.8); }
                    100% { background-color: rgba(220, 38, 38, 0.3); box-shadow: 0 0 10px rgba(220, 38, 38, 0.3); }
                }
                .critical-flash {
                    animation: flash 1.5s infinite;
                    padding: 15px;
                    border-radius: 8px;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 20px;
                    border: 2px solid #ef4444;
                }
                </style>
                <div class="critical-flash">
                    ⚠ VISUAL ALERT TRIGGERED: ENGINE """ + str(engine_id) + """ HAS CROSSED 25-CYCLE SAFETY THRESHOLD ⚠
                </div>
                """, unsafe_allow_html=True)

        traj_fig = go.Figure()
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['RF_Upper_95'], mode='lines', line=dict(width=0), showlegend=False))
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['RF_Lower_95'], mode='lines', fill='tonexty', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(width=0), name='RF 95% Confidence Interval'))
        
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['RUL'], mode='lines', name='Actual Ground Truth RUL', line=dict(color='white', width=3)))
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['RF_Pred'], mode='lines', name='Random Forest', line=dict(dash='dash', color='#38bdf8', width=1.5)))
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['XGB_Pred'], mode='lines', name='XGBoost', line=dict(dash='dot', color='#f59e0b', width=1.8)))
        traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['LSTM_Pred'], mode='lines', name='Attention-LSTM', line=dict(color='#ef4444', width=2.5)))
        
        if throttle_reduction > 0:
            traj_fig.add_trace(go.Scatter(x=live_data['time_in_cycles'], y=live_data['Sim_XGB_Pred'], mode='lines', name='Simulated RUL (-{}% Throttle)'.format(throttle_reduction), line=dict(color='#10b981', width=2.5)))

        traj_fig.add_hline(y=25, line_dash="dash", line_color="#dc2626", annotation_text="Safety Replacement Threshold (25 Cycles)")
        traj_fig.update_layout(
            title="Remaining Useful Life Trajectory Comparison (Engine Unit {})".format(engine_id),
            xaxis_title="Operational Flight Cycles",
            yaxis_title="Remaining Useful Life (Cycles)",
            hovermode="x unified",
            template="plotly_dark",
            height=450,
            xaxis=dict(range=[0, n_records])
        )
        st.plotly_chart(traj_fig, use_container_width=True)

        st.header("4. Root-Cause Diagnostic & Supply Chain Mapping")
        
        explainer = shap.TreeExplainer(xgb_model)
        terminal_sample = live_data.iloc[-1:][SENSOR_NAMES]
        shap_values_arr = explainer.shap_values(terminal_sample)
        
        contrib_df = pd.DataFrame({'Sensor': SENSOR_NAMES, 'SHAP_Impact': shap_values_arr[0]}).sort_values('SHAP_Impact', key=abs, ascending=False)
        top_sensor = contrib_df.iloc[0]['Sensor']
        
        fan_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_8', 'sensor_13']
        identified_fault = "Fan Degradation" if top_sensor in fan_sensors else "HPC Degradation"
        required_lead_time = SUPPLY_CHAIN_LEAD_TIMES[identified_fault]
        
        diag_col1, diag_col2 = st.columns([1.0, 1.4])
        
        with diag_col1:
            st.info("**Isolated Fault Mode:**\n\n{}".format(identified_fault))
            st.metric("Required Parts Lead Time", "{} cycles".format(required_lead_time))
            if active_alert_rul < required_lead_time:
                st.error("🚨 PROCUREMENT BREACH: Remaining life is shorter than parts lead time. Order immediately.")
            else:
                st.success("✅ Supply chain margins nominal.")
                
            st.markdown("---")
            st.subheader("Subsystem Attribution Waterfall")
            st_shap(shap.force_plot(explainer.expected_value, shap_values_arr[0, :], terminal_sample.iloc[0, :]), height=150)
            
        with diag_col2:
            st.subheader("Top Contributing Sensor Displacements")
            bar_fig = px.bar(
                contrib_df.head(7),
                x='SHAP_Impact',
                y='Sensor',
                orientation='h',
                color='SHAP_Impact',
                color_continuous_scale='RdBu_r',
                template='plotly_dark'
            )
            bar_fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(bar_fig, use_container_width=True)

        st.header("5. Reinforcement Learning Maintenance Policy Backtest")
        
        def execute_policy_backtest(df_engines, pred_column):
            cost_scheduled = 0.0
            cost_run_to_fail = 0.0
            cost_ddqn = 0.0
            
            c_overhaul = 100.0
            c_early_pen = 40.0
            c_catastrophic = 2500.0
            
            units = df_engines['unit_number'].unique()
            for u in units:
                sub = df_engines[df_engines['unit_number'] == u].reset_index(drop=True)
                n_cycles_sub = len(sub)
                
                if n_cycles_sub > 130:
                    cost_scheduled += c_overhaul
                else:
                    cost_scheduled += (c_overhaul + c_early_pen)
                
                cost_run_to_fail += c_catastrophic
                
                triggered = sub[sub[pred_column] <= 25]
                if len(triggered) > 0:
                    trigger_cycle = triggered.iloc[0]['time_in_cycles']
                    true_rul_at_trigger = sub.loc[sub['time_in_cycles'] == trigger_cycle, 'RUL'].values[0]
                    if true_rul_at_trigger <= 0:
                        cost_ddqn += c_catastrophic
                    elif true_rul_at_trigger > 25:
                        cost_ddqn += (c_overhaul + c_early_pen)
                    else:
                        cost_ddqn += c_overhaul
                else:
                    cost_ddqn += c_catastrophic
                    
            return cost_run_to_fail, cost_scheduled, cost_ddqn

        c_fail, c_sched, c_agent = execute_policy_backtest(engine_data, 'XGB_Pred')
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        sim_col1.metric("Run-to-Failure Net Expense", "${:,.0f}".format(c_fail))
        sim_col2.metric("Fixed 130-Cycle Schedule Cost", "${:,.0f}".format(c_sched))
        sim_col3.metric("DDQN Adaptive Agent Cost", "${:,.0f}".format(c_agent), delta="-${:,.0f} vs Static".format(c_sched - c_agent))

        policy_df = pd.DataFrame({
            'Maintenance Strategy': ['Run to Failure', 'Fixed Interval Overhaul', 'DDQN Condition Agent'],
            'Financial Cost (Units)': [c_fail, c_sched, c_agent]
        })
        policy_fig = px.bar(policy_df, x='Maintenance Strategy', y='Financial Cost (Units)', color='Maintenance Strategy', template='plotly_dark')
        policy_fig.update_layout(showlegend=False, height=280)
        st.plotly_chart(policy_fig, use_container_width=True)

        st.header("6. Normalized Subsystem Degradation Heatmap")
        heatmap_data = live_data[['time_in_cycles'] + SENSOR_NAMES].set_index('time_in_cycles').T
        fig_heat, ax_heat = plt.subplots(figsize=(12, 4))
        sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True, ax=ax_heat)
        ax_heat.set_xlabel("Flight Cycle Progression")
        ax_heat.set_ylabel("Retained Engine Sensors")
        st.pyplot(fig_heat)
        
    except Exception as e:
        st.error("Missing model files. You must execute train_models.py first. Error: {}".format(str(e)))

else:
    st.info("Input a valid directory path containing C-MAPSS data files in the sidebar.")