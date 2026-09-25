import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from model_handler import CardioModelHandler

# ---------------------------------------------------------
# Page Configuration (Sidebar Collapsed / No Sidebar)
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioDetect AI | Clinical Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Assessment'

if 'active_model' not in st.session_state:
    st.session_state.active_model = 'Gradient Boosting'

if 'form_version' not in st.session_state:
    st.session_state.form_version = 0

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'trigger_scroll' not in st.session_state:
    st.session_state.trigger_scroll = False

if 'preset_values' not in st.session_state:
    st.session_state.preset_values = None

# ---------------------------------------------------------
# Custom Modern Medical UI Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    /* Hide the sidebar completely */
    [data-testid="collapsedControl"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }
    
    /* Top Header Bar */
    .top-header {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }
    
    .brand-badge {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(6, 182, 212, 0.3) 100%);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Model Control Navbar on Risk Assessment Page */
    .model-control-bar {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }

    /* Section Cards */
    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 10px;
    }

    /* Metric Display Cards */
    .metric-card {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-num {
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Risk Outcome Banners */
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10B981;
        padding: 22px 28px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .risk-banner-mod {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
        padding: 22px 28px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #EF4444;
        padding: 22px 28px;
        border-radius: 16px;
        margin-bottom: 24px;
    }

    /* Model Performance Badge in Results */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #0F172A;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        color: #E2E8F0;
        margin-bottom: 16px;
    }

    /* Recommendation Cards */
    .rec-item {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #0EA5E9;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .rec-item-title {
        font-weight: 700;
        font-size: 14px;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .rec-item-text {
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached Clinical Model Handler
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading clinical models from disk...")
def get_handler():
    return CardioModelHandler()

handler = get_handler()

# ---------------------------------------------------------
# TOP NAVIGATION HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div>
        <div class="brand-title">
            <span>🫀 CardioDetect AI</span>
            <span class="brand-badge">Clinical Engine</span>
        </div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 4px;">
            Evidence-based cardiovascular disease risk stratification powered by validated clinical classification models
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

with nav_c1:
    if st.button("🫀 Clinical Risk Assessment", use_container_width=True, type="primary" if st.session_state.active_page == 'Assessment' else "secondary"):
        st.session_state.active_page = 'Assessment'
        st.rerun()

with nav_c2:
    if st.button("⚖️ Model Benchmark & Selection Guide", use_container_width=True, type="primary" if st.session_state.active_page == 'Benchmarks' else "secondary"):
        st.session_state.active_page = 'Benchmarks'
        st.rerun()

with nav_c3:
    if st.button("📊 Population Analytics", use_container_width=True, type="primary" if st.session_state.active_page == 'Analytics' else "secondary"):
        st.session_state.active_page = 'Analytics'
        st.rerun()

with nav_c4:
    if st.button("ℹ️ Clinical Reference & Specs", use_container_width=True, type="primary" if st.session_state.active_page == 'Docs' else "secondary"):
        st.session_state.active_page = 'Docs'
        st.rerun()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# =========================================================
# TAB 1: CLINICAL RISK ASSESSMENT
# =========================================================
if st.session_state.active_page == 'Assessment':
    
    # ---------------------------------------------------------
    # TOP NAVBAR FOR RISK ASSESSMENT (MODEL SELECTION & PRESETS)
    # ---------------------------------------------------------
    available_models = handler.get_available_models()
    if st.session_state.active_model not in available_models and available_models:
        st.session_state.active_model = available_models[0]
        
    st.markdown("<div class='model-control-bar'>", unsafe_allow_html=True)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.8, 2.2, 1.2])
    
    with ctrl_col1:
        # Callback when a new model is selected
        def on_model_selection():
            new_model = st.session_state.model_selector_key
            if new_model != st.session_state.active_model:
                st.session_state.active_model = new_model
                st.session_state.prediction_result = None  # Wipes previous model results
                st.session_state.form_version += 1         # Automatically resets the form fields
                st.session_state.preset_values = None
                st.session_state.trigger_scroll = False

        st.selectbox(
            "🧠 Select Assessment Model",
            available_models,
            index=available_models.index(st.session_state.active_model) if st.session_state.active_model in available_models else 0,
            key="model_selector_key",
            on_change=on_model_selection,
            help="Switching models will immediately reset the form and clear previous results."
        )

    # Active Model Metrics Display Pill
    active_m_metrics = handler.get_model_metrics(st.session_state.active_model)
    with ctrl_col2:
        st.markdown(f"""
        <div style="padding-top: 14px;">
            <div style="font-size:11px; color:#38BDF8; font-weight:700; text-transform:uppercase;">Active Model Telemetry</div>
            <div style="display:flex; gap:12px; margin-top:4px; flex-wrap:wrap;">
                <span style="background:#0F172A; border:1px solid rgba(255,255,255,0.08); padding:3px 10px; border-radius:10px; font-size:12px; color:#F1F5F9;">
                    CV Acc: <b style="color:#34D399;">{active_m_metrics['cv_mean']}%</b>
                </span>
                <span style="background:#0F172A; border:1px solid rgba(255,255,255,0.08); padding:3px 10px; border-radius:10px; font-size:12px; color:#F1F5F9;">
                    Test Acc: <b style="color:#38BDF8;">{active_m_metrics['test_acc']}%</b>
                </span>
                <span style="background:#0F172A; border:1px solid rgba(255,255,255,0.08); padding:3px 10px; border-radius:10px; font-size:12px; color:#F1F5F9;">
                    Precision: <b>{active_m_metrics['precision']}%</b>
                </span>
                <span style="background:#0F172A; border:1px solid rgba(255,255,255,0.08); padding:3px 10px; border-radius:10px; font-size:12px; color:#F1F5F9;">
                    Recall: <b>{active_m_metrics['recall']}%</b>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with ctrl_col3:
        st.markdown("<div style='font-size:11px; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:4px;'>Quick Presets</div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🟢 Low", use_container_width=True, help="Load low-risk sample values"):
                st.session_state.preset_values = {
                    'age': 42, 'gender': "Female", 'height': 168, 'weight': 60.0,
                    'ap_hi': 115, 'ap_lo': 75, 'chol': "Normal", 'gluc': "Normal",
                    'smoke': "Non-Smoker", 'alco': "No / Moderate", 'active': "Physically Active"
                }
                st.session_state.prediction_result = None
                st.session_state.form_version += 1
                st.session_state.trigger_scroll = False
                st.rerun()
        with col_btn2:
            if st.button("🔴 High", use_container_width=True, help="Load high-risk sample values"):
                st.session_state.preset_values = {
                    'age': 63, 'gender': "Male", 'height': 172, 'weight': 98.0,
                    'ap_hi': 165, 'ap_lo': 100, 'chol': "Well Above Normal", 'gluc': "Above Normal",
                    'smoke': "Smoker", 'alco': "Regular / High", 'active': "Inactive / Sedentary"
                }
                st.session_state.prediction_result = None
                st.session_state.form_version += 1
                st.session_state.trigger_scroll = False
                st.rerun()
                
    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve Default Form Values
    defaults = st.session_state.preset_values or {
        'age': 52, 'gender': "Female", 'height': 165, 'weight': 68.0,
        'ap_hi': 125, 'ap_lo': 80, 'chol': "Normal", 'gluc': "Normal",
        'smoke': "Non-Smoker", 'alco': "No / Moderate", 'active': "Physically Active"
    }

    # Form using versioned key to ensure automatic reset upon model switch
    f_ver = st.session_state.form_version
    with st.form(f"patient_assessment_form_{f_ver}"):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
            <div>
                <h3 style="margin:0; font-size:20px; font-weight:700; color:#F8FAFC;">📋 Patient Biometrics & Diagnostic Data</h3>
                <span style="font-size:13px; color:#94A3B8;">Fill in the patient's biometrics to assess cardiovascular risk.</span>
            </div>
            <div class="model-badge">
                <span>Active Model: <b>{st.session_state.active_model}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Section 1: Demographics & Anthropometrics
        st.markdown("<div class='card-title'>🪪 1. Patient Demographics & Body Metrics</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=int(defaults['age']), step=1, key=f"age_{f_ver}")
        with col2:
            gender_opts = ["Female", "Male"]
            gender_idx = gender_opts.index(defaults['gender']) if defaults['gender'] in gender_opts else 0
            gender_str = st.selectbox("Biological Sex", gender_opts, index=gender_idx, key=f"gender_{f_ver}")
            gender = 1 if gender_str == "Female" else 2
        with col3:
            height = st.number_input("Height (cm)", min_value=100, max_value=230, value=int(defaults['height']), step=1, key=f"height_{f_ver}")
        with col4:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=220.0, value=float(defaults['weight']), step=0.5, key=f"weight_{f_ver}")

        # Section 2: Blood Pressure & Clinical Vitals
        st.markdown("<div class='card-title'>🩺 2. Blood Pressure & Clinical Vitals</div>", unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            ap_hi = st.number_input("Systolic BP (ap_hi, mmHg)", min_value=70, max_value=240, value=int(defaults['ap_hi']), step=1, help="Peak arterial pressure", key=f"ap_hi_{f_ver}")
        with col6:
            ap_lo = st.number_input("Diastolic BP (ap_lo, mmHg)", min_value=40, max_value=160, value=int(defaults['ap_lo']), step=1, help="Resting arterial pressure", key=f"ap_lo_{f_ver}")
        with col7:
            chol_opts = ["Normal", "Above Normal", "Well Above Normal"]
            chol_idx = chol_opts.index(defaults['chol']) if defaults['chol'] in chol_opts else 0
            chol_str = st.selectbox("Serum Cholesterol Level", chol_opts, index=chol_idx, key=f"chol_{f_ver}")
            chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            cholesterol = chol_map[chol_str]
        with col8:
            gluc_opts = ["Normal", "Above Normal", "Well Above Normal"]
            gluc_idx = gluc_opts.index(defaults['gluc']) if defaults['gluc'] in gluc_opts else 0
            gluc_str = st.selectbox("Fasting Glucose Level", gluc_opts, index=gluc_idx, key=f"gluc_{f_ver}")
            gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            gluc = gluc_map[gluc_str]

        # Section 3: Lifestyle & Behavioral Factors
        st.markdown("<div class='card-title'>🏃 3. Lifestyle & Behavioral Factors</div>", unsafe_allow_html=True)
        col9, col10, col11 = st.columns(3)
        
        with col9:
            smoke_opts = ["Non-Smoker", "Smoker"]
            smoke_idx = smoke_opts.index(defaults['smoke']) if defaults['smoke'] in smoke_opts else 0
            smoke_str = st.selectbox("Tobacco / Smoking Status", smoke_opts, index=smoke_idx, key=f"smoke_{f_ver}")
            smoke = 1 if smoke_str == "Smoker" else 0
        with col10:
            alco_opts = ["No / Moderate", "Regular / High"]
            alco_idx = alco_opts.index(defaults['alco']) if defaults['alco'] in alco_opts else 0
            alco_str = st.selectbox("Alcohol Consumption", alco_opts, index=alco_idx, key=f"alco_{f_ver}")
            alco = 1 if alco_str == "Regular / High" else 0
        with col11:
            act_opts = ["Physically Active", "Inactive / Sedentary"]
            act_idx = act_opts.index(defaults['active']) if defaults['active'] in act_opts else 0
            active_str = st.selectbox("Physical Activity Level", act_opts, index=act_idx, key=f"act_{f_ver}")
            active = 1 if active_str == "Physically Active" else 0

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔍 Check Assessment & Predict Risk", type="primary", use_container_width=True)

    # Process Form Submission
    if submit_btn:
        patient_payload = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }
        
        # Predict using selected model
        st.session_state.prediction_result = handler.predict(patient_payload, model_name=st.session_state.active_model)
        st.session_state.trigger_scroll = True
        st.rerun()

    # ---------------------------------------------------------
    # Auto-Scroll Anchor & Results Section
    # ---------------------------------------------------------
    st.markdown('<div id="cardio-results-anchor"></div>', unsafe_allow_html=True)

    # Render Results only if prediction exists
    if st.session_state.prediction_result is not None:
        res = st.session_state.prediction_result
        risk_pct = res['risk_percentage']
        model_name = res['model_used']
        metrics = res['model_metrics']
        
        st.markdown("---")
        
        # Risk Banner
        if risk_pct < 35.0:
            st.markdown(f"""
            <div class="risk-banner-low">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <h2 style="margin:0; color:#10B981; font-weight:800; font-size:26px;">🟢 Low Cardiovascular Risk ({risk_pct}%)</h2>
                        <p style="margin:4px 0 0 0; color:#A7F3D0; font-size:14px;">The patient exhibits a low estimated probability of cardiovascular disease based on current markers.</p>
                    </div>
                    <div style="background:rgba(16,185,129,0.25); border:1px solid #10B981; padding:6px 16px; border-radius:30px; font-weight:700; color:#34D399; font-size:13px;">
                        Negative for CVD Propensity
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif 35.0 <= risk_pct < 65.0:
            st.markdown(f"""
            <div class="risk-banner-mod">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <h2 style="margin:0; color:#F59E0B; font-weight:800; font-size:26px;">🟡 Moderate Cardiovascular Risk ({risk_pct}%)</h2>
                        <p style="margin:4px 0 0 0; color:#FDE68A; font-size:14px;">Borderline and elevated risk factors detected. Targeted lifestyle modification and clinical monitoring advised.</p>
                    </div>
                    <div style="background:rgba(245,158,11,0.25); border:1px solid #F59E0B; padding:6px 16px; border-radius:30px; font-weight:700; color:#FBBF24; font-size:13px;">
                        Intermediate Monitoring Needed
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-high">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <h2 style="margin:0; color:#EF4444; font-weight:800; font-size:26px;">🔴 High Cardiovascular Risk ({risk_pct}%)</h2>
                        <p style="margin:4px 0 0 0; color:#FCA5A5; font-size:14px;">High probability of cardiovascular risk indicated. Comprehensive clinical evaluation and diagnostic testing recommended.</p>
                    </div>
                    <div style="background:rgba(239,68,68,0.25); border:1px solid #EF4444; padding:6px 16px; border-radius:30px; font-weight:700; color:#F87171; font-size:13px;">
                        Clinical Alert: Elevated Risk
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Main Results Dashboard Columns
        col_res_left, col_res_right = st.columns([1.1, 0.9])
        
        with col_res_left:
            st.markdown(f"""
            <div class="model-badge">
                <span>Inference Model: <b>{model_name}</b> | 5-Fold CV Mean: <b>{metrics['cv_mean']}%</b> | Test Acc: <b>{metrics['test_acc']}%</b></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={'suffix': "%", 'font': {'size': 44, 'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#0EA5E9", 'thickness': 0.26},
                    'bgcolor': "#0F172A",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.22)'},
                        {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.22)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.22)'}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 4},
                        'thickness': 0.8,
                        'value': risk_pct
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=25, r=25, t=25, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Two Key Biometric Status Cards
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                bmi_color = "#10B981" if res['bmi_severity'] == "success" else ("#38BDF8" if res['bmi_severity'] == "info" else ("#F59E0B" if res['bmi_severity'] == "warning" else "#EF4444"))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-sub">Body Mass Index (BMI)</div>
                    <div class="metric-num" style="color: {bmi_color};">{res['bmi']} <span style="font-size:14px; font-weight:500;">kg/m²</span></div>
                    <div style="font-size:13px; margin-top:4px; font-weight:700; color:#E2E8F0;">{res['bmi_category']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_m2:
                bp_color = "#10B981" if res['bp_severity'] == "success" else ("#38BDF8" if res['bp_severity'] == "info" else ("#F59E0B" if res['bp_severity'] == "warning" else "#EF4444"))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-sub">Blood Pressure Stage</div>
                    <div class="metric-num" style="color: {bp_color}; font-size:22px; margin-top:8px;">{res['bp_category']}</div>
                    <div style="font-size:12px; margin-top:6px; color:#94A3B8;">{res['bp_desc']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_res_right:
            st.markdown("### 💡 Evidence-Based Clinical Guidance")
            
            recs = []
            if res['bp_category'] != "Normal":
                recs.append(("🩺 Blood Pressure Management", f"Patient presents with <b>{res['bp_category']}</b> ({res['bp_desc']}). Recommend Dietary Approaches to Stop Hypertension (DASH diet), sodium intake < 1,500 mg/day, and ambulatory BP monitoring."))
            else:
                recs.append(("✅ Optimal Blood Pressure", "Blood pressure is in the optimal physiological range. Continue routine annual screenings."))
                
            if res['bmi'] >= 25.0:
                recs.append(("⚖️ Weight & Metabolic Optimization", f"Calculated BMI is <b>{res['bmi']} kg/m² ({res['bmi_category']})</b>. A structured 5–10% body weight reduction significantly reduces cardiac workload and vascular resistance."))
            else:
                recs.append(("✅ Healthy Body Mass Index", f"BMI of <b>{res['bmi']} kg/m²</b> is optimal. Maintain physical conditioning."))

            if res['feature_contributions'].get('cholesterol', 0) > 0.05:
                recs.append(("🧪 Lipid Profile Attention", "Elevated cholesterol levels detected. Evaluate fasting lipid panel (LDL-C, HDL-C, Triglycerides) and discuss statin therapy if indicated."))

            if res['feature_contributions'].get('active', 0) > 0.05:
                recs.append(("🏃 Physical Activity Recommendation", "Sedentary habit increases cardiovascular mortality risk. Target at least 150 minutes of moderate aerobic activity weekly."))

            for r_title, r_desc in recs:
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-item-title">{r_title}</div>
                    <div class="rec-item-text">{r_desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # Feature Impact Waterfall Breakdown
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown(f"### 🔍 Explainable AI: Feature Risk Contributions ({model_name})")
        st.caption("Displays the patient factors that push the cardiovascular risk score higher (red) versus factors that are protective or within normal physiological baselines (green/slate).")
        
        feature_labels = {
            'ap_hi': 'Systolic Blood Pressure (ap_hi)',
            'ap_lo': 'Diastolic Blood Pressure (ap_lo)',
            'age': 'Patient Age',
            'cholesterol': 'Serum Cholesterol',
            'BMI': 'Body Mass Index (BMI)',
            'weight': 'Weight (kg)',
            'gluc': 'Blood Glucose',
            'smoke': 'Smoking / Tobacco',
            'active': 'Physical Inactivity Penalty',
            'alco': 'Alcohol Consumption',
            'gender': 'Biological Sex',
            'height': 'Height'
        }
        
        contrib_list = []
        for k, label in feature_labels.items():
            impact = res['feature_contributions'].get(k, 0.0)
            contrib_list.append({'Feature': label, 'Impact': impact})
            
        df_impact = pd.DataFrame(contrib_list).sort_values(by='Impact', ascending=True)
        
        colors = []
        for v in df_impact['Impact']:
            if v > 0.02:
                colors.append('#EF4444')
            elif v < -0.02:
                colors.append('#10B981')
            else:
                colors.append('#64748B')
                
        fig_waterfall = go.Figure(go.Bar(
            x=df_impact['Impact'],
            y=df_impact['Feature'],
            orientation='h',
            marker_color=colors,
            text=df_impact['Impact'].apply(lambda x: f"{x:+.3f}" if abs(x) > 0.01 else "Baseline"),
            textposition='auto'
        ))
        fig_waterfall.update_layout(
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'},
            xaxis=dict(title="Relative Risk Contribution (+ = Risk Elevator, - = Protective / Baseline)", gridcolor='#334155'),
            yaxis=dict(title=""),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # Trigger Smooth Scroll down to results if requested
    if st.session_state.trigger_scroll and st.session_state.prediction_result is not None:
        components.html(
            """
            <script>
                setTimeout(function() {
                    const el = window.parent.document.getElementById('cardio-results-anchor');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            </script>
            """,
            height=0,
            width=0
        )
        st.session_state.trigger_scroll = False

# =========================================================
# TAB 2: MODEL BENCHMARK & CLINICAL SELECTION GUIDE
# =========================================================
elif st.session_state.active_page == 'Benchmarks':
    st.markdown("## ⚖️ Model Benchmark & Clinical Selection Guide")
    st.caption("Detailed performance comparison and clinical decision matrix for all trained models.")
    
    summary_df = handler.get_model_summary_df()
    
    # 1. Overall Models Comparison Table
    st.markdown("### 📋 1. Classification Models Performance Matrix")
    st.dataframe(
        summary_df,
        use_container_width=True
    )
    
    # Visual comparison bar chart
    candidate_metrics = ['CV Acc Mean (%)', '5-Fold CV Mean (%)', 'Test Acc (%)', 'Test Precision (%)', 'Test Recall (%)', 'Test F1-score (%)']
    metric_cols = [c for c in candidate_metrics if c in summary_df.columns]
    if metric_cols:
        st.markdown("### 📊 2. Visual Model Comparison Across Metrics")
        fig_comp = px.bar(
            summary_df,
            x='Model',
            y=metric_cols,
            barmode='group',
            color_discrete_sequence=['#0EA5E9', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#38BDF8']
        )
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'},
            yaxis=dict(range=[60, 80], gridcolor='#334155', title="Metric Score (%)"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=1.15, title="")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # 3. Clinical Decision Guide & Model Selection Matrix (Replaces granular fold table)
    st.markdown("---")
    st.markdown("### 🧭 3. Clinical Model Selection & Decision Guide")
    st.caption("Guidance on selecting the optimal model architecture depending on clinical screening priorities:")
    
    guide_data = [
        {
            "Model": "Gradient Boosting",
            "Best Clinical Scenario": "Primary Outpatient Cardiovascular Screening",
            "Key Diagnostic Advantage": "Highest sensitivity (69.0%) & accuracy (73.7%); best captures non-linear risk interactions.",
            "Clinical Trade-Off": "More complex tree ensembles require feature waterfall plots for full explainability."
        },
        {
            "Model": "Random Forest (Bagging)",
            "Best Clinical Scenario": "General Population Risk Stratification",
            "Key Diagnostic Advantage": "High stability against extreme blood pressure spikes and noise across 100 trees.",
            "Clinical Trade-Off": "Slightly more conservative risk estimates near borderline thresholds."
        },
        {
            "Model": "Decision Tree",
            "Best Clinical Scenario": "Point-of-Care Triage & Clinical Pathways",
            "Key Diagnostic Advantage": "Transparent step-by-step decision rules that mirror clinical flowcharts (max_depth=5).",
            "Clinical Trade-Off": "Higher sensitivity to single-feature cutoffs compared to ensembles."
        },
        {
            "Model": "AdaBoost (Boosting)",
            "Best Clinical Scenario": "High-Precision Secondary Confirmation",
            "Key Diagnostic Advantage": "Highest precision (76.4%), effectively minimizing costly false-positive alarms.",
            "Clinical Trade-Off": "Lower recall (64.3%) makes it less suited for initial broad screening."
        },
        {
            "Model": "Logistic Regression",
            "Best Clinical Scenario": "Epidemiological Studies & Medical Audits",
            "Key Diagnostic Advantage": "Direct coefficient interpretability, standard clinical baseline with calibrated probabilities.",
            "Clinical Trade-Off": "Assumes linear relationships between scaled vitals and log-odds."
        }
    ]
    st.dataframe(pd.DataFrame(guide_data), use_container_width=True)
    
    # Sensitivity vs Precision Trade-off Chart
    st.markdown("##### 🎯 Clinical Sensitivity (Recall) vs. Precision Trade-off")
    fig_tradeoff = px.scatter(
        summary_df,
        x='Test Precision (%)',
        y='Test Recall (%)',
        text='Model',
        size=[18]*len(summary_df),
        color='Model',
        color_discrete_sequence=['#0EA5E9', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
    )
    fig_tradeoff.update_traces(textposition='top center')
    fig_tradeoff.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'},
        xaxis=dict(gridcolor='#334155', title="Precision (Minimizing False Positives) %", range=[74.5, 77.0]),
        yaxis=dict(gridcolor='#334155', title="Recall / Sensitivity (Catching True Cases) %", range=[63.0, 70.5]),
        showlegend=False
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

# =========================================================
# TAB 3: POPULATION ANALYTICS
# =========================================================
elif st.session_state.active_page == 'Analytics':
    st.markdown("## 📊 Population Analytics & Epidemiology")
    st.caption("Exploratory analysis of 70,000 cardiovascular disease records.")
    
    csv_paths = [
        os.path.join(os.path.dirname(__file__), "cardio_train.csv"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "cardio_train.csv")
    ]
    df_raw = None
    for cp in csv_paths:
        if os.path.exists(cp):
            try:
                df_raw = pd.read_csv(cp, sep=";")
                df_raw['age_years'] = (df_raw['age'] / 365.25).round().astype(int)
                break
            except Exception:
                pass
                
    if df_raw is not None:
        col_an1, col_an2 = st.columns(2)
        
        with col_an1:
            st.markdown("##### Age Distribution vs. CVD Prevalence")
            fig_hist = px.histogram(
                df_raw.sample(min(10000, len(df_raw)), random_state=42),
                x="age_years",
                color="cardio",
                barmode="overlay",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={"cardio": "Disease Outcome", "age_years": "Age (Years)"},
                nbins=25
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'},
                legend=dict(title="", orientation="h", y=1.1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_an2:
            st.markdown("##### Blood Pressure Stratification vs Disease Probability")
            bp_sample = df_raw[(df_raw['ap_hi'] >= 90) & (df_raw['ap_hi'] <= 200)].copy()
            bp_sample['BP_Bracket'] = pd.cut(
                bp_sample['ap_hi'],
                bins=[90, 120, 130, 140, 180, 220],
                labels=['Normal (<120)', 'Elevated (120-129)', 'Stage 1 (130-139)', 'Stage 2 (140-180)', 'Crisis (>180)']
            )
            bp_agg = bp_sample.groupby('BP_Bracket', observed=False)['cardio'].mean().reset_index()
            bp_agg['cardio_pct'] = bp_agg['cardio'] * 100
            
            fig_bp = px.bar(
                bp_agg,
                x='BP_Bracket',
                y='cardio_pct',
                color='cardio_pct',
                color_continuous_scale=['#38BDF8', '#F59E0B', '#EF4444'],
                text_auto='.1f',
                labels={'cardio_pct': 'Disease Rate (%)', 'BP_Bracket': 'Systolic BP Category'}
            )
            fig_bp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'},
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bp, use_container_width=True)

# =========================================================
# TAB 4: CLINICAL DOCUMENTATION & ARCHITECTURE (RESTORED PREVIOUS STYLE)
# =========================================================
elif st.session_state.active_page == 'Docs':
    st.markdown("## ℹ️ Model Architecture & Technical Details")
    
    st.markdown("""
    - **Model Architecture**: Scikit-Learn Supervised Classification Suite (Gradient Boosting, Random Forest, Decision Tree, AdaBoost, Logistic Regression)
    - **Preprocessing Pipeline**: `StandardScaler` applied to continuous variables (`age`, `ap_hi`, `ap_lo`, `height`, `weight`, `BMI`)
    - **Training Dataset**: Kaggle Cardiovascular Disease Dataset (70,000 clinical records)
    - **Cross-Validation Framework**: Stratified 5-Fold Cross Validation evaluated on Accuracy, Precision, Recall, and F1-Score
    """)

    st.markdown("---")
    st.markdown("#### 🔬 Learned Feature Importance & Clinical Diagnostic Weights")
    st.caption("Relative impact of biometric features learned during model training:")
    
    # Feature importance table
    feat_data = [
        {"Clinical Feature": "Systolic Blood Pressure (ap_hi)", "Feature Type": "Hemodynamic", "Clinical Importance": "Primary Driver (Highest Weight)"},
        {"Clinical Feature": "Diastolic Blood Pressure (ap_lo)", "Feature Type": "Hemodynamic", "Clinical Importance": "High Impact"},
        {"Clinical Feature": "Patient Age", "Feature Type": "Demographic", "Clinical Importance": "High Impact"},
        {"Clinical Feature": "Serum Cholesterol", "Feature Type": "Metabolic Biomarker", "Clinical Importance": "Moderate Impact"},
        {"Clinical Feature": "Body Mass Index (BMI)", "Feature Type": "Anthropometric", "Clinical Importance": "Moderate Impact"},
        {"Clinical Feature": "Blood Glucose Level", "Feature Type": "Metabolic Biomarker", "Clinical Importance": "Moderate Impact"},
        {"Clinical Feature": "Physical Inactivity", "Feature Type": "Behavioral", "Clinical Importance": "Risk Modifier"},
        {"Clinical Feature": "Smoking Status", "Feature Type": "Behavioral", "Clinical Importance": "Risk Modifier"},
        {"Clinical Feature": "Alcohol Consumption", "Feature Type": "Behavioral", "Clinical Importance": "Secondary Factor"},
        {"Clinical Feature": "Biological Sex", "Feature Type": "Demographic", "Clinical Importance": "Baseline Factor"}
    ]
    st.dataframe(pd.DataFrame(feat_data), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🩺 Clinical Classification Standards")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        ##### Blood Pressure Categories (ACC/AHA 2017 Guidelines)
        - **Normal**: Systolic < 120 mmHg and Diastolic < 80 mmHg
        - **Elevated**: Systolic 120–129 mmHg and Diastolic < 80 mmHg
        - **Stage 1 Hypertension**: Systolic 130–139 mmHg or Diastolic 80–89 mmHg
        - **Stage 2 Hypertension**: Systolic ≥ 140 mmHg or Diastolic ≥ 90 mmHg
        - **Hypertensive Crisis**: Systolic > 180 mmHg and/or Diastolic > 120 mmHg
        """)
    with col_d2:
        st.markdown("""
        ##### Body Mass Index Categories (WHO Standards)
        - **Underweight**: BMI < 18.5 kg/m²
        - **Normal Weight**: BMI 18.5 – 24.9 kg/m²
        - **Overweight**: BMI 25.0 – 29.9 kg/m²
        - **Obesity**: BMI ≥ 30.0 kg/m²
        """)

# ---------------------------------------------------------
# Footer & Clinical Disclaimer
# ---------------------------------------------------------
st.markdown("""
<br><hr>
<div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); color: #FCA5A5; font-size: 12px; padding: 14px 20px; border-radius: 10px;">
    <b>⚠️ Important Clinical Disclaimer:</b> CardioDetect AI is a clinical decision-support and educational tool. Risk scores are statistical estimates derived from population data and must not be used as an independent clinical diagnosis. Always consult a board-certified physician or cardiologist for medical evaluation and diagnostic testing.
</div>
""", unsafe_allow_html=True)
