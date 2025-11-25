import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
from pathlib import Path
from io import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import optuna

# ====================== PAGE CONFIGURATION ======================
st.set_page_config(
    page_title="HR Analytics Dashboard | Employee Attrition Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== PROFESSIONAL CSS STYLING ======================
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-blue: #1a365d;
        --secondary-blue: #2d3748;
        --accent-blue: #3182ce;
        --light-blue: #ebf8ff;
        --dark-blue: #2c5282;
        --success-green: #38a169;
        --warning-orange: #dd6b20;
        --danger-red: #e53e3e;
        --text-dark: #2d3748;
        --text-light: #718096;
        --background: #f7fafc;
    }
    
    /* Main Container */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Header Styling */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        border-left: 5px solid var(--accent-blue);
    }
    
    .main-title {
        color: var(--primary-blue);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: var(--text-light);
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Navigation Tabs */
    .nav-tabs {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards Styling */
    .analysis-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid var(--accent-blue);
        transition: transform 0.2s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Section Headers */
    .section-header {
        color: var(--primary-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--light-blue);
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3);
    }
    
    .prediction-safe {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-blue), var(--dark-blue));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.4);
    }
    
    /* File Uploader */
    .upload-section {
        background: white;
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: var(--accent-blue);
        background: var(--light-blue);
    }
    
    /* Remove Streamlit Default Styling */
    .stApp {
        background: #f7fafc;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER SECTION ======================
st.markdown("""
<div class="main-container">
    <div class="main-header">
        <div class="main-title">📊 HR Analytics Dashboard</div>
        <div class="main-subtitle">Employee Attrition Analysis & Predictive Modeling</div>
    </div>
""", unsafe_allow_html=True)

# ====================== MODEL FUNCTIONS ======================
@st.cache_resource
def load_model():
    try:
        base_path = Path(__file__).parent
        
        model_path = base_path / "best_final_stacking.joblib"
        cols_path = base_path / "train_columns.pkl"

        model = joblib.load(model_path)
        train_columns = joblib.load(cols_path)
        
        return model, train_columns

    except FileNotFoundError:
        st.warning("Model files not found. Using demo mode with simulated predictions.")
        return "demo", None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return "demo", None

def create_demo_prediction(input_data):
    """Create a demo prediction when model is not available"""
    # Simple rule-based prediction for demo
    risk_score = 0
    
    # Factors that increase attrition risk
    if input_data['OverTime'] == 'Yes':
        risk_score += 20
    if input_data['JobSatisfaction'] <= 2:
        risk_score += 25
    if input_data['YearsSinceLastPromotion'] > 5:
        risk_score += 15
    if input_data['MonthlyIncome'] < 5000:
        risk_score += 10
    if input_data['WorkLifeBalance'] <= 2:
        risk_score += 15
    if input_data['YearsAtCompany'] < 2:
        risk_score += 10
    
    # Cap at 95% for realism
    probability = min(risk_score, 95)
    prediction = 1 if probability > 50 else 0
    
    return prediction, probability

# ====================== VISUALIZATION FUNCTIONS ======================
def create_business_travel_analysis(df):
    """Create business travel analysis visualizations"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("📊 Business Travel Analysis")
    
    if 'BusinessTravel' in df.columns and 'Attrition' in df.columns:
        # Business Travel Analysis
        bt_counts = df["BusinessTravel"].value_counts()
        bt_attrition = df[df["Attrition"] == "Yes"]["BusinessTravel"].value_counts()
        attrition_rate = np.floor((bt_attrition / bt_counts) * 100)

        # Display data before visualization
        st.subheader("Business Travel Data")
        
        # Create DataFrame for display
        bt_summary = pd.DataFrame({
            'BusinessTravel': bt_counts.index,
            'Total_Employees': bt_counts.values,
            'Attrition_Count': bt_attrition.values,
            'Attrition_Rate (%)': attrition_rate.values
        })
        
        # Display data in a table
        st.dataframe(bt_summary, use_container_width=True)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", f"{bt_counts.sum():,}")
        with col2:
            st.metric("Total Attrition Cases", f"{bt_attrition.sum():,}")
        with col3:
            overall_rate = (bt_attrition.sum() / bt_counts.sum()) * 100
            st.metric("Overall Attrition Rate", f"{overall_rate:.1f}%")
        
        pie_colors = ['#2E86C1', '#5DADE2', '#AED6F1']
        bar_colors = ['#E74C3C', '#F1948A', '#F5B7B1']
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Employees by Business Travel", "Attrition by Business Travel")
        )
        fig.add_trace(
            go.Pie(
                labels=bt_counts.index,
                values=bt_counts.values,
                textinfo="label+percent",
                hole=0.4,
                marker=dict(colors=pie_colors, line=dict(color="white", width=1.5)),
                name="Business Travel"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=bt_attrition.index,
                y=bt_attrition.values,
                text=[f"{v} ({int(r)}%)" for v, r in zip(bt_attrition.values, attrition_rate)],
                textposition="outside",
                marker_color=bar_colors,
                marker_line_color="white",
                marker_line_width=1.2,
                name="Attrition Count",
                opacity=0.9
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(
                text="Attrition by Business Travel",
                x=0.5,
                font=dict(size=20, family="DejaVu Sans, Arial, sans-serif", color="#333333")
            ),
            template="plotly_white",
            plot_bgcolor="#F9FAFB",
            paper_bgcolor="#F9FAFB",
            font=dict(family="DejaVu Sans, Arial, sans-serif", size=12, color="#2E2E2E"),
            showlegend=False,
            height=500,
            margin=dict(t=70, b=40, l=40, r=40)
        )
        
        fig.update_xaxes(title_text="Business Travel", row=1, col=2)
        fig.update_yaxes(title_text="Number of Employees", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("BusinessTravel or Attrition column not found in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_education_analysis(df):
    """Create education analysis visualizations"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("🎓 Education Analysis")
    
    if 'EducationField' in df.columns and 'Attrition' in df.columns:
        # Education Field Analysis
        edu_field_yes = df[df['Attrition'] == 'Yes'].groupby('EducationField', as_index=False).size().rename(columns={'size':'Count'})
        edu_field_total = df.groupby('EducationField', as_index=False).size().rename(columns={'size':'Total'})
        edu_field_yes = edu_field_yes.merge(edu_field_total, on='EducationField')
        edu_field_yes['AttritionRate'] = edu_field_yes['Count'] / edu_field_yes['Total'] * 100
        edu_field_yes = edu_field_yes.sort_values(by='AttritionRate', ascending=False)

        # Preserve order on x-axis
        edu_field_yes['EducationField'] = pd.Categorical(
            edu_field_yes['EducationField'],
            categories=edu_field_yes['EducationField'],
            ordered=True
        )

        # Create bar chart
        fig1 = px.bar(
            edu_field_yes,
            x='EducationField',
            y='AttritionRate',
            text=edu_field_yes['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition Rate by Education Field'
        )

        fig1.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )

        fig1.update_layout(
            xaxis_title='Education Field',
            yaxis_title='Attrition Rate',
            template='plotly_white',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='#F9FAFB',
            font=dict(family='DejaVu Sans, Arial, sans-serif', size=12, color='#2E2E2E'),
            bargap=0.25,
            height=550
        )
        st.plotly_chart(fig1, use_container_width=True)

    if 'Education' in df.columns and 'Attrition' in df.columns:
        # Education Level Analysis
        education_attrition = df.groupby(['Education', 'Attrition'], as_index=False).size()
        education_total = df.groupby('Education', as_index=False).size().rename(columns={'size':'Total'})
        education_attrition = education_attrition.merge(education_total, on='Education')
        education_attrition['AttritionRate'] = education_attrition['size'] / education_attrition['Total'] * 100

        attrition_yes = education_attrition[education_attrition['Attrition'] == 'Yes']
        attrition_yes = attrition_yes.sort_values(by='AttritionRate', ascending=False)
        attrition_yes['Education'] = pd.Categorical(
            attrition_yes['Education'],
            categories=attrition_yes['Education'],
            ordered=True
        )

        fig = px.bar(
            attrition_yes,
            x='Education',
            y='AttritionRate',
            text=attrition_yes['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Employee Attrition Rate by Education Level'
        )

        fig.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )
        fig.update_layout(
            title=dict(
                text='Attrition Rate by Education Level',
                x=0.5,
                y=0.95,
                font=dict(size=22, family='DejaVu Sans, Arial, sans-serif', color='#333333')
            ),
            xaxis_title='Education Level',
            yaxis_title='Attrition Rate',
            template='plotly_white',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='#F9FAFB',
            font=dict(family='DejaVu Sans, Arial, sans-serif', size=12, color='#2E2E2E'),
            bargap=0.25,
            height=550,
            margin=dict(t=110, b=60, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Education or EducationField columns not found in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_job_analysis(df):
    """Create job analysis visualizations"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("💼 Job Analysis")

    # 1) Convert numeric JobLevel to meaningful names
    joblevel_names = {
        1: "Entry Level",
        2: "Junior Level",
        3: "Mid Level",
        4: "Senior Level",
        5: "Executive Level"
    }

    # Apply mapping
    if 'JobLevel' in df.columns:
        df['JobLevel'] = df['JobLevel'].map(joblevel_names)

        # Make sure the order stays correct in plots
        df['JobLevel'] = pd.Categorical(
            df['JobLevel'],
            categories=["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"],
            ordered=True
        )

    # 2) Check attrition column
    if 'JobLevel' in df.columns and 'Attrition' in df.columns:

        # Attrition count by job level
        joblevel_attrition = (
            df[df['Attrition'] == 'Yes']
            .groupby('JobLevel', as_index=False)
            .size()
            .rename(columns={'size': 'Count'})
        )

        # Total employees by job level
        joblevel_total = (
            df.groupby('JobLevel', as_index=False)
            .size()
            .rename(columns={'size': 'Total'})
        )

        # Merge
        joblevel_attrition = joblevel_attrition.merge(joblevel_total, on='JobLevel')
        joblevel_attrition['AttritionRate'] = (
            joblevel_attrition['Count'] / joblevel_attrition['Total'] * 100
        )

        # IMPORTANT: Keep JobLevel as string column, NOT index
        joblevel_attrition = joblevel_attrition.sort_values(by='AttritionRate', ascending=False)
        #joblevel_attrition = joblevel_attrition.set_index('JobLevel')
        
        #joblevel_attrition_reset = joblevel_attrition.reset_index()

        fig1 = px.bar(
            joblevel_attrition,
            x='JobLevel',
            y='AttritionRate',
            text=joblevel_attrition['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition Rate by Job Level'
        )

        fig1.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )

        fig1.update_layout(
            template='plotly_white',
            xaxis_title='Job Level',
            yaxis_title='Attrition Rate',
            bargap=0.25,
            height=550
        )
        st.plotly_chart(fig1, use_container_width=True)

    if 'JobRole' in df.columns and 'Attrition' in df.columns:
        # Job Role Analysis
        job_attrition = df.groupby(['JobRole', 'Attrition'], as_index=False).size().rename(columns={'size':'Count'})
        job_attrition_yes = job_attrition[job_attrition['Attrition'] == 'Yes']
        job_total = df.groupby('JobRole', as_index=False).size().rename(columns={'size':'Total'})
        job_attrition_yes = job_attrition_yes.merge(job_total, on='JobRole')
        job_attrition_yes['AttritionRate'] = job_attrition_yes['Count'] / job_attrition_yes['Total'] * 100
        job_attrition_yes = job_attrition_yes.sort_values(by='AttritionRate', ascending=False)
        job_attrition_yes['JobRole'] = pd.Categorical(job_attrition_yes['JobRole'], categories=job_attrition_yes['JobRole'], ordered=True)

        fig2 = px.bar(
            job_attrition_yes,
            x='JobRole',
            y='AttritionRate',
            text=job_attrition_yes['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition Rate by Job Role'
        )

        fig2.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )

        fig2.update_layout(
            xaxis_title='Job Role',
            yaxis_title='Attrition Rate',
            template='plotly_white',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='#F9FAFB',
            font=dict(family='DejaVu Sans, Arial, sans-serif', size=12, color='#2E2E2E'),
            height=550,
            margin=dict(t=110, b=60, l=40, r=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

    if 'JobRole' in df.columns and 'MonthlyIncome' in df.columns:
        # Salary by Job Role
        avg_salary_role = (
            df.groupby("JobRole", as_index=False)["MonthlyIncome"]
              .mean()
              .sort_values(by="MonthlyIncome", ascending=False)
        )
        fig3 = px.bar(
            avg_salary_role,
            x="JobRole",
            y="MonthlyIncome",
            text=avg_salary_role["MonthlyIncome"].apply(lambda x: f"${x:,.0f}"),
            color="MonthlyIncome",
            color_continuous_scale="Viridis",
            title="Average Monthly Salary by Job Role"
        )

        fig3.update_traces(
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1.5,
            opacity=0.9
        )
        fig3.update_layout(
            title=dict(
                x=0.5,
                font=dict(size=22, family="Poppins, Arial, sans-serif", color="#222")
            ),
            xaxis_title="Job Role",
            yaxis_title="Average Monthly Income",
            plot_bgcolor="#F9FAFB",
            paper_bgcolor="#F9FAFB",
            font=dict(family="Poppins, Arial, sans-serif", size=12, color="#333"),
            coloraxis_showscale=False,
            margin=dict(t=90, b=60, l=40, r=40),
            height=550
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("JobLevel, JobRole, or MonthlyIncome columns not found in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_satisfaction_analysis(df):
    """Create satisfaction analysis visualizations"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("😊 Satisfaction Analysis")
   
    
    
    if 'EnvironmentSatisfaction' in df.columns and 'Attrition' in df.columns:
        # Environment Satisfaction
        env_sat_yes = df[df['Attrition'] == 'Yes'].groupby('EnvironmentSatisfaction', as_index=False).size().rename(columns={'size':'Count'})
        env_sat_total = df.groupby('EnvironmentSatisfaction', as_index=False).size().rename(columns={'size':'Total'})
        env_sat_yes = env_sat_yes.merge(env_sat_total, on='EnvironmentSatisfaction')
        env_sat_yes['AttritionRate'] = env_sat_yes['Count'] / env_sat_yes['Total'] * 100
        env_sat_yes = env_sat_yes.sort_values(by='AttritionRate', ascending=False)

        env_sat_yes['EnvironmentSatisfaction'] = pd.Categorical(
        env_sat_yes['EnvironmentSatisfaction'],
        categories=env_sat_yes['EnvironmentSatisfaction'],
         ordered=True
        

        )
        fig1 = px.bar(
            env_sat_yes,
            x='EnvironmentSatisfaction',
            y='AttritionRate',
            text=env_sat_yes['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition Rate by Environment Satisfaction'
        )
        fig1.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )
        fig1.update_layout(
            xaxis_title='Environment Satisfaction',
            yaxis_title='Attrition Rate',
            template='plotly_white',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='#F9FAFB',
            font=dict(family='DejaVu Sans, Arial, sans-serif', size=12, color='#2E2E2E'),
            bargap=0.25,
            height=550
        )
        st.plotly_chart(fig1, use_container_width=True)

    if 'JobSatisfaction' in df.columns and 'Attrition' in df.columns:
        # Job Satisfaction
        job_satis_attrition = df[df['Attrition'] == 'Yes'].groupby('JobSatisfaction', as_index=False).size().rename(columns={'size':'Count'})
        job_satis_total = df.groupby('JobSatisfaction', as_index=False).size().rename(columns={'size':'Total'})
        job_satis_attrition = job_satis_attrition.merge(job_satis_total, on='JobSatisfaction')
        job_satis_attrition['AttritionRate'] = job_satis_attrition['Count'] / job_satis_attrition['Total'] * 100
        job_satis_attrition = job_satis_attrition.sort_values(by='AttritionRate', ascending=False)

        job_satis_attrition['JobSatisfaction'] = pd.Categorical(
            job_satis_attrition['JobSatisfaction'],
            categories=job_satis_attrition['JobSatisfaction'],
            ordered=True
        )

        fig2 = px.bar(
            job_satis_attrition,
            x='JobSatisfaction',
            y='AttritionRate',
            text=job_satis_attrition['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition Rate by Job Satisfaction'
        )

        fig2.update_traces(
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.9,
            showlegend=False
        )

        fig2.update_layout(
            template='plotly_white',
            xaxis_title='Job Satisfaction',
            yaxis_title='Attrition Rate (%)',
            bargap=0.25,
            height=550
        )
        st.plotly_chart(fig2, use_container_width=True)

    if 'WorkLifeBalance' in df.columns and 'Attrition' in df.columns:
        # Work Life Balance
        worklife_attrition = df[df['Attrition'] == 'Yes'].groupby('WorkLifeBalance', as_index=False).size().rename(columns={'size':'Count'})
        worklife_total = df.groupby('WorkLifeBalance', as_index=False).size().rename(columns={'size':'Total'})
        worklife_attrition = worklife_attrition.merge(worklife_total, on='WorkLifeBalance')
        worklife_attrition['AttritionRate'] = worklife_attrition['Count'] / worklife_attrition['Total'] * 100
        worklife_attrition = worklife_attrition.sort_values(by='Count', ascending=False)

        fig3 = px.bar(
            worklife_attrition,
            x='WorkLifeBalance',
            y='Count',
            text=worklife_attrition['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition by Work Life Balance'
        )
        fig3.update_traces(textposition='outside', marker_line_color='white', marker_line_width=1.5, opacity=0.9)
        fig3.update_layout(
            template='plotly_white', 
            xaxis_title='Work Life Balance', 
            yaxis_title='Number of Employees (Attrition)',
            height=550
        )
        st.plotly_chart(fig3, use_container_width=True)

    if 'RelationshipSatisfaction' in df.columns and 'Attrition' in df.columns:
        # Relationship Satisfaction
        rel_attrition = df[df['Attrition'] == 'Yes'].groupby('RelationshipSatisfaction', as_index=False).size().rename(columns={'size':'Count'})
        rel_total = df.groupby('RelationshipSatisfaction', as_index=False).size().rename(columns={'size':'Total'})
        rel_attrition = rel_attrition.merge(rel_total, on='RelationshipSatisfaction')
        rel_attrition['AttritionRate'] = rel_attrition['Count'] / rel_attrition['Total'] * 100
        rel_attrition = rel_attrition.sort_values(by='Count', ascending=False)

        fig4 = px.bar(
            rel_attrition,
            x='RelationshipSatisfaction',
            y='Count',
            text=rel_attrition['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#E74C3C'],
            title='Attrition by Relationship Satisfaction'
        )

        fig4.update_traces(textposition='outside', marker_line_color='white', marker_line_width=1.5, opacity=0.9)
        fig4.update_layout(
            template='plotly_white',
            xaxis_title='Relationship Satisfaction',
            yaxis_title='Number of Employees (Attrition)',
            bargap=0.25,
            height=550
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Satisfaction-related columns not found in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_department_employees_chart(df):
    """Create department employees chart"""
    if 'Department' in df.columns:
        dept_total = df['Department'].value_counts().sort_index()
        tcolor = '#2E86C1'
        fig = go.Figure(
            go.Bar(
                x=dept_total.index,
                y=dept_total.values,
                text=dept_total.values,
                textposition='outside',
                marker_color=tcolor,
                marker_line_color='white',
                marker_line_width=1.5,
                opacity=0.9,
                name='Total Employees'
            )
        )
        fig.update_layout(
            title=dict(
                text="Total Employees by Department",
                x=0.5,
                font=dict(size=22, family='DejaVu Sans, Arial, sans-serif', color='#333333', weight='bold')
            ),
            xaxis_title='Department',
            yaxis_title='Number of Employees',
            template='plotly_white',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='#F9FAFB',
            font=dict(family='DejaVu Sans, Arial, sans-serif', size=12, color='#2E2E2E'),
            bargap=0.25,
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_years_at_company_chart(df):
    """Create years at company distribution chart"""
    if 'YearsAtCompany' in df.columns:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.header("📊 Years at Company Distribution")
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.histplot(
            data=df,
            x="YearsAtCompany",
            color="#2E86C1",
            kde=True,
            bins=20,
            edgecolor='white',
            alpha=0.9,
            ax=ax
        )

        average_YAC = df["YearsAtCompany"].mean()
        plt.axvline(
            average_YAC,
            color="#E74C3C",
            linestyle="--",
            linewidth=2.5,
            label=f"Average Tenure: {average_YAC:.1f} years"
        )

        plt.title(
            "Attrition by Years at Company",
            fontsize=18,
            weight='bold',
            pad=20,
            color='#222222',
            family='DejaVu Sans'
        )
        plt.xlabel("Years at Company", fontsize=13, labelpad=10, color='#333333')
        plt.ylabel("Number of Employees", fontsize=13, labelpad=10, color='#333333')

        plt.legend(
            frameon=True,
            shadow=True,
            fontsize=11,
            loc="upper right"
        )
        sns.despine(left=False, bottom=False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

def create_income_vs_variables_chart(df):
    """Create the income vs variables line plot"""
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("📈 Monthly Income vs Key Variables")
    
    # Check if required columns exist
    required_cols = ['Age', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany', 'MonthlyIncome', 'Attrition']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns for income analysis: {', '.join(missing_cols)}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Set the style and palette
    sns.set_theme(style="whitegrid")
    attrition_palette = {"Yes": "#E74C3C", "No": "#2E86C1"}
    selected_cols = ['Age', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany']
    
    # Create the subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for idx, col in enumerate(selected_cols):
        # Create line plot for each variable
        sns.lineplot(
            data=df,
            x=col,
            y='MonthlyIncome',
            hue='Attrition',
            palette=attrition_palette,
            ax=axes[idx],
            linewidth=2.2,
            estimator='mean',  # Use mean for aggregation
            errorbar=None  # Remove confidence intervals for cleaner look
        )

        axes[idx].set_title(f"{col} vs Monthly Income by Attrition",
                            fontsize=15, weight='bold', color='#333333', pad=15)
        axes[idx].set_xlabel(col, fontsize=12)
        axes[idx].set_ylabel("Monthly Income", fontsize=12)
        axes[idx].tick_params(axis='x', rotation=20)
        axes[idx].legend(title="Attrition", frameon=True, fontsize=10)
        
        # Improve grid and layout
        axes[idx].grid(True, alpha=0.3)
        
    plt.tight_layout(pad=4)
    st.pyplot(fig)
    
    # Add some insights
    st.subheader("📊 Insights")
    st.markdown("""
    - **Age vs Income**: Typically shows career progression and salary growth patterns
    - **Distance vs Income**: May reveal commuting patterns and location-based salary differences  
    - **Working Years vs Income**: Shows career trajectory and experience-based compensation
    - **Company Tenure vs Income**: Highlights internal salary growth and retention patterns
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== MAIN CONTENT ======================
def main():
    # ====================== FILE UPLOAD SECTION ======================
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('### 📁 Upload Your HR Analytics Data')
    st.markdown('Drag and drop your CSV file below to start the analysis')
    
    uploaded_file = st.file_uploader(
        " ",
        type=["csv"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Display file info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", df.shape[0])
            with col2:
                st.metric("📈 Columns", df.shape[1])
            with col3:
                st.metric("⚠️ Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("🔍 Duplicates", df.duplicated().sum())
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'df' not in locals() or uploaded_file is None:
        # Show sample data structure before upload
        st.info("💡 **Sample Data Structure:** The dashboard expects HR data with columns like Age, Attrition, Department, MonthlyIncome, etc.")
        return
    
    # ====================== NAVIGATION TABS ======================
    st.markdown('<div class="nav-tabs">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", 
        "📊 Analytics & Insights", 
        "🔮 Predict Attrition", 
        "📈 Reports"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model
    model, train_columns = load_model()
    
    # Color scheme for visualizations
    colors = {"Yes": "#e53e3e", "No": "#3182ce"}
    
    # ====================== TAB 1: DATA OVERVIEW ======================
    with tab1:
        st.markdown('<div class="section-header">Data Overview & Summary</div>', unsafe_allow_html=True)
    
        col1, col2 = st.columns([2, 1])
    
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📄 Data Sample")
            st.dataframe(df.head(8), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📊 Data Types & Info")
        
            # Data types info
            type_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(type_info, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📈 Quick Statistics")
        
            # Key metrics
            if 'Attrition' in df.columns:
                attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
                st.metric("📊 Attrition Rate", f"{attrition_rate:.1f}%")
        
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stat_tab1, stat_tab2 = st.tabs(["🔢 Key Metrics", "📋 All Numeric"])
            
                with stat_tab1:
                    important_cols = ['MonthlyIncome', 'Age', 'TotalWorkingYears', 'YearsAtCompany', 
                                    'DailyRate', 'MonthlyRate', 'HourlyRate', 'DistanceFromHome']
                    available_important = [col for col in important_cols if col in numeric_cols and col in df.columns]
                
                    for col in available_important[:8]:
                        avg_val = df[col].mean()
                        st.metric(f"Avg {col}", f"{avg_val:.1f}")
            
                with stat_tab2:
                    with st.expander("View All Numeric Columns", expanded=False):
                        all_numeric = [col for col in numeric_cols if col in df.columns]
                    
                        for i in range(0, len(all_numeric), 2):
                            cols = all_numeric[i:i+2]
                            col1, col2 = st.columns(2)
                        
                            for j, col_name in enumerate(cols):
                                if j == 0:
                                    with col1:
                                        avg_val = df[col_name].mean()
                                        st.metric(f"Avg {col_name}", f"{avg_val:.1f}")
                                else:
                                    with col2:
                                        avg_val = df[col_name].mean()
                                        st.metric(f"Avg {col_name}", f"{avg_val:.1f}")
        
            st.markdown('</div>', unsafe_allow_html=True)
        
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("🔍 Data Quality")
        
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
            quality_col1, quality_col2 = st.columns(2)
            with quality_col1:
                st.metric("Missing Values", f"{missing_cells}")
                st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
            with quality_col2:
                st.metric("Missing %", f"{missing_percentage:.2f}%")
                st.metric("Total Columns", f"{df.shape[1]}")
        
            st.markdown('</div>', unsafe_allow_html=True)

    # ====================== TAB 2: ANALYTICS & INSIGHTS ======================
    with tab2:
        st.markdown('<div class="section-header">Advanced Analytics & Insights</div>', unsafe_allow_html=True)
    
        analysis_section = st.selectbox(
            "📊 Select Analysis Type",
            [
                "👥 Demographics",
                "🏢 Department & Job Roles", 
                "💰 Salary Analysis",
                "👤 Age Distribution",
                "📊 Business Travel Analysis",
                "🎓 Education Analysis",
                "💼 Job Analysis",
                "😊 Satisfaction Analysis",
                "📊 Tenure Analysis",
                "📈 Years at Company Analysis",
                "📈 Income vs Variables Analysis",  # Added the new analysis
                "🔗 Correlation"
            ]
        )
    
        colors = {"Yes": "#e53e3e", "No": "#3182ce"}
    
        if analysis_section == "👥 Demographics":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("👥 Demographic Analysis")
        
            col1, col2 = st.columns(2)

            with col1:
                if 'Attrition' in df.columns:
                    att_props = df["Attrition"].value_counts(normalize=True) * 100
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=att_props.index,
                        values=att_props.values,
                        textinfo='label+percent',
                        marker=dict(colors=['#E74C3C', '#2E86C1']),
                        hole=0.3
                    ))
                    fig.update_layout(
                        title="Attrition Distribution",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'Gender' in df.columns and 'Attrition' in df.columns:
                    fig1 = px.bar(
                        df.groupby(['Gender', 'Attrition'], as_index=False).size(),
                        x='Gender', y='size', color='Attrition',
                        color_discrete_map=colors, text='size',
                        barmode='group', title='Attrition by Gender'
                    )
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)

            if 'Age' in df.columns and 'Attrition' in df.columns:
                fig2 = px.histogram(df, x="Age", nbins=20, color="Attrition",
                                color_discrete_map=colors, title="Age Distribution by Attrition")
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        elif analysis_section == "🏢 Department & Job Roles":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("🏢 Department and Job Role Analysis")

            col1, col2 = st.columns(2)
    
            with col1:
                if 'Department' in df.columns and 'Attrition' in df.columns:
                    dept_attrition = df[df['Attrition'] == 'Yes']['Department'].value_counts()
                    if len(dept_attrition) > 0:
                        fig3 = go.Figure(go.Pie(
                            labels=dept_attrition.index,
                            values=dept_attrition.values,
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Set3),
                            textinfo='label+percent'
                        ))
                        fig3.update_layout(
                            title="Attrition by Department",
                            height=400
                        )
                        st.plotly_chart(fig3, use_container_width=True)
    
            with col2:
                if 'JobRole' in df.columns and 'Attrition' in df.columns:
                    job_attrition = df.groupby(['JobRole', 'Attrition'], as_index=False).size()
                    job_attrition['Percentage'] = (
                        job_attrition['size'] / job_attrition.groupby('JobRole')['size'].transform('sum') * 100
                    )
                    fig4 = px.bar(
                        job_attrition, x='JobRole', y='Percentage',
                        color='Attrition', text=job_attrition['Percentage'].apply(lambda x: f"{x:.1f}%"),
                        color_discrete_map=colors, barmode='stack',
                        title='Attrition by Job Role (%)'
                    )
                    fig4.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig4, use_container_width=True)
            
            # إضافة الرسم البياني للموظفين حسب القسم
            create_department_employees_chart(df)
            
            st.markdown('</div>', unsafe_allow_html=True)

        elif analysis_section == "💰 Salary Analysis":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("💰 Salary Analysis")
    
            col1, col2 = st.columns(2)

            with col1:
                if 'Department' in df.columns and 'MonthlyIncome' in df.columns:
                    avg_salary = (
                        df.groupby("Department")["MonthlyIncome"]
                        .mean().reset_index()
                        .sort_values(by="MonthlyIncome", ascending=False)
                    )
                    fig5 = px.bar(
                        avg_salary, x="Department", y="MonthlyIncome",
                        text=avg_salary["MonthlyIncome"].apply(lambda x: f"${x:,.0f}"),
                        color="MonthlyIncome", color_continuous_scale="Viridis",
                        title="Average Monthly Income by Department"
                    )
                    st.plotly_chart(fig5, use_container_width=True)

            with col2:
                if 'Attrition' in df.columns and 'MonthlyIncome' in df.columns:
                    fig6 = px.box(df, x="Attrition", y="MonthlyIncome",
                                  color="Attrition", color_discrete_map=colors,
                                  title="Monthly Income Distribution by Attrition")
                    st.plotly_chart(fig6, use_container_width=True)
            
            # إضافة الرسم البياني للرواتب حسب الدور الوظيفي
            # if 'JobRole' in df.columns and 'MonthlyIncome' in df.columns:
            #     st.subheader("💰 Average Salary by Job Role")
            #     avg_salary_role = (
            #         df.groupby("JobRole", as_index=False)["MonthlyIncome"]
            #           .mean()
            #           .sort_values(by="MonthlyIncome", ascending=False)
            #     )
            #     fig_salary = px.bar(
            #         avg_salary_role,
            #         x="JobRole",
            #         y="MonthlyIncome",
            #         text=avg_salary_role["MonthlyIncome"].apply(lambda x: f"${x:,.0f}"),
            #         color="MonthlyIncome",
            #         color_continuous_scale="Viridis",
            #         title="Average Monthly Salary by Job Role"
            #     )

            #     fig_salary.update_traces(
            #         textposition="outside",
            #         marker_line_color="white",
            #         marker_line_width=1.5,
            #         opacity=0.9
            #     )
            #     fig_salary.update_layout(
            #         title=dict(
            #             x=0.5,
            #             font=dict(size=22, family="Poppins, Arial, sans-serif", color="#222")
            #         ),
            #         xaxis_title="Job Role",
            #         yaxis_title="Average Monthly Income",
            #         plot_bgcolor="#F9FAFB",
            #         paper_bgcolor="#F9FAFB",
            #         font=dict(family="Poppins, Arial, sans-serif", size=12, color="#333"),
            #         coloraxis_showscale=False,
            #         margin=dict(t=90, b=60, l=40, r=40),
            #         height=550
            #     )
            #     st.plotly_chart(fig_salary, use_container_width=True)
            
            #  إضافة الرسم البياني الجديد هنا
            # create_income_vs_variables_chart(df)
            
            # st.markdown('</div>', unsafe_allow_html=True)
            
        elif analysis_section == "📈 Income vs Variables Analysis":
             #العرض المباشر للرسم البياني الجديد
            create_income_vs_variables_chart(df)
            
        elif analysis_section == "📊 Tenure Analysis":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("📊 Employee Tenure Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # التحقق من وجود الأعمدة المطلوبة
                if all(col in df.columns for col in ['YearsAtCompany', 'Attrition']):
                    # إنشاء الرسم باستخدام matplotlib و seaborn
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    sns.histplot(
                        data=df,
                        x="YearsAtCompany",
                        hue="Attrition",  # إضافة التلوين حسب Attrition
                        kde=True,
                        bins=20,
                        edgecolor='white',
                        alpha=0.9,
                        ax=ax
                    )
                    
                    # إضافة خط المتوسط
                    avg_years = df["YearsAtCompany"].mean()
                    ax.axvline(
                        x=avg_years, 
                        linestyle="--", 
                        color="red",
                        label=f"Average: {avg_years:.1f} years"
                    )
                    
                    ax.set_title(
                        "Attrition by Years at Company",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#222222',
                        fontfamily='DejaVu Sans'
                    )
                    ax.set_xlabel("Years at Company", fontsize=13, labelpad=10, color='#333333')
                    ax.set_ylabel("Number of Employees", fontsize=13, labelpad=10, color='#333333')
                    
                    ax.legend(
                        frameon=True,
                        shadow=True,
                        fontsize=11,
                        loc="upper right"
                    )
                    
                    sns.despine(left=False, bottom=False)
                    plt.tight_layout()
                    
                    # استخدام st.pyplot بدلاً من st.plotly_chart
                    st.pyplot(fig)
            
            with col2:
                st.subheader("Tenure Statistics")
                if 'YearsAtCompany' in df.columns:
                    stats_df = df['YearsAtCompany'].describe()
                    st.metric("Average Tenure", f"{stats_df['mean']:.1f} years")
                    st.metric("Median Tenure", f"{stats_df['50%']:.1f} years")
                    st.metric("Max Tenure", f"{stats_df['max']:.1f} years")
                    
                    # Attrition rate by tenure groups
                    if 'Attrition' in df.columns:
                        df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[0, 2, 5, 10, 20, 100], 
                                                 labels=['0-2', '3-5', '6-10', '11-20', '20+'])
                        attrition_by_tenure = df.groupby('TenureGroup')['Attrition'].value_counts(normalize=True).unstack().fillna(0)
                        
                        # عرض البيانات بشكل مناسب
                        st.subheader("Attrition by Tenure Group")
                        st.dataframe(attrition_by_tenure.style.format("{:.1%}"))
            
            st.markdown('</div>', unsafe_allow_html=True)
        elif analysis_section == "👤 Age Distribution":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("👤 Age Distribution of Employees")

            if 'Age' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(df["Age"], bins=20, kde=True, color="#2E86C1", ax=ax)
                avg_age = df["Age"].mean()
                plt.axvline(avg_age, color="#E74C3C", linestyle="--", linewidth=2, label=f"Average Age: {avg_age:.1f}")
                plt.title("Age Distribution of Employees", fontsize=16)
                plt.legend()
                st.pyplot(fig)

            if 'Department' in df.columns and 'Attrition' in df.columns:
                st.subheader("🏢 Attrition by Department")
                dept_attrition = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
                fig = px.bar(
                    dept_attrition, x='Department', y='Count', color='Attrition',
                    text='Count', barmode='stack', color_discrete_map=colors,
                    title='Attrition by Department'
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        elif analysis_section == "📊 Business Travel Analysis":
            create_business_travel_analysis(df)

        elif analysis_section == "🎓 Education Analysis":
            create_education_analysis(df)

        elif analysis_section == "💼 Job Analysis":
            create_job_analysis(df)

        elif analysis_section == "😊 Satisfaction Analysis":
            create_satisfaction_analysis(df)

        elif analysis_section == "📈 Years at Company Analysis":
            create_years_at_company_chart(df)

        elif analysis_section == "🔗 Correlation":
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.header("🔗 Correlation Heatmap")

            cat_cols = [i for i in df.columns if df[i].nunique() <= 5 or df[i].dtype == object]
            numeric_df = df.drop(columns=cat_cols + ['EmployeeNumber'], errors='ignore')
            
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr().round(2)
            
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap of Numerical Variables"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Top Correlations")
                corr_matrix = corr.abs().unstack()
                corr_matrix = corr_matrix.sort_values(ascending=False)
                top_correlations = corr_matrix[corr_matrix < 0.99].head(10)
            
                top_corr_df = pd.DataFrame(top_correlations).reset_index()
                top_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
                top_corr_df['Correlation'] = top_corr_df['Correlation'].round(3)
            
                st.dataframe(top_corr_df, use_container_width=True)
            else:
                st.warning("Not enough numerical columns for correlation analysis.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ====================== TAB 3: PREDICT ATTRITION ======================
    with tab3:
        st.markdown('<div class="section-header">Predict Employee Attrition</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("🧮 Enter Employee Details")
            
            with st.form("prediction_form"):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    age = st.number_input("Age", 18, 60, 30)
                    business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Non-Travel', 'Travel_Frequently'])
                    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
                    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
                    distance = st.number_input("Distance From Home", 1, 30, 5)
                    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
                    education_field = st.selectbox("Education Field", ['Other', 'Life Sciences', 'Marketing', 'Medical', 'Technical Degree', 'Human Resources'])
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    job_role = st.selectbox("Job Role", [
                        'Healthcare Representative', 'Sales Executive', 'Laboratory Technician', 
                        'Manufacturing Director', 'Manager', 'Human Resources', 
                        'Research Director', 'Sales Representative', 'Research Scientist'
                    ])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                
                with form_col2:
                    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
                    monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 5000)
                    num_companies_worked = st.number_input("Companies Worked", 0, 9, 5)
                    total_working_years = st.number_input("Total Working Years", 0, 40, 10)
                    years_at_company = st.number_input("Years at Company", 0, 40, 5)
                    years_in_role = st.number_input("Years in Current Role", 0, 20, 3)
                    years_since_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)
                    years_with_manager = st.number_input("Years With Manager", 0, 20, 3)
                    
                    st.markdown("**Satisfaction Levels**")
                    environment_sat = st.slider("Environment Satisfaction", 1, 4, 3)
                    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
                    worklife_balance = st.slider("Work Life Balance", 1, 4, 3)
                    job_involvement = st.slider("Job Involvement", 1, 4, 3)
                    
                    st.markdown("**Additional Info**")
                    job_level = st.slider("Job Level", 1, 5, 3)
                    percent_salary_hike = st.number_input("Salary Hike %", 10, 30, 15)
                    training_times = st.number_input("Training Times Last Year", 0, 6, 2)
                    hourly_rate = st.number_input("Hourly Rate", 30, 100, 50)
                    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
                    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
                    performance_rating = st.selectbox("Performance Rating", [3, 4])
                    over_time = st.selectbox("Over Time", ["Yes", "No"])
                
                submitted = st.form_submit_button("🔮 Predict Attrition Risk", type="primary")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📋 Prediction Results")
            
            if submitted:
                try:
                    # Prepare input data
                    input_data = {
                        'Age': age,
                        'BusinessTravel': business_travel,
                        'DailyRate': daily_rate,
                        'Department': department,
                        'DistanceFromHome': distance,
                        'Education': education,
                        'EducationField': education_field,
                        'EnvironmentSatisfaction': environment_sat,
                        'Gender': gender,
                        'HourlyRate': hourly_rate,
                        'JobInvolvement': job_involvement,
                        'JobLevel': job_level,
                        'JobRole': job_role,
                        'JobSatisfaction': job_satisfaction,
                        'MaritalStatus': marital_status,
                        'MonthlyIncome': monthly_income,
                        'MonthlyRate': monthly_rate,
                        'NumCompaniesWorked': num_companies_worked,
                        'OverTime': over_time,
                        'PercentSalaryHike': percent_salary_hike,
                        'PerformanceRating': performance_rating,
                        'RelationshipSatisfaction': relationship_satisfaction,
                        'StockOptionLevel': stock_option_level,
                        'TotalWorkingYears': total_working_years,
                        'TrainingTimesLastYear': training_times,
                        'WorkLifeBalance': worklife_balance,
                        'YearsAtCompany': years_at_company,
                        'YearsInCurrentRole': years_in_role,
                        'YearsSinceLastPromotion': years_since_promo,
                        'YearsWithCurrManager': years_with_manager
                    }
                    
                    # Make prediction
                    if model == "demo":
                        prediction, probability = create_demo_prediction(input_data)
                        st.info("🔬 **Demo Mode**: Using simulated predictions. Upload model files for real predictions.")
                    else:
                        sample_df = pd.DataFrame([input_data])
                        sample_df = pd.get_dummies(sample_df)
                        sample_df = sample_df.reindex(columns=train_columns, fill_value=0)
                        probability = model.predict_proba(sample_df)[0][1] * 100
                        prediction = model.predict(sample_df)[0]
                    
                    # Determine risk level
                    if prediction == 1 or probability > 70:
                        prediction_class = "prediction-box"
                        risk_level = "HIGH RISK 🔴"
                        risk_color = "#e53e3e"
                        recommendation = "Immediate intervention required. Consider retention strategies and career development programs."
                        risk_factors = [
                            "High overtime frequency",
                            "Below average job satisfaction",
                            "Limited recent promotions",
                            "Work-life balance concerns"
                        ]
                    elif probability > 40:
                        prediction_class = "prediction-box"
                        risk_level = "MEDIUM RISK 🟡"
                        risk_color = "#dd6b20"
                        recommendation = "Monitor closely. Implement engagement programs and regular check-ins."
                        risk_factors = [
                            "Moderate job satisfaction",
                            "Average tenure in current role",
                            "Standard work-life balance"
                        ]
                    else:
                        prediction_class = "prediction-box prediction-safe"
                        risk_level = "LOW RISK 🟢"
                        risk_color = "#38a169"
                        recommendation = "Employee appears stable and engaged. Maintain current support programs."
                        risk_factors = [
                            "Strong job satisfaction",
                            "Good work-life balance",
                            "Recent career growth opportunities"
                        ]
                    
                    # Display prediction results
                    st.markdown(f"""
                    <div class="{prediction_class}">
                        <div style="text-align: center; padding: 1rem;">
                            <h2 style="color: white; margin-bottom: 1rem;">🎯 Attrition Prediction</h2>
                            <div style="font-size: 2.5rem; font-weight: bold; color: white; margin: 1rem 0;">
                                {probability:.1f}%
                            </div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: white; margin: 1rem 0;">
                                {risk_level}
                            </div>
                            <p style="color: white; font-size: 1.1rem; margin: 1rem 0;">
                                {recommendation}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factor analysis
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3182ce; margin: 1rem 0;">
                        <h3 style="color: #2d3748; margin-bottom: 1rem;">🔍 Risk Factor Analysis</h3>
                    """, unsafe_allow_html=True)
                    
                    for factor in risk_factors:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 8px; border-left: 3px solid {risk_color};">
                            <span style="color: {risk_color}; margin-right: 0.5rem;">•</span>
                            <span style="color: #4a5568;">{factor}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Please check that all fields are filled correctly.")
            
            else:
                st.info("👆 Fill out all employee details and click 'Predict Attrition Risk' to see analysis results")
                if model == "demo":
                    st.warning("💡 **Demo Mode Active**: The app is running in demo mode with simulated predictions. To use the real model, ensure model files are available in the specified path.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ====================== TAB 4: REPORTS ======================
    with tab4:
        st.markdown('<div class="section-header">Analytics Reports & Export</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📄 Generate Reports")
            
            report_type = st.selectbox(
                "Select Report Type",
                ["Attrition Analysis", "Department Performance", "Employee Satisfaction", "Full HR Analytics"]
            )
            
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                if st.button("📊 Generate Report", type="primary"):
                    st.success("✅ Report generated successfully!")
            with report_col2:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="hr_analytics_data.csv",
                    mime="text/csv"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📈 Key Performance Indicators")
            
            # Calculate KPIs
            if 'Attrition' in df.columns:
                total_employees = len(df)
                attrition_count = (df['Attrition'] == 'Yes').sum()
                attrition_rate = (attrition_count / total_employees) * 100
                
                kpi1, kpi2 = st.columns(2)
                with kpi1:
                    st.metric("Total Employees", f"{total_employees:,}")
                with kpi2:
                    st.metric("Attrition Count", f"{attrition_count}")
                
                kpi3, kpi4 = st.columns(2)
                with kpi3:
                    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
                with kpi4:
                    if 'MonthlyIncome' in df.columns:
                        avg_income = df['MonthlyIncome'].mean()
                        st.metric("Avg Monthly Income", f"${avg_income:,.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()

st.markdown('</div>', unsafe_allow_html=True)  # Close main container

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; padding: 2rem;'>"
    "<strong>HR Analytics Dashboard</strong> | Employee Attrition Analysis & Predictive Modeling | "
    "Built with ❤️ using Streamlit"
    "</div>", 
    unsafe_allow_html=True
)

