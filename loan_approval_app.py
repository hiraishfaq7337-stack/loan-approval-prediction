# ============================================================
# Loan Approval Prediction
# Streamlit App — Complete Code
# ============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {font-size:2rem; font-weight:700; color:#1a1a2e;}
    .metric-card {background:#f8f9fa; border-radius:12px; padding:16px; text-align:center;}
    .section-title {font-size:1.3rem; font-weight:600; border-left:4px solid #2E75B6; padding-left:10px; margin:20px 0 12px;}
    .approved {background:#d4edda; border-radius:16px; padding:28px; text-align:center;}
    .rejected {background:#f8d7da; border-radius:16px; padding:28px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    st.title("Loan Analytics")
    st.caption("Loan Approval Prediction System")
    st.divider()

    page = st.radio("Navigation", [
        "📊 Overview",
        "🔍 Data Cleaning",
        "📈 EDA — Exploratory Analysis",
        "🤖 Model Training & Evaluation",
        "🎯 User Prediction System"
    ])

    st.divider()
    uploaded = st.file_uploader("Upload loan_approval_dataset.csv", type=["csv"])

# ── 2. Load Dataset ──────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

try:
    df_raw = load_data(uploaded)
except:
    st.error("Dataset file nahi mili! Sidebar se CSV upload karein.")
    st.stop()

# ── 3. Data Cleaning ─────────────────────────────────────────
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    if 'loan_id' in df.columns:
        df.drop(columns=['loan_id'], inplace=True)
    return df

df = clean_data(df_raw)

# ── 6. Encoding ──────────────────────────────────────────────
le_edu  = LabelEncoder()
le_emp  = LabelEncoder()
le_stat = LabelEncoder()

df["education_enc"]    = le_edu.fit_transform(df["education"])
df["self_employed_enc"]= le_emp.fit_transform(df["self_employed"])
df["loan_status_enc"]  = le_stat.fit_transform(df["loan_status"])

# ── 5 & 7. Feature Selection + Train-Test Split ──────────────
FEATURES = [
    "cibil_score",
    "income_annum",
    "loan_amount",
    "loan_term",
    "no_of_dependents",
    "education_enc",
    "self_employed_enc"
]
TARGET = "loan_status_enc"

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 8. Train Model ───────────────────────────────────────────
@st.cache_resource
def train_model(X_tr, y_tr):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    return model

model = train_model(X_train, y_train)

# ── Save Model to .pkl ───────────────────────────────────────
MODEL_PATH = "loan_approval_model.pkl"
if not os.path.exists(MODEL_PATH):
    joblib.dump(model, MODEL_PATH)

# ── 9. Evaluate Model ────────────────────────────────────────
y_pred  = model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
cm      = confusion_matrix(y_test, y_pred)
feat_imp= pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

FEAT_LABELS = {
    "cibil_score":       "CIBIL Score",
    "income_annum":      "Annual Income",
    "loan_amount":       "Loan Amount",
    "loan_term":         "Loan Term",
    "no_of_dependents":  "No. of Dependents",
    "education_enc":     "Education",
    "self_employed_enc": "Self Employed"
}

# ════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="main-title">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)
    st.caption("Kaggle — Loan Approval Dataset | Random Forest Classifier")
    st.divider()

    approved_count  = (df["loan_status"] == "Approved").sum()
    rejected_count  = (df["loan_status"] == "Rejected").sum()
    approval_rate   = round(approved_count / len(df) * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Applications", len(df))
    c2.metric("Approved", approved_count)
    c3.metric("Rejected", rejected_count)
    c4.metric("Approval Rate", f"{approval_rate}%")
    c5.metric("Features Used", len(FEATURES))

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-title">Pipeline Steps</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Import Libraries",           "Done"),
        ("2", "Load Dataset",               "Done"),
        ("3", "Data Cleaning",              "Done"),
        ("4", "Exploratory Data Analysis",  "EDA tab"),
        ("5", "Feature Selection (7 features)", "Auto"),
        ("6", "Encoding",                   "Auto"),
        ("7", "Train-Test Split (80/20)",   "Done"),
        ("8", "Train Model (Random Forest)","Done"),
        ("9", "Evaluate Model",             "Accuracy, Precision, Recall, F1"),
        ("10","Actual vs Predicted Graph",  "Model tab"),
        ("11","Feature Importance Graph",   "Model tab"),
        ("12","User Prediction System",     "Predict tab"),
    ]
    step_df = pd.DataFrame(steps, columns=["Step", "Description", "Status"])
    st.dataframe(step_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ════════════════════════════════════════════════════════════
elif page == "🔍 Data Cleaning":
    st.markdown('<div class="main-title">🔍 Data Cleaning Report</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
        mv = df_raw.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        mv["Missing %"] = (mv["Missing"] / len(df_raw) * 100).round(2)
        st.dataframe(mv, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Data Types</div>', unsafe_allow_html=True)
        dt = df_raw.dtypes.reset_index()
        dt.columns = ["Column", "DType"]
        st.dataframe(dt, use_container_width=True)

    st.markdown('<div class="section-title">Duplicate Rows</div>', unsafe_allow_html=True)
    st.info(f"Duplicate rows found: **{df_raw.duplicated().sum()}** — removed ✅")

    st.markdown('<div class="section-title">Encoding: Categorical → Numeric</div>', unsafe_allow_html=True)
    enc_df = df[["education","education_enc","self_employed","self_employed_enc","loan_status","loan_status_enc"]].drop_duplicates().sort_values("loan_status")
    st.dataframe(enc_df, use_container_width=True)

    st.markdown('<div class="section-title">After Cleaning — Shape</div>', unsafe_allow_html=True)
    st.success(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

# ════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════
elif page == "📈 EDA — Exploratory Analysis":
    st.markdown('<div class="main-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.divider()

    # ── Loan Status Distribution
    st.markdown('<div class="section-title">Loan Status Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        vc = df["loan_status"].value_counts().reset_index()
        vc.columns = ["Status", "Count"]
        fig = px.pie(vc, names="Status", values="Count",
                     title="Approved vs Rejected",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(vc, x="Status", y="Count",
                     title="Loan Status Count",
                     color="Status",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Histograms
    st.markdown('<div class="section-title">Histograms</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, container in zip(
        ["cibil_score", "income_annum", "loan_amount"],
        [col1, col2, col3]
    ):
        fig = px.histogram(df, x=col, nbins=30, title=col,
                           color_discrete_sequence=["#2E75B6"])
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=40, b=20, l=20, r=20))
        container.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    for col, container in zip(["loan_term", "no_of_dependents"], [col1, col2]):
        fig = px.histogram(df, x=col, nbins=20, title=col,
                           color_discrete_sequence=["#1D9E75"])
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=40, b=20, l=20, r=20))
        container.plotly_chart(fig, use_container_width=True)

    # ── Scatter Plots
    st.markdown('<div class="section-title">Scatter Plots</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="cibil_score", y="loan_amount",
                         color="loan_status",
                         title="CIBIL Score vs Loan Amount",
                         color_discrete_sequence=["#2ECC71","#E74C3C"], height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x="income_annum", y="loan_amount",
                         color="loan_status",
                         title="Income vs Loan Amount",
                         color_discrete_sequence=["#2ECC71","#E74C3C"], height=320)
        st.plotly_chart(fig, use_container_width=True)

    # ── Boxplots
    st.markdown('<div class="section-title">Boxplots</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="loan_status", y="cibil_score",
                     color="loan_status",
                     title="CIBIL Score by Loan Status",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="loan_status", y="income_annum",
                     color="loan_status",
                     title="Annual Income by Loan Status",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["cibil_score","income_annum","loan_amount",
                "loan_term","no_of_dependents","loan_status_enc"]
    corr = df[num_cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax,
                linewidths=0.5, fmt=".2f")
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

    # ── Education & Self Employed
    st.markdown('<div class="section-title">Categorical Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        edu_df = df.groupby(["education","loan_status"]).size().reset_index(name="count")
        fig = px.bar(edu_df, x="education", y="count", color="loan_status",
                     barmode="group", title="Education vs Loan Status",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        emp_df = df.groupby(["self_employed","loan_status"]).size().reset_index(name="count")
        fig = px.bar(emp_df, x="self_employed", y="count", color="loan_status",
                     barmode="group", title="Self Employed vs Loan Status",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=300)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Evaluation":
    st.markdown('<div class="main-title">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.caption("Algorithm: Random Forest Classifier | Split: 80% Train / 20% Test")
    st.divider()

    # ── 9. Metrics
    st.markdown('<div class="section-title">Model Performance Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall",    f"{rec*100:.2f}%")
    c4.metric("F1 Score",  f"{f1*100:.2f}%")

    # ── Confusion Matrix
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected","Approved"],
                    yticklabels=["Rejected","Approved"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.markdown("**Classification Report**")
        report = classification_report(y_test, y_pred,
                                       target_names=["Rejected","Approved"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

    # ── 10. Actual vs Predicted
    st.markdown('<div class="section-title">Actual vs Predicted Graph</div>', unsafe_allow_html=True)
    avp_df = pd.DataFrame({
        "Index":     range(len(y_test)),
        "Actual":    le_stat.inverse_transform(y_test),
        "Predicted": le_stat.inverse_transform(y_pred)
    }).head(100)
    fig = px.scatter(avp_df, x="Index", y="Actual",
                     title="Actual vs Predicted (first 100 test samples)",
                     color="Predicted",
                     color_discrete_sequence=["#2ECC71","#E74C3C"], height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── 11. Feature Importance
    st.markdown('<div class="section-title">Feature Importance Graph</div>', unsafe_allow_html=True)
    fi_df = feat_imp.reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fi_df["Feature"] = fi_df["Feature"].map(FEAT_LABELS)
    fig = px.bar(fi_df, x="Feature", y="Importance",
                 title="Feature Importance — Random Forest",
                 color="Importance",
                 color_continuous_scale="Blues", height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model Details"):
        st.write(f"**Algorithm:** Random Forest Classifier")
        st.write(f"**Estimators:** 100 | **Random State:** 42")
        st.write(f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}")
        st.write(f"**Features used:** {', '.join([FEAT_LABELS[f] for f in FEATURES])}")
        st.write(f"**Model saved as:** loan_approval_model.pkl ✅")

# ════════════════════════════════════════════════════════════
# PAGE: USER PREDICTION SYSTEM  (Step 12)
# ════════════════════════════════════════════════════════════
elif page == "🎯 User Prediction System":
    st.markdown('<div class="main-title">🎯 User Prediction System</div>', unsafe_allow_html=True)
    st.caption("Applicant ki details bhar kar loan approval predict karo")
    st.divider()

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Applicant Information</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            cibil        = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)
            income       = st.number_input("Annual Income (Rs)", min_value=100000, max_value=50000000, value=5000000, step=100000)
        with col2:
            loan_amt     = st.number_input("Loan Amount (Rs)", min_value=100000, max_value=50000000, value=10000000, step=100000)
            loan_term    = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)
        with col3:
            dependents   = st.number_input("No. of Dependents", min_value=0, max_value=10, value=2)
            education    = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp     = st.selectbox("Self Employed", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Loan Status", use_container_width=True)

    if submitted:
        edu_enc = 1 if education == "Graduate" else 0
        emp_enc = 1 if self_emp == "Yes" else 0

        inp   = np.array([[cibil, income, loan_amt, loan_term, dependents, edu_enc, emp_enc]])
        pred  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        label = le_stat.inverse_transform([pred])[0]
        conf  = round(max(proba) * 100, 1)

        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if label == "Approved":
                st.markdown(f"""
                <div class="approved">
                    <div style="font-size:3rem;">✅</div>
                    <div style="font-size:2rem;font-weight:700;color:#155724;">APPROVED</div>
                    <div style="font-size:1rem;color:#555;margin-top:8px;">Loan likely to be approved</div>
                    <div style="font-size:1.1rem;margin-top:10px;">Confidence: <b>{conf}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected">
                    <div style="font-size:3rem;">❌</div>
                    <div style="font-size:2rem;font-weight:700;color:#721c24;">REJECTED</div>
                    <div style="font-size:1rem;color:#555;margin-top:8px;">Loan likely to be rejected</div>
                    <div style="font-size:1.1rem;margin-top:10px;">Confidence: <b>{conf}%</b></div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Input Summary**")
        st.table(pd.DataFrame({
            "Feature": ["CIBIL Score","Annual Income","Loan Amount","Loan Term","Dependents","Education","Self Employed"],
            "Value":   [cibil, f"Rs {income:,}", f"Rs {loan_amt:,}", f"{loan_term} years", dependents, education, self_emp]
        }))
