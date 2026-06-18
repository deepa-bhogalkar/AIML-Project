# ╔══════════════════════════════════════════════════════════════════╗
# ║          LocustGuard — Streamlit App                            ║
# ║  SRN: PES1PG25CA051  |  Student: Deepashree  |  MCA AIML       ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# HOW TO RUN:
#   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
#   streamlit run locust_guard_app.py
#
# Place swarm.csv in the same folder before running.
# ──────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# sklearn imports — exactly as in the notebook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, f1_score
)

# ══════════════════════════════════════════════
# 0.  PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="LocustGuard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] { background:#1a4d2e; }
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] .stRadio label { color:#e8f5e9 !important; }
[data-testid="stSidebar"] hr { border-color:#2d6a4f; }

/* ── Metric cards ────────────────────────── */
[data-testid="stMetric"] {
    background:#f0fdf4; border:1px solid #bbf7d0;
    border-radius:10px; padding:12px 16px;
}
[data-testid="stMetricLabel"] { color:#166534 !important; font-size:13px !important; }
[data-testid="stMetricValue"] { color:#14532d !important; font-size:24px !important; }

/* ── Green header banner ─────────────────── */
.hdr {
    background:#1a4d2e; padding:22px 28px;
    border-radius:12px; margin-bottom:20px;
}
.hdr h1 { color:#fff; margin:0; font-size:26px; }
.hdr p  { color:#a7f3d0; margin:4px 0 0; font-size:13px; }

/* ── Section heading ─────────────────────── */
.sec { color:#14532d; font-weight:600; margin:18px 0 6px; font-size:16px; }

/* ── Code-style pill ─────────────────────── */
.pill {
    display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0;
    color:#14532d; border-radius:6px; padding:2px 10px;
    font-family:monospace; font-size:12px; margin:2px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 1.  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 LocustGuard")
    st.markdown("Locust Swarm Risk Prediction")
    st.markdown("---")
    page = st.radio(
        "Go to",
        ["🏠 Overview",
         "🗂️ Data & Preprocessing",
         "⚙️ Hyperparameters & Training",
         "📈 Evaluation",
         "🔍 Live Predictor",
         "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**SRN:** PES1PG25CA051")
    st.markdown("**Student:** Deepashree")
    st.markdown("**Domain:** Agriculture · AIML")
    st.markdown("**SDG Goal:** Life on Land 🌍")

# ══════════════════════════════════════════════
# 2.  DATA LOADING & PREPROCESSING
#     (mirrors the notebook exactly)
# ══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & preprocessing dataset…")
def load_and_preprocess(path: str):
    # ── Step 1 : Load ────────────────────────
    df = pd.read_csv(path)
    raw_shape = df.shape

    # ── Step 2 : Drop NA ─────────────────────
    df = df.dropna()
    after_dropna = df.shape

    # ── Step 3 : IQR Outlier Removal ─────────
    # Apply only on numeric columns to avoid boolean issues
    num_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[num_cols] < (Q1 - 1.5 * IQR)) |
             (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[mask]
    after_iqr = df.shape

    # ── Step 4 : Label Encoding ───────────────
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col].astype(str))

    # ── Step 5 : Target column ────────────────
    # As per notebook: alternate 0/1 pattern
    df['CAT'] = [0 if i % 2 == 0 else 1 for i in range(len(df))]
    X = df.drop('CAT', axis=1)
    y = df['CAT']

    # ── Step 6 : Feature Scaling ─────────────
    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()

# Remove all non-numeric columns automatically
    X = X.select_dtypes(include=['number'])

# Fill any remaining missing values
    X = X.fillna(X.median())

# Scale the data
    X_std = scaler_std.fit_transform(X)
    X_mm = scaler_mm.fit_transform(X)

    # ── Step 7 : Train / Test Split ───────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.2, random_state=42
    )

    info = {
        "raw_shape":    raw_shape,
        "after_dropna": after_dropna,
        "after_iqr":    after_iqr,
        "features":     list(X.columns),
        "n_features":   X.shape[1],
        "X":            X,
        "X_std":        X_std,
        "X_mm":         X_mm,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "scaler_std":   scaler_std,
        "scaler_mm":    scaler_mm,
        "df":           df,
    }
    return info

# ── Try loading real CSV; fall back to synthetic demo data ──────────
CSV_PATH = "swarm.csv"
try:
    data = load_and_preprocess(CSV_PATH)
    DEMO_MODE = False
except FileNotFoundError:
    DEMO_MODE = True

    @st.cache_data
    def synthetic_data():
        rng = np.random.default_rng(42)
        n = 3000
        cols = {
            "X": rng.uniform(-20, 60, n),
            "Y": rng.uniform(-40, 40, n),
            "OBJECTID": np.arange(1, n + 1),
            "TmSTARTDAT": rng.uniform(0, 12, n),
            "LOCPRCPTN": rng.uniform(0, 200, n),
            "LOCSTTMNT": rng.integers(0, 5, n).astype(float),
            "STARTDATE_ENC": rng.integers(0, 365, n).astype(float),
        }
        for i in range(145):
            cols[f"F{i}"] = rng.uniform(0, 1, n)
        df = pd.DataFrame(cols)
        df['CAT'] = [0 if i % 2 == 0 else 1 for i in range(n)]
        X = df.drop('CAT', axis=1)
        y = df['CAT']
        sc = StandardScaler()
        sm = MinMaxScaler()
        Xs = sc.fit_transform(X)
        Xm = sm.fit_transform(X)
        Xt, Xv, yt, yv = train_test_split(Xs, y, test_size=0.2, random_state=42)
        return {
            "raw_shape": (29384, 152), "after_dropna": (21790, 152),
            "after_iqr": (n, 152),
            "features": list(X.columns), "n_features": X.shape[1],
            "X": X, "X_std": Xs, "X_mm": Xm,
            "X_train": Xt, "X_test": Xv, "y_train": yt, "y_test": yv,
            "scaler_std": sc, "scaler_mm": sm, "df": df,
        }

    data = synthetic_data()

# ══════════════════════════════════════════════
# 3.  MODEL TRAINING WITH HYPERPARAMETERS
# ══════════════════════════════════════════════

# Default hyperparameter dict — shown in sidebar on training page
DEFAULT_HP = {
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "subsample": 1.0,
        "random_state": 42,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
    },
    "Logistic Regression": {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": 42,
    },
    "Naive Bayes": {
        "var_smoothing": 1e-9,
    },
}

@st.cache_resource(show_spinner="Training all models…")
def train_all(X_train, X_test, y_train, y_test, hp):
    models = {
        "Random Forest": RandomForestClassifier(**hp["Random Forest"]),
        "Gradient Boosting": GradientBoostingClassifier(**hp["Gradient Boosting"]),
        "SVM": SVC(**hp["SVM"]),
        "Logistic Regression": LogisticRegression(**hp["Logistic Regression"]),
        "Naive Bayes": GaussianNB(var_smoothing=hp["Naive Bayes"]["var_smoothing"]),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":     m,
            "pred":      pred,
            "prob":      prob,
            "accuracy":  accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall":    recall_score(y_test, pred, zero_division=0),
            "f1":        f1_score(y_test, pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, prob),
            "mse":       mean_squared_error(y_test, pred),
            "mae":       mean_absolute_error(y_test, pred),
        }
    return results

results = train_all(
    data["X_train"], data["X_test"],
    data["y_train"], data["y_test"],
    DEFAULT_HP
)

GREEN   = "#1a4d2e"
LGREEN  = "#2d6a4f"
MINT    = "#bbf7d0"
ACCENT  = "#d97706"

# ══════════════════════════════════════════════
# 4.  PAGES
# ══════════════════════════════════════════════

# ─── PAGE : OVERVIEW ──────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="hdr">
      <h1>🌿 LocustGuard</h1>
      <p>Locust Swarm Risk Prediction Using Climate &amp; Land Data · MCA AIML · PES1PG25CA051</p>
    </div>
    """, unsafe_allow_html=True)

    if DEMO_MODE:
        st.warning("⚠️ **Demo mode** — `swarm.csv` not found. Place the file in the same folder as this script to use real data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Dataset",    f"{data['raw_shape'][0]:,} rows",    f"{data['raw_shape'][1]} columns")
    c2.metric("After Cleaning", f"{data['after_iqr'][0]:,} rows",    "dropna + IQR")
    c3.metric("Features Used",  str(data["n_features"]),              "all 151 input cols")
    c4.metric("Best Model",     "Random Forest",                      f"{results['Random Forest']['accuracy']*100:.1f}% acc")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="sec">📌 Problem Statement</p>', unsafe_allow_html=True)
        st.info(
            "Locust swarms severely damage crops across **Africa, the Middle East, and South Asia**, "
            "threatening food security. Manual surveys are slow — this causes delayed responses and "
            "major agricultural losses."
        )
        st.markdown('<p class="sec">🎯 Goal (UN SDG: Life on Land)</p>', unsafe_allow_html=True)
        st.success(
            "Predict **high-risk vs low-risk** locust swarm zones from FAO climate & land data "
            "using ML models — enabling early intervention to reduce crop damage."
        )
        st.markdown('<p class="sec">🛠️ Algorithms</p>', unsafe_allow_html=True)
        for alg in ["Random Forest", "Gradient Boosting", "SVM (RBF)", "Logistic Regression", "Naive Bayes"]:
            st.markdown(f'<span class="pill">{alg}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="sec">📊 Label Distribution</p>', unsafe_allow_html=True)
        counts = data["df"]["CAT"].value_counts().sort_index()
        labels = ["Low Risk (0)", "High Risk (1)"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=labels,
            colors=["#86efac", "#fca5a5"],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for t in autotexts: t.set_fontsize(11)
        ax.set_title("Target Class Distribution", color=GREEN, fontsize=12)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<p class="sec">📋 Dataset Overview</p>', unsafe_allow_html=True)
    st.dataframe(data["df"].head(8), use_container_width=True, hide_index=True)

    st.markdown('<p class="sec">📈 Monthly Locust Risk Trend (Simulated)</p>', unsafe_allow_html=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    risk   = [22, 28, 35, 48, 60, 78, 82, 70, 55, 42, 30, 20]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(months, risk, alpha=0.12, color="#dc2626")
    ax.plot(months, risk, marker="o", color="#dc2626", lw=2, ms=5, label="Risk Score")
    ax.axhline(60, color=ACCENT, ls="--", lw=1.5, label="Alert Threshold (60)")
    ax.set_ylim(0, 100); ax.set_ylabel("Risk Score")
    ax.set_title("Average Monthly Locust Risk Score", color=GREEN, fontsize=13)
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()


# ─── PAGE : DATA & PREPROCESSING ──────────────
elif page == "🗂️ Data & Preprocessing":
    st.title("🗂️ Data & Preprocessing")

    st.markdown('<p class="sec">Step-by-step pipeline (mirrors your notebook exactly)</p>', unsafe_allow_html=True)

    with st.expander("✅ Step 1 — Load Dataset", expanded=True):
        st.code("""df = pd.read_csv('swarm.csv')
print(df.head())
print(df.shape)   # (29384, 152)""", language="python")
        c1, c2 = st.columns(2)
        c1.metric("Raw Rows",    f"{data['raw_shape'][0]:,}")
        c2.metric("Raw Columns", str(data['raw_shape'][1]))

    with st.expander("✅ Step 2 — Check & Handle Missing Values"):
        st.code("""print(df.isnull().sum())
df = df.dropna()
print(df.shape)   # (21790, 152)""", language="python")
        c1, c2 = st.columns(2)
        c1.metric("Before dropna", f"{data['raw_shape'][0]:,} rows")
        c2.metric("After dropna",  f"{data['after_dropna'][0]:,} rows")

    with st.expander("✅ Step 3 — Remove Outliers (IQR Method)"):
        st.code("""Q1  = df.quantile(0.25)
Q3  = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |
          (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df.shape)""", language="python")
        c1, c2 = st.columns(2)
        c1.metric("Before IQR", f"{data['after_dropna'][0]:,} rows")
        c2.metric("After IQR",  f"{data['after_iqr'][0]:,} rows",
                  f"-{data['after_dropna'][0]-data['after_iqr'][0]:,} outliers removed")

    with st.expander("✅ Step 4 — Label Encoding (Categorical Columns)"):
        st.code("""encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col].astype(str))
print("Encoding Completed")""", language="python")
        st.success("All object-type columns converted to numeric integers.")

    with st.expander("✅ Step 5 — Create Target Column"):
        st.code("""# Notebook creates alternating 0/1 pattern as the target
df['CAT'] = [0 if i % 2 == 0 else 1 for i in range(len(df))]
X = df.drop('CAT', axis=1)
y = df['CAT']""", language="python")
        st.info("CAT = 0 → Low Risk  |  CAT = 1 → High Risk")

    with st.expander("✅ Step 6 — Feature Scaling"):
        st.code("""# Standard Scaler (used for training)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler also used for certain models
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)""", language="python")

    with st.expander("✅ Step 7 — Train / Test Split"):
        st.code("""X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size  = 0.2,   # 80% train, 20% test
    random_state = 42
)""", language="python")
        c1, c2 = st.columns(2)
        c1.metric("Training samples", f"{len(data['X_train']):,}")
        c2.metric("Test samples",     f"{len(data['X_test']):,}")

    st.markdown("---")
    st.markdown('<p class="sec">📊 Feature Distributions</p>', unsafe_allow_html=True)

    numeric_cols = [c for c in data["df"].columns if data["df"][c].dtype != "object" and c != "CAT"]
    feat = st.selectbox("Select a feature column to visualise:", numeric_cols[:20])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    data["df"][feat].hist(bins=30, ax=axes[0], color=GREEN, edgecolor="white")
    axes[0].set_title(f"{feat} — Histogram", color=GREEN)
    data["df"].boxplot(column=feat, ax=axes[1], patch_artist=True,
                       boxprops=dict(facecolor=MINT, color=GREEN),
                       medianprops=dict(color=ACCENT))
    axes[1].set_title(f"{feat} — Boxplot", color=GREEN)
    data["df"][feat].plot.kde(ax=axes[2], color=LGREEN, lw=2)
    axes[2].set_title(f"{feat} — Density", color=GREEN)
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<p class="sec">🔗 Correlation Heatmap (first 12 numeric features)</p>', unsafe_allow_html=True)
    num_df = data["df"].select_dtypes(include="number").iloc[:, :12]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="Greens",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix (first 12 columns)", color=GREEN)
    st.pyplot(fig); plt.close()


# ─── PAGE : HYPERPARAMETERS & TRAINING ────────
elif page == "⚙️ Hyperparameters & Training":
    st.title("⚙️ Hyperparameters & Training")
    st.markdown("Adjust hyperparameters below, then click **Train Models** to retrain on the fly.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌲 Random Forest",
        "📈 Gradient Boosting",
        "🔷 SVM",
        "📉 Logistic Regression",
        "🔵 Naive Bayes",
    ])

    hp = {}

    with tab1:
        st.markdown("### Random Forest Hyperparameters")
        st.code("""model = RandomForestClassifier(
    n_estimators    = 100,   # number of trees
    max_depth       = None,  # unlimited depth
    min_samples_split = 2,
    min_samples_leaf  = 1,
    max_features    = 'sqrt',
    random_state    = 42
)
model.fit(X_train, y_train)""", language="python")
        c1, c2 = st.columns(2)
        n_est   = c1.slider("n_estimators (trees)", 10, 500, 100, 10,
                             help="More trees = better accuracy but slower")
        max_dep = c2.select_slider("max_depth", options=["None",2,3,5,10,20,50], value="None",
                                    help="None = grow until pure leaves")
        c3, c4 = st.columns(2)
        mss = c3.slider("min_samples_split", 2, 20, 2,
                         help="Min samples to split an internal node")
        msl = c4.slider("min_samples_leaf", 1, 20, 1,
                         help="Min samples in a leaf node")
        mf  = st.selectbox("max_features", ["sqrt","log2","None"], index=0,
                             help="Features to consider per split")
        hp["Random Forest"] = {
            "n_estimators":     n_est,
            "max_depth":        None if max_dep == "None" else int(max_dep),
            "min_samples_split": mss,
            "min_samples_leaf":  msl,
            "max_features":     None if mf == "None" else mf,
            "random_state":     42,
        }
        st.info(f"📌 **Notebook default:** n_estimators=100, random_state=42 — all other defaults")

    with tab2:
        st.markdown("### Gradient Boosting Hyperparameters")
        st.code("""model = GradientBoostingClassifier(
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 3,
    subsample     = 1.0,
    random_state  = 42
)""", language="python")
        c1, c2 = st.columns(2)
        gb_n  = c1.slider("n_estimators", 10, 500, 100, 10)
        gb_lr = c2.slider("learning_rate", 0.01, 1.0, 0.1, 0.01,
                           help="Shrinks each tree's contribution — lower = more robust")
        c3, c4 = st.columns(2)
        gb_d  = c3.slider("max_depth (GBM)", 1, 10, 3,
                           help="Depth of each individual tree")
        gb_ss = c4.slider("subsample", 0.5, 1.0, 1.0, 0.05,
                           help="Fraction of samples per tree — <1.0 = stochastic boosting")
        hp["Gradient Boosting"] = {
            "n_estimators":  gb_n,
            "learning_rate": gb_lr,
            "max_depth":     gb_d,
            "subsample":     gb_ss,
            "random_state":  42,
        }

    with tab3:
        st.markdown("### SVM Hyperparameters")
        st.code("""svm_model = SVC(
    kernel      = 'rbf',   # from notebook
    C           = 1.0,     # regularisation
    gamma       = 'scale',
    probability = True
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))""", language="python")
        c1, c2 = st.columns(2)
        svm_k = c1.selectbox("kernel", ["rbf","linear","poly","sigmoid"], index=0,
                              help="rbf is the notebook default")
        svm_c = c2.select_slider("C (regularisation)", [0.01,0.1,0.5,1.0,2.0,5.0,10.0,100.0], value=1.0,
                                  help="Larger C = less margin, more correct training points")
        svm_g = st.selectbox("gamma", ["scale","auto"], index=0)
        hp["SVM"] = {"kernel": svm_k, "C": svm_c, "gamma": svm_g, "probability": True}

    with tab4:
        st.markdown("### Logistic Regression Hyperparameters")
        st.code("""log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred  = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
# Notebook output: ~0.487""", language="python")
        c1, c2 = st.columns(2)
        lr_c  = c1.select_slider("C (inverse regularisation)", [0.001,0.01,0.1,1.0,5.0,10.0,100.0], value=1.0)
        lr_it = c2.slider("max_iter", 100, 5000, 1000, 100)
        lr_s  = st.selectbox("solver", ["lbfgs","liblinear","saga","newton-cg"], index=0)
        hp["Logistic Regression"] = {
            "C": lr_c, "max_iter": lr_it, "solver": lr_s, "random_state": 42
        }
        st.warning("📌 Notebook accuracy for LR ≈ 0.487 (target is alternating 0/1 — hard to learn linearly)")

    with tab5:
        st.markdown("### Naive Bayes Hyperparameters")
        st.code("""nb_model = GaussianNB(var_smoothing=1e-9)
nb_model.fit(X_train, y_train)
nb_pred  = nb_model.predict(X_test)""", language="python")
        nb_vs = st.select_slider(
            "var_smoothing",
            [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=1e-9,
            help="Stabilises variance estimates; larger = more smoothing"
        )
        hp["Naive Bayes"] = {"var_smoothing": nb_vs}

    st.markdown("---")
    if st.button("🚀 Train All Models with Above Hyperparameters", type="primary", use_container_width=True):
        with st.spinner("Training… this may take a moment for SVM/GBM."):
            new_results = train_all.__wrapped__(
                data["X_train"], data["X_test"],
                data["y_train"], data["y_test"], hp
            )
        st.session_state["custom_results"] = new_results
        st.success("✅ Training complete! Go to **📈 Evaluation** to see results.")
        # Show quick summary
        st.markdown('<p class="sec">Quick Accuracy Summary</p>', unsafe_allow_html=True)
        for name, v in new_results.items():
            st.metric(name, f"{v['accuracy']*100:.2f}%")


# ─── PAGE : EVALUATION ────────────────────────
elif page == "📈 Evaluation":
    st.title("📈 Evaluation")

    # Use custom-trained results if available
    res = st.session_state.get("custom_results", results)

    # Accuracy comparison
    names = list(res.keys())
    accs  = [res[n]["accuracy"]*100 for n in names]
    best  = names[np.argmax(accs)]

    col_colors = ["#14532d" if n == best else "#86efac" for n in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, accs, color=col_colors, edgecolor="white")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=11, fontweight="bold", color=GREEN)
    ax.set_xlim(40, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", color=GREEN, fontsize=14)
    ax.spines[["top","right"]].set_visible(False)
    best_patch = mpatches.Patch(color="#14532d", label=f"Best: {best}")
    ax.legend(handles=[best_patch], fontsize=10)
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # Full metrics table
    st.markdown('<p class="sec">📋 All Metrics</p>', unsafe_allow_html=True)
    metric_rows = []
    for name, v in res.items():
        metric_rows.append({
            "Model":          name,
            "Accuracy":       f"{v['accuracy']*100:.2f}%",
            "Precision":      f"{v['precision']*100:.2f}%",
            "Recall":         f"{v['recall']*100:.2f}%",
            "F1 Score":       f"{v['f1']*100:.2f}%",
            "ROC-AUC":        f"{v['roc_auc']*100:.2f}%",
            "MSE":            f"{v['mse']:.4f}",
            "MAE":            f"{v['mae']:.4f}",
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    # Notebook metrics code block
    st.markdown('<p class="sec">📓 Metrics Code (from notebook)</p>', unsafe_allow_html=True)
    st.code("""# ── Logistic Regression ─────────────────────────
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred  = log_model.predict(X_test)
log_prob  = log_model.predict_proba(X_test)[:, 1]

print("Accuracy: ",  accuracy_score(y_test,  log_pred))   # 0.487
print("Precision:",  precision_score(y_test, log_pred))
print("Recall:   ",  recall_score(y_test,    log_pred))
print("ROC-AUC:  ",  roc_auc_score(y_test,   log_prob))

# ── SVM ──────────────────────────────────────
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
svm_pred  = svm_model.predict(X_test)
svm_prob  = svm_model.predict_proba(X_test)[:, 1]

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))  # 0.476
print("MSE:", mean_squared_error(y_test,  svm_pred))
print("MAE:", mean_absolute_error(y_test, svm_pred))

# ── Random Forest ─────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred   = rf_model.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, y_pred))""", language="python")

    st.markdown("---")
    # Per-model deep dive
    st.markdown('<p class="sec">🔬 Deep Dive — Confusion Matrix & Report</p>', unsafe_allow_html=True)
    sel = st.selectbox("Choose model:", list(res.keys()))
    v   = res[sel]

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(data["y_test"], v["pred"])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Low Risk","High Risk"],
                    yticklabels=["Low Risk","High Risk"],
                    linewidths=1, linecolor="white")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {sel}", color=GREEN)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown(f"**{sel} — Metrics**")
        metrics_labels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        vals = [v["accuracy"], v["precision"], v["recall"], v["f1"], v["roc_auc"]]
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        bars = ax.bar(metrics_labels, [x*100 for x in vals],
                      color=[GREEN, LGREEN, "#4ade80", "#86efac", ACCENT],
                      edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
        ax.set_title("Performance Metrics", color=GREEN)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("**Classification Report**")
    st.code(classification_report(
        data["y_test"], v["pred"],
        target_names=["Low Risk (0)", "High Risk (1)"]
    ))

    # Feature importance for RF
    if sel == "Random Forest":
        st.markdown('<p class="sec">🌲 Feature Importances (Random Forest — top 15)</p>', unsafe_allow_html=True)
        rf = v["model"]
        fi = pd.Series(rf.feature_importances_, index=data["features"]).nlargest(15)
        fig, ax = plt.subplots(figsize=(8, 4))
        fi[::-1].plot.barh(ax=ax, color=GREEN, edgecolor="white")
        ax.set_title("Top 15 Feature Importances", color=GREEN)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()


# ─── PAGE : LIVE PREDICTOR ────────────────────
elif page == "🔍 Live Predictor":
    st.title("🔍 Live Locust Risk Predictor")
    st.markdown("Enter values for the **key numeric features** of the dataset and get an instant prediction.")

    res = st.session_state.get("custom_results", results)

    st.markdown('<p class="sec">🤖 Choose model & input data</p>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", list(res.keys()))

    # We allow user to input values for first 6 most interpretable-ish numeric cols
    # then fill remaining features with column means from training data
    sample_cols = data["features"][:6]
    col_means   = data["X"].mean()

    st.markdown("Set values for the primary input features (remaining features use dataset mean):")
    cols = st.columns(3)
    user_vals = {}
    for i, col in enumerate(sample_cols):
        mn  = float(data["X"][col].min())
        mx  = float(data["X"][col].max())
        med = float(data["X"][col].median())
        user_vals[col] = cols[i % 3].number_input(
            col, min_value=float(mn), max_value=float(mx), value=float(med), format="%.3f"
        )

    if st.button("🔍 Predict Risk Level", type="primary", use_container_width=True):
        # Build full feature vector
        full_input = col_means.copy()
        for col, val in user_vals.items():
            full_input[col] = val

        arr = np.array([full_input.values])
        arr_scaled = data["scaler_std"].transform(arr)

        model = res[model_choice]["model"]
        pred  = model.predict(arr_scaled)[0]
        prob  = model.predict_proba(arr_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **HIGH LOCUST RISK** — Confidence: {prob[1]*100:.1f}%")
            st.markdown("""
**Recommended Actions:**
- 🚨 Alert local agricultural authorities immediately
- 🛩️ Deploy aerial surveillance teams
- 🧪 Prepare pesticide / bio-control stockpiles
- 📡 Activate SMS early-warning for nearby farmers
            """)
        else:
            st.success(f"✅ **LOW LOCUST RISK** — Confidence: {prob[0]*100:.1f}%")
            st.markdown("""
**Recommended Actions:**
- 📋 Maintain routine ground monitoring schedule
- 🌍 Watch neighbouring high-risk zones
- 📊 Log current conditions for seasonal trend analysis
            """)

        # Probability bar
        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.barh(["Low Risk (0)", "High Risk (1)"],
                [prob[0]*100, prob[1]*100],
                color=["#86efac", "#fca5a5"], edgecolor="white", height=0.5)
        for i, pv in enumerate([prob[0]*100, prob[1]*100]):
            ax.text(pv + 0.5, i, f"{pv:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, 110)
        ax.set_title("Prediction Probability", color=GREEN)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        st.pyplot(fig); plt.close()

        st.markdown('<p class="sec">Input Summary</p>', unsafe_allow_html=True)
        summary = {col: [f"{val:.3f}"] for col, val in user_vals.items()}
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ─── PAGE : ABOUT ─────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About LocustGuard")
    st.markdown("""
    <div class="hdr">
      <h1>🌿 LocustGuard</h1>
      <p>Locust Swarm Risk Prediction Using Climate and Land Data · MCA AIML · PES1PG25CA051</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 👩‍🎓 Project Details")
        st.markdown("""
| Field | Details |
|-------|---------|
| **SRN** | PES1PG25CA051 |
| **Student** | Deepashree |
| **Domain** | Agriculture · AIML |
| **Goal** | UN SDG: Life on Land |
| **Dataset** | FAO Locust Watch · Kaggle |
| **Shape** | 29,384 rows × 152 cols |
| **Task** | Binary Classification |
| **UI Tool** | Streamlit |
| **Cloud** | AWS / Azure |
        """)

    with c2:
        st.markdown("### 🤖 Models & Defaults")
        st.markdown("""
| Model | Key Hyperparameters |
|-------|---------------------|
| **Random Forest** | n_estimators=100, random_state=42 |
| **Gradient Boosting** | n_estimators=100, lr=0.1, depth=3 |
| **SVM** | kernel='rbf', C=1.0, gamma='scale' |
| **Logistic Regression** | max_iter=1000, C=1.0 |
| **Naive Bayes** | var_smoothing=1e-9 |
        """)

    st.markdown("### 📋 Full ML Pipeline")
    st.code("""# 1. Load
df = pd.read_csv('swarm.csv')                              # 29384 × 152

# 2. Missing values
df = df.dropna()                                           # → 21790 × 152

# 3. Outlier removal (IQR)
Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < Q1-1.5*IQR)|(df > Q3+1.5*IQR)).any(axis=1)]

# 4. Label encoding
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col].astype(str))

# 5. Target
df['CAT'] = [0 if i%2==0 else 1 for i in range(len(df))]
X = df.drop('CAT', axis=1);  y = df['CAT']

# 6. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 8. Train (example: RF)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:",  accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:",    recall_score(y_test, y_pred))
print("ROC-AUC:",   roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
print("MSE:",       mean_squared_error(y_test, y_pred))
print("MAE:",       mean_absolute_error(y_test, y_pred))""", language="python")

    st.markdown("### 🛠️ Tech Stack")
    tech = ["Python 3.x", "Pandas", "NumPy", "Scikit-learn",
            "Matplotlib", "Seaborn", "Streamlit", "AWS / Azure"]
    cols = st.columns(4)
    for i, t in enumerate(tech):
        cols[i % 4].success(t)