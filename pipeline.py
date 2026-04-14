import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

# --- UI Configuration ---
st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide")
st.title("🚀 Advanced ML Pipeline Dashboard")
st.markdown("---")

# --- 1. Problem Type Selection ---
with st.sidebar:
    st.header("Step 1: Configuration")
    problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Horizontal Pipeline Layout
    tabs = st.tabs([
        "📊 Data Input & PCA", "🔍 EDA", "🛠️ Engineering", 
        "🎯 Feature Selection", "🧪 Split & Train", "📈 Evaluation & Tuning"
    ])

    # --- Tab 1: Input Data & PCA ---
    with tabs[0]:
        st.subheader("Data Preview & Dimensionality")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            target_col = st.selectbox("Select Target Feature", df.columns)
            features = st.multiselect("Select Input Features", 
                                    [c for c in df.columns if c != target_col],
                                    default=[c for c in df.columns if c != target_col][:5])
        
        if features:
            X_raw = df[features].select_dtypes(include=[np.number]).dropna()
            if not X_raw.empty:
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_raw)
                fig_pca = px.scatter(components, x=0, y=1, color=df.loc[X_raw.index, target_col],
                                   title="PCA Visualization (2D)", template="plotly_dark")
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.warning("Please ensure features are numerical for PCA.")

    # --- Tab 2: EDA ---
    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.write("Statistics", df.describe())
        with col_eda2:
            fig_corr = px.imshow(df.corr(numeric_only=True), text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- Tab 3: Data Engineering & Outliers ---
    with tabs[2]:
        st.subheader("Data Cleaning & Outlier Detection")
        
        # Imputation
        strategy = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=strategy)
            df[features] = imputer.fit_transform(df[features])
            st.success("Missing values handled!")

        # Outlier Detection
        outlier_method = st.selectbox("Detection Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        outliers = np.zeros(len(df))
        
        if st.button("Detect Outliers"):
            numeric_df = df[features].select_dtypes(include=[np.number])
            if outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05)
                outliers = iso.fit_predict(numeric_df) == -1
            elif outlier_method == "IQR":
                Q1 = numeric_df.quantile(0.25)
                Q3 = numeric_df.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            
            st.write(f"Detected {sum(outliers)} outliers.")
            if sum(outliers) > 0:
                if st.button("Remove Selected Outliers"):
                    df = df[~outliers]
                    st.experimental_rerun()

    # --- Tab 4: Feature Selection ---
    with tabs[3]:
        st.subheader("Feature Importance & Selection")
        method = st.selectbox("Selection Method", ["Variance Threshold", "Information Gain"])
        
        X = df[features].select_dtypes(include=[np.number])
        y = df[target_col]
        if problem_type == "Classification":
            y = LabelEncoder().fit_transform(y)

        if method == "Information Gain":
            scores = mutual_info_classif(X, y) if problem_type == "Classification" else mutual_info_regression(X, y)
            feat_imp = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feat_imp)

    # --- Tab 5: Data Split & Model Selection ---
    with tabs[4]:
        st.subheader("Training Setup")
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        model_choice = st.selectbox("Choose Model", 
            ["Linear/Logistic Regression", "SVM", "Random Forest", "K-Means"])
        
        k_fold = st.number_input("K-Fold Value", 2, 10, 5)
        
        # Model mapping
        if model_choice == "Random Forest":
            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
        elif model_choice == "SVM":
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)
        else:
            model = LogisticRegression() if problem_type == "Classification" else LinearRegression()

        if st.button("Train Model"):
            cv_results = cross_validate(model, X_train, y_train, cv=k_fold, return_train_score=True)
            st.write(f"Mean Training Score: {cv_results['train_score'].mean():.4f}")
            st.write(f"Mean Validation Score: {cv_results['test_score'].mean():.4f}")
            
            # Check for Overfitting
            diff = cv_results['train_score'].mean() - cv_results['test_score'].mean()
            if diff > 0.15:
                st.warning("⚠️ Warning: Model might be Overfitting!")
            elif cv_results['train_score'].mean() < 0.6:
                st.info("ℹ️ Note: Model might be Underfitting.")

    # --- Tab 6: Hyperparameter Tuning ---
    with tabs[5]:
        st.subheader("Fine-Tuning (Grid Search)")
        if model_choice == "Random Forest":
            n_estimators = st.multiselect("n_estimators", [50, 100, 200], default=[100])
            param_grid = {'n_estimators': n_estimators}
            
            if st.button("Run Grid Search"):
                grid = GridSearchCV(model, param_grid, cv=3)
                grid.fit(X_train, y_train)
                st.write("Best Parameters:", grid.best_params_)
                st.write("Best Score:", grid.best_score_)
        else:
            st.write("Tuning options for this model coming soon.")

else:
    st.info("Please upload a CSV file to begin.")