import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Customer Lifetime Value Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("📊 Customer Lifetime Value Prediction")
st.subheader("Customer Analytics & Machine Learning Dashboard")

st.markdown("""
This dashboard analyzes customer purchasing behavior,
customer segmentation and Customer Lifetime Value (CLV)
using RFM analysis and Machine Learning.
""")

st.divider()

# -----------------------------------
# 2. LOAD DATA
# -----------------------------------

@st.cache_data
def load_csv(filename):
    path = BASE_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


final_data = load_csv("final_customer_clv.csv")
results_data = load_csv("Customer_Lifetime_Value_Results.csv")
prediction_data = load_csv("xgboost_clv_predictions.csv")
rfm_data = load_csv("rfm_t_features.csv")
cleaned_data = load_csv("cleaned_usc_data.csv")


# -----------------------------------
# 3. LOAD MACHINE LEARNING MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    model_files = [
        "xgboost_clv_model.pkl",
        "customer_value_model.pkl"
    ]

    for filename in model_files:
        path = BASE_DIR / filename

        if path.exists():
            try:
                return joblib.load(path), filename
            except Exception:
                continue

    return None, None


model, model_name = load_model()


# -----------------------------------
# 4. SIDEBAR
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Customer Data",
        "Customer Analysis",
        "CLV Prediction",
        "Model Information"
    ]
)


# -----------------------------------
# 5. HELPER FUNCTIONS
# -----------------------------------

def find_clv_column(df):

    if df.empty:
        return None

    candidates = [
        "predicted_clv",
        "predicted_customer_lifetime_value",
        "customer_lifetime_value",
        "lifetime_value",
        "clv",
        "predicted_value",
        "predicted_clv_value"
    ]

    column_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in candidates:
        if name in column_map:
            return column_map[name]

    for col in df.columns:
        name = str(col).lower()

        if "clv" in name or "lifetime_value" in name:
            return col

    return None


def get_active_data():

    for df in [final_data, results_data, prediction_data]:
        if not df.empty:
            return df

    return pd.DataFrame()


# -----------------------------------
# 6. OVERVIEW DASHBOARD
# -----------------------------------

if page == "Overview":

    st.header("📈 Business Overview")

    active_data = get_active_data()

    total_customers = 0
    average_clv = 0
    total_clv = 0

    if not active_data.empty:

        total_customers = len(active_data)

        clv_column = find_clv_column(active_data)

        if clv_column:

            clv_values = pd.to_numeric(
                active_data[clv_column],
                errors="coerce"
            ).dropna()

            if not clv_values.empty:
                average_clv = clv_values.mean()
                total_clv = clv_values.sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Average CLV",
        f"{average_clv:,.2f}"
    )

    col3.metric(
        "Total CLV",
        f"{total_clv:,.2f}"
    )

    col4.metric(
        "Model Status",
        "Loaded" if model is not None else "Not Found"
    )

    st.divider()

    st.subheader("Project Summary")

    st.write("""
    The Customer Lifetime Value project helps businesses
    understand how much revenue a customer can generate
    throughout their relationship with the company.

    The project uses:

    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - RFM Analysis
    - Customer Segmentation
    - XGBoost Machine Learning
    - Customer Lifetime Value Prediction
    """)

    if not active_data.empty:

        clv_column = find_clv_column(active_data)

        if clv_column:

            st.subheader("Customer Lifetime Value Distribution")

            values = pd.to_numeric(
                active_data[clv_column],
                errors="coerce"
            ).dropna()

            if not values.empty:

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.histplot(
                    values,
                    kde=True,
                    ax=ax
                )

                ax.set_title("CLV Distribution")
                ax.set_xlabel("Customer Lifetime Value")
                ax.set_ylabel("Number of Customers")

                st.pyplot(fig)
                plt.close(fig)


# -----------------------------------
# 7. CUSTOMER DATA
# -----------------------------------

elif page == "Customer Data":

    st.header("📁 Customer Dataset")

    dataset_options = {
        "Final CLV Data": final_data,
        "Customer Results": results_data,
        "XGBoost Predictions": prediction_data,
        "RFM-T Features": rfm_data,
        "Cleaned Dataset": cleaned_data
    }

    selected_dataset = st.selectbox(
        "Select Dataset",
        list(dataset_options.keys())
    )

    df = dataset_options[selected_dataset]

    if not df.empty:

        st.success("Dataset loaded successfully!")

        st.write("Dataset Shape:", df.shape)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])

        with col2:
            st.write("Missing Values:", int(df.isnull().sum().sum()))
            st.write("Duplicate Rows:", int(df.duplicated().sum()))

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        st.download_button(
            "Download Dataset CSV",
            data=df.to_csv(index=False),
            file_name="customer_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Selected dataset is not available.")


# -----------------------------------
# 8. CUSTOMER ANALYSIS
# -----------------------------------

elif page == "Customer Analysis":

    st.header("👥 Customer Analysis")

    df = rfm_data.copy()

    if df.empty:
        df = get_active_data()

    if not df.empty:

        st.success("Customer data loaded.")

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select Feature for Analysis",
                numeric_columns
            )

            st.subheader(
                f"Distribution of {selected_column}"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(
                df[selected_column].dropna(),
                kde=True,
                ax=ax
            )

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Correlation Heatmap")

            if len(numeric_columns) >= 2:

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.heatmap(
                    df[numeric_columns].corr(),
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    ax=ax
                )

                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Feature Comparison")

            if len(numeric_columns) >= 2:

                x_feature = st.selectbox(
                    "X Axis",
                    numeric_columns,
                    key="x_feature"
                )

                y_feature = st.selectbox(
                    "Y Axis",
                    numeric_columns,
                    key="y_feature"
                )

                fig, ax = plt.subplots(figsize=(9, 5))

                sns.scatterplot(
                    data=df,
                    x=x_feature,
                    y=y_feature,
                    ax=ax
                )

                st.pyplot(fig)
                plt.close(fig)

        else:
            st.warning("No numerical columns found.")

    else:
        st.warning("No customer analysis data available.")


# -----------------------------------
# 9. CUSTOMER LIFETIME VALUE PREDICTION
# -----------------------------------

elif page == "CLV Prediction":

    st.header("🤖 Customer Lifetime Value Prediction")

    if model is not None:

        st.success(f"Machine Learning Model Loaded: {model_name}")

        st.write("""
        Enter customer feature values below to generate
        a Customer Lifetime Value prediction.
        """)

        feature_names = getattr(
            model,
            "feature_names_in_",
            None
        )

        if feature_names is None:
            st.warning(
                "The model does not expose feature_names_in_. "
                "Individual prediction requires the exact training features."
            )

        else:

            with st.form("prediction_form"):

                input_values = {}

                st.subheader("Customer Features")

                columns = st.columns(2)

                for i, feature in enumerate(feature_names):

                    with columns[i % 2]:

                        input_values[feature] = st.number_input(
                            str(feature),
                            value=0.0,
                            format="%.4f"
                        )

                submitted = st.form_submit_button(
                    "Predict Customer Lifetime Value"
                )

            if submitted:

                try:

                    input_df = pd.DataFrame(
                        [input_values],
                        columns=feature_names
                    )

                    prediction = model.predict(input_df)

                    st.divider()

                    st.success("Prediction Generated Successfully!")

                    st.metric(
                        "Predicted Customer Lifetime Value",
                        f"{float(np.asarray(prediction).flatten()[0]):,.2f}"
                    )

                    st.dataframe(
                        input_df,
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"Prediction Error: {e}"
                    )

    else:

        st.error("Machine Learning model not found.")

    st.divider()

    st.subheader("Existing Customer Predictions")

    if not prediction_data.empty:

        st.dataframe(
            prediction_data.head(100),
            use_container_width=True
        )

        st.download_button(
            "Download Predictions",
            data=prediction_data.to_csv(index=False),
            file_name="CLV_Predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("No saved prediction dataset available.")


# -----------------------------------
# 10. MODEL INFORMATION
# -----------------------------------

elif page == "Model Information":

    st.header("⚙️ Model Information")

    if model is not None:

        st.success("Model successfully loaded.")

        st.write("Model File:", model_name)

        st.write(
            "Model Type:",
            type(model).__name__
        )

        if hasattr(model, "feature_names_in_"):

            st.subheader("Model Input Features")

            st.write(
                list(model.feature_names_in_)
            )

        if hasattr(model, "n_features_in_"):

            st.metric(
                "Total Input Features",
                model.n_features_in_
            )

    else:

        st.error("Model could not be loaded.")

    st.divider()

    st.subheader("Project Files")

    files = [
        "final_customer_clv.csv",
        "Customer_Lifetime_Value_Results.csv",
        "xgboost_clv_predictions.csv",
        "xgboost_clv_model.pkl",
        "customer_value_model.pkl",
        "rfm_t_features.csv",
        "cleaned_usc_data.csv",
        "final_customer_clv.json",
        "README.md",
        "requirements.txt"
    ]

    for filename in files:

        path = BASE_DIR / filename

        if path.exists():
            st.write("✅", filename)
        else:
            st.write("❌", filename)


# -----------------------------------
# 11. FOOTER
# -----------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>Customer Lifetime Value Prediction Project</p>
        <p>Developed using Python, Streamlit, Pandas and XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
    