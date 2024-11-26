import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np

import joblib

# Set wide mode as the default layout for Streamlit
st.set_page_config(layout="wide")

# Define named functions for the transformations
def log10_transform(x):
    return np.log10(x)

def inverse_log10_transform(x):
    return np.power(10, x)

def load_model(filename):
    try:
        # Save the model to the specified file
        model = joblib.load(filename)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")

# Load actual data
full = pd.read_csv("Full/full_3.csv", index_col=0)

full_err_df = pd.read_csv("Full/full_err.csv", index_col=0)
full_err_df['accumulated_error'] = np.abs(full_err_df['accumulated_error'])
interval = 100  # Define the interval for I
sampled_err_df = full_err_df[full_err_df['I'] % interval == 0]  # Sample rows where I is a multiple of the interval

# Load the selected model
model_files = {
    "Precision":{
        "model_poly_final": "Models/Precision/model_poly_final.joblib",
        "model_dt_final": "Models/Precision/model_dt_final.joblib",
        "model_huber_final": "Models/Precision/model_huber_final.joblib",
        "model_gb_final": "Models/Precision/model_gb_final.joblib",
        "model_svr_final": "Models/Precision/model_svr_final.joblib",
        "model_ert_final": "Models/Precision/model_ert_final.joblib",
        "model_poly_final_test": "Models/Precision/model_poly_final_test.joblib"

    },
    "Execution":{
        "model_dt_final": "Models/Execution/model_dt_final.joblib",
        "model_ert_final": "Models/Execution/model_ert_final.joblib",
        "model_huber_final": "Models/Execution/model_huber_final.joblib",
        "model_poly_final": "Models/Execution/model_poly_final.joblib",
        "model_svr_final": "Models/Execution/model_svr_final.joblib",
    }
}

# Create a layout with two columns
t1, t2 = st.columns([3, 7])# [3, 7]


# Set the title of the Streamlit app
with t1:
    st.title("Bonsai Estimate")

# Create a selectbox to choose between execution and precision models
with t2:
    # Display the filter criteria    
    cc1, cc2, cc3 = st.columns([1, 8, 1])
    with cc1:
        st.write(' ')

    with cc2:
        prediction_type = st.selectbox("Select Prediction Type", ["Execution", "Precision"])

    with cc3:
        st.write(' ')

# Create a layout with two columns
col1, col2 = st.columns([3, 7])

with col1:

    # select model to use
    model_options = list(model_files[prediction_type].keys())
    model_type = st.selectbox("Select Model", model_options)
    model = load_model(model_files[prediction_type][model_type])

    # Extract column names from the sklearn model
    column_names = model.feature_names_in_
    # st.write("Column Names:", column_names)

    # Set dataset to be used
    if prediction_type == "Execution":
        df = full
        target = 'exec_time_avg'
    elif prediction_type == "Precision":
        df = sampled_err_df
        target = 'accumulated_error'


    # Create a streamlit selector for choosing the plot type
    plot_type = st.selectbox("Select Plot Type", [f"{column} vs {target}" for column in column_names])

    # Set the src and target variables based on the selected plot type
    for column in column_names:
        if plot_type == f"{column} vs {target}":
            src = column
            break

    # Define the filter criteria
    filter_criteria = {}

    # List of input columns
    input_columns = column_names # ['theta', 'N', 'dt', 'I']

    # Define the step and format based on the src value
    step_format_dict = {
        'theta': (0.01, "%0.2f",df['theta'].min()),
        'dt': (0.00625, "%0.7f", df['dt'].min()),
    }
    
    # Set input values to the filter_criteria for features not in src
    for feature in input_columns:
        if feature != src:
            stp, fmt, val = step_format_dict.get(feature, (1, "%d", df[feature].iloc[-1]))
            value = st.number_input(feature, value=val, step=stp, format=fmt, placeholder="Input...")
            filter_criteria[feature] = [round(value, 7)] 
            # st.write(f"The current {feature} value is ", round(value, 7))

    stp, fmt, val = step_format_dict.get(src, (1, "%d", df[src].max()))

    new_src_range = st.slider(f"Select Range of {src} prediction", 
                              min_value=df[src].min(), max_value=df[src].max()*2, value=(df[src].min(), df[src].max()), step=stp)

    # Generate new_src_values within the selected range
    if isinstance(stp, int):
        new_src_values = np.linspace(new_src_range[0], new_src_range[1], 50).round().astype(int)
    else:
        new_src_values = np.linspace(new_src_range[0], new_src_range[1], 50)

    dicts = {}
    for feature, values in filter_criteria.items():
        dicts[feature] = values[0]
    dicts[src] = new_src_values

    generated_df = pd.DataFrame(dicts)
    
    st.write("actual_df:",df[input_columns])

    # Apply filtering based on the filter criteria
    for feature, values in filter_criteria.items():
        df = df[df[feature].isin(values)]

    generated_df[f'predicted_{target}'] = model.predict(generated_df[input_columns])

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Actual
    actual_scatter = ax.scatter(df[src], df[target], color='blue', label='Actual')
    ax.plot(df[src], df[target],  linestyle='--', color='blue')

    # Predicted
    predicted_scatter = ax.scatter(generated_df[src], generated_df[f'predicted_{target}'], color= 'red', marker='x', label=f'{model_type}')
    ax.plot(generated_df[src], generated_df[f'predicted_{target}'], linestyle='-', color= 'red')

    # Set title
    # ax.set_title(f'{src} vs accumulated_error\n{filter_criteria}')
    ax.set_title(f'{src} vs {target}')

    # Set x-axis label
    ax.set_xlabel(src)

    # Set y-axis label
    ax.set_ylabel(target)

    # Set legend
    ax.legend(facecolor='white', framealpha=1)

    # Set grid
    ax.grid(linestyle='-', linewidth=1, alpha=0.5, zorder=0)

    # Add mouse position plugin
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14, fmt='.0f'))

    # Convert figure to HTML
    fig_html = mpld3.fig_to_html(fig)

    # Display the figure
    components.html(fig_html, height=620)

    # Display the filter criteria    
    c1, c2, c3 = st.columns([1.5, 7, 1.5])
    with c1:
        st.write(' ')

    with c2:
        st.write(pd.DataFrame(filter_criteria).reset_index(drop=True))

    with c3:
        st.write(' ')

