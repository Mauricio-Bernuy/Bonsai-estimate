import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sklearn

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
full_err_df = full_err_df.query('N > 100')
interval = 100  # Define the interval for I
sampled_err_df = full_err_df[full_err_df['I'] % interval == 0]  # Sample rows where I is a multiple of the interval

# Load the selected model
model_files = {
    "Precision":{
        "model_ert_final (MAPE 26.11%)": "Models/Precision/model_ert_final.joblib",
        "model_ert_final_TS (MAPE 17.55%)": "Models/Precision/model_ert_final_TS.joblib",
        "model_svr_final (MAPE 24.00%)": "Models/Precision/model_svr_final.joblib",
        "model_svr_final_TS (MAPE 12.46%)": "Models/Precision/model_svr_final_TS.joblib",
        "model_poly_final (MAPE 38.50%)": "Models/Precision/model_poly_final.joblib",
        "model_poly_final_TS (MAPE 10.75%)": "Models/Precision/model_poly_final_TS.joblib",
        "model_dt_final (MAPE 46.76%)": "Models/Precision/model_dt_final.joblib",
        "model_dt_final_TS (MAPE 17.55%)": "Models/Precision/model_dt_final_TS.joblib",
        "model_huber_final (MAPE 543.86%)": "Models/Precision/model_huber_final.joblib",
        "model_huber_final_TS (MAPE 313.37%)": "Models/Precision/model_huber_final_TS.joblib",

    },
    "Execution":{
        "model_ert_final (MAPE 6.26%)": "Models/Execution/model_ert_final.joblib",
        "model_ert_final_GPU (MAPE 22.34%)": "Models/Execution/model_ert_final_GPU.joblib",
        "model_svr_final (MAPE 9.82%)": "Models/Execution/model_svr_final.joblib",
        "model_svr_final_GPU (MAPE 33.56%)": "Models/Execution/model_svr_final_GPU.joblib",
        "model_poly_final (MAPE 6.32%)": "Models/Execution/model_poly_final.joblib",
        "model_poly_final_GPU (MAPE 33.41%)": "Models/Execution/model_poly_final_GPU.joblib",
        "model_dt_final (MAPE 6.04%)": "Models/Execution/model_dt_final.joblib",
        "model_dt_final_GPU (MAPE 19.38%)": "Models/Execution/model_dt_final_GPU.joblib",
        "model_huber_final (MAPE 66.89%)": "Models/Execution/model_huber_final.joblib",
        "model_huber_final_GPU (MAPE 64.64%)": "Models/Execution/model_huber_final_GPU.joblib",
    }
}

GPUS = {
    "GTX 1060" : {
        "Name": "NVIDIA GeForce GTX 1060 6GB",
        "Compute Capability": "6.1",
        "Total Memory (MB)": 6065,
        "Multiprocessors (SMs)": 10,
        "Max Threads Per SM": 2048,
        "Total Cores": 1280,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 32,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 96,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 96,
        "L2 Cache Size (KB)": 1536,
        "Memory Bus Width (bits)": 192,
        "Memory Bandwidth (GB/s)": 192,
        "Clock Rate (MHz)": 1771,
        "Warps Per SM": 64,
        "Blocks Per SM": 32,
        "Half Precision FLOP/s": 2111,
        "Single Precision FLOP/s": 1055,
        "Double Precision FLOP/s": 1055,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 192,
        "Global Memory Size (MB)": 6065,
        "L2 Cache Size": 1536,
        "Memcpy Engines": 2
    },
    "RTX 3070" : {
        "Name": "NVIDIA GeForce RTX 3070",
        "Compute Capability": "8.6",
        "Total Memory (MB)": 7877,
        "Multiprocessors (SMs)": 46,
        "Max Threads Per SM": 1536,
        "Total Cores": 5888,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 16,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 100,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 100,
        "L2 Cache Size (KB)": 4096,
        "Memory Bus Width (bits)": 256,
        "Memory Bandwidth (GB/s)": 448,
        "Clock Rate (MHz)": 1755,
        "Warps Per SM": 48,
        "Blocks Per SM": 16,
        "Half Precision FLOP/s": 9623,
        "Single Precision FLOP/s": 4811,
        "Double Precision FLOP/s": 4811,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 448,
        "Global Memory Size (MB)": 7877,
        "L2 Cache Size": 4096,
        "Memcpy Engines": 2
    },
    "RTX 3060" : {
        "Name": "NVIDIA GeForce RTX 3060",
        "Compute Capability": "8.6",
        "Total Memory (MB)": 11939,
        "Multiprocessors (SMs)": 28,
        "Max Threads Per SM": 1536,
        "Total Cores": 3584,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 16,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 100,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 100,
        "L2 Cache Size (KB)": 2304,
        "Memory Bus Width (bits)": 192,
        "Memory Bandwidth (GB/s)": 360,
        "Clock Rate (MHz)": 1777,
        "Warps Per SM": 48,
        "Blocks Per SM": 16,
        "Half Precision FLOP/s": 5931,
        "Single Precision FLOP/s": 2965,
        "Double Precision FLOP/s": 2965,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 360,
        "Global Memory Size (MB)": 11939,
        "L2 Cache Size": 2304,
        "Memcpy Engines": 2
    },
    "RTX 2070 SUPER" : {
        "Name": "NVIDIA GeForce RTX 2070 SUPER",
        "Compute Capability": "7.5",
        "Total Memory (MB)": 7790,
        "Multiprocessors (SMs)": 40,
        "Max Threads Per SM": 1024,
        "Total Cores": 5120,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 16,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 64,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 64,
        "L2 Cache Size (KB)": 4096,
        "Memory Bus Width (bits)": 256,
        "Memory Bandwidth (GB/s)": 448,
        "Clock Rate (MHz)": 1785,
        "Warps Per SM": 32,
        "Blocks Per SM": 16,
        "Half Precision FLOP/s": 8511,
        "Single Precision FLOP/s": 4255,
        "Double Precision FLOP/s": 4255,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 448,
        "Global Memory Size (MB)": 7790,
        "L2 Cache Size": 4096,
        "Memcpy Engines": 3
    },
    "RTX 4060 Ti" : {
        "Name": "NVIDIA GeForce RTX 4060 Ti",
        "Compute Capability": "8.9",
        "Total Memory (MB)": 7843,
        "Multiprocessors (SMs)": 34,
        "Max Threads Per SM": 1536,
        "Total Cores": 4352,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 24,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 100,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 100,
        "L2 Cache Size (KB)": 32768,
        "Memory Bus Width (bits)": 128,
        "Memory Bandwidth (GB/s)": 288,
        "Clock Rate (MHz)": 2565,
        "Warps Per SM": 48,
        "Blocks Per SM": 24,
        "Half Precision FLOP/s": 10396,
        "Single Precision FLOP/s": 5198,
        "Double Precision FLOP/s": 5198,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 288,
        "Global Memory Size (MB)": 7843,
        "L2 Cache Size": 32768,
        "Memcpy Engines": 2
    },
    "Tesla T4" : {
        "Name": "Tesla T4",
        "Compute Capability": "7.5",
        "Total Memory (MB)": 15102,
        "Multiprocessors (SMs)": 40,
        "Max Threads Per SM": 1024,
        "Total Cores": 5120,
        "Warp Size": 32,
        "Max Threads Per Block": 1024,
        "Max Blocks Per SM": 16,
        "Shared Memory Per Block (KB)": 48,
        "Shared Memory Per SM (KB)": 64,
        "Registers Per Block": 65536,
        "Registers Per SM": 65536,
        "L1 Cache Size (KB)": 64,
        "L2 Cache Size (KB)": 4096,
        "Memory Bus Width (bits)": 256,
        "Memory Bandwidth (GB/s)": 320,
        "Clock Rate (MHz)": 1590,
        "Warps Per SM": 32,
        "Blocks Per SM": 16,
        "Half Precision FLOP/s": 7581,
        "Single Precision FLOP/s": 3790,
        "Double Precision FLOP/s": 3790,
        "Concurrent Kernels": 1,
        "Threads Per Warp": 32,
        "Global Memory Bandwidth (GB/s)": 320,
        "Global Memory Size (MB)": 15102,
        "L2 Cache Size": 4096,
        "Memcpy Engines": 3
    },
    "Tesla P100-PCIE-16GB" : {
        'Name': 'Tesla P100-PCIE-16GB',
        'Compute Capability': '6.0',
        'Total Memory (MB)': 16269,
        'Multiprocessors (SMs)': 56,
        'Max Threads Per SM': 2048,
        'Total Cores': 7168,
        'Warp Size': 32,
        'Max Threads Per Block': 1024,
        'Max Blocks Per SM': 32,
        'Shared Memory Per Block (KB)': 48,
        'Shared Memory Per SM (KB)': 64,
        'Registers Per Block': 65536,
        'Registers Per SM': 65536,
        'L1 Cache Size (KB)': 64,
        'L2 Cache Size (KB)': 4096,
        'Memory Bus Width (bits)': 4096,
        'Memory Bandwidth (GB/s)': 732,
        'Clock Rate (MHz)': 1328,
        'Warps Per SM': 64,
        'Blocks Per SM': 32,
        'Half Precision FLOP/s': 8865,
        'Single Precision FLOP/s': 4432,
        'Double Precision FLOP/s': 4432,
        'Concurrent Kernels': 1,
        'Threads Per Warp': 32,
        'Global Memory Bandwidth (GB/s)': 732,
        'Global Memory Size (MB)': 16269,
        'L2 Cache Size': 4096,
        'Memcpy Engines': 2
    },
    "A100": {
        'Clock Rate (MHz)': 1065,
        'Multiprocessors (SMs)': 108,
        'L2 Cache Size (KB)': 81920,
        'Half Precision FLOP/s': 13711,
        'Single Precision FLOP/s': 6855,
        'Double Precision FLOP/s': 6855
    },
    "A100 x 8 (SELENE Cluster)": {
        'Clock Rate (MHz)': 1065,
        'Multiprocessors (SMs)': 108,
        'L2 Cache Size (KB)': 81920 * 8,
        'Half Precision FLOP/s': 13711 * 8,
        'Single Precision FLOP/s': 6855 * 8,
        'Double Precision FLOP/s': 6855 * 8
    },
    "A100 x 4320 (SELENE Supercomputer)": {
        'Clock Rate (MHz)': 1065,
        'Multiprocessors (SMs)': 108,
        'L2 Cache Size (KB)': 81920 * 8 * 540,
        'Half Precision FLOP/s': 13711 * 8 * 540,
        'Single Precision FLOP/s': 6855 * 8 * 540,
        'Double Precision FLOP/s': 6855 * 8 * 540
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

    stp, fmt, val = step_format_dict.get(src, (1, "%d", df[src].max()))

    new_src_range = st.slider(f"Select Range of {src} prediction", 
                              min_value=df[src].min(), max_value=df[src].max()*2, value=(df[src].min(), df[src].max()), step=stp)
    # Generate new_src_values within the selected range
    if isinstance(stp, int):
        new_src_values = np.linspace(new_src_range[0], new_src_range[1], 100).round().astype(int)
    else:
        new_src_values = np.linspace(new_src_range[0], new_src_range[1], 100)

    st.write("#### Parameters")

    if prediction_type == "Execution":
        # gpu_options = [None] + list(GPUS.keys())
        gpu_options = GPUS.keys()
        selected_gpu = st.selectbox("*(optional) Select GPU*", gpu_options)
        # st.write(GPUS[selected_gpu])
    else:
        selected_gpu = list(GPUS.keys())[0]

    numcol1, numcol2, numcol3 = st.columns(3)

    # Set input values to the filter_criteria for features not in src
    for i, feature in enumerate([c for c in input_columns if c != src]):
        stp, fmt, val = step_format_dict.get(feature, (1, "%d", df[feature].iloc[-1]))

        if selected_gpu is not None:
            if GPUS[selected_gpu].keys().__contains__(feature):
                val = GPUS[selected_gpu][feature]
        # val = 123 if feature == 'N' else val
        # Distribute inputs across the three columns
        if i % 3 == 0:
            with numcol1:
                value = st.number_input(feature, value=val, step=stp, format=fmt, placeholder="Input...", key=feature)
        elif i % 3 == 1:
            with numcol2:
                value = st.number_input(feature, value=val, step=stp, format=fmt, placeholder="Input...", key=feature)
        else:
            with numcol3:
                value = st.number_input(feature, value=val, step=stp, format=fmt, placeholder="Input...", key=feature)

        filter_criteria[feature] = [round(value, 7)] 

    dicts = {}
    for feature, values in filter_criteria.items():
        dicts[feature] = values[0]
    dicts[src] = new_src_values

    generated_df = pd.DataFrame(dicts)
    
    st.write("#### Actual data")

    st.write(df[list(input_columns)+[target]])

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

    # st.write(str(filter_criteria))
    # # Display the filter criteria    
    # c1, c2, c3 = st.columns([1.5, 7, 1.5])
    # with c1:
    #     st.write(' ')

    # with c2:
    #     st.write(pd.DataFrame(filter_criteria).reset_index(drop=True))

    # with c3:
    #     st.write(' ')

