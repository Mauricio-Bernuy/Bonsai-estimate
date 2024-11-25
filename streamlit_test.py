import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np

import joblib

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



from matplotlib.backend_bases import MouseEvent

class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
            self.ax.figure.canvas.draw()


# Load the selected model
model_files = {
    "model_poly_final": "Models/Precision/model_poly_final.joblib",
    "model_dt_final": "Models/Precision/model_dt_final.joblib",
    "model_huber_final": "Models/Precision/model_huber_final.joblib",
    "model_gb_final": "Models/Precision/model_gb_final.joblib",
    "model_svr_final": "Models/Precision/model_svr_final.joblib",
    "model_ert_final": "Models/Precision/model_ert_final.joblib",
    "model_poly_final_test": "Models/Precision/model_poly_final_test.joblib"
}
# Set wide mode as the default layout for Streamlit
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("Bonsai Estimate")

# Create a layout with two columns
col1, col2 = st.columns([3, 7])

with col1:
    model_options = list(model_files.keys())
    model_type = st.selectbox("Select Model", model_options)
    model = load_model(model_files[model_type])

    full = pd.read_csv("Full/full_3.csv", index_col=0)
    full_err_df = pd.read_csv("Full/full_err.csv", index_col=0)
    full_err_df['accumulated_error'] = np.abs(full_err_df['accumulated_error'])



    interval = 100  # Define the interval for I
    sampled_err_df = full_err_df[full_err_df['I'] % interval == 0]  # Sample rows where I is a multiple of the interval\
    df = sampled_err_df

    # df = full_err_df

    # Create a streamlit selector for choosing the plot type
    plot_type = st.selectbox("Select Plot Type", ["I vs accumulated_error", "N vs accumulated_error", "dt vs accumulated_error", "theta vs accumulated_error"])

    # Set the src and target variables based on the selected plot type
    if plot_type == "I vs accumulated_error":
        src = 'I'
        target = 'accumulated_error'
    elif plot_type == "N vs accumulated_error":
        src = 'N'
        target = 'accumulated_error'
    elif plot_type == "dt vs accumulated_error":
        src = 'dt'
        target = 'accumulated_error'
    elif plot_type == "theta vs accumulated_error":
        src = 'theta'
        target = 'accumulated_error'

    # Define the filter criteria
    filter_criteria = {}

    # List of input columns
    input_columns = ['theta', 'N', 'dt', 'I']

    # Set input values to the filter_criteria for features not in src
    for feature in input_columns:
        if feature != src:
            if feature == 'N':
                value = st.number_input(feature, value=100, step=1, format="%d", placeholder="Type an integer...")
            elif feature == 'theta':
                value = st.number_input(feature, value=0.20, step=0.01, format="%0.2f", placeholder="Type a number...")
            elif feature == 'I':
                value = st.number_input(feature, value=5000, step=1, format="%d", placeholder="Type an integer...")
            elif feature == 'dt':
                value = st.number_input(feature, value=0.0625, step=0.00625, format="%0.5f", placeholder="Type a number...")
            filter_criteria[feature] = [value]
            st.write(f"The current {feature} value is ", value)


    # Add an input to select the range of new_src_values
    if src == 'N':
        stp = 1
    elif src == 'theta':
        stp = 0.01
    elif src == 'I':
        stp = 1
    elif src == 'dt':
        stp = 0.00625

    new_src_range = st.slider("Select Range of new_src_values", 
                              min_value=df[src].min(), max_value=df[src].max()*2, value=(df[src].min(), df[src].max()), step=stp)

    # Generate new_src_values within the selected range
    new_src_values = np.linspace(new_src_range[0], new_src_range[1], 50)
    # new_src_values = np.linspace(df[src].min(), df[src].max(), 50)  # Generate src values in the range of existing data

    generated_df = pd.DataFrame({
        'N': filter_criteria.get('N', [100])[0],
        'theta': filter_criteria.get('theta', [0.20])[0],
        'dt': filter_criteria.get('dt', [0.0625])[0],
        'I': filter_criteria.get('I', [5000])[0],
        src: new_src_values
    })

    # Apply filtering based on the filter criteria
    for feature, values in filter_criteria.items():
        df = df[df[feature].isin(values)]

    generated_df['predicted_accumulated_error'] = model.predict(generated_df[['N', 'theta', 'dt', 'I']])

with col2:
    fig, ax = plt.subplots(figsize=(10, 6)) # 

    actual_scatter = ax.scatter(df[src], df[target], color='blue', label='Actual')
    ax.plot(df[src], df[target], color='blue')

    predicted_scatter = ax.scatter(generated_df[src], generated_df['predicted_accumulated_error'], color= 'red', marker='x', label=f'{model_type}')
    ax.plot(generated_df[src], generated_df['predicted_accumulated_error'], linestyle='-', color= 'red')
    
    ax.set_title(f'{src} vs accumulated_error, {filter_criteria}')
    ax.legend(facecolor='white', framealpha=1)
    ax.grid(linestyle='-', linewidth=1, alpha=0.5, zorder=0)



    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14,fmt='.5f'))
    
    fig_html = mpld3.fig_to_html(fig)
    
    # Display the current screen size
    components.html(fig_html, height=800)

