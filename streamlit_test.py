import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd


full = pd.read_csv("Full/full_3.csv", index_col=0)
full_err_df = pd.read_csv("Full/full_err.csv", index_col=0)

import pandas as pd
import matplotlib.pyplot as plt

interval = 1000  # Define the interval for I
sampled_err_df = full_err_df[full_err_df['I'] % interval == 0]  # Sample rows where I is a multiple of the interval\
df = sampled_err_df

src = 'I'
target = 'accumulated_error'


N_VAL = st.number_input(
    "N", value=100, placeholder="Type a number..."
)
st.write("The current N particle size is ", N_VAL)

THETA_VAL = st.number_input(
    "theta", value=0.2, placeholder="Type a number..."
)
st.write("The current theta hyperparameter is ", THETA_VAL)


# Define the filter criteria
filter_criteria = {
    'theta': [THETA_VAL],
    'N': [N_VAL],
    # 'dt': [0.00625,]
    # 'I': [10000],
}

# Apply filtering based on the filter criteria
for feature, values in filter_criteria.items():
    df = df[df[feature].isin(values)]

# Group by a specific feature
grouped = df.groupby('dt')

# Plot X vs y for each group in the histogram
colors = ['blue', 'green', 'red', 'purple','yellow']  # Specify colors for each group if known

# Create a figure and axis
fig, ax = plt.subplots() # figsize=(10, 6)

# Plot X vs y with scatter points and connect them with dashed lines for each group
for (group_name, group_data), color in zip(grouped, colors):
    ax.scatter(group_data[src], group_data[target], color=color, label=f'{group_name}')
    ax.plot(group_data[src], group_data[target], linestyle='--', color=color)  # Dashed line

# Add grid and logarithmic scales
ax.grid(linestyle='-', linewidth=1)
# ax.set_yscale("logit")         # Logarithmic scale for y-axis
# ax.set_xscale('symlog')      # Symmetric log scale for x-axis
# 'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'
# Add labels, title, and legend
ax.set_xlabel(src)
ax.set_ylabel(target)
ax.set_title('I vs accumulated_error, theta: [0.2], dt: [0.0625]')
ax.legend()

# Set x and y limits to cover the range of points
# ax.set_xlim(df[src].min() - 1, df['I'].max() + 1)
# ax.set_ylim(df[target].min() * 0.8, df[target].max() * 1.2)


# plt.show()



#create your figure and get the figure object returned
# fig = plt.figure() 
# plt.plot([1, 2, 3, 4, 5]) 

fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)

