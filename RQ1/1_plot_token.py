import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings
# Suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Suppress all deprecation warnings from matplotlib
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


df = pd.read_csv("./rq1_results/1_token_counts_per_turn.csv")
print("Analyzing data by turn")


# Filter rows where both user_token_length and assistant_token_length are >= 3
df_filtered = df[(df["user_token_length"] >= 1)] #& (df["assistant_token_length"] >= 3)]

# Calculate the number of remaining conversations or turns
remaining_items = len(df_filtered)
print(f"Number of remaining turns: {remaining_items}")

# Calculate and print statistics for user (input) tokens
user_min = df_filtered["user_token_length"].min()
user_max = df_filtered["user_token_length"].max()
user_median = df_filtered["user_token_length"].median()
user_mean = df_filtered["user_token_length"].mean()
print("\nDeveloper Prompts (Input) Token Statistics:")
print(f"Minimum: {user_min}")
print(f"Maximum: {user_max}")
print(f"Median: {user_median}")
print(f"Average: {user_mean:.2f}")

# Calculate and print statistics for assistant (output) tokens
assistant_min = df_filtered["assistant_token_length"].min()
assistant_max = df_filtered["assistant_token_length"].max()
assistant_median = df_filtered["assistant_token_length"].median()
print("\nLLM Responses (Output) Token Statistics:")
print(f"Minimum: {assistant_min}")
print(f"Maximum: {assistant_max}")
print(f"Median: {assistant_median}")
assistant_mean = df_filtered["assistant_token_length"].mean()
print(f"..........Average: {assistant_mean:.2f}")

# Calculate ratio statistics
df_filtered["ratio"] = df_filtered["assistant_token_length"] / df_filtered["user_token_length"]
df_filtered["ratio"] = df_filtered["ratio"].replace(0, 1e-10)
ratio_median = df_filtered["ratio"].median()
ratio_min = df_filtered["ratio"].min()
ratio_max = df_filtered["ratio"].max()
print("\nToken Length Ratio Statistics (Output/Input):")
print(f"Minimum ratio: {ratio_min}")
print(f"Maximum ratio: {ratio_max}")
print(f"Median ratio: {ratio_median}")
# Set plot style
# sns.set(style="whitegrid")

# Create a figure with two subplots, using width_ratios to adjust their sizes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1.5, 1]})  # 1:1:1 ratio for boxes

# Left figure: Boxplot for token lengths (log scale)
box = ax1.boxplot(
    [df_filtered["user_token_length"], df_filtered["assistant_token_length"]], 
    labels=["Developer prompts", "LLM response"], 
    patch_artist=True,  # Enable patching for color filling
    widths=0.6,         # Set the width of the boxes
    boxprops=dict(
        facecolor='white',  # Fill color of the box
        edgecolor='black',  # Edge color of the box
        linewidth=1.5       # Width of the box edges
    ),
    flierprops=dict(
        marker='o',         # Use circle markers for outliers
        markersize=3,       # Reduce the size of the outliers
        markerfacecolor='lightgray',  # Make the fill color lighter
        markeredgecolor='gray',      # Make the edge color lighter
        markeredgewidth=1            # Reduce the edge width
    ),
    medianprops=dict(
        color='black',      # Color of the median line
        linewidth=2         # Width of the median line
    ),
    whiskerprops=dict(
        color='black',      # Color of the whiskers
        linewidth=1.5       # Width of the whiskers
    ),
    capprops=dict(
        color='black',      # Color of the caps
        linewidth=1.5       # Width of the caps
    )
)

# Create a gradient color map (from green to blue)
cmap = LinearSegmentedColormap.from_list("custom", ["white", "white"])
colors = cmap(np.linspace(0, 1, 2))  # Two colors for User and Assistant

# Apply colors to each box in the box plot
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Apply colors to the median lines
for median in box['medians']:
    median.set_color("black")  # Set median line color
    median.set_linewidth(2)    # Increase median line thickness

# Customize the axes and title
ax1.set_ylabel("Token Length", fontsize=18)  # Y-axis label for token lengths
ax1.set_xlabel("(a)", fontsize=16)  # X-axis label for token types

# Set the x-axis labels with adjusted font size
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)

# Use log scale for the y-axis
# ax1.set_yscale("log")
ax1.set_yscale('symlog', linthresh=0.1) 
ax1.grid(False)

# Right figure: Boxplot for token length ratio (real ratio values, but symlog-transformed for boxplot)
df_filtered["ratio"] = df_filtered["assistant_token_length"] / df_filtered["user_token_length"]

# Handle division by zero or very small values by replacing 0 with a small positive value
df_filtered["ratio"] = df_filtered["ratio"].replace(0, 1e-10)
original = df_filtered["ratio"] 
assistant_mid = np.percentile(df_filtered['ratio'], 50) 
print("Median ratio: ", assistant_mid)

# Apply symlog transformation to the ratio for the boxplot
df_filtered["log_ratio"] = df_filtered["ratio"]

# Create the boxplot using the symlog-transformed data
box = ax2.boxplot(
    df_filtered["log_ratio"], 
    labels=["LLM response /Developer prompts"], 
    patch_artist=True,  # Enable patching for color filling
    widths=0.45,         # Set the width of the boxes
    boxprops=dict(
        facecolor='white',  # Fill color of the box
        edgecolor='black',  # Edge color of the box
        linewidth=1.5       # Width of the box edges
    ),
    flierprops=dict(
        marker='o',         # Use circle markers for outliers
        markersize=3,       # Reduce the size of the outliers
        markerfacecolor='lightgray',  # Make the fill color lighter
        markeredgecolor='gray',      # Make the edge color lighter
        markeredgewidth=1            # Reduce the edge width
    ),
    medianprops=dict(
        color='black',      # Color of the median line
        linewidth=2         # Width of the median line
    ),
    whiskerprops=dict(
        color='black',      # Color of the whiskers
        linewidth=1.5       # Width of the whiskers
    ),
    capprops=dict(
        color='black',      # Color of the caps
        linewidth=1.5       # Width of the caps
    )
)

# Create a gradient color map (from green to blue)
cmap = LinearSegmentedColormap.from_list("custom", ["white", "white"])
colors = cmap(np.linspace(0, 1, 1))  # One color for the single box

# Apply colors to the box in the box plot
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Apply colors to the median lines
for median in box['medians']:
    median.set_color("black")  # Set median line color
    median.set_linewidth(2)    # Increase median line thickness

# Customize the axes and title
ax2.set_ylabel("Token Ratio", fontsize=18)  # Y-axis label for the ratio
ax2.set_xlabel("(b)", fontsize=16)  # X-axis label for the ratio

# Set the x-axis labels with adjusted font size
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

# Set y-axis to show real ratio values
# Map the symlog-transformed data back to real values for the y-axis labels
def inverse_symlog(y, linthresh=0.01, base=10):
    linear_region = np.abs(y) < 1
    log_region = np.abs(y) >= 1

    result = np.zeros_like(y)
    result[log_region] = np.sign(y[log_region]) * linthresh * (base ** np.abs(y[log_region]))
    result[linear_region] = y[linear_region] * linthresh

    return result
 
plt.yscale('symlog', linthresh=0.1)  
# Define your custom tick positions in the symlog space
plt.grid(False)

# Add title indicating analysis level
# if ANALYZE_BY_TURN:
#     fig.suptitle("Token Length Analysis by Turn", y=1.05, fontsize=16)
# else:
#     fig.suptitle("Token Length Analysis by Conversation", y=1.05, fontsize=16)

# Adjust layout and show plot
plt.tight_layout()
plt.show()