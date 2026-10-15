# %% [markdown]
# #### Test Order Generation

# %%
import random

def generate_random_partition(total_sum, num_parts):
    dividers = sorted(random.randint(0, total_sum) for _ in range(num_parts - 1))
    points = [0] + dividers + [total_sum]
    return [points[i+1] - points[i] for i in range(num_parts)]

def generate_test_orders_of_boxamt(boxamt):
    for row_id in range(boxamt*100, boxamt*100+100):
        values = generate_random_partition(boxamt, 10)
        row_data = [row_id] + values
        print(",".join(map(str, row_data)))

for boxamt in range(40):
    generate_test_orders_of_boxamt(boxamt)

# %% [markdown]
# #### Test Order runs CSV concatenation

# %%
import pandas as pd
from pathlib import Path
from collections import defaultdict

test_names = ['optg', 'topx', 'algo']

for test_name in test_names:
    print(f'\n\nRunning concatenation for {test_name} CSVs...')
    
    input_path = Path(f'./results/{test_name}_comparisons/test_orders')
    output_path = Path(f'./results/{test_name}_comparisons/collated/')
    csv_files = list(input_path.glob('*.csv'))
    grouped_files = defaultdict(list)

    # Empty output dir before filling it again
    if output_path.exists():
        for file in output_path.iterdir():
            file.unlink()

    if test_name == 'optg':
        split_letter = 'T'
        sort_column = 'order_id'
        complete_row_amt = 100
    elif test_name == 'topx':
        split_letter = 'O'
        sort_column = 'topx_limit'
        complete_row_amt = 1600
    elif test_name == 'algo':
        split_letter = 'T'
        sort_column = 'algorithm'
        complete_row_amt = 400

    # Group files by box amount
    for file in csv_files:
        try:
            box_amt = file.name.split(split_letter)[1][:2]
            grouped_files[box_amt].append(file)
        except:
            print(f"Skipping {file.name}: naming scheme mismatch")
            continue

    # Concatenate and sort each group
    for box_amt, files in grouped_files.items():
        file_amt = len(files)
        combined_df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        combined_df.sort_values(by=['order_id', sort_column], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        order_range = f"{box_amt}00-{box_amt}99"
        
        # Check data completeness
        is_complete = (file_amt == 100) and (len(combined_df) == complete_row_amt)
        if is_complete:
            output_file = Path(f'{output_path}/{test_name}_comparison_{order_range}.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"[{order_range}] Successfully combined 100 files")
        elif file_amt >= 50:
            output_file = Path(f'{output_path}/{test_name}_comparison_{order_range}_incomplete.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"[{order_range}] Saved as incomplete: Found {file_amt}/100 files and {len(combined_df) + 1}/{complete_row_amt} lines")
        else:
            print(f"[{order_range}] Not saved: Found {file_amt}/100 files and {len(combined_df) + 1}/{complete_row_amt} lines")

# %% [markdown]
# #### CSV Missing Orders Finder

# %%
import pandas as pd
from pathlib import Path

def find_missing_orders(start_id, end_id, test_name, order_type):
    directory = Path(f"./results/{test_name}_comparisons/{order_type}_orders")
    files = list(directory.glob('*.csv'))
    if not files:
        print(f"No CSV files found in {directory}")
        return

    found_order_ids = set()
    for file in files:
        try:
            df = pd.read_csv(file, usecols=['order_id'])
            found_order_ids.update(df['order_id'].dropna().unique())
        except ValueError:
            print(f"Skipping {file.name}: 'order_id' column not found.")
        except Exception as e:
            print(f"Skipping {file.name} due to error: {e}")

    expected_order_ids = set(range(start_id, end_id + 1))
    missing_orders = sorted(list(expected_order_ids - found_order_ids))
    

    if not missing_orders:
        line = f"{test_name}_missing_{order_type}_orders = []"
        print(f"All expected orders ({start_id} to {end_id}) are present. Nothing is missing!")
    else:
        line = f"{test_name}_missing_{order_type}_orders = {missing_orders}"
        print(f"Missing {len(missing_orders)} orders:")
        print(line)
        print("")

    return line

test_names = ['optg', 'topx', 'algo']
order_types = ['given', 'test']
paste = ""

for test_name in test_names:
    for order_type in order_types:
        if order_type == 'given':
            start_id = 1
            end_id = 40
        elif test_name == 'algo':
            start_id = 1000
            end_id = 3999
        else:
            start_id = 1000
            end_id = 2999

        print(f"Checking {test_name} {order_type} order files...")
        paste += f"\n{find_missing_orders(start_id, end_id, test_name, order_type)}"

print("Paste the following to the boxoptimizer testing area:")
print(paste)

# %% [markdown]
# #### CSV Metric Extreme Difference Finder

# %%
import os
import glob
import pandas as pd

def get_extreme_differences(directory, metric, compare_tuple, n_results=10, ascending=False):
    
    # Determine the differentiator column based on directory name
    expected_diff_col = 'topx_limit' if 'topx' in directory.lower() else 'algorithm'
    
    # Gather all CSVs
    search_path = os.path.join(directory, "*.csv")
    all_files = glob.glob(search_path)
    
    if not all_files:
        print(f"No CSV files found in directory: {directory}")
        return None
        
    # Read and combine data
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        
        # Fallback: If 'topx_limit' or 'algorithm' is missing, assume the 2nd column is the differentiator
        if expected_diff_col not in temp_df.columns:
            actual_diff_col = temp_df.columns[1]
            temp_df.rename(columns={actual_diff_col: expected_diff_col}, inplace=True)
            
        df_list.append(temp_df)
        
    combined_df = pd.concat(df_list, ignore_index=True)

    # Failed BnB run data filtering
    bad_order_ids = combined_df[combined_df['cog_z'] == -1.0]['order_id'].unique()
    if len(bad_order_ids) > 0:
        print(f"Ignoring {len(bad_order_ids)} orders due to failed runs (cog_z == -1.0)")
        combined_df = combined_df[~combined_df['order_id'].isin(bad_order_ids)]
    
    # Check if the metric actually exists in the data
    if metric not in combined_df.columns:
        raise ValueError(f"Metric '{metric}' not found in the CSV columns: {list(combined_df.columns)}")

    # Ensure the diff column is treated as a string for consistent pivoting if comparing mixed types
    combined_df[expected_diff_col] = combined_df[expected_diff_col].astype(str)
    item_1, item_2 = str(compare_tuple[0]), str(compare_tuple[1])

    # Pivot the dataframe so orders are rows and conditions are columns
    pivot_df = combined_df.pivot_table(
        index='order_id', 
        columns=expected_diff_col, 
        values=metric
    ).reset_index()
    
    # Check if the items we want to compare actually exist in the data
    if item_1 not in pivot_df.columns or item_2 not in pivot_df.columns:
        raise ValueError(f"Could not find both '{item_1}' and '{item_2}' in the {expected_diff_col} column. Available options: {list(pivot_df.columns[1:])}")

    # Calculate the difference
    diff_col_name = f"Diff: {item_1} - {item_2}"
    pivot_df[diff_col_name] = pivot_df[item_1] - pivot_df[item_2]
    
    # Drop rows where we don't have data for both sides of the comparison
    result_df = pivot_df.dropna(subset=[diff_col_name])
    
    # Sort by the difference
    result_df = result_df.sort_values(by=diff_col_name, ascending=ascending)
    
    # Select only the relevant columns to display nicely
    final_cols = ['order_id', item_1, item_2, diff_col_name]
    
    return result_df[final_cols].head(n_results).reset_index(drop=True)

TESTING_DIR = './results/algo_comparisons/test_orders/'
get_extreme_differences(TESTING_DIR, 'score (max_z)', ('BFD', 'BNB'), 736, False)

# %% [markdown]
# #### BnB Optimality Guarantee Test Results Processing

# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRINT_MODE = False
SAVE_MODE = True
TESTING_NEW_DATA = True

if TESTING_NEW_DATA:
    directory = './results/optg_comparisons/collated'
else:
    directory = './results/old_results/bnb_old_symbreak/bnb_comparisons'

csv_files = glob.glob(os.path.join(directory, '*.csv'))
results_dict = {}

for filepath in csv_files:
    filename = os.path.basename(filepath)

    if TESTING_NEW_DATA:
        box_count = filename[16:18]
    else:
        box_count = filename[25:27]
    
    df = pd.read_csv(filepath)

    initial_row_count = len(df)
    df = df[(df['score_no_guarantee'] != 0) & (df['score_with_guarantee'] != 0)]
    dropped_rows = initial_row_count - len(df)

    if PRINT_MODE:
        print(f"\n--- Analysis of {box_count}-box orders ---")
        if dropped_rows > 0:
            print(f"    !! Dropped {dropped_rows} failed run(s) with 0 score !!")
    
    # Initialize the dictionary for this box group
    results_dict[box_count] = {
        'completeness_status':          'good',
        'mismatch_count':               0,
        'mismatch_percent':             0.0,
        'avg_score_diff':               0.0,
        'percent_improvement':          0.0,
        'speed_multiplier':             0.0,
        'mean_nodes_no_guarantee':      0,
        'sigma_nodes_no_guarantee':     0,
        'mean_nodes_with_guarantee':    0,
        'sigma_nodes_with_guarantee':   0,
        'mean_score_no_guarantee':      0.0,  # Added for performance comparison
        'mean_score_with_guarantee':     0.0,  # Added for performance comparison
    }
    
    # Check all non- *_factor columns for missing values
    cols_to_check = [c for c in df.columns if not c.endswith('_factor')]
    missing_data_cols = df[cols_to_check].isnull().any(axis=1)
    orders_included = len(df)

    if orders_included == 100 and not missing_data_cols.any():
        if PRINT_MODE:
            print("- Completeness:                 good")
    else:
        results_dict[box_count]['completeness_status'] = False
        if PRINT_MODE:
            print(f"- Completeness:                 !missing data!")
            if len(df) != 100:
                print(f"    !! File contains {orders_included}/100 rows !!")
            if missing_data_cols.any():
                missing_ids = df[missing_data_cols]['order_id'].tolist()
                print(f"    !! Missing data in order IDs:")
                print(f"      {missing_ids}")

    # Count guarantee on/off score mismatches in orders and percentage of total
    mismatches = df[df['score_no_guarantee'] != df['score_with_guarantee']]
    mismatch_count = len(mismatches)
    results_dict[box_count]['mismatch_count'] = mismatch_count

    mismatch_percent = (mismatch_count / orders_included) * 100 if orders_included > 0 else 0
    results_dict[box_count]['mismatch_percent'] = mismatch_percent
    
    if mismatch_count > 0:
        avg_score_diff = abs(mismatches['score_no_guarantee'] - mismatches['score_with_guarantee']).mean()
        results_dict[box_count]['avg_score_diff'] = avg_score_diff
        if PRINT_MODE:
            print(f"- Score mismatches:             {mismatch_count}")
            print(f"- Average mismatch size:        {avg_score_diff:.2f}")
    else:
        if PRINT_MODE:
            print("- Score mismatches:             none")

    # Calculate node means and sigmas
    mean_nodes_no_guarantee    = df['nodes_no_guarantee'].mean()
    sigma_nodes_no_guarantee   = df['nodes_no_guarantee'].std()
    mean_nodes_with_guarantee  = df['nodes_with_guarantee'].mean()
    sigma_nodes_with_guarantee = df['nodes_with_guarantee'].std()

    # Node evaluation number comparison through ratio of means
    if pd.notna(mean_nodes_no_guarantee) and mean_nodes_no_guarantee > 0:
        times_fewer = mean_nodes_with_guarantee / mean_nodes_no_guarantee
        percent_improvement = (1 - (mean_nodes_no_guarantee / mean_nodes_with_guarantee)) * 100
        
        results_dict[box_count]['percent_improvement'] = percent_improvement
        results_dict[box_count]['speed_multiplier'] = times_fewer
        
        if PRINT_MODE:
            print(f"- Node improvement:             {percent_improvement:.2f}%")
            print(f"- No guarantee speedup:         {times_fewer:.2f}x")
    else:
        print("- Node improvement:             !error!")

    # Average node amounts and standard deviation recording
    results_dict[box_count]['mean_nodes_no_guarantee']    = mean_nodes_no_guarantee
    results_dict[box_count]['sigma_nodes_no_guarantee']   = sigma_nodes_no_guarantee
    results_dict[box_count]['mean_nodes_with_guarantee']  = mean_nodes_with_guarantee
    results_dict[box_count]['sigma_nodes_with_guarantee'] = sigma_nodes_with_guarantee

    # Calculate and record mean packing scores/heights
    results_dict[box_count]['mean_score_no_guarantee']   = df['score_no_guarantee'].mean()
    results_dict[box_count]['mean_score_with_guarantee']  = df['score_with_guarantee'].mean()

# Sort the dictionary by box_count
sorted_items = sorted(results_dict.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

# Extract data for plotting
box_counts_list         = [item[0]                               for item in sorted_items]
mismatch_percents_list  = [item[1]['mismatch_percent']             for item in sorted_items]
avg_score_diff_list     = [item[1]['avg_score_diff']                 for item in sorted_items]
mean_score_no_g_list    = [item[1]['mean_score_no_guarantee']    for item in sorted_items]
mean_score_with_g_list  = [item[1]['mean_score_with_guarantee']   for item in sorted_items]
speedup_list            = [item[1]['speed_multiplier']           for item in sorted_items]
mean_nodes_no_g_list    = [item[1]['mean_nodes_no_guarantee']    for item in sorted_items]
sigma_nodes_no_g_list   = [item[1]['sigma_nodes_no_guarantee']   for item in sorted_items]
mean_nodes_with_g_list  = [item[1]['mean_nodes_with_guarantee']  for item in sorted_items]
sigma_nodes_with_g_list = [item[1]['sigma_nodes_with_guarantee'] for item in sorted_items]

# Create multiplot and set x axis for all plots
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 16), layout="constrained")
thin_bar_width = 0.35
x_axis = np.arange(len(box_counts_list))
plt.style.use('seaborn-v0_8-whitegrid')

# Percentage of score mismatches
bars1 = ax1.bar(x_axis, mismatch_percents_list, color='#0D9488', edgecolor='black')
ax1.set_title('Frequency of Sub-Optimal Outcomes Without Guarantee', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage of Mismatches (%)', fontsize=11)
ax1.set_xticks(x_axis)
ax1.set_xticklabels(box_counts_list)
ax1.bar_label(bars1, padding=3, fmt='%.1f%%', fontsize=9)

# Average mismatch size
bars2 = ax2.bar(x_axis, avg_score_diff_list, color='#0D9488', edgecolor='black')
ax2.set_title('Average Max-Z Reduction With Guarantee', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Packed Max-Z Difference', fontsize=11)
ax2.set_xticks(x_axis)
ax2.set_xticklabels(box_counts_list)
ax2.bar_label(bars2, padding=3, fmt=lambda x: f'{x:.2f}' if x > 0 else '', fontsize=9)

# Performance
rects1 = ax3.bar(x_axis - 0.2, mean_score_no_g_list, thin_bar_width, label='No Guarantee', color="#E11D48", edgecolor='black')
rects2 = ax3.bar(x_axis + 0.2, mean_score_with_g_list, thin_bar_width, label='With Guarantee', color="#16A34A", edgecolor='black')
ax3.set_title('Average Max-Z Height', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Max-Z', fontsize=11)
ax3.set_xticks(x_axis)
ax3.set_xticklabels(box_counts_list)
ax3.set_ylim(200, 320)
ax3.legend()
ax3.bar_label(rects1, padding=3, fmt='%.1f', fontsize=8, rotation=90)
ax3.bar_label(rects2, padding=3, fmt='%.1f', fontsize=8, rotation=90)

# Speedup factor
bars4 = ax4.bar(x_axis, speedup_list, color='#16A34A', edgecolor='black')
ax4.set_title('Average no Guarantee Search Space Reduction', fontsize=12, fontweight='bold')
ax4.set_ylabel('Search Space Reduction Factor', fontsize=11)
ax4.set_xticks(x_axis)
ax4.set_xticklabels(box_counts_list)
ax4.bar_label(bars4, padding=3, fmt='%.2fx', fontsize=9)

# Average nodes per order 
rects5_1 = ax5.bar(x_axis - 0.2, mean_nodes_no_g_list, thin_bar_width, label='No Guarantee', color="#E11D48", edgecolor='black', log=True)
rects5_2 = ax5.bar(x_axis + 0.2, mean_nodes_with_g_list, thin_bar_width, label='With Guarantee', color="#16A34A", edgecolor='black', log=True)
ax5.set_title('Average Explored Nodes', fontsize=12, fontweight='bold')
ax5.set_ylabel('Average Nodes Traversed (Log)', fontsize=11)
ax5.set_xticks(x_axis)
ax5.set_xticklabels(box_counts_list)
ax5.legend()
ax5.bar_label(rects5_1, padding=3, fmt='%.0f', fontsize=8, rotation=45)
ax5.bar_label(rects5_2, padding=3, fmt='%.0f', fontsize=8, rotation=45)

# Node sigma
rects6_1 = ax6.bar(x_axis - 0.2, sigma_nodes_no_g_list, thin_bar_width, label='No Guarantee', color="#E11D48", edgecolor='black', log=True)
rects6_2 = ax6.bar(x_axis + 0.2, sigma_nodes_with_g_list, thin_bar_width, label='With Guarantee', color="#16A34A", edgecolor='black', log=True)
ax6.set_title('σ of Explored Nodes', fontsize=12, fontweight='bold')
ax6.set_ylabel('Standard Deviation of Nodes Traversed (Log)', fontsize=11)
ax6.set_xticks(x_axis)
ax6.set_xticklabels(box_counts_list)
ax6.legend()
ax6.bar_label(rects6_1, padding=3, fmt='%.0f', fontsize=8, rotation=45)
ax6.bar_label(rects6_2, padding=3, fmt='%.0f', fontsize=8, rotation=45)

# Shared X-labels
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlabel('Boxes per Order', fontsize=11)

# Save the figure if SAVE_MODE is True
if SAVE_MODE:
    save_path = './results/figures/bnb_optg_test.png'
    plt.savefig(save_path, dpi=300)
    if PRINT_MODE:
        print(f"\n Figure successfully saved to: {save_path}")

plt.show()

# %% [markdown]
# #### BnB Top-X Test Results Processing

# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
SAVE_MODE   = True
csv_dir     = './results/topx_comparisons/collated/'
num_lines   = 10

# Initializations
csv_files   = glob.glob(os.path.join(csv_dir, '*topx*.csv'))
results_dict = {}
aggregated_limit_data = []

for filepath in csv_files:
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath)
    
    # Extract box count from the first order_id
    if len(df) == 0:
        continue
    box_count = str(df['order_id'].iloc[0] // 100)

    # PRECALCULATIONS
    # Timing 
    df['total_seconds'] = df['time_taken (min)'] * 60 + df['(seconds)']                                 # Convert min,sec time to seconds

    # DF manipulation
    baseline_df = df[df['topx_limit'] == -1][['order_id', 'score', 'nodes']].copy()                     # Extract no-limit (-1 limit) data
    baseline_df.rename(columns={'score': 'base_score', 'nodes': 'base_nodes'}, inplace=True)            # Rename baseline columns to prevent overwriting with merge
    df = df.merge(baseline_df, on='order_id', how='left')                                               # Add baseline data back to df for easy comparison
    limits_df = df[df['topx_limit'] != -1].copy()                                                       # Get df without baseline rows for calculations on limited runs
    
    # Node improvement
    limits_df['node_improvement_factor'] = limits_df['base_nodes'] / limits_df['nodes']                 # Calculate node improvement and put it in new column
    
    # Unlimited score matches
    limits_df['matches_unlimited'] = limits_df['score'] == limits_df['base_score']                      # Check if score matches unlimited and put it in new column
    unlimited_matches = limits_df[limits_df['matches_unlimited']]                                       # Filter only rows that matched unlimited
    min_limit_per_order = unlimited_matches.groupby('order_id')['topx_limit'].min()                     # Get the minimal limit per order from filtered df
    avg_min_limit = min_limit_per_order.mean()                                                          # Get the average across orders
    perfect_match_count = len(min_limit_per_order)                                                      # Get the amount of matches of matches with unlimited
    
    groups_by_limit = limits_df.groupby('topx_limit').agg(                                              # Aggregate groups by limit to extract avg time, avg improvement, and score match rate
        avg_time=('total_seconds', 'mean'),
        avg_node_improvement=('node_improvement_factor', 'mean'),
        unlimited_match_rate=('matches_unlimited', 'mean')
    ).reset_index()
    
    # Store aggregated line data for the charts
    groups_by_limit['box_count'] = box_count
    aggregated_limit_data.append(groups_by_limit)
    
    # Store single-value metrics for the bar charts
    results_dict[box_count] = {
        'avg_min_limit': avg_min_limit,
        'perfect_match_count': perfect_match_count
    }

# Combine all limit data into one df
all_limits_df = pd.concat(aggregated_limit_data, ignore_index=True)
box_counts_sorted = sorted(results_dict.keys(), key=int)

# Get representative samples to plot to prevent too many overlaid lines
if len(box_counts_sorted) > num_lines:
    indices = np.linspace(0, len(box_counts_sorted) - 1, num_lines, dtype=int)
    box_counts_line_plots = [box_counts_sorted[i] for i in indices]

    if num_lines > 10:
        cmap = plt.get_cmap('tab20c')
    else:
        cmap = plt.get_cmap('tab10')

else:
    box_counts_line_plots = box_counts_sorted
    cmap = plt.get_cmap('tab20c')

# VISUALIZATIONS
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
colors = [cmap(i % 20) for i in range(len(box_counts_sorted))]

# Figure 1: time vs limit value
for i, bc in enumerate(box_counts_line_plots):
    data = all_limits_df[all_limits_df['box_count'] == bc]
    ax1.plot(data['topx_limit'], data['avg_time'], linewidth=2.5, label=f'{bc} Boxes', color=colors[i])

ax1.set_title('Average Solve Time vs. Top-X Limit', fontsize=14, fontweight='bold')
ax1.set_xlabel('Top-X Candidate Limit', fontsize=12)
ax1.set_ylabel('Average Time (Seconds)', fontsize=12)
ax1.set_yscale('log')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Order Size')

# Figure 2: node reduction vs limit value
for i, bc in enumerate(box_counts_line_plots):
    data = all_limits_df[all_limits_df['box_count'] == bc]
    ax2.plot(data['topx_limit'], data['avg_node_improvement'], linewidth=2.5, label=f'{bc} Boxes', color=colors[i])

ax2.set_title('Node Reduction Factor vs. Top-X Limit', fontsize=14, fontweight='bold')
ax2.set_xlabel('Top-X Candidate Limit', fontsize=12)
ax2.set_ylabel('Nodes Saved Factor (x times fewer)', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(title='Order Size')

# Figure 3: average min limit value to reach matched score
x_axis = np.arange(len(box_counts_sorted))
avg_min_limits = [results_dict[bc]['avg_min_limit'] for bc in box_counts_sorted]

bars = ax3.bar(x_axis, avg_min_limits, color='#2e7dc7', edgecolor='black', width=0.6)
ax3.set_title('Average Minimum Top-X Limit Needed to Match No-limit Score', fontsize=14, fontweight='bold')
ax3.set_xlabel('Boxes per Order', fontsize=12)
ax3.set_ylabel('Top-X Limit', fontsize=12)
ax3.set_xticks(x_axis)
ax3.set_xticklabels(box_counts_sorted)
ax3.grid(axis='y', linestyle='--', alpha=0.6)

# Put values on bars
for index, value in enumerate(avg_min_limits):
    ax3.text(index, value + 0.25, f"{value:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Figure 4: order percentage matching no-limit score vs limit value
for i, bc in enumerate(box_counts_line_plots):
    data = all_limits_df[all_limits_df['box_count'] == bc]
    ax4.plot(data['topx_limit'], data['unlimited_match_rate'] * 100, linewidth=2.5, label=f'{bc} Boxes', color=colors[i])

ax4.set_title('% Orders Matching Non-limited Score vs. Top-X Limit', fontsize=14, fontweight='bold')
ax4.set_xlabel('Top-X Candidate Limit', fontsize=12)
ax4.set_ylabel('% of Orders Matching unlimited', fontsize=12)
ax4.set_ylim(0, 105)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(title='Order Size')

plt.tight_layout(pad=3.0)

# Save the figure if SAVE_MODE is True
if SAVE_MODE:
    os.makedirs('./results', exist_ok=True)
    save_path = './results/figures/topx_performance_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

# %% [markdown]
# #### Algorithm Comparison Test Results Processing

# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configuration
TESTING_TEST_ORDERS = True
SAVE_MODE           = True
SELECTIVE_MEDIAN    = True

if TESTING_TEST_ORDERS:
    data_path = "./results/algo_comparisons/test_orders/algorithm_comparison_T*_to_T*.csv"
else:
    data_path = "./results/algo_comparisons/given_orders/algorithm_comparison_O*_to_O*.csv"

if SELECTIVE_MEDIAN:
    agg_criterion = 'median'
else:
    agg_criterion = 'mean'

# Data loading and preprocessing by glob
all_files = glob.glob(data_path)

if not all_files:
    print(f"No CSV files found in directory: {data_path}")
else:
    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Failed BnB run data filtering
    bad_order_ids = combined_df[combined_df['cog_z'] == -1.0]['order_id'].unique()
    if len(bad_order_ids) > 0:
        print(f"Ignoring {len(bad_order_ids)} orders due to failed runs (cog_z == -1.0)")
        combined_df = combined_df[~combined_df['order_id'].isin(bad_order_ids)]

    # Pivot DF so each row is an order_id and columns are algorithms
    pivot_df = combined_df.pivot_table(index='order_id', columns='algorithm', values='score (max_z)').reset_index()

    # Calculate how much BNB reduced the height compared to BFD
    pivot_df['bnb_improvement'] = pivot_df['BFD'] - pivot_df['BNB']                                                     
    
    # Extract the numeric portion of the order_id for correct sorting 
    pivot_df['order_num'] = pivot_df['order_id'].astype(str).str.extract(r'(\d+)').astype(float)
    pivot_df = pivot_df.sort_values('order_num').dropna(subset=['bnb_improvement'])

    # Extract exact box amount from order_id
    combined_df['boxamt'] = combined_df['order_id'] // 100
    
    # Filter for all type II test orders
    combined_df = combined_df[(combined_df['boxamt'] >= 10) & (combined_df['boxamt'] <= 39)]    

    # Create range bins and labels
    bins = [10, 15, 20, 25, 30, 35, 40]
    bin_labels = ['10-14', '15-19', '20-24', '25-29', '30-34', '35-39']
    combined_df['boxamt_range'] = pd.cut(combined_df['boxamt'], bins=bins, labels=bin_labels, right=False)

    # Data aggregation grouping by both box amount range and algorithm
    agg_df = combined_df.groupby(['boxamt_range', 'algorithm'], observed=False).agg({
        'nodes': agg_criterion,
        'time_taken (s)': agg_criterion,
        'score (max_z)': 'mean',
        'cog_z': 'mean',
        'volume_util (%)': 'mean',
        'packing_score': 'mean'
    }).reset_index()

    # Order algorithms
    algo_order = ['RANDOM', 'FFD', 'BFD', 'BNB']
    agg_df['algorithm'] = pd.Categorical(agg_df['algorithm'], categories=algo_order, ordered=True)
    agg_df = agg_df.sort_values(['boxamt_range', 'algorithm'])
    # Graph creation
    fig = plt.figure(figsize=(16, 20), layout="constrained")
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.55])

    # 6 normal plots
    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    ]
    
    # 7th plot is double-wide
    ax7 = fig.add_subplot(gs[3, :])

    colors = ["#E11D48", "#F59E0B", "#0D9488", "#16A34A"]
    metrics = [
        ('nodes', f'M{agg_criterion[1:]} Nodes Traversed', 'Node Count [Log Scale]', True),
        ('time_taken (s)', f'M{agg_criterion[1:]} Time Taken (s)', 'Time (s) [Log Scale]', True),
        ('score (max_z)', 'Average Box Height Score (Max Z)', 'Score (Lower is better)', False),
        ('cog_z', 'Average Center of Gravity (Z)', 'CoG Z-Height (Lower is better)', False),
        ('volume_util (%)', 'Average Volume Utilization', 'Volume Utilization(%)', False),
        ('packing_score', 'Average Packing Score', 'Packing Score (0 to 1)', False)
    ]

    # Configuration for the X-axis positioning
    target_ranges = bin_labels
    x_axis = np.arange(len(target_ranges))
    bar_width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for idx, (metric, title, ylabel, log_y) in enumerate(metrics):
        ax = axes[idx]
        
        for i, algo in enumerate(algo_order):
            # Extract data for this algorithm 
            algo_data = agg_df[agg_df['algorithm'] == algo]
            algo_data = algo_data.set_index('boxamt_range').reindex(target_ranges).reset_index()
            y_values = algo_data[metric]
            
            x_pos = x_axis + offsets[i] * bar_width
            bars = ax.bar(x_pos, y_values, bar_width, label=algo, color=colors[i], edgecolor='black')
            
            # Formatter for the top of the bars
            labels = []
            for val in y_values:
                if pd.isna(val) or val == 0:
                    labels.append("")
                elif val < 0.1:
                    labels.append(f"{val:.3f}")
                elif val < 10:
                    labels.append(f"{val:.2f}")
                else:
                    labels.append(f"{val:.1f}")
            
            ax.bar_label(bars, labels=labels, padding=4, fontsize=9, rotation=90)

        # Plot Styling updates
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Box Count Range', fontsize=12)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(target_ranges)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=10)
        
        # Scaling limits to make room for vertical bar labels
        if log_y:
            ax.set_yscale('log')
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top * 10) 
        else:
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top * 1.3)

    # Plot 7
    x_positions = np.arange(len(pivot_df))
    ax7.bar(x_positions, pivot_df['bnb_improvement'], color='#4da6ff', edgecolor='none')
    
    # Add trendline
    if len(pivot_df) > 1 and TESTING_TEST_ORDERS:
        z = np.polyfit(x_positions, pivot_df['bnb_improvement'], 1)
        p = np.poly1d(z)
        ax7.plot(x_positions, p(x_positions), color='red', linewidth=2.5, linestyle='--', label='Trendline')
        ax7.legend(loc='upper left')

    ax7.set_title('BnB Max-Z Reduction vs. BFD per Order, Sorted by Box Count', fontsize=16, fontweight='bold')
    ax7.set_xlabel('Order ID', fontsize=14)
    ax7.set_ylabel('Height Reduction (Max Z Saved)', fontsize=14)
    ax7.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax7.set_axisbelow(True)
    
    # Don't show labels if there are too many bars
    if len(pivot_df) > 50:
        ax7.set_xticks([])
    else:
        ax7.set_xticks(x_positions)
        ax7.set_xticklabels(pivot_df['order_id'], rotation=45, ha='right')

    if SAVE_MODE:
        os.makedirs('./results/figures', exist_ok=True)
        if TESTING_TEST_ORDERS:
            save_filename = 'algorithm_test_orders_per_boxamt_range.png'
        else:
            save_filename = 'algorithm_given_orders_multigraph_per_boxamt_range.png'
        save_path = f'./results/figures/{save_filename}'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved successfully to: {save_path}")
    
    plt.show()


