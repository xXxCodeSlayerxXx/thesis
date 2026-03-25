# %% [markdown]
# #### Imports

# %%
import sys
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

NOTEBOOK_MODE = hasattr(sys, 'ps1')                                             # Detect whether diplay modules are to be loaded for notebook or terminal running (trick from https://stackoverflow.com/questions/1212779/detecting-when-a-python-script-is-being-run-interactively-in-ipython)
if NOTEBOOK_MODE:
    print("Running in notebook mode...")
    from tqdm.notebook import tqdm, trange
else: 
    print("Running in terminal mode...")
    from tqdm import tqdm, trange

# %% [markdown]
# #### Enum Instantiation

# %%
class Criterion(Enum):                                                          # Enum for box sorting criterion options
    LENGTH = "x"
    WIDTH = "y"
    HEIGHT = "z"
    AREA = "a"
    VOLUME = "v"

class Metric(Enum):                                                             # Enum for optimization metric options
    PACKING_SCORE = "ps"
    VOLUME_UTILIZATION = "vu"
    COG_Z = "cogz"
    MAX_Z = "maxz"
    ALL = "all"

class Algorithm(Enum):                                                          # Enum for box placement algorithm options
    RANDOM = "random"
    FFD = "ffd"
    BFD = "bfd"
    BNB = "bnb"

# %% [markdown]
# #### Global Settings

# %%
PALLET_DIMS = (1000, 1400, 1400)                                                # Length, width, height (X, Y, Z, respectively) in mm
DEFAULT_MAX_ATTEMPTS = 2000                                                     # Default cutoff for random attempts to place boxes
DEFAULT_CRITERION = Criterion.VOLUME                                            # Default criterion for box sorting
DEFAULT_OPTIMIZATION_METRIC = Metric.MAX_Z                                      # Default score to optimize best fit algorithms for
DEFAULT_ALGORITHM = Algorithm.BFD                                               # Default box placement algorithm
ML_OBSERVATION_SCALE_FACTOR = 10                                                # Scale factor to scale down height map observations for ML model input
SUPPORTED_AREA_PERCENTAGE = 70                                                  # Percentage of pallet area that must be supported under a box for it to be placed
HOR_ROTATION_ALLOWED_DEFAULT = True                                             # Default value of whether boxes may be rotated horizontally
VER_ROTATION_ALLOWED_DEFAULT = True                                             # Default value of whether boxes may be rotated vertically
BNB_OPTIMALITY_GUARANTEE = False                                                # Disable any code that would sacrifice the guarantee that the BnB algorithm's outcome is the optimal one

# %% [markdown]
# #### Data loading and precomputing

# %%
boxtypes = pd.read_csv("boxtypes.csv")                                          # Load box type dimensions
orders = pd.read_csv("orders.csv")                                              # Load orders data
test_orders = pd.read_csv("orders_test.csv")                                    # Load test orders data

# Turn dataframes into dicts for instant access without searching 
boxtypes_dict = boxtypes.set_index('ID').to_dict('index')

orders_dict = {}
for _, row in orders.iterrows():
    order_id = row['order_id']
    box_list = []
    for i in range(1, 11):
        col = f'amt_{i}'
        if col in row and pd.notna(row[col]):
            box_list.extend([i] * int(row[col]))
    orders_dict[order_id] = box_list

test_orders_dict = {}
for _, row in test_orders.iterrows():
    order_id = row['order_id']
    box_list = []
    for i in range(1, 11):
        col = f'amt_{i}'
        if col in row and pd.notna(row[col]):
            box_list.extend([i] * int(row[col]))
    test_orders_dict[order_id] = box_list

# %% [markdown]
# #### Environment Definition

# %%
class Pallet:
    # Basic functionality methods
    def __init__(self, dims=PALLET_DIMS):                                       # Initialize pallet with given dimensions
        # unpack pallet dimensions
        self.size_x, self.size_y, self.size_z = dims
        # Initialize list to store boxes placed on the pallet
        self.boxes = []
        # Initialize heightmap to track box heights at each (x, y) position
        self.heightmap = np.zeros((self.size_x, self.size_y), dtype=np.int32)
        # Initialize sets candidate coordinates (existing box edges) for all potential placements
        self.candidates_x = {0}
        self.candidates_y = {0}
        # Initialize set of candidate coordinate pairs at corner points to optimize placement (in corners between boxes)
        self.extpts = {(0, 0)}
        # Initialize total occupied volume under heightmap counter (boxes + wasted volume)
        self.heightmap_sum = 0
        # Initialize storage of max Z
        self.max_z = 0

    def reset(self):                                                            # Emtpy the pallet
        # Clear the list of boxes
        self.boxes = []
        # Reset the heightmap to all zeros
        self.heightmap = np.zeros((self.size_x, self.size_y), dtype=np.int32)
        # Reset sets of candidate coordinates
        self.candidates_x = {0}
        self.candidates_y = {0}
        # Reset set of extreme points
        self.extpts = {(0, 0)}
        # Reset occupied volume counter
        self.heightmap_sum = 0
        # Reset storage of max Z
        self.max_z = 0

    # Box placement logic
    def get_in_bounds_status(self, x, y, z):                                    # Check if the given (x, y, z) position is within the pallet boundaries
        return (x <= self.size_x) and (y <= self.size_y) and (z <= self.size_z)
    
    def get_in_box_status(self, x, y, z):                                       # Check if the given (x, y, z) position is inside any box on the pallet
        for box in self.boxes:
            if (box['x'] <= x < box['x'] + box['dx'] and
                box['y'] <= y < box['y'] + box['dy'] and
                box['z'] <= z < box['z'] + box['dz']):
                return True
        return False

    def check_box_placement_validity(self, box_dims, x, y):                     # Check if a box of given dimensions may be placed at position (x, y) on the pallet
        # Unpack box dimensions
        dx, dy, dz = box_dims

        # Get the height the bottom of the box will rest on
        z = self.get_max_height_in_area(x, y, dx, dy)
        if z == -1:
            return False

        # Check if the box fits within the pallet boundaries, or return False
        if not self.get_in_bounds_status(x+dx, y+dy, z+dz):
            return False

        # Check if there is enough support under the box, or return False
        box_area = dx * dy
        area_supported = np.sum(self.heightmap[x:x+dx, y:y+dy] == z)
        support_percentage = (area_supported / box_area) * 100
        if support_percentage < SUPPORTED_AREA_PERCENTAGE:
            return False
        
        return True

    def place_box(self, box_dims, x, y):                                        # Attempt to place a box of given dimensions at (x, y) position on the pallet. Return a delta (dict of information to reverse move) if successful, False if not.
        # Unpack box dimensions and get z value to place it at
        dx, dy, dz = box_dims
        z = self.get_max_height_in_area(x, y, dx, dy)

        # Check box placement prerequisites, fail if not met
        if not self.check_box_placement_validity(box_dims, x, y):
            return False
        
        # Save a 'before' snapshot of changed part of the heightmap for delta
        heightmap_backup = self.heightmap[x:x+dx, y:y+dy].copy()

        # Check if we will be adding new candidate coordinates by placing this box
        adding_candidate_x = (x + dx < self.size_x) and ((x + dx) not in self.candidates_x)
        adding_candidate_y = (y + dy < self.size_y) and ((y + dy) not in self.candidates_y)

        # Compute change in local part of heightmap by placing box
        old_local_sum = int(np.sum(heightmap_backup))
        new_local_sum = dx * dy * (z + dz)
        heightmap_sum_delta = new_local_sum - old_local_sum
        
        # Place the box: update the heightmap and max z (if it is increased) and store the dimensions in the boxlist
        self.heightmap[x:x+dx, y:y+dy] = z + dz
        self.heightmap_sum += heightmap_sum_delta
        self.max_z = max(self.max_z, z + dz)
        self.boxes.append({
            'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy, 'dz': dz
        })

        # Add box edges to candidate coordinate lists if prerequisites are met
        if adding_candidate_x: self.candidates_x.add(x + dx)
        if adding_candidate_y: self.candidates_y.add(y + dy)

        # Save old set of extreme points calculate the post-move set
        old_extpts = self.extpts.copy()
        self.calculate_extreme_points()

        # Put together all the information needed to reverse this move in a delta dict
        delta = {
            'x': x, 
            'y': y,
            'z': z,
            'dx': dx, 
            'dy': dy,
            'dz': dz,
            'heightmap_backup': heightmap_backup,
            'heightmap_sum_delta': heightmap_sum_delta,
            'x_candidate_added': adding_candidate_x,
            'y_candidate_added': adding_candidate_y,
            'added_extpts': self.extpts.difference(old_extpts),
            'removed_extpts': old_extpts.difference(self.extpts)
        }

        # Return delta to indicate successful placement
        return delta
    
    def remove_box(self, delta):                                                # Remove a box based on the information in the given delta dict
        # Load values from delta dict
        x, y, dx, dy = delta['x'], delta['y'], delta['dx'], delta['dy']
        self.heightmap[x:x+dx, y:y+dy] = delta['heightmap_backup']

        # Revert change to heightmap sum
        self.heightmap_sum -= delta['heightmap_sum_delta']

        # Recompute max_z if removed box was the tallest one
        if delta['z'] + delta['dz'] >= self.max_z:
            self.max_z = np.max(self.heightmap)
        
        # Remove the last box off the list
        self.boxes.pop()
        
        # Remove candidate coordinates if they were added by this move
        if delta['x_candidate_added']: self.candidates_x.remove(x + dx)
        if delta['y_candidate_added']: self.candidates_y.remove(y + dy)

        # Revert extreme points set with information from delta: remove added points, and restore removed points
        self.extpts = self.extpts.difference(delta['added_extpts'])
        self.extpts = self.extpts.union(delta['removed_extpts'])

    # Pallet state and analysis methods    
    def get_max_height_in_area(self, x, y, dx, dy):                             # Get the maximum height in a rectangular area of the heightmap
        # Ensure we don't go out of bounds
        x_end = min(x + dx, self.size_x)                                          
        y_end = min(y + dy, self.size_y)

        # Specify the rectangle being checked
        region = self.heightmap[x:x_end, y:y_end]

        # Return -1 (fail) if the region is of size 0 or smaller
        if region.size <= 0:
            return -1
        
        # Otherwise, return the maximum height in the region
        return np.max(region)
    
    def count_boxes(self):                                                      # Return the number of boxes currently on the pallet
        return len(self.boxes)

    def get_area_usage_at_z(self, z):                                           # Return percentage of area used at specific height (z-value)
        # Get total area and initialize accumulator for used area
        total_area = self.size_x * self.size_y
        used_area_acc = 0
        
        # Check every box for intersection at chosen z, if it is, add its area to used area accumulator
        for box in self.boxes:
            box_bottom = box['z']
            box_top = box_bottom + box['dz']

            if box_bottom <= z <= box_top:
                box_area = box['dx'] * box['dy']
                used_area_acc += box_area

        # Return used/total ratio * 100 (percentage)
        return round(used_area_acc / total_area * 100, 2)

    def calculate_extreme_points(self):                                         # Calculate the set of extreme points (corners of the top layer of boxes) for use as candidate placements
        # Start with the origin as an extreme point
        new_extpts = {(0, 0)} 
    
        for box in self.boxes:
            # Allow stacking boxes on top of each other by including the top corners of boxes as extreme points
            new_extpts.add((box['x'], box['y']))

            # Find existing box edges to use as starting points for projections
            x_end = box['x'] + box['dx']
            y_end = box['y'] + box['dy']
            
            # Trace right edge to Y=0 (find max Y of boxes behind it)
            y_projection_dist = 0
            for other in self.boxes:
                # If the other box is strictly behind right edge, and overlaps in X dimension, and is further to the back, overwrite projection distance
                if other['y'] + other['dy'] <= box['y'] and other['x'] < x_end < other['x'] + other['dx']:
                    y_projection_dist = max(y_projection_dist, other['y'] + other['dy'])
            # Add the point at the end of the right edge and its projection back to Y=0 as extreme point
            new_extpts.add((x_end, y_projection_dist))
            
            # Trace front edge to X=0 (find max X of boxes to the left of it)
            x_projection_dist = 0
            for other in self.boxes:
                # If the other box is strictly to the left of front edge, and overlaps in Y dimension, and is further to the left, overwrite projection distance
                if other['x'] + other['dx'] <= box['x'] and other['y'] < y_end < other['y'] + other['dy']:
                    x_projection_dist = max(x_projection_dist, other['x'] + other['dx'])
            # Add the point at the end of the front edge and its projection back to X=0 as extreme point
            new_extpts.add((x_projection_dist, y_end))
        
        # Remove any points that are out of bounds and set as new extreme points
        for x, y, in new_extpts.copy():
            if x < 0 or y < 0 or x > self.size_x or y > self.size_y:
                new_extpts.remove((x, y))
        self.extpts = new_extpts

    # Pallet metrics and visualization methods
    def get_max_height(self):                                                   # Max height of boxes on the pallet, value to optimize for
        return self.max_z

    def get_min_height(self):                                                   # Min height of boxes on the pallet (usually 0 as it is unlikely boxes will cover every mm^2)
        return np.min(self.heightmap)

    def check_order_fullfillment(self, orderID, order_dict):                    # Check if all boxes in an order are placed on the pallet, if not, returns the percentage
        placed_box_count = self.count_boxes()
        required_box_count = len(get_box_list_from_order(orderID, order_dict))
        if placed_box_count >= required_box_count:
            return 100.0
        else:
            return round(placed_box_count / required_box_count * 100, 2)

    def visualize_heightmap(self, title, ax=None):                              # Visualize the heightmap and return as axis object
        # If no axis object is provided, create a new one
        if ax is None:
            fig, ax = plt.subplots()
        
        image = ax.imshow(self.heightmap.T, origin='lower', cmap='viridis', extent=[0, self.size_x, 0, self.size_y])
        ax.figure.colorbar(image, ax=ax, label='Height (mm)')
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        return ax

    def visualize_boxes(self, title, ax=None):                                  # Make 3D plot of pallet and return as axis object
        # If no axis object is provided, create a new one with 3D technology
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        for box in self.boxes:
            # Create a 3D block for each box
            ax.bar3d(box['x'], box['y'], box['z'], box['dx'], box['dy'], box['dz'], alpha=0.7)

        # Set graph size to accurately reflect pallet dimensions
        ax.set_xlim(0, self.size_x)
        ax.set_ylim(0, self.size_y)
        ax.set_zlim(0, self.size_z)

        # Set graph labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        return ax

    def get_volume_utilization(self):                                           # Return percentage of volume (up to max used z) filled with box
        # Get ceiling and volume of considered space
        max_z = self.get_max_height()
        total_volume = self.size_x * self.size_y * max_z
        
        # Sum up the volume of every box on the pallet
        occupied_volume = 0
        for box in self.boxes:
            box_v = box['dx'] * box['dy'] * box['dz']
            occupied_volume += box_v

        return round(occupied_volume / total_volume * 100, 2)

    def get_center_of_gravity_z(self):                                          # Return the height of the average position of box volume on the pallet (as a proxy for mass, which we have no data about)
        # Initialize accumulators for total box volume and the volume-height product
        total_box_volume_acc = 0
        volume_at_height_acc = 0

        for box in self.boxes:
            # Get each box's volume and add to the total
            box_v = box['dx'] * box['dy'] * box['dz']
            total_box_volume_acc += box_v

            # Get each box's CoG by taking the bottom of the box (z) and adding half the box's height, 
            # then add the box's volume times that height to the total
            box_cog_height = box['z'] + (box['dz'] / 2)
            volume_at_height_acc += box_v * box_cog_height

        # Return -1 (fail) if no boxes are detected
        if total_box_volume_acc == 0:
            return -1
        
        # Take total volume component back out of score to get representation of height of center of gravity
        return round(volume_at_height_acc / total_box_volume_acc, 2)

    def get_packing_score(self):                                                # Calculate score (0-1, higher is better) based on per-layer utilization, each level counting less (100% at the bottom to 0% at the top of used space)
        max_height = self.get_max_height()
        if max_height == 0:
            return 0.0

        total_area = self.size_x * self.size_y
        total_weighted_score = 0.0
        total_possible_weight = 0.0

        # Calculate the mathematical contribution of each box
        for box in self.boxes:
            z_bottom = box['z']
            z_top = box['z'] + box['dz']
            box_area = box['dx'] * box['dy']
            
            # The average weight factor for this specific box's height span
            # (1.0 at bottom of pallet, approaching 0.0 at max_height)
            weight_bottom = 1.0 - (z_bottom / max_height)
            weight_top = 1.0 - (z_top / max_height)
            avg_weight = (weight_bottom + weight_top) / 2.0
            
            # Multiply volume by the average weight
            box_volume = box_area * box['dz']
            total_weighted_score += box_volume * avg_weight
            
        # Calculate what the score would be if the pallet was 100% full up to max_height
        total_possible_volume = total_area * max_height
        total_possible_weight = total_possible_volume * 0.5 # Average weight of a full block is 0.5

        if total_possible_weight == 0:
            return 0.0
            
        return round((total_weighted_score / total_possible_weight), 3)

    def get_total_box_volume(self):                                             # Calculate combined volume of all boxes placed on the pallet
        total_box_volume = 0
        for box in self.boxes:
            volume = box['dx'] * box['dy'] * box['dz']
            total_box_volume += volume
        return total_box_volume

    def get_wasted_space(self):                                                 # Calculate space not occupied by boxes under the heightmap
        total_box_volume = self.get_total_box_volume()
        total_occupied_volume = np.sum(self.heightmap)
        wasted_space = total_occupied_volume - total_box_volume
        return wasted_space

    def get_pallet_results(self, algo, orderID, order_dict, print_mode=False, save_mode=False, bnb_stats=None):  # Get metrics for the pallet, optionally display them, and/or save a composite results image to ./results/
        # Get metrics
        fulfillment = self.check_order_fullfillment(orderID, order_dict)
        volume_util = self.get_volume_utilization()
        area_usage_at_z0 = self.get_area_usage_at_z(0)
        cog_z = self.get_center_of_gravity_z()
        packing_score = self.get_packing_score()
        max_z = self.get_max_height()

        # Determine order set used
        if order_dict == orders_dict:
            set_name = "GivenOrders"
            set_name_display = "Given Orders"
        else:
            set_name = "TestOrders"
            set_name_display = "Test Orders"

        # Build composite figure if build or save mode is active
        if print_mode or save_mode:
            fig = plt.figure(figsize=(19, 7))

            # Heightmap panel
            ax1 = fig.add_subplot(1, 3, 1)
            self.visualize_heightmap(
                f"Heightmap\n{algo.value.upper()} | Order {orderID} | {set_name_display}",
                ax=ax1
            )

            # 3D visualization panel
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            self.visualize_boxes(
                f"3D View\n{algo.value.upper()} | Order {orderID}",
                ax=ax2
            )

            # Metrics (text) panel
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.axis('off')
            ax3.set_title("Metrics Summary", fontsize=13, pad=14)

            # Include BnB optimality guarantee tag when relevant
            if algo == Algorithm.BNB:
                optimal_guarantee = bnb_stats.get('optimality_guarantee', None) if bnb_stats else None
                if optimal_guarantee is True:
                    optimal_tag = "ON  (exact)"
                elif optimal_guarantee is False:
                    optimal_tag = "OFF (fast)"
                else:
                    optimal_tag = f"{'ON' if BNB_OPTIMALITY_GUARANTEE else 'OFF'} (global)"
            else:
                optimal_tag = "N/A"

            # Format metrics into neat block of text
            metrics_lines = [
                "  -- Run Information ------------------",
                f"  Algorithm:              {algo.value.upper()}",
                f"  Order ID:               {orderID}",
                f"  Boxes in order:         {len(order_dict[orderID])}",
                f"  Dataset:                {set_name_display}",
                f"  BnB Opt. Guarantee:     {optimal_tag}",
                "",
                "  -- Packing Metrics ------------------",
                f"  Max Z Height:           {max_z}        mm",
                f"  CoG Z-height:           {cog_z}     mm",
                f"  Packing Score:          {packing_score}",
                f"  Order Fulfillment:      {fulfillment}      %",
                f"  Volume Utilization:     {volume_util}      %",
                f"  Area Filled at z=0:     {area_usage_at_z0}      %",
            ]

            # Add BnB pruning statistics if BnB is used
            if algo == Algorithm.BNB and bnb_stats is not None:
                total_pruned = (bnb_stats['pruned_rule1'] + bnb_stats['pruned_rule4'] + bnb_stats['pruned_dedupe'] + bnb_stats['pruned_symbreak'])
                metrics_lines += [
                    "",
                    " --- BnB Search Stats -----------------",
                    f"  Nodes evaluated:        {bnb_stats['nodes']:,}",
                    f"  Nodes pruned by rule:\n"
                    f"  Rule 1 (trivial):       {bnb_stats['pruned_rule1']:,}",
                    f"  Rule 4 (tall-low):      {bnb_stats['pruned_rule4']:,}",
                    f"  Deduplication:          {bnb_stats['pruned_dedupe']:,}",
                    f"  Symmetry breaking       {bnb_stats['pruned_symbreak']:,}",
                    "",
                    f"  Total pruned:           {total_pruned:,}",
                ]

            # Join all lines of metrics text into one variable
            metrics_text = "\n".join(metrics_lines)

            # Add metrics text to panel 3 in a text box
            ax3.text(
                0.05, 0.56, metrics_text,
                transform=ax3.transAxes,
                fontsize=10.5,
                verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.7',
                    facecolor='#f0f4f8',
                    edgecolor="#4aa56d",
                    linewidth=1.8
                )
            )

            fig.suptitle(
                f"Pallet Results:  {algo.value.upper()}  |  Order {orderID}  |  {set_name_display}",
                fontsize=13, fontweight='bold', y=1.01
            )
            plt.tight_layout()

            # Save the figure to ./results/ with run details in filename
            if save_mode:
                import os
                os.makedirs("./results", exist_ok=True)
                opt_suffix = ""
                if algo == Algorithm.BNB and bnb_stats is not None:
                    g = bnb_stats.get('optimality_guarantee', None)
                    opt_suffix = f"_opt{'ON' if g else 'OFF'}"
                filename = (
                    f"{algo.value.upper()}{opt_suffix}"
                    f"_{set_name}"
                    f"_order{orderID}"
                    f"_ff{int(fulfillment)}"
                    f"_vu{volume_util}"
                    f"_maxz{max_z}"
                    ".png"
                )
                filepath = os.path.join("./results", filename)
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved results at: {filepath}")

            # Show the figure in print mode
            if print_mode:
                plt.show()
            else:
                plt.close(fig)

        # Return metrics (with BnB stats if BnB was run and stats are available)
        if algo == Algorithm.BNB and bnb_stats is not None:
            return (fulfillment, volume_util, area_usage_at_z0, cog_z, packing_score, max_z, bnb_stats)
        return (fulfillment, volume_util, area_usage_at_z0, cog_z, packing_score, max_z)

    def simulate_placement(self, x, y, box_dims, requested_metric=Metric.ALL):  # Simulate a box placement on a pallet and return either a specific metric or a dictionary of all metrics from the result
        # Place the box and get a delta to reverse the move after scoring
        delta = self.place_box(box_dims, x, y)
        
        # Return -1 if move is impossible
        if delta == False:
            return -1 
        
        # Calculate requested metric(s)
        if requested_metric == Metric.PACKING_SCORE:
            score = self.get_packing_score()
        elif requested_metric == Metric.VOLUME_UTILIZATION:
            score = self.get_volume_utilization()
        elif requested_metric == Metric.COG_Z:
            score = self.get_center_of_gravity_z()
        elif requested_metric == Metric.MAX_Z:
            score = self.get_max_height()
        else:
            score = {
                "ps": self.get_packing_score(),
                "vu": self.get_volume_utilization(),
                "cogz": self.get_center_of_gravity_z(),
                "maxz": self.get_max_height()
            }
            
        # Undo move to preserve original pallet state
        self.remove_box(delta)
        
        return score


# %% [markdown]
# #### Helper Functions

# %%
def get_box_properties_from_id(boxid):                                                                                                  # Retrieve length (x), width (y), height (z), area (a) and volume (v) of a box from its ID
    box = boxtypes_dict[boxid]
    
    x = box['LENGTH']
    y = box['WIDTH']
    z = box['HEIGHT']
    
    a = x * y
    v = a * z

    return x, y, z, a, v

def get_box_list_from_order(orderid, order_dict=orders_dict):                                                                           # Retrieve list of box IDs from a given order ID
    return order_dict.get(orderid, [])

def sort_box_list_by_size(box_list, criterion=DEFAULT_CRITERION, invert=False):                                                         # Return a list of box IDs sorted by size (default = largest to smallest). Arguments: sortby to choose sorting criterion (length ("x"), width ("y"), height ("z"), area ("a"), volume ("v")), invert to sort smallest to largest, 
    # The get_box_properties_from_id function returns (x, y, z, a, v), 
    # so map sortby to the correct index of those outputs
    criterion_to_index_dict = {
        "x": 0, "length": 0,
        "y": 1, "width": 1,
        "z": 2, "height": 2,
        "a": 3, "area": 3,
        "v": 4, "volume": 4
    }

    # Extract string value if it's an Enum
    if isinstance(criterion, Enum):
        criterion_value = criterion.value
    else:
        criterion_value = criterion

    # Make sure no bogus value is used
    if criterion_value not in criterion_to_index_dict:
        raise ValueError("Invalid sortby value. Use 'x', 'y', 'z', 'a', or 'v'.")

    # Sort the box list based on the chosen criterion with lambda function
    sorted_list = sorted(
        box_list, 
        key=lambda boxid: get_box_properties_from_id(boxid)[criterion_to_index_dict[criterion_value]], 
        reverse=not invert
    )
    
    return sorted_list
    
def process_order(order, algo, max_attempts=DEFAULT_MAX_ATTEMPTS, criterion=DEFAULT_CRITERION, metric=DEFAULT_OPTIMIZATION_METRIC, order_dict=orders_dict, leave_tqdm=True, optimality_guarantee=None):     # Process a given order using the specified algorithm, max attempts, and sorting criterion. For BnB, returns resultant pallet and node+pruning stats, returns resultant pallet only for other algos
    box_list = get_box_list_from_order(order, order_dict)
    pallet = Pallet()

    if algo == Algorithm.RANDOM:
        place_box_list_random(pallet, box_list, max_attempts)
        return pallet

    elif algo == Algorithm.FFD:
        place_box_list_first_fit_decreasing(pallet, box_list, criterion=criterion)
        return pallet

    elif algo == Algorithm.BFD:
        place_box_list_best_fit_decreasing(pallet, box_list, criterion=criterion, opt_metric=metric)
        return pallet

    elif algo == Algorithm.BNB:
        bnb_stats = place_box_list_branch_and_bound(pallet, box_list, criterion=criterion, opt_metric=metric, leave_tqdm=leave_tqdm, optimality_guarantee=optimality_guarantee)
        return pallet, bnb_stats

def get_box_orientations(dx, dy, dz, rot_h=HOR_ROTATION_ALLOWED_DEFAULT, rot_v=VER_ROTATION_ALLOWED_DEFAULT):                           # Get a list of possible orientations for a box with given dimensions, based on allowed rotations
    # initialize list of valid orientations with the original orientation
    orientations = [(dx, dy, dz)]

    # If horizontal rotation is allowed, add the orientation with length and width swapped
    if rot_h:
        orientations.append((dy, dx, dz))

    # If vertical rotation is allowed, add orientations with length and height swapped, and width and height swapped
    if rot_v:
        orientations.append((dx, dz, dy))  # Width and height swapped
        orientations.append((dz, dy, dx))  # Length and height swapped

        # If both horizontal and vertical rotation are allowed, also add the orientations with both rotations applied
        if rot_h:
            orientations.append((dz, dx, dy))  # Length and height swapped, then horizontal rotation
            orientations.append((dy, dz, dx))  # Width and height swapped, then horizontal rotation
    
    return list(set(orientations))  # Remove duplicates if any rotations result in the same orientation

def calculate_box_volumes_list(box_list):                                                                                               # Get a list of volume values corresponding to a given box list
    # Initialize list, append every volume value in the box list to it and return
    box_volumes = []
    for box in box_list:
        _, _, _, _, v = get_box_properties_from_id(box)
        box_volumes.append(v)
    return box_volumes

def calculate_min_height_footprint_list(box_list):                                                                                      # Get a list of footprint areas for each box in the list. The minimum height footprint is the box's largest face area, i.e. the area of the box oriented in such a way that it contibutes to the z-dimension the least
    footprints = []
    for boxid in box_list:
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)
        min_footprint = max(dx*dy, dx*dz, dy*dz)
        footprints.append(min_footprint)
    return footprints

def calculate_footprint_to_go_dict(box_list):                                                                                           # From a box list, calculate a dict of box index to total minimum height footprint area still to place (sum of largest face of each remaining box)
    footprints = calculate_min_height_footprint_list(box_list)
    total_footprint = sum(footprints)
    footprint_to_go_dict = {}
    cumulative_counter = 0
    for i in range(len(box_list)):
        footprint_to_go_dict[i] = total_footprint - cumulative_counter
        cumulative_counter += footprints[i]
    return footprint_to_go_dict

def calculate_cumulative_volume_dicts(box_list):                                                                                        # From a box list, calculate a dict of box index to cumulative volume placed and a dict of box index to volume still to place
    # Initialize box volume list and dicts, count total box volume and initialize running counter
    box_volumes = calculate_box_volumes_list(box_list)
    cumulative_volume_dict = {}
    volume_to_go_dict = {}
    total_volume = sum(box_volumes)
    cumulative_counter = 0
    
    # Fill dicts with values while keeping count of processed volume
    for i, boxid in enumerate(box_list):
        cumulative_volume_dict[i] = cumulative_counter
        volume_to_go_dict[i] = total_volume - cumulative_counter
        cumulative_counter += box_volumes[i]
        
    return cumulative_volume_dict, volume_to_go_dict

# %% [markdown]
# #### Box Placing Algorithms

# %%
def place_random_boxes(pallet, num_boxes, box_size_range):                                                                          # Place a number of randomly sized boxes on the pallet
    for _ in range(num_boxes):
        # Generate random box dimensions within the specified range
        dx = np.random.randint(box_size_range[0], box_size_range[1])
        dy = np.random.randint(box_size_range[0], box_size_range[1])
        dz = np.random.randint(box_size_range[0], box_size_range[1])

        # Generate random (x, y) position for the box
        x = np.random.randint(0, pallet.size_x)
        y = np.random.randint(0, pallet.size_y)

        # Attempt to place the box on the pallet
        pallet.place_box((dx, dy, dz), x, y)

def place_box_list_random(pallet, box_list, max_attempts):                                                                          # Place boxes from a given box list randomly on the pallet
    for boxid in box_list:
        # Get box dimensions
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)

        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            # Generate random (x, y) position for the box
            x = np.random.randint(0, pallet.size_x)
            y = np.random.randint(0, pallet.size_y)

            # Attempt to place the box on the pallet
            placed = pallet.place_box((dx, dy, dz), x, y)
            attempts += 1

def place_box_list_first_fit_decreasing(pallet, box_list, criterion=DEFAULT_CRITERION):                                             # Baseline (naive) algorithm: place boxes from a box list on the pallet, biggest boxes (by criterion x, y, z, a, or v) first, with lowest x (then y) values possible
    # Sort the box list by size (largest to smallest)
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)

    for boxid in tqdm(sorted_box_list, leave=False):
        # Get base dimensions
        original_dx, original_dy, original_dz, _, _ = get_box_properties_from_id(boxid)
        
        # Get all valid orientations for this box
        orientations = get_box_orientations(original_dx, original_dy, original_dz)

        placed = False

        # Sort candidates to ensure we pack from (0,0) outwards
        cand_x = sorted(list(pallet.candidates_x))
        cand_y = sorted(list(pallet.candidates_y))

        # Try to place box at all (x, y) candidates, and stop once the first one is found
        for x in cand_x:
            if placed: break
            for y in cand_y:
                if placed: break
                for dims in orientations:
                    if pallet.place_box(dims, x, y):
                        placed = True
                        break

def place_box_list_best_fit_decreasing(pallet, box_list, criterion=DEFAULT_CRITERION, opt_metric=DEFAULT_OPTIMIZATION_METRIC):      # Baseline (naive) algorithm: place boxes from a box list on the pallet, finding the best place for the boxes based on best value of optimization metric after placement
    # Sort the box list by size (largest to smallest)
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)

    # Define which metrics should be maximized vs minimized (higher is better vs lower is better)
    maximize_metrics = [Metric.PACKING_SCORE, Metric.VOLUME_UTILIZATION]         
    minimize_metrics = [Metric.COG_Z, Metric.MAX_Z]

    for boxid in tqdm(sorted_box_list, desc="Placing boxes (Best Fit)", leave=False):
        # Get base dimensions
        original_dx, original_dy, original_dz, _, _ = get_box_properties_from_id(boxid)
        
        # Get all valid orientations for this box
        orientations = get_box_orientations(original_dx, original_dy, original_dz)

        # Initialize best score trackers for this specific box
        if opt_metric in maximize_metrics:
            best_score = -1                     # Worse than any possible score as minimum for both maximization metrics is 0, making -1 a safe worst score.
        else:
            best_score = PALLET_DIMS[2] + 100   # Worse than any possible score as maximum for both minimization metrics is max z height.
        
        # Initialize storage for best placement found for this box and whether one has been found
        best_placement = None
        found_valid_move = False

        # Sort candidates to ensure we check from (0,0) outwards
        cand_x = sorted(list(pallet.candidates_x))
        cand_y = sorted(list(pallet.candidates_y))

        # Check all possible placements
        for dims in orientations:
            for x in cand_x:
                for y in cand_y:
                    # Precheck conditions for place_box() before running it to save compute
                    # If prechecks pass, simulate the placement compare the resulting metric score without affecting the real pallet state
                    if pallet.check_box_placement_validity(dims, x, y):
                        # Calculate the score based on the chosen metric
                        score = pallet.simulate_placement(x, y, dims, opt_metric)

                        # Compare score against best found so far
                        if opt_metric in maximize_metrics:
                            if score > best_score:
                                best_score = score
                                best_placement = (dims, x, y)
                                found_valid_move = True
                        else: # Minimize metrics
                            if score < best_score:
                                best_score = score
                                best_placement = (dims, x, y)
                                found_valid_move = True

        # After finding best placement for box, place it on the real pallet if a valid move was found
        if found_valid_move and best_placement:
            b_dims, b_x, b_y = best_placement
            pallet.place_box(b_dims, b_x, b_y)
        else:
            print(f"Could not place box {boxid} anywhere.")
            pass

def place_box_list_branch_and_bound(pallet, box_list, criterion=DEFAULT_CRITERION, opt_metric=DEFAULT_OPTIMIZATION_METRIC, leave_tqdm=True, optimality_guarantee=None):   # Recursive backtracking algorithm to find the optimal placement of boxes from a box list on the pallet based on optimization metric, with pruning based on bounding functions
    # Sort the box list by size (largest to smallest)
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)

    # Check if optimality guarantee was overridden and use override if so, otherwise use global default
    use_guarantee = BNB_OPTIMALITY_GUARANTEE if optimality_guarantee is None else optimality_guarantee

    # Define which metrics should be maximized vs minimized (higher is better vs lower is better)
    maximize_metrics = [Metric.PACKING_SCORE, Metric.VOLUME_UTILIZATION]         
    minimize_metrics = [Metric.COG_Z, Metric.MAX_Z]

    # Initialize non-tqdm counters for nodes and pruning for metrics extraction
    count_nodes    = 0
    count_bound1   = 0
    count_bound4   = 0
    count_dedupe   = 0
    count_symbreak = 0
    count_hm_rec = 0

    # Initialize best score tracker for the entire placement
    # For MAX_Z, run the BFD algorithm first to get a decent initial score to beat, making pruning more aggressive.
    if opt_metric == Metric.MAX_Z:
        temp_pallet = Pallet()
        place_box_list_best_fit_decreasing(temp_pallet, sorted_box_list, criterion=criterion)
        best_score = temp_pallet.get_max_height() + 1 # Make sure BnB algorithm doesn't fail catastrophically if unable to find a better solution than BFD
    elif opt_metric in maximize_metrics:
        best_score = -1                     # Worse than any possible score as minimum for both maximization metrics is 0
    elif opt_metric in minimize_metrics:
        best_score = PALLET_DIMS[2] + 100   # Worse than any possible score as maximum for both minimization metrics is max z height.

    # Initialize a list of deltas for the current sequence of placements to traverse both ways
    current_sequence = []
    # Initialize storage for the best sequence of placements found
    best_sequence = None

    # Precalculate volume and footprint to go at each box index for bounding case 2 [DEPRECATED]
    #_, volume_to_go_dict = calculate_cumulative_volume_dicts(sorted_box_list)
    #footprint_to_go_dict = calculate_footprint_to_go_dict(sorted_box_list)
    #pallet_area = pallet.size_x * pallet.size_y

    # Keep global dict of seen heightmap hashes keyed by tree depth for heightmap recognition pruning
    # seen_heightmaps_by_depth_dict = {}

    # Make list of box dimensions tuples for symmetry breaking rule
    dimension_tuples = []
    for boxid in sorted_box_list:
        dimension_tuples.append(tuple(sorted(get_box_properties_from_id(boxid)[:3])))

    # Precalculate box orientations for this order
    box_orientations_dict = {}
    for boxid in set(sorted_box_list):
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)
        box_orientations_dict[boxid] = get_box_orientations(dx, dy, dz)

    # Precalculate tallest remaining box orientations for bounding rule 4
    tallest_remaining_orientations = []
    for i in range(len(sorted_box_list)):
        tallest_remaining_id = max(
            sorted_box_list[i:],
            key=lambda bid: min(get_box_properties_from_id(bid)[:3])
        )
        tallest_remaining_orientations.append(box_orientations_dict[tallest_remaining_id])

    def recursive_place(box_index):         # Define recursive function to place boxes one by one and prune as needed
        # Take the best_score and best_sequence variables, along with the node and pruning counters into
        # local scope of the recursive function so they can be updated without having to make them global
        nonlocal best_score, best_sequence, count_nodes, count_bound1, count_bound4, count_dedupe, count_symbreak, count_hm_rec

        # BASE CASE: if we've placed all boxes, evaluate score and update best if needed
        if box_index == len(sorted_box_list):
            if opt_metric == Metric.PACKING_SCORE:
                current_score = pallet.get_packing_score()
            elif opt_metric == Metric.VOLUME_UTILIZATION:
                current_score = pallet.get_volume_utilization()
            elif opt_metric == Metric.COG_Z:
                current_score = pallet.get_center_of_gravity_z()
            elif opt_metric == Metric.MAX_Z:
                current_score = pallet.get_max_height()

            # Update best score and sequence if this is the best found so far
            is_better = (opt_metric in maximize_metrics and current_score > best_score) or (opt_metric in minimize_metrics and current_score < best_score)
            if is_better:
                #print(f"New best score: {current_score}")
                best_score = current_score
                best_sequence = list(current_sequence)
            return
        
        # BOUNDING CASES 
        # MAX_Z bounding cases
        if opt_metric == Metric.MAX_Z:
            # Bounding rule 1 (trivial rule): Calculate if the current partial state is already worse than the best known full state and prune if so
            if pallet.get_max_height() >= best_score:
                counter_bound1.update(1)
                count_bound1 += 1
                return

            # Bounding rule 2 (volume bound) [DEPRECATED]: Check if remaining box volume fits under current max z. If it doesn't, check how much max z would be raised by adding the volume over the remaining box footprints (flat box packing) or the total pallet area, whichever is lowest. If the max z is raised over the current best, prune branch
            # volume_to_go = volume_to_go_dict[box_index]
            # current_max = pallet.get_max_height()
            # free_volume_under_current_max = current_max * pallet_area - pallet.heightmap_sum
            # if volume_to_go > free_volume_under_current_max:
            #     overflow = volume_to_go - free_volume_under_current_max
            #     footprint_to_go = footprint_to_go_dict[box_index]
            #     effective_footprint = min(footprint_to_go, pallet_area)
            #     lower_bound = current_max + overflow / effective_footprint
            #     if lower_bound >= best_score:
            #         counter_bound2.update(1)
            #         count_bound2 += 1
            #         return

            # Bounding rule 3 (look-ahead bound) [DEPRECATED]: Check one box ahead to see if the best-case placement does not push the height over best_score
            # next_boxid = sorted_box_list[box_index]
            # next_dx, next_dy, next_dz, _, _ = get_box_properties_from_id(next_boxid)
            # best_case_next_height = pallet.get_min_height() + min(next_dx, next_dy, next_dz)
            # if best_case_next_height >= best_score:
            #     counter_bound3.update(1)
            #     return
            
            # Bounding rule 4 (tallest-lowest): Check if tallest remaining box placed at its lowest possible candidate position exceeds best_score
            t_orientations = tallest_remaining_orientations[box_index]

            min_landing_z = PALLET_DIMS[2] + 100
            for x, y in pallet.extpts:
                for dims in t_orientations:
                    if pallet.check_box_placement_validity(dims, x, y):
                        landing_z = pallet.get_max_height_in_area(x, y, dims[0], dims[1])
                        min_landing_z = min(min_landing_z, landing_z + dims[2])

            if PALLET_DIMS[2] + 100 > min_landing_z >= best_score:
                counter_bound4.update(1)
                count_bound4 += 1
                return

        # BRANCHING CASE: try to place the next box in all possible orientations and positions, and recursively place the next box after each valid placement
        boxid = sorted_box_list[box_index]
        orientations = box_orientations_dict[boxid]

        # Initialize set (every member must be unique) of seen profiles for deduplication rule
        seen_profiles = set() if not use_guarantee else None

        # Check if the current box matches the dimensions of the previous box and record its dimensions and placed position as a comparison key for symmetry breaking rule
        if dimension_tuples[box_index] == dimension_tuples[box_index - 1]:
            symbreak_key = current_sequence[-1]   # ((dx, dy, dz), x, y) of the previous identical box
        else:
            symbreak_key = None

        # Iterate through sorted extreme points and orientations
        sorted_extpts = sorted(pallet.extpts)
        for dims in orientations:
            for x, y in sorted_extpts:
                # Check whether this placement is valid
                if pallet.check_box_placement_validity(dims, x, y):

                    # Symmetry breaking rule: if a comparison key is registered and the candidate placement is smaller than the key (checked value by value in the tuples ((dx, dy, dz), x, y) ), prune branch as it would lead to an identical resultant pallet in terms of dimensions but with different boxes (of the same or different box IDs, like a type 8 or 10 box, which are dimensionally identical) occupying the same place.
                    if symbreak_key is not None and symbreak_key > (dims, x, y):
                        counter_symbreak.update(1)
                        count_symbreak += 1
                        continue

                    # Deduplication rule: if optimality need not be guaranteed, check if placing the box in the same orientation at another extreme point leads to the same resultant z-height. This is likely to lead to a very similar heightmap structure, so prune all these candidate branches except for one representative. 
                    if not use_guarantee:
                        z = pallet.get_max_height_in_area(x, y, dims[0], dims[1])
                        profile_key = (dims, z)
                        if profile_key in seen_profiles:
                            counter_dedupe.update(1)
                            count_dedupe += 1
                            continue
                        seen_profiles.add(profile_key)

                    delta = pallet.place_box(dims, x, y)
                    if not delta:
                        continue

                    # Heightmap recognition rule [DEPRECATED]: if current heightmap has already been seen at this search depth (no matter on which branch), prune branch as it will lead to copied search space down-tree
                    # heightmap_hash = hash(pallet.heightmap.tobytes())
                    # hashes_at_current_depth = seen_heightmaps_by_depth_dict.setdefault(box_index, set())
                    # if heightmap_hash in hashes_at_current_depth:
                    #     pallet.remove_box(delta)
                    #     count_hm_rec += 1
                    #     counter_hm_rec.update(1)
                    #     continue
                    # # If hash has not been seen at this depth, add it to the set of seen hashes at this depth
                    # hashes_at_current_depth.add(heightmap_hash)

                    # If branch has not been pruned, add placement to sequence and recurse
                    current_sequence.append((dims, x, y))
                    pbar.update(1)
                    count_nodes += 1
                    recursive_place(box_index + 1)
                    current_sequence.pop()
                    pallet.remove_box(delta)
    
    # Start the recursive search starting with the first box (index 0) and keep track of branches and bounds
    pbar             = tqdm(desc="Evaluating Placements", unit=" nodes", leave=leave_tqdm, total=1766249) # Total is set to previous best to estimate time
    counter_bound1   = tqdm(desc="Number of branches pruned by bounding rule 1", unit=" prunes", leave=leave_tqdm)
    #counter_bound2  = tqdm(desc="Number of branches pruned by bounding rule 2", unit=" prunes", leave=leave_tqdm)
    #counter_bound3  = tqdm(desc="Number of branches pruned by bounding rule 3", unit=" prunes", leave=leave_tqdm)
    counter_bound4   = tqdm(desc="Number of branches pruned by bounding rule 4", unit=" prunes", leave=leave_tqdm)
    counter_dedupe   = tqdm(desc="Number of branches pruned by deduplication rule", unit=" prunes", leave=leave_tqdm)
    counter_symbreak = tqdm(desc="Number of branches pruned by symmetry breaking rule",unit=" prunes", leave=leave_tqdm)
    #counter_hm_rec   = tqdm(desc="Number of branches pruned by heightmap recognition rule",unit=" prunes", leave=leave_tqdm)
    recursive_place(0)

    # Reconstruct the optimal pallet state using the best sequence found
    if best_sequence:
        for dims, x, y in best_sequence:
            pallet.place_box(dims, x, y)
    else:
        print("Branch and Bound failed. No valid placements found.")

    # Put node and pruning stats into a dict and return for metrics extraction
    bnb_stats = {
        'nodes':                count_nodes,
        'pruned_rule1':         count_bound1,
        'pruned_rule4':         count_bound4,
        'pruned_dedupe':        count_dedupe,
        'pruned_symbreak':      count_symbreak,
        'best_score':           best_score,
        'optimality_guarantee': use_guarantee,
    }
    return bnb_stats


# %% [markdown]
# #### Testing Functions

# %%
def plot_random_performance(trials, max_max_attempts, step):                    # Plot performance of random box placement over multiple trials. Arguments: trials - amount of runs per step, max_max_attempts - maximum max_attempts value to try, step - step size between max_attempts values
    # Final lists for plotting, max_attempts values on x axis, average fullfilment on y-axis
    x_axis = [] 
    y_axis = []

    # Number of orders to cycle through
    number_of_orders = 40

    # Use max and step size to get final range of max_attempts values (x-axis)
    max_attempts_range = range(step, max_max_attempts + 1, step)

    for max_attempts in tqdm(max_attempts_range, desc="Testing max_attempts values", position=0, leave=True):
        # Keep track of scores for this max_attempts value
        scores = []

        for i in trange(trials, desc=f"Running trials (max_attempts={max_attempts})...", position=1, leave=False):
            # Pick different order every time (round robin) and get its box_list
            order_id = (i % number_of_orders) + 1
            box_list = get_box_list_from_order(order_id)

            # Initialize pallet and fill it up with the random algorithm
            pallet = Pallet()
            place_box_list_random(pallet, box_list, max_attempts)

            # Get fulfillment score (and change from True to 100 if all boxes are accounted for)
            score = pallet.check_order_fullfillment(order_id)
            if score == True:
                score = 100
            
            # Add score to scores list
            scores += [score]

        # Average the scores for this max_attempts value
        avg_score = np.mean(scores)

        # Add the datapoints to the final lists
        x_axis += [max_attempts]
        y_axis += [avg_score]

    # Plot the results
    plt.figure()
    plt.plot(x_axis, y_axis)

    plt.title(f"Random placement algorithm fulfillment scores ({trials} trials/step)")
    plt.xlabel("Maximum attempts per box")
    plt.ylabel('Average order fulfillment percentage')
    plt.ylim(0, 105)

    plt.show()

def run_optimality_guarantee_test(start_order=1, end_order=None, order_dict=test_orders_dict, criterion=DEFAULT_CRITERION, metric=Metric.MAX_Z, print_pallets=False, save_pallets=False):    # Run BnB with and without the optimality guarantee on a range of orders and export a comparison CSV
    # If no final order is specified, let the final order be the last entry in the dict
    if end_order is None:
        end_order = max(order_dict.keys())

    # Get the order IDs to test between the start and end order IDs
    order_ids = [order_id for order_id in sorted(order_dict.keys()) if start_order <= order_id <= end_order]

    result_rows = []

    for order_id in tqdm(order_ids, desc="Testing orders", unit=" orders"):
        print(f"\n----------------------------------------------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"Running order {order_id} with guarantee off...")

        # Run order with no optimality guarantee and get stats
        pallet_no_guarantee = Pallet()
        box_list = get_box_list_from_order(order_id, order_dict)
        stats_no_guarantee = place_box_list_branch_and_bound(
            pallet_no_guarantee, box_list,
            criterion=criterion, opt_metric=metric,
            leave_tqdm=False, optimality_guarantee=False
        )
        score_no_guarantee = pallet_no_guarantee.get_max_height() if metric == Metric.MAX_Z else None
        pallet_no_guarantee.get_pallet_results(algo=Algorithm.BNB, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=stats_no_guarantee)

        print(f"\n----------------------------------------------------------------------------------------------------------------------------")
        print(f"Running order {order_id} with guarantee on...")

        # Run order with optimality guarantee and get stats
        pallet_with_guarantee = Pallet()
        stats_with_guarantee= place_box_list_branch_and_bound(
            pallet_with_guarantee, box_list,
            criterion=criterion, opt_metric=metric,
            leave_tqdm=False, optimality_guarantee=True
        )
        score_with_guarantee = pallet_with_guarantee.get_max_height() if metric == Metric.MAX_Z else None
        pallet_no_guarantee.get_pallet_results(algo=Algorithm.BNB, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=stats_with_guarantee)

        # Calculate differences, absolute and relative
        def calculate_row_difference(val_no_guarantee, val_with_guarantee):
            abs_diff = abs(val_no_guarantee - val_with_guarantee)
            factor   = round(val_no_guarantee / val_with_guarantee, 4) if val_with_guarantee != 0 else "N/A" # Avoid division by 0
            return abs_diff, factor

        nodes_diff,    nodes_factor    = calculate_row_difference(stats_no_guarantee['nodes'],           stats_with_guarantee['nodes'])
        r1_diff,       r1_factor       = calculate_row_difference(stats_no_guarantee['pruned_rule1'],    stats_with_guarantee['pruned_rule1'])
        r4_diff,       r4_factor       = calculate_row_difference(stats_no_guarantee['pruned_rule4'],    stats_with_guarantee['pruned_rule4'])
        dd_diff,       dd_factor       = calculate_row_difference(stats_no_guarantee['pruned_dedupe'],   stats_with_guarantee['pruned_dedupe'])
        sb_diff,       sb_factor       = calculate_row_difference(stats_no_guarantee['pruned_symbreak'], stats_with_guarantee['pruned_symbreak'])

        result_rows.append({
            'order_id':                 order_id,
            # Packing result
            'score_no_guarantee':       score_no_guarantee,
            'score_with_guarantee':     score_with_guarantee,
            # Nodes
            'nodes_no_guarantee':       stats_no_guarantee['nodes'],
            'nodes_with_guarantee':     stats_with_guarantee['nodes'],
            'nodes_diff_abs':           nodes_diff,
            'nodes_diff_factor':        nodes_factor,
            # Pruned by rule 1
            'r1_no_guarantee':          stats_no_guarantee['pruned_rule1'],
            'r1_with_guarantee':        stats_with_guarantee['pruned_rule1'],
            'r1_diff_abs':              r1_diff,
            'r1_diff_factor':           r1_factor,
            # Pruned by rule 4
            'r4_no_guarantee':          stats_no_guarantee['pruned_rule4'],
            'r4_with_guarantee':        stats_with_guarantee['pruned_rule4'],
            'r4_diff_abs':              r4_diff,
            'r4_diff_factor':           r4_factor,
            # Pruned by deduplication
            'dedupe_no_guarantee':      stats_no_guarantee['pruned_dedupe'],
            'dedupe_with_guarantee':    stats_with_guarantee['pruned_dedupe'],
            'dedupe_diff_abs':          dd_diff,
            'dedupe_diff_factor':       dd_factor,
            # Pruned by symmetry breaking
            'symbreak_no_guarantee':    stats_no_guarantee['pruned_symbreak'],
            'symbreak_with_guarantee':  stats_with_guarantee['pruned_symbreak'],
            'symbreak_diff_abs':        sb_diff,
            'symbreak_diff_factor':     sb_factor,
        })

        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"Score  : {score_no_guarantee} (no guarantee) vs {score_with_guarantee} (with guarantee)")
        print(f"Nodes  : {stats_no_guarantee['nodes']:,} vs {stats_with_guarantee['nodes']:,}  (Difference: {nodes_diff:+,}, Factor: {nodes_factor})")

    # Export to CSV
    output_csv=f"./results/bnb_comparisons/bnb_guarantee_comparison_{start_order}_to_{end_order}.csv"
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\n----------------------------------------------------------------------------------------------------------------------------")
    print(f"----------------------------------------------------------------------------------------------------------------------------")
    print(f"Results saved to: {output_csv}")
    return results_df

# %% [markdown]
# #### Testing Area

# %%
current_order_dict = test_orders_dict
current_orderID = 25
current_algo = Algorithm.BNB
current_criterion = Criterion.VOLUME
current_metric = Metric.MAX_Z

if NOTEBOOK_MODE == True:
    if current_algo == Algorithm.BNB:
        testpallet, bnb_stats = process_order(current_orderID, algo=current_algo, criterion=current_criterion, order_dict=current_order_dict, metric=current_metric)
        testpallet.get_pallet_results(current_algo, current_orderID, current_order_dict, print_mode=True, bnb_stats=bnb_stats)
    else:
        testpallet = process_order(current_orderID, algo=current_algo, criterion=current_criterion, order_dict=current_order_dict, metric=current_metric)
        testpallet.get_pallet_results(current_algo, current_orderID, current_order_dict, print_mode=True)
else:
    run_optimality_guarantee_test(start_order=1800, end_order=1800, order_dict=test_orders_dict, print_pallets=True, save_pallets=False)



