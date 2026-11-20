# %% [markdown]
# #### Imports

# %%
# Pin BLAS/OpenMP thread pools to 1 before importing NumPy so spawning up to
# 128 Python worker processes never causes internal thread oversubscription.
import os
for _blas_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_env_var, "1")

import sys
import time
import math
import heapq
import random
import signal
import hashlib
import traceback
import numpy as np
import pandas as pd
from enum import Enum
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
from operator import itemgetter
from multiprocessing.connection import wait as mp_wait

NOTEBOOK_MODE = hasattr(sys, 'ps1')                                             # Detect whether diplay modules are to be loaded for notebook or terminal running (trick from https://stackoverflow.com/questions/1212779/detecting-when-a-python-script-is-being-run-interactively-in-ipython)
if NOTEBOOK_MODE:
    print("Running in notebook mode...")
    from tqdm.notebook import tqdm, trange
else: 
    print("Running in terminal mode...")
    from tqdm import tqdm, trange

# %% [markdown]
# #### Instantiations & Categorizations

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

_BNB_MC_CONTEXT = None                                                          # Read-only context inherited copy-on-write by forked BnB workers
_FIRST_ITEM = itemgetter(0)

MAXIMIZE_METRICS = [Metric.PACKING_SCORE, Metric.VOLUME_UTILIZATION]            # Define which metrics should be maximized vs minimized (higher is better vs lower is better)
MINIMIZE_METRICS = [Metric.COG_Z, Metric.MAX_Z]

# %% [markdown]
# #### Global Settings

# %%
DEFAULT_MP_CORES                = max(2, os.cpu_count() or 2)                   # Default number of cores to use for multiprocessing tasks
PALLET_DIMS                     = (1000, 1400, 1400)                            # Length, width, height (X, Y, Z, respectively) in mm
DEFAULT_CRITERION               = Criterion.VOLUME                              # Default criterion for box sorting
DEFAULT_OPTIMIZATION_METRIC     = Metric.MAX_Z                                  # Default score to optimize best fit algorithms for
DEFAULT_ALGORITHM               = Algorithm.BFD                                 # Default box placement algorithm
HOR_ROTATION_ALLOWED_DEFAULT    = True                                          # Default value of whether boxes may be rotated horizontally
VER_ROTATION_ALLOWED_DEFAULT    = True                                          # Default value of whether boxes may be rotated vertically
BNB_OPTIMALITY_GUARANTEE        = False                                         # Disable any code that would sacrifice the guarantee that the BnB algorithm's outcome is the optimal one
DEFAULT_MAX_ATTEMPTS            = 2000                                          # Default cutoff for random attempts to place boxes
SUPPORTED_AREA_PERCENTAGE       = 70                                            # Percentage of pallet area that must be supported under a box for it to be placed
BNB_TOPX_DEFAULT_LIMIT          = 5                                             # Default limit amount for BnB Filter 5
BNB_MC_MIN_SPLIT_REMAINING      = 2                                             # Multi-core BnB: only donate subtrees with at least this many boxes left to place
BNB_MC_MAX_DONATE_PER_SPLIT     = 8                                             # Multi-core BnB: maximum sibling tasks donated in a single work-splitting burst

# %% [markdown]
# #### Data loading and precomputing

# %%
boxtypes = pd.read_csv("./data/boxtypes.csv")                                   # Load box type dimensions
orders = pd.read_csv("./data/orders.csv")                                       # Load orders data
test_orders = pd.read_csv("./data/orders_test.csv")                             # Load test orders data

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
        # Compact tuple list (x, y, x2, y2, top_z) for exact O(k) geometric queries
        self._box_rects = []
        # Lazy heightmap materialized on demand (avoids 5.6 MB array scans/copies during search)
        self._heightmap = None
        self._heightmap_dirty = True
        self._custom_heightmap = False
        self._has_zero_height_box = False
        # Initialize sets of candidate coordinates (existing box edges) for all potential placements
        self.candidates_x = {0}
        self.candidates_y = {0}
        # Initialize set of candidate coordinate pairs at corner points to optimize placement (in corners between boxes)
        self.extpts = {(0, 0)}
        # Initialize storage of max Z
        self.max_z = 0

    @property
    def heightmap(self):                                                        # Materialize exact int32 heightmap lazily on access
        if self._heightmap is None:
            self._heightmap = np.zeros((self.size_x, self.size_y), dtype=np.int32)
            if self._box_rects:
                for bx, by, bx2, by2, btop in self._box_rects:
                    self._heightmap[bx:bx2, by:by2] = btop
            self._heightmap_dirty = False
        elif self._heightmap_dirty:
            self._heightmap.fill(0)
            for bx, by, bx2, by2, btop in self._box_rects:
                self._heightmap[bx:bx2, by:by2] = btop
            self._heightmap_dirty = False
        return self._heightmap

    @heightmap.setter
    def heightmap(self, value):                                                 # Preserve compatibility if caller assigns heightmap directly
        self._heightmap = value
        self._heightmap_dirty = False
        self._custom_heightmap = True

    @property
    def heightmap_sum(self):                                                    # Total occupied volume under heightmap (computed lazily)
        return int(np.sum(self.heightmap))

    def reset(self):                                                            # Empty the pallet
        # Clear the list of boxes and compact geometric index
        self.boxes = []
        self._box_rects = []
        # Mark heightmap dirty without reallocating a 5.6 MB array
        self._heightmap_dirty = True
        self._custom_heightmap = False
        self._has_zero_height_box = False
        # Reset sets of candidate coordinates
        self.candidates_x = {0}
        self.candidates_y = {0}
        # Reset set of extreme points
        self.extpts = {(0, 0)}
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

    def get_valid_placement_z(self, box_dims, x, y):                            # Exact O(k) placement check: returns landing z-height if valid, else -1
        dx, dy, dz = box_dims
        x2 = x + dx
        y2 = y + dy

        # Fast bounds rejection before scanning placed boxes
        if x < 0 or y < 0 or dx <= 0 or dy <= 0 or x2 > self.size_x or y2 > self.size_y:
            return -1

        # Exact numpy fallback only if caller manually replaced heightmap or placed a 0-height box
        if self._custom_heightmap or self._has_zero_height_box:
            z = self.get_max_height_in_area(x, y, dx, dy)
            if z == -1 or z + dz > self.size_z:
                return -1
            area_supported = int(np.sum(self.heightmap[x:x2, y:y2] == z))
            if (area_supported / (dx * dy)) * 100 < SUPPORTED_AREA_PERCENTAGE:
                return -1
            return z

        # Single O(k) pass over placed boxes computes BOTH landing height z AND exact
        # supported area simultaneously. Because heights increase strictly monotonically
        # on placement (dz > 0), all placed boxes sharing the same top height btop == z
        # are pairwise disjoint in (x, y), so their overlap areas sum directly with zero
        # double-counting and exact integer equivalence to np.sum(heightmap == z).
        max_allowed_z = self.size_z - dz
        if max_allowed_z < 0:
            return -1

        z = 0
        area_supported = 0
        for bx, by, bx2, by2, btop in self._box_rects:
            if bx < x2 and x < bx2 and by < y2 and y < by2:
                if btop > z:
                    if btop > max_allowed_z:
                        return -1
                    z = btop
                    ox = (x2 if x2 < bx2 else bx2) - (x if x > bx else bx)
                    oy = (y2 if y2 < by2 else by2) - (y if y > by else by)
                    area_supported = ox * oy
                elif btop == z:
                    ox = (x2 if x2 < bx2 else bx2) - (x if x > bx else bx)
                    oy = (y2 if y2 < by2 else by2) - (y if y > by else by)
                    area_supported += ox * oy

        if z == 0:
            return 0 if 100 >= SUPPORTED_AREA_PERCENTAGE else -1

        if (area_supported / (dx * dy)) * 100 < SUPPORTED_AREA_PERCENTAGE:
            return -1

        return z

    def check_box_placement_validity(self, box_dims, x, y):                     # Check if a box of given dimensions may be placed at position (x, y) on the pallet
        return self.get_valid_placement_z(box_dims, x, y) != -1

    def place_box(self, box_dims, x, y):                                        # Attempt to place a box of given dimensions at (x, y) position on the pallet. Return a delta (dict of information to reverse move) if successful, False if not.
        # Unpack box dimensions and validate placement + compute landing z in one pass
        dx, dy, dz = box_dims
        z = self.get_valid_placement_z(box_dims, x, y)
        if z == -1:
            return False

        x2 = x + dx
        y2 = y + dy
        top_z = z + dz

        # Check if we will be adding new candidate coordinates by placing this box
        adding_candidate_x = (x2 < self.size_x) and (x2 not in self.candidates_x)
        adding_candidate_y = (y2 < self.size_y) and (y2 not in self.candidates_y)

        # Save lightweight rollback state (O(1) for max_z and extpts)
        prev_max_z = self.max_z
        prev_extpts = self.extpts
        prev_zero_flag = self._has_zero_height_box
        heightmap_backup = None
        if self._custom_heightmap:
            heightmap_backup = self._heightmap[x:x2, y:y2].copy()
            self._heightmap[x:x2, y:y2] = top_z
        else:
            self._heightmap_dirty = True

        if dz <= 0:
            self._has_zero_height_box = True

        # Place the box: update max_z, box list, and compact geometric index
        if top_z > self.max_z:
            self.max_z = top_z
        self.boxes.append({
            'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy, 'dz': dz
        })
        self._box_rects.append((x, y, x2, y2, top_z))

        # Add box edges to candidate coordinate lists if prerequisites are met
        if adding_candidate_x: self.candidates_x.add(x2)
        if adding_candidate_y: self.candidates_y.add(y2)

        # Calculate the post-move extreme points (creates a fresh set, leaving prev_extpts intact)
        self.calculate_extreme_points()

        # Put together all the information needed to reverse this move in a delta dict
        delta = {
            'x': x,
            'y': y,
            'z': z,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'prev_max_z': prev_max_z,
            'prev_extpts': prev_extpts,
            'prev_zero_flag': prev_zero_flag,
            'heightmap_backup': heightmap_backup,
            'x_candidate_added': adding_candidate_x,
            'y_candidate_added': adding_candidate_y,
        }

        # Return delta to indicate successful placement
        return delta
    
    def remove_box(self, delta):                                                # Remove a box based on the information in the given delta dict
        x, y, dx, dy = delta['x'], delta['y'], delta['dx'], delta['dy']

        if delta['heightmap_backup'] is not None:
            self._heightmap[x:x+dx, y:y+dy] = delta['heightmap_backup']
        else:
            self._heightmap_dirty = True

        # Restore max_z and zero-height flag in O(1)
        self.max_z = delta['prev_max_z']
        self._has_zero_height_box = delta['prev_zero_flag']
        
        # Remove the last box off the list and geometric index
        self.boxes.pop()
        self._box_rects.pop()
        
        # Remove candidate coordinates if they were added by this move
        if delta['x_candidate_added']: self.candidates_x.remove(x + dx)
        if delta['y_candidate_added']: self.candidates_y.remove(y + dy)

        # Restore previous extreme points set in O(1)
        self.extpts = delta['prev_extpts']

    # Pallet state and analysis methods    
    def get_max_height_in_area(self, x, y, dx, dy):                             # Get the maximum height in a rectangular area of the heightmap
        if self._custom_heightmap or x < 0 or y < 0:
            x_end = min(x + dx, self.size_x)
            y_end = min(y + dy, self.size_y)
            region = self.heightmap[x:x_end, y:y_end]
            if region.size <= 0:
                return -1
            return int(np.max(region))

        x_end = x + dx
        if x_end > self.size_x:
            x_end = self.size_x
        y_end = y + dy
        if y_end > self.size_y:
            y_end = self.size_y

        if x >= x_end or y >= y_end:
            return -1

        max_h = 0
        for bx, by, bx2, by2, btop in self._box_rects:
            if bx < x_end and x < bx2 and by < y_end and y < by2:
                if btop > max_h:
                    max_h = btop
        return max_h
    
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
        size_x = self.size_x
        size_y = self.size_y
        rects = self._box_rects
        new_extpts = {(0, 0)}

        for bx, by, x_end, y_end, _ in rects:
            # Allow stacking boxes on top of each other by including the top corners of boxes as extreme points
            if 0 <= bx <= size_x and 0 <= by <= size_y:
                new_extpts.add((bx, by))

            # Trace right edge to Y=0 and front edge to X=0 in a single pass
            y_projection_dist = 0
            x_projection_dist = 0
            for ox, oy, ox2, oy2, _ in rects:
                if oy2 <= by and ox < x_end < ox2:
                    if oy2 > y_projection_dist:
                        y_projection_dist = oy2
                if ox2 <= bx and oy < y_end < oy2:
                    if ox2 > x_projection_dist:
                        x_projection_dist = ox2

            if 0 <= x_end <= size_x and 0 <= y_projection_dist <= size_y:
                new_extpts.add((x_end, y_projection_dist))
            if 0 <= x_projection_dist <= size_x and 0 <= y_end <= size_y:
                new_extpts.add((x_projection_dist, y_end))

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

        try:
            return round(occupied_volume / total_volume * 100, 2)
        except:
            return 0

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

    def get_pallet_results(self, algo, orderID, order_dict, print_mode=False, save_mode=False, bnb_stats=None, nodes=0):  # Get metrics for the pallet, optionally display them, and/or save a composite results image to ./results/
        # Get metrics
        fulfillment = self.check_order_fullfillment(orderID, order_dict)
        volume_util = self.get_volume_utilization()
        area_usage_at_z0 = self.get_area_usage_at_z(0)
        cog_z = self.get_center_of_gravity_z()
        packing_score = self.get_packing_score()
        max_z = self.get_max_height()

        if bnb_stats is not None:
            nodes_evaluated = bnb_stats['nodes']
        else:
            nodes_evaluated = nodes 

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

            # Include BnB optimality guarantee tag and candidate limit when relevant
            if algo == Algorithm.BNB and bnb_stats is not None:
                optimal_guarantee = bnb_stats.get('optimality_guarantee', None)
                topx_limit = bnb_stats.get('topx_limit', None)
                if optimal_guarantee is True:
                    optimal_tag = "ON  (exact)"
                elif optimal_guarantee is False:
                    optimal_tag = "OFF (fast)"
                else:
                    optimal_tag = f"{'ON' if BNB_OPTIMALITY_GUARANTEE else 'OFF'} (global)"
            else:
                optimal_tag = "N/A"
                topx_limit = "N/A"

            # Format metrics into neat block of text
            metrics_lines = [
                "  -- Run Information ------------------",
                f"  Algorithm:              {algo.value.upper()}",
                f"  Order ID:               {orderID}",
                f"  Boxes in order:         {len(order_dict[orderID])}",
                f"  Dataset:                {set_name_display}",
                f"  BnB Opt. Guarantee:     {optimal_tag}",
                f"  BnB Candidate Limit:    {topx_limit}",
                f"  Nodes evaluated:        {nodes_evaluated:,}",
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
                total_pruned = (bnb_stats['pruned_rule1'] + bnb_stats['pruned_rule4'] + bnb_stats['pruned_filt1'] + bnb_stats['pruned_filt2'] + bnb_stats['pruned_filt5'])
                metrics_lines += [
                    "",
                    " --- BnB Search Stats -----------------",
                    f"  Branches pruned by rule:\n"
                    f"  Rule 1 (trivial):         {bnb_stats['pruned_rule1']:,}",
                    f"  Rule 4 (tall-low):        {bnb_stats['pruned_rule4']:,}",
                    f"  Branches blocked by filter:\n"
                    f"  Deduplication:            {bnb_stats['pruned_filt1']:,}",
                    f"  Symmetry breaking:        {bnb_stats['pruned_filt2']:,}",
                    f"  Top-x candidate limiting: {bnb_stats['pruned_filt5']:,}",
                    "",
                    f"  Total pruned:             {total_pruned:,}",
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
        # Fast exact path for MAX_Z: no need to mutate boxes or recompute extreme points
        if requested_metric == Metric.MAX_Z:
            z = self.get_valid_placement_z(box_dims, x, y)
            if z == -1:
                return -1
            top_z = z + box_dims[2]
            return top_z if top_z > self.max_z else self.max_z

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
    
    dx = box['LENGTH']
    dy = box['WIDTH']
    dz = box['HEIGHT']
    
    a = dx * dy
    v = a * dz

    return dx, dy, dz, a, v

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
    
def process_order(order, algo, max_attempts=DEFAULT_MAX_ATTEMPTS, criterion=DEFAULT_CRITERION, metric=DEFAULT_OPTIMIZATION_METRIC, order_dict=orders_dict, leave_tqdm=True, optimality_guarantee=None, num_extpts_to_try=None, use_mp=True):     # Process a given order using the specified algorithm, max attempts, and sorting criterion. For BnB, returns resultant pallet and node+pruning stats, returns resultant pallet only for other algos
    box_list = get_box_list_from_order(order, order_dict)
    pallet = Pallet()

    if algo == Algorithm.RANDOM:
        nodes = place_box_list_random(pallet, box_list, max_attempts)
        return pallet, nodes

    elif algo == Algorithm.FFD:
        nodes = place_box_list_first_fit_decreasing(pallet, box_list, criterion=criterion)
        return pallet, nodes

    elif algo == Algorithm.BFD:
        nodes = place_box_list_best_fit_decreasing(pallet, box_list, criterion=criterion, opt_metric=metric)
        return pallet, nodes

    elif algo == Algorithm.BNB:
        if use_mp == False:
            bnb_stats = place_box_list_branch_and_bound(pallet, box_list, criterion=criterion, leave_tqdm=leave_tqdm, optimality_guarantee=optimality_guarantee, num_extpts_to_try=num_extpts_to_try)
        elif use_mp == True:
            bnb_stats = place_box_list_branch_and_bound_mc(pallet, box_list, criterion=criterion, leave_tqdm=leave_tqdm, optimality_guarantee=optimality_guarantee, num_extpts_to_try=num_extpts_to_try, cores=DEFAULT_MP_CORES)
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

def get_core_counts_list(mp_core_max):
    i = 2
    core_counts = []
    while i <= mp_core_max:
        core_counts += [i]
        i = i *2
    return core_counts

def _bnb_zero_counts():
    return {
        'nodes': 0,
        'pruned_rule1': 0,
        'pruned_rule4': 0,
        'pruned_filt1': 0,
        'pruned_filt2': 0,
        'pruned_filt5': 0,
    }

def _bnb_merge_cert_front(cert_front, path_key, score, initial_best_score):
    """Insert a certified (path_key, score) into the Pareto-front.

    Invariant: `cert_front` is a list of `(path_key, score)` tuples strictly
    increasing in `path_key` and strictly decreasing in `score`, all with
    `score < initial_best_score`.
    Returns `(updated_front, changed)`.
    """
    if score >= initial_best_score:
        return cert_front, False

    for p, s in cert_front:
        if p <= path_key and s <= score:
            return cert_front, False
        if p > path_key:
            break

    new_front = []
    inserted = False
    for p, s in cert_front:
        if p >= path_key and s >= score:
            continue
        if not inserted and p > path_key:
            new_front.append((path_key, score))
            inserted = True
        new_front.append((p, s))
    if not inserted:
        new_front.append((path_key, score))
    return new_front, True

def _bnb_cert_front_limits(cert_front, current_path, initial_best_score):
    """Compute exact sound (r1_limit, r4_limit) for a node at `current_path`.

    - Rule 1 (max_z) is monotonic/admissible: any certified global best `g_score`
      prunes with `g_score` to its right (`current_path > g_path`) and with
      `g_score + 1` to its left (preserving earlier winning ties).
    - Rule 4 (tall-low) is NOT monotonic across extreme-point changes: to never
      prune a solution that single-core DFS would reach, Rule 4 only uses
      certified solutions strictly to the LEFT of `current_path`.
    """
    if not cert_front:
        return initial_best_score, initial_best_score

    g_path, g_score = cert_front[-1]
    if current_path > g_path:
        return g_score, g_score

    r1_limit = g_score + 1
    r4_limit = initial_best_score
    for p, s in cert_front:
        if p < current_path:
            r4_limit = s
        else:
            break
    return r1_limit, r4_limit

def _bnb_first_disproved_spike_depth(spikes, cert_front):
    """Return the shallowest ancestor depth whose Rule-4 spike is disproved by `cert_front`, or None."""
    if not spikes or not cert_front:
        return None
    for anc_path, anc_r4 in spikes:
        for cp, cs in cert_front:
            if cp < anc_path:
                if cs <= anc_r4:
                    return len(anc_path)
            else:
                break
    return None

def _bnb_spikes_disproved_by_front(spikes, cert_front):
    """Return True iff `cert_front` already proves an ancestor spike was pruned by Rule 4."""
    return _bnb_first_disproved_spike_depth(spikes, cert_front) is not None

def _bnb_compute_rule4_min_z(pallet, tallest_orientations, early_exit_threshold=-1):
    """Exact Rule 4 (tall-low) minimum landing height matching single-core BnB.

    If `early_exit_threshold >= 0`, stops as soon as any valid placement achieves
    `top_z <= early_exit_threshold` (guaranteed neither to prune nor to spike).
    """
    min_landing_z = pallet.size_z + 100
    get_valid_z = pallet.get_valid_placement_z
    for x, y in pallet.extpts:
        for dims in tallest_orientations:
            landing_z = get_valid_z(dims, x, y)
            if landing_z != -1:
                top_z = landing_z + dims[2]
                if top_z < min_landing_z:
                    min_landing_z = top_z
                    if top_z <= early_exit_threshold:
                        return min_landing_z
    return min_landing_z

def _bnb_generate_valid_children(pallet, box_index, current_sequence,
                                 sorted_box_list, use_guarantee,
                                 num_extpts_to_try, box_orientations_dict,
                                 dimension_tuples, counts):
    """Generate ordered valid child placements for `box_index` (Filters 5, 1, 2).

    Applies the exact same candidate generation, top-X sorting/slicing,
    deduplication (Filter 1), and symmetry breaking (Filter 2) order as
    `place_box_list_branch_and_bound()`, returning `[(child_idx, (dims, x, y), top_z), ...]`.
    """
    boxid = sorted_box_list[box_index]
    orientations = box_orientations_dict[boxid]
    sorted_extpts = sorted(pallet.extpts)
    seen_profile_keys = set() if not use_guarantee else None

    if box_index > 0 and dimension_tuples[box_index] == dimension_tuples[box_index - 1]:
        symmetry_key = current_sequence[-1]
    else:
        symmetry_key = None

    candidate_placements = []
    get_valid_z = pallet.get_valid_placement_z
    for dims in orientations:
        dz = dims[2]
        for x, y in sorted_extpts:
            landing_z = get_valid_z(dims, x, y)
            if landing_z != -1:
                top_z = landing_z + dz
                candidate_placements.append(((top_z, x + y), dims, x, y, landing_z))

    if not use_guarantee and num_extpts_to_try is not None:
        candidate_placements.sort(key=_FIRST_ITEM)
        filtered = max(0, len(candidate_placements) - num_extpts_to_try)
        candidate_placements = candidate_placements[:num_extpts_to_try]
        if filtered > 0:
            counts['pruned_filt5'] += filtered

    valid_children = []
    child_idx = 0
    for _, dims, x, y, landing_z in candidate_placements:
        if not use_guarantee:
            profile_key = (dims, landing_z)
            if profile_key in seen_profile_keys:
                counts['pruned_filt1'] += 1
                continue

        if symmetry_key is not None and symmetry_key > (dims, x, y):
            counts['pruned_filt2'] += 1
            continue

        if not use_guarantee:
            seen_profile_keys.add(profile_key)

        valid_children.append((child_idx, (dims, x, y), int(landing_z + dims[2])))
        child_idx += 1

    return valid_children

def _bnb_mc_worker_main(worker_id, worker_conn, shared_split_tokens,
                        shared_split_lock, shared_bound_version, context_arg):
    """Persistent worker process for dynamic work-splitting parallel BnB.

    Reuses a single Pallet instance across tasks with longest-common-prefix
    incremental backtracking (eliminating repeated 5.6 MB heightmap allocations
    and kernel TLB shootdowns across 128 cores) and uses an explicit DFS stack
    so shallowest unexplored sibling subtrees can be donated on demand whenever
    any peer core goes idle.
    """
    try:
        ctx = context_arg if context_arg is not None else _BNB_MC_CONTEXT
        (
            pallet_dims,
            sorted_box_list,
            use_guarantee,
            num_extpts_to_try,
            box_orientations_dict,
            tallest_remaining_orientations,
            dimension_tuples,
            initial_best_score,
        ) = ctx

        n_boxes = len(sorted_box_list)
        size_z_plus_100 = pallet_dims[2] + 100
        max_split_b_idx = n_boxes - BNB_MC_MIN_SPLIT_REMAINING

        pallet = Pallet(dims=pallet_dims)
        cur_seq = []
        cur_deltas = []
        counts = _bnb_zero_counts()

        local_cert_front = []
        local_bound_version = 0

        def merge_front_updates(incoming_front):
            nonlocal local_cert_front
            for p, s in incoming_front:
                local_cert_front, _ = _bnb_merge_cert_front(
                    local_cert_front, p, s, initial_best_score
                )

        def poll_bounds():
            nonlocal local_bound_version
            v = shared_bound_version.value
            if v != local_bound_version:
                while worker_conn.poll():
                    msg = worker_conn.recv()
                    if msg[0] == "BOUND":
                        merge_front_updates(msg[1])
                        local_bound_version = msg[2]

        def transition_pallet_to(target_seq):
            """Move persistent `pallet` from `cur_seq` to `target_seq` via LCP."""
            lcp = 0
            limit = min(len(cur_seq), len(target_seq))
            while lcp < limit and cur_seq[lcp] == target_seq[lcp]:
                lcp += 1

            while len(cur_seq) > lcp:
                pallet.remove_box(cur_deltas.pop())
                cur_seq.pop()

            for move in target_seq[lcp:]:
                dims, x, y = move
                delta = pallet.place_box(dims, x, y)
                if not delta:
                    return False
                cur_deltas.append(delta)
                cur_seq.append(move)
            return True

        def search_subtree(root_path, root_seq, root_spikes):
            nonlocal local_cert_front
            if not transition_pallet_to(root_seq):
                return

            root_depth = len(root_seq)
            if root_depth > 0:
                counts['nodes'] += 1

            spikes = list(root_spikes)

            def enter_node(b_idx, cur_path):
                nonlocal local_cert_front
                poll_bounds()
                r1_limit, r4_limit = _bnb_cert_front_limits(
                    local_cert_front, cur_path, initial_best_score
                )

                # Base case: all boxes placed
                if b_idx == n_boxes:
                    sc = int(pallet.get_max_height())
                    if sc < r1_limit:
                        active_spikes = tuple(
                            (ap, ar4) for ap, ar4 in spikes if ar4 > sc
                        )
                        if not active_spikes:
                            local_cert_front, changed = _bnb_merge_cert_front(
                                local_cert_front, cur_path, sc, initial_best_score
                            )
                            if changed:
                                worker_conn.send(
                                    ("SOLUTION", worker_id, True, cur_path, list(cur_seq), sc, ())
                                )
                        elif not _bnb_spikes_disproved_by_front(active_spikes, local_cert_front):
                            worker_conn.send(
                                ("SOLUTION", worker_id, False, cur_path, list(cur_seq), sc, active_spikes)
                            )
                    return None

                # Rule 1 (Trivial Bounding)
                cur_max_z = int(pallet.get_max_height())
                if cur_max_z >= r1_limit:
                    counts['pruned_rule1'] += 1
                    return None

                # Rule 4 (Tall-Low Bounding)
                # Can early-exit once any valid placement has top_z <= cur_max_z
                # and top_z < r4_limit (neither prunes nor creates a Rule-4 spike).
                r4_early_exit = cur_max_z if cur_max_z < r4_limit else (r4_limit - 1)
                min_landing_z = int(_bnb_compute_rule4_min_z(
                    pallet, tallest_remaining_orientations[b_idx], r4_early_exit
                ))
                if size_z_plus_100 > min_landing_z >= r4_limit:
                    counts['pruned_rule4'] += 1
                    return None

                pushed_spike = False
                if size_z_plus_100 > min_landing_z > cur_max_z:
                    spikes.append((cur_path, min_landing_z))
                    pushed_spike = True

                # Branching (Filters 5, 1, 2)
                valid_children = _bnb_generate_valid_children(
                    pallet,
                    b_idx,
                    cur_seq,
                    sorted_box_list,
                    use_guarantee,
                    num_extpts_to_try,
                    box_orientations_dict,
                    dimension_tuples,
                    counts,
                )
                if not valid_children:
                    if pushed_spike:
                        spikes.pop()
                    return None

                # Frame: [b_idx, path_key, valid_children, next_idx, pushed_spike, active_delta, base_max_z]
                return [b_idx, cur_path, valid_children, 0, pushed_spike, None, cur_max_z]

            first_frame = enter_node(root_depth, root_path)
            if first_frame is None:
                return

            dfs_stack = [first_frame]

            while dfs_stack:
                # Poll incoming bound updates before splitting or expanding more siblings
                poll_bounds()
                top = dfs_stack[-1]

                # Undo the child we just returned from
                if top[5] is not None:
                    pallet.remove_box(top[5])
                    cur_deltas.pop()
                    cur_seq.pop()
                    top[5] = None

                # Immediate stack unwinding ONLY down to the shallowest disproved
                # spike depth (never discarding shallower sibling frames whose
                # ancestors were not disproved!).
                disproved_depth = _bnb_first_disproved_spike_depth(spikes, local_cert_front)
                if disproved_depth is not None:
                    while dfs_stack and dfs_stack[-1][0] >= disproved_depth:
                        popped = dfs_stack.pop()
                        if popped[5] is not None:
                            pallet.remove_box(popped[5])
                            cur_deltas.pop()
                            cur_seq.pop()
                        if popped[4]:
                            spikes.pop()
                    if not dfs_stack:
                        break
                    continue

                top_r1_limit, _ = _bnb_cert_front_limits(
                    local_cert_front, top[1], initial_best_score
                )
                if int(pallet.get_max_height()) >= top_r1_limit:
                    dfs_stack.pop()
                    if top[4]:
                        spikes.pop()
                    continue

                # Dynamic top-of-stack work splitting when peer cores need work
                if shared_split_tokens.value > 0:
                    for frame in dfs_stack:
                        f_depth = frame[0]
                        if f_depth > max_split_b_idx:
                            break
                        f_children = frame[2]
                        f_next = frame[3]
                        f_active = frame[5] is not None
                        start_idx = f_next if f_active else (f_next + 1)
                        avail = len(f_children) - start_idx
                        if avail <= 0:
                            continue

                        with shared_split_lock:
                            tok = shared_split_tokens.value
                            if tok > 0:
                                n_donate = min(avail, tok, BNB_MC_MAX_DONATE_PER_SPLIT)
                                shared_split_tokens.value = tok - n_donate
                            else:
                                n_donate = 0

                        if n_donate > 0:
                            donated_slice = f_children[start_idx : start_idx + n_donate]
                            if f_active:
                                frame[3] = start_idx + n_donate
                            else:
                                del f_children[start_idx : start_idx + n_donate]

                            prefix_seq = cur_seq[:f_depth]
                            prefix_spikes = tuple(
                                (ap, ar4) for ap, ar4 in spikes if len(ap) <= f_depth
                            )
                            f_path = frame[1]
                            f_base_max_z = frame[6]
                            donated_tasks = [
                                (
                                    f_path + (orig_idx,),
                                    prefix_seq + [move],
                                    prefix_spikes,
                                    max(f_base_max_z, child_top_z),
                                )
                                for orig_idx, move, child_top_z in donated_slice
                            ]
                            worker_conn.send(("DONATE", worker_id, donated_tasks))
                        break

                # Finished all children of `top` -> pop frame
                if top[3] >= len(top[2]):
                    dfs_stack.pop()
                    if top[4]:
                        spikes.pop()
                    continue

                # Descend into next child
                orig_child_idx, move, child_top_z = top[2][top[3]]
                top[3] += 1
                next_path = top[1] + (orig_child_idx,)
                child_r1_limit, _ = _bnb_cert_front_limits(
                    local_cert_front, next_path, initial_best_score
                )
                # Fast Rule-1 pre-check before touching pallet heightmap
                if max(top[6], child_top_z) >= child_r1_limit:
                    counts['nodes'] += 1
                    counts['pruned_rule1'] += 1
                    continue

                dims, x, y = move
                delta = pallet.place_box(dims, x, y)
                if not delta:
                    continue

                top[5] = delta
                cur_deltas.append(delta)
                cur_seq.append(move)
                counts['nodes'] += 1

                child_frame = enter_node(top[0] + 1, next_path)
                if child_frame is not None:
                    dfs_stack.append(child_frame)

        # Main idle/task loop
        while True:
            msg = worker_conn.recv()
            tag = msg[0]
            if tag == "BOUND":
                merge_front_updates(msg[1])
                local_bound_version = msg[2]
            elif tag == "TASK":
                task_path, task_seq, task_spikes, _ = msg[1]
                search_subtree(task_path, task_seq, task_spikes)
                worker_conn.send(("DONE", worker_id))
            elif tag == "STOP":
                worker_conn.send(("FINAL_STATS", worker_id, counts))
                break
    except Exception:
        try:
            worker_conn.send(("ERROR", worker_id, traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            worker_conn.close()
        except Exception:
            pass

def _bnb_mc_verify_anomalous_candidates(pallet_dims, sorted_box_list, use_guarantee,
                                        num_extpts_to_try, box_orientations_dict,
                                        tallest_remaining_orientations, dimension_tuples,
                                        initial_best_score, best_cert_path, best_cert_seq,
                                        best_cert_score, surviving_anomalies):
    """Exact single-core prefix check for ultra-rare non-monotonic Rule-4 candidates.

    Only invoked when a speculative worker found a solution whose path had an
    ancestor Rule-4 spike (`r4 > final_score`) AND that candidate beats or ties
    from the left the best certified solution without being disproved by
    `cert_front`. Replays exact single-core DFS strictly up to the rightmost
    unresolved candidate, terminating early as soon as all candidates are
    disproved or resolved.
    """
    n_boxes = len(sorted_box_list)
    size_z_plus_100 = pallet_dims[2] + 100
    remaining = sorted(surviving_anomalies, key=lambda c: c[0])
    max_target_path = remaining[-1][0]

    pallet = Pallet(dims=pallet_dims)
    cur_seq = []
    dummy_counts = _bnb_zero_counts()
    best_score = initial_best_score
    best_seq = None
    best_path = None
    stop_early = False

    def prune_targets(at_path):
        nonlocal remaining, max_target_path, stop_early
        kept = []
        for cand in remaining:
            cp, _, csc, cspikes = cand
            if cp <= at_path:
                continue
            if csc >= best_score:
                continue
            if any(at_path < ap and best_score <= ar4 for ap, ar4 in cspikes):
                continue
            kept.append(cand)
        remaining = kept
        if not remaining:
            stop_early = True
        else:
            max_target_path = remaining[-1][0]

    def dfs(b_idx, cur_path):
        nonlocal best_score, best_seq, best_path
        if stop_early or cur_path > max_target_path:
            return

        if b_idx == n_boxes:
            sc = int(pallet.get_max_height())
            if sc < best_score:
                best_score = sc
                best_seq = list(cur_seq)
                best_path = cur_path
                prune_targets(cur_path)
            return

        if pallet.get_max_height() >= best_score:
            return

        min_landing_z = _bnb_compute_rule4_min_z(
            pallet, tallest_remaining_orientations[b_idx], best_score - 1
        )
        if size_z_plus_100 > min_landing_z >= best_score:
            return

        valid_children = _bnb_generate_valid_children(
            pallet, b_idx, cur_seq, sorted_box_list, use_guarantee,
            num_extpts_to_try, box_orientations_dict, dimension_tuples,
            dummy_counts,
        )
        for child_idx, (dims, x, y), _ in valid_children:
            if stop_early:
                return
            next_path = cur_path + (child_idx,)
            if next_path > max_target_path:
                break
            delta = pallet.place_box(dims, x, y)
            if not delta:
                continue
            cur_seq.append((dims, x, y))
            dfs(b_idx + 1, next_path)
            cur_seq.pop()
            pallet.remove_box(delta)

    dfs(0, ())

    if best_seq is not None and (
        best_cert_seq is None
        or best_score < best_cert_score
        or (best_score == best_cert_score and best_path < best_cert_path)
    ):
        return best_seq, best_score
    return best_cert_seq, best_cert_score

def _extract_pallet_metric(pallet_to_measure, metric):
    if metric == Metric.MAX_Z:
        return pallet_to_measure.get_max_height()
    if metric == Metric.PACKING_SCORE:
        return pallet_to_measure.get_packing_score()
    if metric == Metric.VOLUME_UTILIZATION:
        return pallet_to_measure.get_volume_utilization()
    if metric == Metric.COG_Z:
        return pallet_to_measure.get_center_of_gravity_z()
    if metric == Metric.ALL:
        return None
    raise ValueError(f"Unsupported metric: {metric}")

def _canonical_boxes(pallet_to_compare):
    return sorted(
        (
            int(b['x']), int(b['y']), int(b['z']),
            int(b['dx']), int(b['dy']), int(b['dz']),
        )
        for b in pallet_to_compare.boxes
    )

def _write_csv_atomic(df, csv_path):
    """Write a DataFrame to `csv_path` atomically via temp file + os.replace."""
    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{csv_path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

def _find_existing_speed_csv(output_dir, prefix, order_id):
    """Return existing non-empty speed-comparison CSV path for `order_id`, or None."""
    for stem in ("spee_comparison", "speed_comparison"):
        candidate = os.path.join(output_dir, f"{stem}_{prefix}{order_id}.csv")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None

def _bnb_speed_test_worker(conn, order_id, cores, box_list, criterion, metric,
                           bnb_topx, optimality_guarantee=False):
    """Isolated process worker for a single (order, core_count) benchmark run.

    Creates its own POSIX process group so the scheduler can cleanly terminate
    both this worker and any child multi-core pool processes on interruption
    without leaving orphaned CPU-burning processes behind.

    Uses `time.process_time()` for 1-core serial runs so brief SIGSTOP pauses
    during exclusive full-machine multi-core bursts do not inflate serial
    benchmark timings, and `time.perf_counter()` for multi-core runs (which are
    never paused or oversubscribed).
    """
    try:
        if hasattr(os, "setpgrp"):
            try:
                os.setpgrp()
            except OSError:
                pass

        os.environ["TQDM_DISABLE"] = "1"
        pallet = Pallet()

        if cores == 1:
            start_time = time.process_time()
            stats = place_box_list_branch_and_bound(
                pallet,
                box_list,
                criterion=criterion,
                leave_tqdm=False,
                optimality_guarantee=optimality_guarantee,
                num_extpts_to_try=bnb_topx,
            )
            elapsed_seconds = time.process_time() - start_time
        else:
            start_time = time.perf_counter()
            stats = place_box_list_branch_and_bound_mc(
                pallet,
                box_list,
                criterion=criterion,
                leave_tqdm=False,
                optimality_guarantee=optimality_guarantee,
                num_extpts_to_try=bnb_topx,
                cores=cores,
            )
            elapsed_seconds = time.perf_counter() - start_time

        metric_val = _extract_pallet_metric(pallet, metric)
        canonical_boxes = _canonical_boxes(pallet)
        heightmap_hash = hashlib.sha256(pallet.heightmap.tobytes()).hexdigest()

        conn.send({
            'order_id': order_id,
            'cores': cores,
            'seconds': float(elapsed_seconds),
            'metric_value': float(metric_val) if metric_val is not None else None,
            'max_z': int(pallet.get_max_height()),
            'box_count': len(pallet.boxes),
            'stats': stats,
            'canonical_boxes': canonical_boxes,
            'heightmap_hash': heightmap_hash,
            'error': None,
        })
    except Exception as e:
        try:
            conn.send({
                'order_id': order_id,
                'cores': cores,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _build_speed_comparison_rows(order_id, prefix, metric, all_core_counts, results_map):
    """Build exact CSV row dicts comparing single-core and multi-core results."""
    serial_res = results_map[1]
    serial_pallet_boxes = serial_res['canonical_boxes']
    serial_heightmap_hash = serial_res['heightmap_hash']
    serial_max_z = serial_res['max_z']
    serial_seconds = serial_res['seconds']
    serial_stats = serial_res['stats']

    rows = [{
        'order_id': order_id,
        'order_label': f"{prefix}{order_id}",
        'implementation': 'single_core',
        'cores': 1,
        'seconds': serial_seconds,
        'speedup_vs_single_core': 1.0,
        'metric': metric.value,
        'metric_value': serial_res['metric_value'],
        'max_z': serial_max_z,
        'box_count': serial_res['box_count'],
        'nodes': serial_stats['nodes'],
        'pruned_rule1': serial_stats['pruned_rule1'],
        'pruned_rule4': serial_stats['pruned_rule4'],
        'pruned_filt1': serial_stats['pruned_filt1'],
        'pruned_filt2': serial_stats['pruned_filt2'],
        'pruned_filt5': serial_stats['pruned_filt5'],
        'matches_single_core': True,
        'boxes_match': True,
        'heightmap_match': True,
        'max_z_match': True,
    }]

    all_matches = True
    speedup_summary = []

    for cores in sorted(c for c in all_core_counts if c > 1):
        mc_res = results_map[cores]
        mc_seconds = mc_res['seconds']
        mc_stats = mc_res['stats']

        boxes_match = (mc_res['canonical_boxes'] == serial_pallet_boxes)
        heightmap_match = (mc_res['heightmap_hash'] == serial_heightmap_hash)
        max_z_match = (mc_res['max_z'] == serial_max_z)
        pallets_match = bool(boxes_match and heightmap_match and max_z_match)

        if not pallets_match:
            all_matches = False

        speedup = serial_seconds / mc_seconds if mc_seconds > 0 else math.inf
        speedup_summary.append(f"MP{cores}:{speedup:.1f}x")

        rows.append({
            'order_id': order_id,
            'order_label': f"{prefix}{order_id}",
            'implementation': 'multi_core',
            'cores': cores,
            'seconds': mc_seconds,
            'speedup_vs_single_core': speedup,
            'metric': metric.value,
            'metric_value': mc_res['metric_value'],
            'max_z': mc_res['max_z'],
            'box_count': mc_res['box_count'],
            'nodes': mc_stats['nodes'],
            'pruned_rule1': mc_stats['pruned_rule1'],
            'pruned_rule4': mc_stats['pruned_rule4'],
            'pruned_filt1': mc_stats['pruned_filt1'],
            'pruned_filt2': mc_stats['pruned_filt2'],
            'pruned_filt5': mc_stats['pruned_filt5'],
            'matches_single_core': pallets_match,
            'boxes_match': boxes_match,
            'heightmap_match': heightmap_match,
            'max_z_match': max_z_match,
        })

    return rows, all_matches, speedup_summary

def _save_order_speed_comparison_csv(order_id, prefix, metric, all_core_counts, results_map, output_dir):
    """Collates serial and multi-core results for one order and atomically writes its CSV."""
    rows, all_matches, speedup_summary = _build_speed_comparison_rows(
        order_id, prefix, metric, all_core_counts, results_map
    )
    serial_seconds = results_map[1]['seconds']
    csv_path = os.path.join(output_dir, f"spee_comparison_{prefix}{order_id}.csv")
    df = pd.DataFrame(rows)
    _write_csv_atomic(df, csv_path)

    summary_str = " | ".join(speedup_summary[-3:])
    match_status = "OK" if all_matches else "MISMATCH!"
    print(
        f"[COMPLETED] Order {prefix}{order_id:<4} -> {csv_path} | "
        f"Serial: {serial_seconds:6.2f}s | {summary_str} | Verified: {match_status}",
        flush=True,
    )
    return all_matches, csv_path

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

def place_box_list_random(pallet, box_list, max_attempts=10000):                                                                    # Place boxes from a given box list randomly on the pallet
    nodes_traversed = 0
    
    for boxid in box_list:
        # Get box dimensions
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)

        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            nodes_traversed += 1
            # Generate random (x, y) position for the box
            x = np.random.randint(0, pallet.size_x)
            y = np.random.randint(0, pallet.size_y)

            # Attempt to place the box on the pallet
            placed = pallet.place_box((dx, dy, dz), x, y)
            attempts += 1

    return nodes_traversed

def place_box_list_first_fit_decreasing(pallet, box_list, criterion=DEFAULT_CRITERION):                                             # Baseline (naive) algorithm: place boxes from a box list on the pallet, biggest boxes (by criterion x, y, z, a, or v) first, with lowest x (then y) values possible
    # Sort the box list by size (largest to smallest)
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)
    nodes_traversed = 0

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
                    nodes_traversed += 1
                    if pallet.place_box(dims, x, y):
                        placed = True
                        break
    
    return nodes_traversed

def place_box_list_best_fit_decreasing(pallet, box_list, criterion=DEFAULT_CRITERION, opt_metric=DEFAULT_OPTIMIZATION_METRIC):      # Baseline (naive) algorithm: place boxes from a box list on the pallet, finding the best place for the boxes based on best value of optimization metric after placement
    # Sort the box list by size (largest to smallest)
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)
    nodes_traversed = 0

    for boxid in tqdm(sorted_box_list, desc="Placing boxes (Best Fit)", leave=False):
        # Get base dimensions
        original_dx, original_dy, original_dz, _, _ = get_box_properties_from_id(boxid)
        
        # Get all valid orientations for this box
        orientations = get_box_orientations(original_dx, original_dy, original_dz)

        # Initialize best score trackers for this specific box
        if opt_metric in MAXIMIZE_METRICS:
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
                    nodes_traversed += 1

                    # Precheck conditions for place_box() before running it to save compute
                    # If prechecks pass, simulate the placement compare the resulting metric score without affecting the real pallet state
                    if pallet.check_box_placement_validity(dims, x, y):
                        # Calculate the score based on the chosen metric
                        score = pallet.simulate_placement(x, y, dims, opt_metric)

                        # Compare score against best found so far
                        if opt_metric in MAXIMIZE_METRICS:
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

    return nodes_traversed

def place_box_list_branch_and_bound(pallet, box_list, criterion=DEFAULT_CRITERION, leave_tqdm=True, optimality_guarantee=None, num_extpts_to_try=None):   # Recursive backtracking algorithm to find the optimal placement of boxes from a box list on the pallet, with pruning based on bounding functions and branching filters
    # PREWORK AND INITIALIZATIONS
    sorted_box_list = sort_box_list_by_size(box_list, criterion=criterion, invert=False)                    # Sort the box list by size (largest to smallest)
    use_guarantee = BNB_OPTIMALITY_GUARANTEE if optimality_guarantee is None else optimality_guarantee      # Override global optimality guarantee if specified
    current_sequence = []                                                                                   # Initialize a list of deltas for the current sequence of placements to traverse both ways
    best_sequence = None                                                                                    # Initialize storage for the best sequence of placements found

    temp_pallet = Pallet()                                                                                  # Initialize best score tracker for the entire placement
    place_box_list_best_fit_decreasing(temp_pallet, sorted_box_list, criterion=criterion)                   # Run the BFD algorithm first to get a decent initial score to beat, making pruning more aggressive
    best_score = temp_pallet.get_max_height() + 1                                                           # Make sure BnB algorithm doesn't fail catastrophically if unable to find a better solution than BFD

    box_count = len(box_list)
    node_count_estimate = 139828400000 + (4413.08 - 139828400000) / (1 + (box_count/210.6191) ** 7.855901)  # Equation derived from curve fitting average BnB node counts per box amount

    # PRECALCULATIONS
    box_orientations_dict = {}                                                                              # Precalculate box orientations for this order
    for boxid in set(sorted_box_list):
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)
        box_orientations_dict[boxid] = get_box_orientations(dx, dy, dz)
    
    tallest_remaining_orientations = []                                                                     # Rule 4 (Tall-Low): Precalculate tallest remaining box orientations by depth
    for i in range(len(sorted_box_list)):
        tallest_remaining_id = max(
            sorted_box_list[i:],
            key=lambda bid: min(get_box_properties_from_id(bid)[:3])
        )
        tallest_remaining_orientations.append(box_orientations_dict[tallest_remaining_id])

    dimension_tuples = []                                                                                   # Filter 2 (Symmetry Breaking): Make list of box dimensions tuples
    for boxid in sorted_box_list:
        dimension_tuples.append(tuple(sorted(get_box_properties_from_id(boxid)[:3])))

    def recursive_place(box_index):                                                                         # Define recursive function to place boxes one by one and prune as needed
        # Take variables into local scope of the recursive function
        nonlocal best_score, best_sequence, count_nodes, count_bound1, count_bound4, count_filt1, count_filt2, count_filt5

        # BASE CASE: if all boxes are placed, evaluate score and update best if needed
        if box_index == len(sorted_box_list):
            current_score = pallet.get_max_height()
            if current_score < best_score:
                best_score = current_score
                best_sequence = list(current_sequence)
            return
        
        # BOUNDING CASES
        # Rule 1 (Trivial Rule)
        if pallet.get_max_height() >= best_score:                                                           # Calculate if the current partial state is already worse than the best known full state and prune if so
            counter_bound1.update(1)
            count_bound1 += 1
            return
        
        # Rule 4 (Tall-Low)
        t_orientations = tallest_remaining_orientations[box_index]                                          # Check if tallest remaining box placed at its lowest possible candidate position exceeds best_score and prune if so
        min_landing_z = PALLET_DIMS[2] + 100
        get_valid_z = pallet.get_valid_placement_z
        
        for x, y in pallet.extpts:
            for dims in t_orientations:
                landing_z = get_valid_z(dims, x, y)
                if landing_z != -1:
                    top_z = landing_z + dims[2]
                    if top_z < min_landing_z:
                        min_landing_z = top_z
                        if min_landing_z < best_score:
                            break
            if min_landing_z < best_score:
                break

        if PALLET_DIMS[2] + 100 > min_landing_z >= best_score:
            counter_bound4.update(1)
            count_bound4 += 1
            return

        # BRANCHING CASE
        boxid = sorted_box_list[box_index]
        orientations = box_orientations_dict[boxid]
        sorted_extpts = sorted(pallet.extpts)

        seen_profile_keys = set() if not use_guarantee else None                                            # Filter 1 (Deduplication): Initialize set of unique seen profiles

        if dimension_tuples[box_index] == dimension_tuples[box_index - 1]:                                  # Filter 2 (Symmetry Breaking): Check if current box matches dimensions of previous box and record (dims, x, y) as comparison key
            symmetry_key = current_sequence[-1]
        else:
            symmetry_key = None

        candidate_placements = []                                                                           # Make list of valid placement candidates to loop through
        for dims in orientations:
            dz = dims[2]
            for x, y in sorted_extpts:
                landing_z = get_valid_z(dims, x, y)
                if landing_z != -1:
                    top_z     = landing_z + dz
                    score     = (top_z, x + y)
                    candidate_placements.append((score, dims, x, y, landing_z))

        if not use_guarantee and num_extpts_to_try is not None:                                             # Filter 5 (Top-X candidates): if optimality need not be guaranteed and candidate limit is specified, score all orientation + placement candidates by resultant box height, then lowest x+y placement, and only branch on the top X scoring candidates
            candidate_placements.sort(key=_FIRST_ITEM)
            candidate_placements_filtered = max(0, len(candidate_placements) - num_extpts_to_try)
            candidate_placements = candidate_placements[:num_extpts_to_try]
            if candidate_placements_filtered > 0:
                counter_filt5.update(candidate_placements_filtered)
                count_filt5 += candidate_placements_filtered

        # Iterate through valid candidate placements, attempt to place box, and recurse after each valid placement
        for _, dims, x, y, z in candidate_placements:
            if not use_guarantee:                                                                           # Filter 1 (Deduplication): if optimality need not be guaranteed, check if placing the box in the same orientation at another extreme point leads to the same resultant z-height. This is likely to lead to a very similar heightmap structure, so prune all these candidate branches except for one representative. 
                profile_key = (dims, z)
                if profile_key in seen_profile_keys:
                    counter_filt1.update(1)
                    count_filt1 += 1
                    continue

            if symmetry_key is not None and symmetry_key > (dims, x, y):                                    # Filter 2 (Symmetry Breaking): if a comparison key is registered and the candidate placement is smaller than the key (checked value by value in the tuples ((dx, dy, dz), x, y) ), prune branch as it would lead to an identical resultant pallet in terms of dimensions but with different boxes (of the same or different box IDs, like a type 8 or 10 box, which are dimensionally identical) occupying the same place.
                counter_filt2.update(1)
                count_filt2 += 1
                continue

            delta = pallet.place_box(dims, x, y)                                                            # Place box if branch not filtered
            if not delta:
                continue

            if not use_guarantee:                                                                           # Only add comparison key if box was successfully placed
                seen_profile_keys.add(profile_key)

            current_sequence.append((dims, x, y))                                                           # Add placement to sequence and recurse
            pbar.update(1)
            count_nodes += 1
            recursive_place(box_index + 1)
            current_sequence.pop()
            pallet.remove_box(delta)

    # Initialize non-tqdm counters for nodes and pruning for metrics extraction
    count_nodes         = 0
    count_bound1        = 0
    count_bound4        = 0
    count_filt1         = 0
    count_filt2         = 0
    count_filt5         = 0

    # Start the recursive search starting with the first box (index 0) and keep track of branches and bounds
    pbar                = tqdm(desc="Evaluating Placements", unit=" nodes", leave=leave_tqdm, total=round(node_count_estimate))
    counter_bound1      = tqdm(desc="Number of branches pruned by bounding rule 1",     unit=" prunes", leave=leave_tqdm)
    counter_bound4      = tqdm(desc="Number of branches pruned by bounding rule 4",     unit=" prunes", leave=leave_tqdm)
    counter_filt1       = tqdm(desc="Number of branches filtered by deduplication",     unit=" prunes", leave=leave_tqdm)
    counter_filt2       = tqdm(desc="Number of branches filtered by symmetry breaking", unit=" prunes", leave=leave_tqdm)
    counter_filt5       = tqdm(desc="Number of branches filtered by top-x limiting",    unit=" prunes", leave=leave_tqdm)
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
        'pruned_filt1':         count_filt1,
        'pruned_filt2':         count_filt2,
        'pruned_filt5':         count_filt5,
        'best_score':           best_score,
        'optimality_guarantee': use_guarantee,
        'topx_limit':           num_extpts_to_try
    }
    return bnb_stats

def place_box_list_branch_and_bound_mc(pallet, box_list, criterion=DEFAULT_CRITERION, leave_tqdm=False, optimality_guarantee=None, num_extpts_to_try=None, cores=1):
    """Exact multi-core Branch-and-Bound with dynamic work splitting.

    Guarantees results bit-for-bit identical to `place_box_list_branch_and_bound()`
    (exact same placement sequence, heightmap, and max_z):
      - Uses leftmost-first priority dispatch (`heapq` by DFS path key) and
        on-demand top-of-stack dynamic work splitting across workers, avoiding
        both upfront serial BFS bottlenecks and static-partition stragglers.
      - Tracks certified solutions (`max(ancestor Rule-4) <= final score`) in a
        shared Pareto-front so Rule 1 and Rule 4 never prune any branch that
        single-core DFS would explore, and verifies any rare non-monotonic
        Rule-4 candidate exactly before committing the final result.
    """
    global _BNB_MC_CONTEXT

    if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
        raise ValueError("cores must be a positive integer")

    if cores == 1 or len(box_list) <= 2:
        return place_box_list_branch_and_bound(
            pallet,
            box_list,
            criterion=criterion,
            leave_tqdm=leave_tqdm,
            optimality_guarantee=optimality_guarantee,
            num_extpts_to_try=num_extpts_to_try,
        )

    # 1. Prework & precalculations (identical to single-core BnB)
    sorted_box_list = sort_box_list_by_size(
        box_list, criterion=criterion, invert=False
    )
    use_guarantee = (
        BNB_OPTIMALITY_GUARANTEE
        if optimality_guarantee is None
        else optimality_guarantee
    )
    pallet_dims = (pallet.size_x, pallet.size_y, pallet.size_z)
    n_boxes = len(sorted_box_list)

    temp_pallet = Pallet(dims=pallet_dims)
    place_box_list_best_fit_decreasing(
        temp_pallet, sorted_box_list, criterion=criterion
    )
    initial_best_score = int(temp_pallet.get_max_height() + 1)

    box_orientations_dict = {}
    for boxid in set(sorted_box_list):
        dx, dy, dz, _, _ = get_box_properties_from_id(boxid)
        box_orientations_dict[boxid] = get_box_orientations(dx, dy, dz)

    tallest_remaining_orientations = []
    for i in range(n_boxes):
        tallest_remaining_id = max(
            sorted_box_list[i:],
            key=lambda bid: min(get_box_properties_from_id(bid)[:3]),
        )
        tallest_remaining_orientations.append(
            box_orientations_dict[tallest_remaining_id]
        )

    dimension_tuples = [
        tuple(sorted(get_box_properties_from_id(boxid)[:3]))
        for boxid in sorted_box_list
    ]

    # 2. Fast leftmost-first seed expansion (capped at <= 64 expansions, ~0.01-0.3s):
    #    Repeatedly expands ONLY the leftmost node in the priority heap so all
    #    workers start concentrated inside the leftmost branch (discovering tight
    #    bounds immediately instead of wasting cycles speculating on far-right
    #    root siblings before the first incumbent is found). Reuses a single
    #    Pallet with LCP backtracking (zero 5.6 MB allocations) and records full
    #    Rule-4 spikes on every seed task so exactness is 100% preserved.
    seed_counts = _bnb_zero_counts()
    seed_pallet = Pallet(dims=pallet_dims)
    seed_cur_seq = []
    seed_cur_deltas = []
    size_z_plus_100 = pallet_dims[2] + 100

    cert_front = []
    best_cert_path = None
    best_cert_seq = None
    best_cert_score = initial_best_score
    anomalous_candidates = []

    def seed_transition_to(target_seq):
        lcp = 0
        limit = min(len(seed_cur_seq), len(target_seq))
        while lcp < limit and seed_cur_seq[lcp] == target_seq[lcp]:
            lcp += 1
        while len(seed_cur_seq) > lcp:
            seed_pallet.remove_box(seed_cur_deltas.pop())
            seed_cur_seq.pop()
        for move in target_seq[lcp:]:
            dims, x, y = move
            delta = seed_pallet.place_box(dims, x, y)
            if not delta:
                return False
            seed_cur_deltas.append(delta)
            seed_cur_seq.append(move)
        return True

    # Seed heap entries: (path_key, seq, spikes, min_max_z)
    task_heap = [((), [], (), 0)]
    target_seed_tasks = max(cores * 2, 8)
    max_seed_expansions = min(64, max(cores, 4))
    seed_expansions = 0
    max_split_depth = 0

    while task_heap and len(task_heap) < target_seed_tasks and seed_expansions < max_seed_expansions:
        node_path, node_seq, node_spikes, node_min_max_z = heapq.heappop(task_heap)
        if _bnb_spikes_disproved_by_front(node_spikes, cert_front):
            continue
        r1_limit, r4_limit = _bnb_cert_front_limits(
            cert_front, node_path, initial_best_score
        )
        if node_min_max_z >= r1_limit:
            if len(node_seq) > 0:
                seed_counts['nodes'] += 1
            seed_counts['pruned_rule1'] += 1
            continue

        if not seed_transition_to(node_seq):
            continue

        b_idx = len(node_seq)
        if b_idx > 0:
            seed_counts['nodes'] += 1

        if b_idx == n_boxes:
            sc = int(seed_pallet.get_max_height())
            if sc < r1_limit:
                active_spikes = tuple(
                    (ap, ar4) for ap, ar4 in node_spikes if ar4 > sc
                )
                if not active_spikes:
                    cert_front, changed = _bnb_merge_cert_front(
                        cert_front, node_path, sc, initial_best_score
                    )
                    if changed and (
                        best_cert_seq is None
                        or sc < best_cert_score
                        or (sc == best_cert_score and node_path < best_cert_path)
                    ):
                        best_cert_path = node_path
                        best_cert_seq = list(node_seq)
                        best_cert_score = sc
                elif not _bnb_spikes_disproved_by_front(active_spikes, cert_front):
                    anomalous_candidates.append((node_path, list(node_seq), sc, active_spikes))
            continue

        cur_max_z = int(seed_pallet.get_max_height())
        if cur_max_z >= r1_limit:
            seed_counts['pruned_rule1'] += 1
            continue

        r4_early_exit = cur_max_z if cur_max_z < r4_limit else (r4_limit - 1)
        min_landing_z = int(_bnb_compute_rule4_min_z(
            seed_pallet, tallest_remaining_orientations[b_idx], r4_early_exit
        ))
        if size_z_plus_100 > min_landing_z >= r4_limit:
            seed_counts['pruned_rule4'] += 1
            continue

        child_spikes = node_spikes
        if size_z_plus_100 > min_landing_z > cur_max_z:
            child_spikes = node_spikes + ((node_path, min_landing_z),)

        valid_children = _bnb_generate_valid_children(
            seed_pallet, b_idx, seed_cur_seq, sorted_box_list, use_guarantee,
            num_extpts_to_try, box_orientations_dict, dimension_tuples,
            seed_counts,
        )
        seed_expansions += 1
        if b_idx + 1 > max_split_depth:
            max_split_depth = b_idx + 1

        for orig_idx, move, child_top_z in valid_children:
            child_min_max_z = max(cur_max_z, child_top_z)
            heapq.heappush(
                task_heap,
                (node_path + (orig_idx,), node_seq + [move], child_spikes, child_min_max_z),
            )

    # Trivial / dead-end tree resolved completely during seed: no workers needed
    if not task_heap:
        surviving_anomalies = [
            cand for cand in anomalous_candidates
            if (
                (
                    best_cert_seq is None
                    or cand[2] < best_cert_score
                    or (cand[2] == best_cert_score and cand[0] < best_cert_path)
                )
                and not _bnb_spikes_disproved_by_front(cand[3], cert_front)
            )
        ]
        if surviving_anomalies:
            best_sequence, best_score = _bnb_mc_verify_anomalous_candidates(
                pallet_dims, sorted_box_list, use_guarantee, num_extpts_to_try,
                box_orientations_dict, tallest_remaining_orientations,
                dimension_tuples, initial_best_score, best_cert_path,
                best_cert_seq, best_cert_score, surviving_anomalies,
            )
        else:
            best_sequence = best_cert_seq
            best_score = best_cert_score

        if best_sequence:
            for dims, x, y in best_sequence:
                pallet.place_box(dims, x, y)
        else:
            print("Branch and Bound failed. No valid placements found.")
        return {
            'nodes': seed_counts['nodes'],
            'pruned_rule1': seed_counts['pruned_rule1'],
            'pruned_rule4': seed_counts['pruned_rule4'],
            'pruned_filt1': seed_counts['pruned_filt1'],
            'pruned_filt2': seed_counts['pruned_filt2'],
            'pruned_filt5': seed_counts['pruned_filt5'],
            'best_score': best_score,
            'optimality_guarantee': use_guarantee,
            'topx_limit': num_extpts_to_try,
            'tasks': 0,
            'task_depth': max_split_depth,
            'workers': 1,
        }

    # 3. Launch persistent worker pool & run event-driven dynamic work-splitting
    use_fork = "fork" in mp.get_all_start_methods()
    mp_ctx = mp.get_context("fork") if use_fork else mp.get_context()

    context_tuple = (
        pallet_dims,
        sorted_box_list,
        use_guarantee,
        num_extpts_to_try,
        box_orientations_dict,
        tallest_remaining_orientations,
        dimension_tuples,
        initial_best_score,
    )

    worker_count = cores
    shared_split_tokens = mp_ctx.RawValue('i', 0)
    shared_split_lock = mp_ctx.Lock()
    shared_bound_version = mp_ctx.RawValue('i', 0)

    bound_version = 0

    totals = dict(seed_counts)
    total_tasks_dispatched = 0

    workers = []
    master_conns = {}
    conn_to_wid = {}
    idle_workers = []
    busy_workers = set()

    buffer_target = max(2, worker_count // 2)

    def refresh_split_tokens():
        n_idle = len(idle_workers)
        q_len = len(task_heap)
        if n_idle > 0:
            desired = min(worker_count * 2, n_idle + max(1, worker_count // 4))
        elif q_len < buffer_target:
            desired = buffer_target - q_len
        else:
            desired = 0
        with shared_split_lock:
            shared_split_tokens.value = desired

    def dispatch_to_idle():
        nonlocal total_tasks_dispatched
        while idle_workers and task_heap:
            task = heapq.heappop(task_heap)
            task_path, task_seq, task_spikes, task_min_max_z = task
            # Skip any queued subtree whose ancestor spike is already disproved
            if _bnb_spikes_disproved_by_front(task_spikes, cert_front):
                continue
            # Skip any queued subtree already pruned by Rule 1
            r1_limit, _ = _bnb_cert_front_limits(
                cert_front, task_path, initial_best_score
            )
            if task_min_max_z >= r1_limit:
                totals['nodes'] += 1
                totals['pruned_rule1'] += 1
                continue
            wid = idle_workers.pop()
            busy_workers.add(wid)
            master_conns[wid].send(("TASK", task))
            total_tasks_dispatched += 1
        refresh_split_tokens()

    def broadcast_bound():
        nonlocal bound_version
        bound_version += 1
        payload = ("BOUND", list(cert_front), bound_version)
        for conn in master_conns.values():
            conn.send(payload)
        shared_bound_version.value = bound_version

    prev_context = _BNB_MC_CONTEXT
    if use_fork:
        _BNB_MC_CONTEXT = context_tuple

    try:
        for wid in range(worker_count):
            parent_conn, child_conn = mp_ctx.Pipe(duplex=True)
            proc = mp_ctx.Process(
                target=_bnb_mc_worker_main,
                args=(
                    wid,
                    child_conn,
                    shared_split_tokens,
                    shared_split_lock,
                    shared_bound_version,
                    None if use_fork else context_tuple,
                ),
                daemon=False,
            )
            proc.start()
            child_conn.close()
            workers.append(proc)
            master_conns[wid] = parent_conn
            conn_to_wid[parent_conn] = wid
            idle_workers.append(wid)

        # Keep lower worker IDs popped first so worker 0 takes the leftmost task
        idle_workers.reverse()
        if cert_front:
            broadcast_bound()
        dispatch_to_idle()

        active_conns = list(master_conns.values())

        while busy_workers or task_heap:
            ready_conns = mp_wait(active_conns)
            for conn in ready_conns:
                while conn.poll():
                    try:
                        msg = conn.recv()
                    except EOFError:
                        wid = conn_to_wid[conn]
                        raise RuntimeError(
                            f"BnB worker {wid} exited unexpectedly before completing search."
                        )

                    tag = msg[0]
                    if tag == "DONATE":
                        _, _, donated_tasks = msg
                        for d_task in donated_tasks:
                            d_path, d_seq, d_spikes, d_min_max_z = d_task
                            if _bnb_spikes_disproved_by_front(d_spikes, cert_front):
                                continue
                            r1_limit, _ = _bnb_cert_front_limits(
                                cert_front, d_path, initial_best_score
                            )
                            if d_min_max_z >= r1_limit:
                                totals['nodes'] += 1
                                totals['pruned_rule1'] += 1
                                continue
                            d_depth = len(d_seq)
                            if d_depth > max_split_depth:
                                max_split_depth = d_depth
                            heapq.heappush(task_heap, d_task)
                    elif tag == "SOLUTION":
                        _, _, is_cert, s_path, s_seq, s_score, s_spikes = msg
                        if is_cert:
                            cert_front, changed = _bnb_merge_cert_front(
                                cert_front, s_path, s_score, initial_best_score
                            )
                            if changed:
                                if (
                                    best_cert_seq is None
                                    or s_score < best_cert_score
                                    or (s_score == best_cert_score and s_path < best_cert_path)
                                ):
                                    best_cert_path = s_path
                                    best_cert_seq = s_seq
                                    best_cert_score = s_score
                                broadcast_bound()
                        else:
                            if not _bnb_spikes_disproved_by_front(s_spikes, cert_front):
                                anomalous_candidates.append((s_path, s_seq, s_score, s_spikes))
                    elif tag == "DONE":
                        _, wid = msg
                        busy_workers.discard(wid)
                        idle_workers.append(wid)
                    elif tag == "ERROR":
                        _, wid, tb = msg
                        raise RuntimeError(f"BnB worker {wid} failed:\n{tb}")

            dispatch_to_idle()

        # Gracefully stop workers and collect exact aggregated statistics
        with shared_split_lock:
            shared_split_tokens.value = 0

        for wid, conn in master_conns.items():
            conn.send(("STOP",))

        pending_stats = set(master_conns.keys())
        while pending_stats:
            conns_left = [master_conns[w] for w in pending_stats]
            for conn in mp_wait(conns_left):
                while conn.poll():
                    try:
                        msg = conn.recv()
                    except EOFError:
                        pending_stats.discard(conn_to_wid[conn])
                        break
                    if msg[0] == "FINAL_STATS":
                        _, wid, w_counts = msg
                        for k in totals:
                            totals[k] += w_counts[k]
                        pending_stats.discard(wid)
                    elif msg[0] == "ERROR":
                        _, wid, tb = msg
                        raise RuntimeError(f"BnB worker {wid} failed during shutdown:\n{tb}")

        for proc in workers:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
    finally:
        _BNB_MC_CONTEXT = prev_context
        for conn in master_conns.values():
            try:
                conn.close()
            except Exception:
                pass
        for proc in workers:
            if proc.is_alive():
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1.0)
                except Exception:
                    pass

    # 4. Resolve any surviving non-monotonic Rule-4 anomalous candidates exactly
    surviving_anomalies = [
        cand for cand in anomalous_candidates
        if (
            (
                best_cert_seq is None
                or cand[2] < best_cert_score
                or (cand[2] == best_cert_score and cand[0] < best_cert_path)
            )
            and not _bnb_spikes_disproved_by_front(cand[3], cert_front)
        )
    ]

    if surviving_anomalies:
        best_sequence, best_score = _bnb_mc_verify_anomalous_candidates(
            pallet_dims,
            sorted_box_list,
            use_guarantee,
            num_extpts_to_try,
            box_orientations_dict,
            tallest_remaining_orientations,
            dimension_tuples,
            initial_best_score,
            best_cert_path,
            best_cert_seq,
            best_cert_score,
            surviving_anomalies,
        )
    else:
        best_sequence = best_cert_seq
        best_score = best_cert_score

    # 5. Reconstruct optimal pallet state
    if best_sequence:
        for dims, x, y in best_sequence:
            pallet.place_box(dims, x, y)
    else:
        print("Branch and Bound failed. No valid placements found.")

    return {
        'nodes': totals['nodes'],
        'pruned_rule1': totals['pruned_rule1'],
        'pruned_rule4': totals['pruned_rule4'],
        'pruned_filt1': totals['pruned_filt1'],
        'pruned_filt2': totals['pruned_filt2'],
        'pruned_filt5': totals['pruned_filt5'],
        'best_score': best_score,
        'optimality_guarantee': use_guarantee,
        'topx_limit': num_extpts_to_try,
        'tasks': total_tasks_dispatched,
        'task_depth': max_split_depth,
        'workers': worker_count,
    }


# %% [markdown]
# #### Testing Functions

# %%
def run_random_fulfillment_test(start_order=1000, end_order=3999, order_dict=test_orders_dict, trials=500, max_max_attempts=10000, step=10, save_result=True):
    # Final lists for plotting, max_attempts values on x axis, average fulfillment on y-axis
    x_axis = [] 
    y_axis = []

    # If no final order is specified, let the final order be the last entry in the dict
    if end_order is None:
        end_order = max(order_dict.keys())

    # Get the order IDs to test between the start and end order IDs
    order_ids = [order_id for order_id in sorted(order_dict.keys()) if start_order <= order_id <= end_order]
    
    # Prevent running on emtpy list
    if not order_ids:
        print("No orders found in the specified range.")
        return

    # Create an evenly spaced array of indices from 0 to the last index of order_ids
    indices = np.round(np.linspace(0, len(order_ids) - 1, trials)).astype(int)
    
    # Map those indices back to the actual order IDs
    selected_order_ids = [order_ids[i] for i in indices]

    # Use max and step size to get final range of max_attempts values (x-axis)
    max_attempts_range = range(step, max_max_attempts + 1, step)

    for max_attempts in tqdm(max_attempts_range, desc="Testing max_attempts values", position=0, leave=True):
        # Keep track of scores for this max_attempts value
        scores = []

        # Iterate directly through our pre-calculated deterministic list
        for order_id in tqdm(selected_order_ids, leave=False):
            box_list = get_box_list_from_order(order_id, order_dict)

            # Initialize pallet and pack it with the random algorithm
            pallet = Pallet()
            place_box_list_random(pallet, box_list, max_attempts)

            # Get fulfillment score based on chosen order dict
            score = pallet.check_order_fullfillment(order_id, order_dict)
            
            # Add score to scores list
            scores.append(score)

        # Average the scores for this max_attempts value
        avg_score = np.mean(scores)

        # Add the datapoints to the final lists
        x_axis.append(max_attempts)
        y_axis.append(avg_score)

    # Plot the results
    plt.figure()
    plt.plot(x_axis, y_axis)

    plt.title(f"Random placement algorithm fulfillment scores ({trials} trials/step)")
    plt.xlabel("Maximum attempts per box")
    plt.ylabel('Average order fulfillment percentage')
    plt.ylim(0, 105)

    if save_result:
        save_path = './results/figures/random_fulfillment_test.png'
        plt.savefig(save_path, dpi=300)

    plt.show()

def run_optimality_guarantee_test(start_order=1, end_order=None, order_dict=test_orders_dict, criterion=DEFAULT_CRITERION, metric=Metric.MAX_Z, print_pallets=False, save_pallets=False, output_dir=None, verbose=True):    # Run BnB with and without the optimality guarantee on a range of orders and export a comparison CSV
    # If no final order is specified, let the final order be the last entry in the dict
    if end_order is None:
        end_order = max(order_dict.keys())

    # Base final filename on whether given or test orders are tested
    if order_dict == orders_dict:
        prefix = "O"
    else:
        prefix = "T"

    if output_dir is None:
        output_dir = "./results/optg_comparisons"

    # Get the order IDs to test between the start and end order IDs
    order_ids = [order_id for order_id in sorted(order_dict.keys()) if start_order <= order_id <= end_order]

    result_rows = []
    order_iter = tqdm(order_ids, desc="Testing orders", unit=" orders") if verbose else order_ids

    for order_id in order_iter:
        if verbose:
            print(f"\n----------------------------------------------------------------------------------------------------------------------------")
            print(f"----------------------------------------------------------------------------------------------------------------------------")
            print(f"Running order {order_id} with guarantee off...")

        # Run order with no optimality guarantee and get stats
        pallet_no_guarantee = Pallet()
        box_list = get_box_list_from_order(order_id, order_dict)
        stats_no_guarantee = place_box_list_branch_and_bound(
            pallet_no_guarantee, box_list, criterion=criterion, leave_tqdm=False, optimality_guarantee=False
        )
        score_no_guarantee = pallet_no_guarantee.get_max_height() if metric == Metric.MAX_Z else None
        if print_pallets or save_pallets:
            pallet_no_guarantee.get_pallet_results(algo=Algorithm.BNB, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=stats_no_guarantee)

        if verbose:
            print(f"\n----------------------------------------------------------------------------------------------------------------------------")
            print(f"Running order {order_id} with guarantee on...")

        # Run order with optimality guarantee and get stats
        pallet_with_guarantee = Pallet()
        stats_with_guarantee = place_box_list_branch_and_bound(
            pallet_with_guarantee, box_list, criterion=criterion, leave_tqdm=False, optimality_guarantee=True
        )
        score_with_guarantee = pallet_with_guarantee.get_max_height() if metric == Metric.MAX_Z else None
        if print_pallets or save_pallets:
            pallet_with_guarantee.get_pallet_results(algo=Algorithm.BNB, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=stats_with_guarantee)

        # Calculate differences, absolute and relative
        def calculate_row_difference(val_no_guarantee, val_with_guarantee):
            abs_diff = abs(val_no_guarantee - val_with_guarantee)
            factor   = round(val_no_guarantee / val_with_guarantee, 4) if val_with_guarantee != 0 else "N/A" # Avoid division by 0
            return abs_diff, factor

        nodes_diff, nodes_factor    = calculate_row_difference(stats_no_guarantee['nodes'],                 stats_with_guarantee['nodes'])
        r1_diff,    r1_factor       = calculate_row_difference(stats_no_guarantee['pruned_rule1'],          stats_with_guarantee['pruned_rule1'])
        r4_diff,    r4_factor       = calculate_row_difference(stats_no_guarantee['pruned_rule4'],          stats_with_guarantee['pruned_rule4'])
        f1_diff,    f1_factor       = calculate_row_difference(stats_no_guarantee['pruned_filt1'],          stats_with_guarantee['pruned_filt1'])
        f2_diff,    f2_factor       = calculate_row_difference(stats_no_guarantee['pruned_filt2'],          stats_with_guarantee['pruned_filt2'])
        f5_diff,    f5_factor       = calculate_row_difference(stats_no_guarantee['pruned_filt5'],          stats_with_guarantee['pruned_filt5'])

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
            # Filtered by deduplication
            'filt1_no_guarantee':       stats_no_guarantee['pruned_filt1'],
            'filt1_with_guarantee':     stats_with_guarantee['pruned_filt1'],
            'filt1_diff_abs':           f1_diff,
            'filt1_diff_factor':        f1_factor,
            # Filtered by symmetry breaking
            'filt2_no_guarantee':       stats_no_guarantee['pruned_filt2'],
            'filt2_with_guarantee':     stats_with_guarantee['pruned_filt2'],
            'filt2_diff_abs':           f2_diff,
            'filt2_diff_factor':        f2_factor,
            # Filtered by top-x limiting
            'filt5_no_guarantee':       stats_no_guarantee['pruned_filt5'],
            'filt5_with_guarantee':     stats_with_guarantee['pruned_filt5'],
            'filt5_diff_abs':           f5_diff,
            'filt5_diff_factor':        f5_factor,
        })

        if verbose:
            print(f"----------------------------------------------------------------------------------------------------------------------------")
            print(f"Score  : {score_no_guarantee} (no guarantee) vs {score_with_guarantee} (with guarantee)")
            print(f"Nodes  : {stats_no_guarantee['nodes']:,} vs {stats_with_guarantee['nodes']:,}  (Difference: {nodes_diff:+,}, Factor: {nodes_factor})")

    # Export to CSV atomically
    output_csv = os.path.join(output_dir, f"optg_comparison_{prefix}{start_order}_to_{prefix}{end_order}.csv")
    results_df = pd.DataFrame(result_rows)
    _write_csv_atomic(results_df, output_csv)
    if verbose:
        print(f"\n----------------------------------------------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"Results saved to: {output_csv}")
    return results_df

def run_topx_limiting_test(start_order=1, end_order=None, order_dict=test_orders_dict, criterion=DEFAULT_CRITERION, print_pallets=False, save_pallets=False, topx_min=1, topx_max=5, topx_step=1, topx_test_without=True, output_dir=None, verbose=True):     # Run BnB with a range of top-X candidate limits on a range of orders and export a comparison CSV
    # If no final order is specified, run until the end of the dict
    if end_order is None:
        end_order = max(order_dict.keys())

    if output_dir is None:
        output_dir = "./results/topx_comparisons"

    # Get the order IDs to test between the start and end order IDs
    order_ids = [order_id for order_id in sorted(order_dict.keys()) if start_order <= order_id <= end_order]

    # Generate list of limits to test
    topx_values_to_test = list(range(topx_min, topx_max + 1, topx_step))
    if topx_test_without:
        topx_values_to_test.append(None) # None will trigger the standard, unfiltered BnB run

    result_rows = []
    order_iter = tqdm(order_ids, desc="Testing orders", unit=" orders") if verbose else order_ids

    for order_id in order_iter:
        if verbose:
            print(f"\n----------------------------------------------------------------------------------------------------------------------------")
            print(f"----------------------------------------------------------------------------------------------------------------------------")
            print(f"Running order {order_id}...")
        
        box_list = get_box_list_from_order(order_id, order_dict)

        for topx_val in topx_values_to_test:
            starttime = time.perf_counter()
            limit_label = f"Top-{topx_val}" if topx_val is not None else "No Limit"
            if verbose:
                print(f"\n  Running with {limit_label}...")

            pallet = Pallet()
            
            # Run BnB (single-core for exact node/time comparability across limits)
            stats = place_box_list_branch_and_bound(pallet, box_list, criterion=criterion, leave_tqdm=False, optimality_guarantee=False, num_extpts_to_try=topx_val)
            score = pallet.get_max_height()
            
            # Get results based on print and save mode
            if print_pallets or save_pallets:
                pallet.get_pallet_results(algo=Algorithm.BNB, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=stats)

            # Record metrics (sub-second precision preserved in (seconds) column)
            total_pruned = (stats['pruned_rule1'] + stats['pruned_rule4'] + stats['pruned_filt1'] + stats['pruned_filt2'] + stats['pruned_filt5'])
            secs_taken = time.perf_counter() - starttime
            mins_taken = int(secs_taken // 60)
            restseconds = round(secs_taken - (mins_taken * 60), 4)

            result_rows.append({
                'order_id'        : order_id,
                'topx_limit'      : topx_val if topx_val is not None else -1,
                'time_taken (min)': mins_taken,
                '(seconds)'       : restseconds,
                'score'           : score,
                'nodes'           : stats['nodes'],
                'pruned_rule1'    : stats['pruned_rule1'],
                'pruned_rule4'    : stats['pruned_rule4'],
                'pruned_filt1'    : stats['pruned_filt1'],
                'pruned_filt2'    : stats['pruned_filt2'],
                'pruned_filt5'    : stats['pruned_filt5'],
                'total_pruned'    : total_pruned,
            })

            if verbose:
                print(f"Finished! Score: {score} | Nodes: {stats['nodes']:,} | Filter 5 Prunes: {stats['pruned_filt5']:,} | Time taken: {secs_taken:.3f} seconds")

    # Export to CSV atomically (filename uses O{id} for compatibility with data_processing.ipynb)
    output_csv = os.path.join(output_dir, f"topx_test_O{start_order}-O{end_order}_L{topx_min}-L{topx_max}-S{topx_step}.csv")
    results_df = pd.DataFrame(result_rows)
    _write_csv_atomic(results_df, output_csv)
    
    if verbose:
        print(f"\n----------------------------------------------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"Testing complete! Results saved to: {output_csv}")
    
    return results_df

def run_algorithm_comparison_test(start_order=1, end_order=None, order_dict=test_orders_dict, criterion=DEFAULT_CRITERION, metric=Metric.MAX_Z, print_pallets=False, save_pallets=False, random_attempts=DEFAULT_MAX_ATTEMPTS, bnb_topx=BNB_TOPX_DEFAULT_LIMIT, output_dir=None, use_mp=False, verbose=True):     
    # If no final order is specified, run until the end of the dict
    if end_order is None:
        end_order = max(order_dict.keys())

    # Base final filename on whether given or test orders are tested
    if order_dict == orders_dict:
        prefix = "O"
    else:
        prefix = "T"

    if output_dir is None:
        output_dir = "./results/algo_comparisons"

    # Get the order IDs to test between the start and end order IDs
    order_ids = [order_id for order_id in sorted(order_dict.keys()) if start_order <= order_id <= end_order]

    result_rows = []
    order_iter = tqdm(order_ids, desc="Testing orders across algorithms", unit=" orders") if verbose else order_ids

    for order_id in order_iter:
        if verbose:
            print(f"\n----------------------------------------------------------------------------------------------------------------------------")
            print(f"----------------------------------------------------------------------------------------------------------------------------")
            print(f"Running order {order_id}...")
        
        # Iterate over every algorithm in the enum
        for algo in Algorithm:
            if verbose:
                print(f"\n  Running {algo.value.upper()}...")
            starttime = time.perf_counter()
            
            # Process the order based on the algorithm type (use_mp=False keeps single-core comparison fair and pool-safe)
            if algo == Algorithm.BNB:
                pallet, bnb_stats = process_order(order_id, algo, criterion=criterion, metric=metric, order_dict=order_dict, leave_tqdm=False, optimality_guarantee=False, num_extpts_to_try=bnb_topx, use_mp=use_mp)
                nodes = bnb_stats['nodes']
            else:
                pallet, nodes = process_order(order_id, algo, max_attempts=random_attempts, criterion=criterion, metric=metric, order_dict=order_dict, leave_tqdm=False)
                bnb_stats = None
                
            endtime = time.perf_counter()
            secs_taken = round(endtime - starttime, 4)
            
            # Extract metrics
            fulfillment = pallet.check_order_fullfillment(order_id, order_dict)
            volume_util = pallet.get_volume_utilization()
            area_usage_z0 = pallet.get_area_usage_at_z(0)
            cog_z = pallet.get_center_of_gravity_z()
            packing_score = pallet.get_packing_score()
            max_z = pallet.get_max_height()
            
            # Print and/or save the pallet visualizations
            if print_pallets or save_pallets:
                pallet.get_pallet_results(algo=algo, orderID=order_id, order_dict=order_dict, print_mode=print_pallets, save_mode=save_pallets, bnb_stats=bnb_stats, nodes=nodes)

            # Collate run data
            row_data = {
                'order_id': order_id,
                'algorithm': algo.value.upper(),
                'time_taken (s)': secs_taken,
                'score (max_z)': max_z,
                'packing_score': packing_score,
                'fulfillment (%)': fulfillment,
                'volume_util (%)': volume_util,
                'area_usage_z0 (%)': area_usage_z0,
                'cog_z': cog_z,
                'nodes': nodes,
            }
            
            # Add BnB specific stats if applicable, otherwise fill with N/A
            if algo == Algorithm.BNB and bnb_stats is not None:
                total_pruned = (bnb_stats['pruned_rule1'] + bnb_stats['pruned_rule4'] + bnb_stats['pruned_filt1'] + bnb_stats['pruned_filt2'] + bnb_stats['pruned_filt5'])
                row_data.update({
                    'pruned_rule1': bnb_stats['pruned_rule1'],
                    'pruned_rule4': bnb_stats['pruned_rule4'],
                    'pruned_filt1': bnb_stats['pruned_filt1'],
                    'pruned_filt2': bnb_stats['pruned_filt2'],
                    'pruned_filt5': bnb_stats['pruned_filt5'],
                    'total_pruned': total_pruned
                })
            else:
                row_data.update({
                    'pruned_rule1': None, 
                    'pruned_rule4': None,
                    'pruned_filt1': None,
                    'pruned_filt2': None,
                    'pruned_filt5': None,
                    'total_pruned': None
                })
                
            result_rows.append(row_data)
            if verbose:
                print(f"    -> Finished {algo.value.upper()} | Score: {max_z} | Fulfillment: {fulfillment}% | Time: {secs_taken}s")

    # Export to csv atomically
    output_csv = os.path.join(output_dir, f"algorithm_comparison_{prefix}{start_order}_to_{prefix}{end_order}.csv")
    results_df = pd.DataFrame(result_rows)
    _write_csv_atomic(results_df, output_csv)
    
    if verbose:
        print(f"\n----------------------------------------------------------------------------------------------------------------------------")
        print(f"----------------------------------------------------------------------------------------------------------------------------")
        print(f"Testing complete! Results saved to: {output_csv}")
    
    return results_df

def run_bnb_mc_speed_comparison(mp_core_max=DEFAULT_MP_CORES, start_order=1, end_order=None, order_dict=test_orders_dict, criterion=DEFAULT_CRITERION, metric=Metric.MAX_Z, print_pallets=False, save_pallets=False, bnb_topx=BNB_TOPX_DEFAULT_LIMIT, optimality_guarantee=False, output_dir="./results/spee_comparisons"):
    """Compare serial and multi-core BnB runtimes and verify identical pallets.

    Runs one serial baseline per order followed by one multi-core run for each
    power-of-two core count up to `mp_core_max`. Verifies exact equality of
    canonical box placements, heightmap, and max_z against the serial baseline
    and atomically saves the combined CSV.
    """
    if end_order is None:
        end_order = max(order_dict.keys())

    prefix = "O" if order_dict == orders_dict else "T"

    order_ids = [
        order_id for order_id in sorted(order_dict.keys())
        if start_order <= order_id <= end_order
    ]
    if not order_ids:
        raise ValueError(
            f"No orders found from {start_order} through {end_order} in order_dict"
        )

    core_counts = get_core_counts_list(mp_core_max)
    all_core_counts = [1] + core_counts
    result_rows = []

    for order_id in tqdm(order_ids, desc="Testing orders", unit=" orders"):
        print("\n" + "-" * 124)
        print("-" * 124)
        print(f"Running serial baseline for order {prefix}{order_id}...")

        box_list = get_box_list_from_order(order_id, order_dict)
        serial_pallet = Pallet()

        serial_start = time.perf_counter()
        serial_stats = place_box_list_branch_and_bound(
            serial_pallet,
            box_list,
            criterion=criterion,
            leave_tqdm=False,
            optimality_guarantee=optimality_guarantee,
            num_extpts_to_try=bnb_topx,
        )
        serial_seconds = time.perf_counter() - serial_start
        serial_metric = _extract_pallet_metric(serial_pallet, metric)
        serial_boxes = _canonical_boxes(serial_pallet)
        serial_hm_hash = hashlib.sha256(serial_pallet.heightmap.tobytes()).hexdigest()

        serial_pallet.get_pallet_results(
            algo=Algorithm.BNB,
            orderID=order_id,
            order_dict=order_dict,
            print_mode=print_pallets,
            save_mode=save_pallets,
            bnb_stats=serial_stats,
        )

        results_map = {
            1: {
                'order_id': order_id,
                'cores': 1,
                'seconds': float(serial_seconds),
                'metric_value': float(serial_metric) if serial_metric is not None else None,
                'max_z': int(serial_pallet.get_max_height()),
                'box_count': len(serial_pallet.boxes),
                'stats': serial_stats,
                'canonical_boxes': serial_boxes,
                'heightmap_hash': serial_hm_hash,
            }
        }
        print(
            f"Serial finished | {serial_seconds:.6f}s | "
            f"max_z={serial_pallet.get_max_height()} | nodes={serial_stats['nodes']}"
        )

        for cores in core_counts:
            print(f"\nRunning order {prefix}{order_id} with {cores} processes...")
            mc_pallet = Pallet()

            mc_start = time.perf_counter()
            mc_stats = place_box_list_branch_and_bound_mc(
                mc_pallet,
                box_list,
                criterion=criterion,
                leave_tqdm=False,
                optimality_guarantee=optimality_guarantee,
                num_extpts_to_try=bnb_topx,
                cores=cores,
            )
            mc_seconds = time.perf_counter() - mc_start
            mc_metric = _extract_pallet_metric(mc_pallet, metric)
            mc_boxes = _canonical_boxes(mc_pallet)
            mc_hm_hash = hashlib.sha256(mc_pallet.heightmap.tobytes()).hexdigest()

            boxes_match = (mc_boxes == serial_boxes)
            heightmap_match = np.array_equal(mc_pallet.heightmap, serial_pallet.heightmap)
            max_z_match = (mc_pallet.get_max_height() == serial_pallet.get_max_height())
            pallets_match = boxes_match and heightmap_match and max_z_match
            speedup = serial_seconds / mc_seconds if mc_seconds > 0 else math.inf

            if not pallets_match:
                serial_box_set = set(serial_boxes)
                mc_box_set = set(mc_boxes)
                only_serial = sorted(serial_box_set - mc_box_set)
                only_mc = sorted(mc_box_set - serial_box_set)
                differing_cells = int(np.count_nonzero(
                    mc_pallet.heightmap != serial_pallet.heightmap
                ))
                max_heightmap_delta = int(np.max(np.abs(
                    mc_pallet.heightmap.astype(np.int64)
                    - serial_pallet.heightmap.astype(np.int64)
                )))
                print("\n*** PALLET MISMATCH DETECTED ***")
                print(f"Order: {prefix}{order_id} | Cores: {cores}")
                print(
                    f"Matches -> boxes: {boxes_match}, heightmap: {heightmap_match}, "
                    f"max_z: {max_z_match}"
                )
                print(
                    f"Serial max_z={serial_pallet.get_max_height()}, "
                    f"MC max_z={mc_pallet.get_max_height()}"
                )
                print(
                    f"Differing heightmap cells: {differing_cells}, "
                    f"max delta: {max_heightmap_delta}"
                )
                print(f"Only in serial (first 10): {only_serial[:10]}")
                print(f"Only in MC (first 10):     {only_mc[:10]}")
                print(f"Serial stats: {serial_stats}")
                print(f"MC stats:     {mc_stats}")

            mc_pallet.get_pallet_results(
                algo=Algorithm.BNB,
                orderID=order_id,
                order_dict=order_dict,
                print_mode=print_pallets,
                save_mode=save_pallets,
                bnb_stats=mc_stats,
            )

            results_map[cores] = {
                'order_id': order_id,
                'cores': cores,
                'seconds': float(mc_seconds),
                'metric_value': float(mc_metric) if mc_metric is not None else None,
                'max_z': int(mc_pallet.get_max_height()),
                'box_count': len(mc_pallet.boxes),
                'stats': mc_stats,
                'canonical_boxes': mc_boxes,
                'heightmap_hash': mc_hm_hash,
            }
            print(
                f"MP{cores} finished | {mc_seconds:.6f}s | "
                f"speedup={speedup:.3f}x | match={pallets_match}"
            )

        order_rows, _, _ = _build_speed_comparison_rows(
            order_id, prefix, metric, all_core_counts, results_map
        )
        result_rows.extend(order_rows)

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(
        output_dir,
        f"speed_comparison_{prefix}{start_order}_to_{prefix}{end_order}_MP-2-{mp_core_max}.csv",
    )
    results_df = pd.DataFrame(result_rows)
    _write_csv_atomic(results_df, output_csv)

    mismatch_count = int((~results_df['matches_single_core']).sum())
    print("\n" + "-" * 124)
    print("-" * 124)
    print(f"Testing complete! Results saved to: {output_csv}")
    print(f"Detected multi-core pallet mismatches: {mismatch_count}")
    return results_df

def run_smart_concurrent_bnb_mc_speed_comparison(
    order_list=None,
    mp_core_max=128,
    order_dict=None,
    criterion=DEFAULT_CRITERION,
    metric=Metric.MAX_Z,
    bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
    output_dir="./results/spee_comparisons/test_orders",
    total_machine_cores=None,
    max_active_orders=None,
    optimality_guarantee=False,
    status_interval_seconds=30.0,
):
    """Bounded-WIP, finishing-first overnight scheduler targeting ~100% CPU use.

    Designed for shared high-core-count compute servers (e.g. 128 cores):
      1. Resumable & crash-safe: skips orders that already have a saved CSV
         (checking both `spee_comparison_*.csv` and `speed_comparison_*.csv`)
         and writes each finished order atomically via temp file + `os.replace`.
      2. Bounded work-in-progress (WIP): keeps only a small sliding window of
         active orders (default ~16-32 on 128 cores instead of 128) and always
         prioritizes finishing active orders over starting new ones so CSVs
         flush steadily and interrupting early wastes minimal partial work.
      3. Two-phase exclusive/packed execution for pristine benchmark timings:
         - Full-width runs (`cores == total_machine_cores`, e.g. 128c) run in
           clean exclusive bursts with 100% of cores dedicated to one task
           (briefly `SIGSTOP`-pausing active 1-core tasks, which time via
           `time.process_time()` so pauses never distort measurements).
         - Sub-machine runs (`1, 2, 4, ..., C/2`) are bin-packed together with
           zero core oversubscription and zero pauses on multi-core tasks.
      4. Clean signal handling & process-group cleanup: on Ctrl+C / SIGTERM,
         terminates every worker's entire process group (`os.killpg`) so zero
         orphaned worker processes ever remain on the shared server.
    """
    if order_list is None:
        order_list = spee_missing_test_orders
    if order_dict is None:
        order_dict = test_orders_dict

    prefix = "O" if order_dict == orders_dict else "T"
    os.makedirs(output_dir, exist_ok=True)

    if total_machine_cores is None:
        total_machine_cores = os.cpu_count() or mp_core_max
    total_machine_cores = max(1, int(total_machine_cores))

    all_test_core_counts = [1] + [
        c for c in get_core_counts_list(mp_core_max) if c <= total_machine_cores
    ]
    full_width_cores = max(all_test_core_counts)
    has_full_width = (full_width_cores == total_machine_cores and full_width_cores > 1)
    supports_sigstop = hasattr(signal, "SIGSTOP") and hasattr(signal, "SIGCONT")

    # Compute Little's-Law bounded WIP window: enough concurrent orders to keep
    # 100% of cores fed without hoarding dozens of half-done orders in memory.
    if max_active_orders is None:
        soft_active_orders = max(4, min(20, math.ceil(total_machine_cores / 8)))
        hard_active_orders = max(soft_active_orders, min(32, math.ceil(total_machine_cores / 4)))
        if total_machine_cores < 4:
            soft_active_orders = max(1, total_machine_cores)
            hard_active_orders = soft_active_orders
    else:
        hard_active_orders = max(1, int(max_active_orders))
        soft_active_orders = min(hard_active_orders, max(4, min(20, math.ceil(total_machine_cores / 8))))
        if hard_active_orders < soft_active_orders:
            soft_active_orders = hard_active_orders

    # 1. Deduplicate and skip already-completed orders on disk
    pending_orders = []
    seen_oids = set()
    skipped_count = 0
    for oid in order_list:
        if oid in seen_oids:
            continue
        seen_oids.add(oid)
        if _find_existing_speed_csv(output_dir, prefix, oid) is not None:
            skipped_count += 1
        else:
            pending_orders.append(oid)

    print("=" * 100)
    print("SMART BnB MULTI-CORE BENCHMARK SCHEDULER (BOUNDED-WIP / FINISHING-FIRST)")
    print(f"Machine Core Capacity : {total_machine_cores} cores")
    print(f"Active Order Window   : soft={soft_active_orders}, hard={hard_active_orders}")
    print(f"Core Counts per Order : {all_test_core_counts}")
    print(f"Total Orders Given    : {len(seen_oids)}")
    print(f"Already Completed     : {skipped_count} (skipped)")
    print(f"Remaining to Process  : {len(pending_orders)}")
    print(f"Results Directory     : {output_dir}")
    print("=" * 100, flush=True)

    if not pending_orders:
        print("All target orders already have results in the output directory. Done!")
        return {
            'completed': 0,
            'skipped': skipped_count,
            'remaining': 0,
            'failed': [],
            'mismatches': [],
            'interrupted': False,
        }

    active_orders = []           # order_ids in admission order
    uncompleted_tasks = {}       # order_id -> set(core counts not yet completed)
    running_tasks = {}           # order_id -> set(core counts currently running)
    order_results = {}           # order_id -> {cores: result_dict}
    task_retries = {}            # (order_id, cores) -> crash retry count
    failed_orders = []           # order_ids retired due to unrecoverable errors
    completed_session = 0
    mismatch_orders = []

    # Active child processes: proc -> (conn, order_id, cores, start_time)
    active_processes = {}
    conn_to_proc = {}
    paused_serial_procs = []     # LIFO list of paused 1-core procs (newest paused first)
    paused_serial_set = set()

    # Scheduler mode:
    #   "FULL_WIDTH": draining MC to 0 (or running exclusive full-width bursts)
    #   "PACKING":    co-scheduling 1c + 2..C/2 tasks with zero oversubscription
    drain_for_full_width = has_full_width
    max_crash_retries = 1

    mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    def admit_order(oid):
        active_orders.append(oid)
        uncompleted_tasks[oid] = set(all_test_core_counts)
        running_tasks[oid] = set()
        order_results[oid] = {}

    def retire_order(oid):
        if oid in active_orders:
            active_orders.remove(oid)
        uncompleted_tasks.pop(oid, None)
        running_tasks.pop(oid, None)
        order_results.pop(oid, None)

    def kill_proc_group(proc, conn=None):
        """Cleanly terminate `proc` and its entire process group (zero orphans)."""
        pid = proc.pid
        if pid is not None:
            if supports_sigstop and proc in paused_serial_set:
                try:
                    os.kill(pid, signal.SIGCONT)
                except OSError:
                    pass
            if hasattr(os, "killpg"):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            else:
                try:
                    proc.terminate()
                except Exception:
                    pass
            proc.join(timeout=1.5)
            if proc.is_alive():
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                else:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                proc.join(timeout=1.0)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def launch_task(order_id, cores_to_run):
        box_list = get_box_list_from_order(order_id, order_dict)
        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
        proc = mp_ctx.Process(
            target=_bnb_speed_test_worker,
            args=(
                child_conn,
                order_id,
                cores_to_run,
                box_list,
                criterion,
                metric,
                bnb_topx,
                optimality_guarantee,
            ),
            daemon=False,
        )
        proc.start()
        child_conn.close()
        active_processes[proc] = (parent_conn, order_id, cores_to_run, time.monotonic())
        conn_to_proc[parent_conn] = proc
        running_tasks[order_id].add(cores_to_run)
        return proc

    def get_core_usage():
        mc_used = 0
        running_1c = []
        for p, (_, _, c, _) in active_processes.items():
            if c > 1:
                mc_used += c
            elif p not in paused_serial_set:
                running_1c.append(p)
        return mc_used, running_1c

    def pause_serial_procs(count_needed, running_1c):
        """SIGSTOP up to `count_needed` newest 1-core tasks so oldest finish first."""
        if not supports_sigstop or count_needed <= 0:
            return 0
        # Sort newest-admitted / most-recently-started first so oldest orders keep running
        ordered = sorted(
            running_1c,
            key=lambda p: (
                active_orders.index(active_processes[p][1])
                if active_processes[p][1] in active_orders else -1,
                active_processes[p][3],
            ),
            reverse=True,
        )
        paused_now = 0
        for p in ordered:
            if paused_now >= count_needed:
                break
            try:
                os.kill(p.pid, signal.SIGSTOP)
                paused_serial_set.add(p)
                paused_serial_procs.append(p)
                paused_now += 1
            except OSError:
                pass
        return paused_now

    def resume_serial_procs(max_to_resume):
        """SIGCONT up to `max_to_resume` oldest paused 1-core tasks."""
        if not supports_sigstop or max_to_resume <= 0 or not paused_serial_procs:
            return 0
        # Resume oldest-admitted orders first so they complete and flush to CSV ASAP
        paused_serial_procs.sort(
            key=lambda p: (
                active_orders.index(active_processes[p][1])
                if p in active_processes and active_processes[p][1] in active_orders
                else 10**9
            )
        )
        resumed = 0
        still_paused = []
        for p in paused_serial_procs:
            if p not in active_processes:
                paused_serial_set.discard(p)
                continue
            if resumed < max_to_resume:
                try:
                    os.kill(p.pid, signal.SIGCONT)
                    paused_serial_set.discard(p)
                    resumed += 1
                    continue
                except OSError:
                    paused_serial_set.discard(p)
                    continue
            still_paused.append(p)
        paused_serial_procs[:] = still_paused
        return resumed

    def order_priority_key(oid):
        """Sort key preferring orders closest to completion, then oldest admission."""
        remaining_count = len(uncompleted_tasks[oid])
        serial_done = 0 if (1 not in uncompleted_tasks[oid]) else 1
        admit_idx = active_orders.index(oid)
        return (serial_done, remaining_count, admit_idx)

    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    prev_sigterm = None
    if hasattr(signal, "SIGTERM"):
        try:
            prev_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
        except (ValueError, OSError):
            prev_sigterm = None

    last_status_time = time.monotonic()
    interrupted = False

    try:
        while active_processes or pending_orders or active_orders:
            # --- A. Reap finished or unexpectedly exited worker processes ---
            finished_procs = []
            for proc, (conn, oid, cores_run, _) in list(active_processes.items()):
                if conn.poll():
                    try:
                        res = conn.recv()
                    except EOFError:
                        res = {
                            'error': 'EOFError reading worker pipe',
                            'traceback': 'Worker process terminated before sending result.',
                        }
                    conn_to_proc.pop(conn, None)
                    conn.close()
                    if proc in paused_serial_set:
                        paused_serial_set.discard(proc)
                        try:
                            paused_serial_procs.remove(proc)
                        except ValueError:
                            pass
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        kill_proc_group(proc)
                    finished_procs.append(proc)

                    if oid in running_tasks:
                        running_tasks[oid].discard(cores_run)
                    if res.get('error'):
                        print(
                            f"\n[ERROR] Task ({prefix}{oid}, {cores_run}c) failed:\n"
                            f"{res.get('traceback', res.get('error'))}",
                            flush=True,
                        )
                        # Retire failed order cleanly (never infinite-loop on broken order)
                        if oid in active_orders:
                            for p_other, (c_other, o_other, _, _) in list(active_processes.items()):
                                if o_other == oid and p_other is not proc:
                                    conn_to_proc.pop(c_other, None)
                                    kill_proc_group(p_other, c_other)
                                    finished_procs.append(p_other)
                                    paused_serial_set.discard(p_other)
                            retire_order(oid)
                            failed_orders.append(oid)
                    elif oid in uncompleted_tasks:
                        uncompleted_tasks[oid].discard(cores_run)
                        order_results[oid][cores_run] = res

                elif not proc.is_alive():
                    conn_to_proc.pop(conn, None)
                    conn.close()
                    if proc in paused_serial_set:
                        paused_serial_set.discard(proc)
                        try:
                            paused_serial_procs.remove(proc)
                        except ValueError:
                            pass
                    proc.join()
                    finished_procs.append(proc)
                    if oid in running_tasks:
                        running_tasks[oid].discard(cores_run)
                    retries = task_retries.get((oid, cores_run), 0) + 1
                    task_retries[(oid, cores_run)] = retries
                    if retries <= max_crash_retries:
                        print(
                            f"\n[WARN] Worker for {prefix}{oid} ({cores_run}c) exited "
                            f"unexpectedly (retry {retries}/{max_crash_retries}).",
                            flush=True,
                        )
                    else:
                        print(
                            f"\n[ERROR] Worker for {prefix}{oid} ({cores_run}c) crashed "
                            f"repeatedly; retiring order {prefix}{oid}.",
                            flush=True,
                        )
                        if oid in active_orders:
                            retire_order(oid)
                            failed_orders.append(oid)

            for proc in finished_procs:
                active_processes.pop(proc, None)

            # --- B. Save and retire any orders with all core counts completed ---
            for oid in list(active_orders):
                if len(order_results[oid]) == len(all_test_core_counts):
                    all_ok, _ = _save_order_speed_comparison_csv(
                        order_id=oid,
                        prefix=prefix,
                        metric=metric,
                        all_core_counts=all_test_core_counts,
                        results_map=order_results[oid],
                        output_dir=output_dir,
                    )
                    completed_session += 1
                    if not all_ok:
                        mismatch_orders.append(oid)
                    retire_order(oid)

            if not active_processes and not pending_orders and not active_orders:
                break

            # --- C. Schedule work without exceeding `total_machine_cores` ---
            mc_used, running_1c = get_core_usage()

            # Check whether we should enter full-width drain mode (only when
            # testing `cores == total_machine_cores`, e.g. 128c on 128-core host).
            if has_full_width and not drain_for_full_width:
                any_order_only_needs_fw = any(
                    uncompleted_tasks[o] == {full_width_cores}
                    and full_width_cores not in running_tasks[o]
                    for o in active_orders
                )
                any_order_needs_fw = any(
                    full_width_cores in uncompleted_tasks[o]
                    and full_width_cores not in running_tasks[o]
                    for o in active_orders
                )
                any_sub_mc_launchable = any(
                    any(
                        1 < c < full_width_cores and c not in running_tasks[o]
                        for c in uncompleted_tasks[o]
                    )
                    for o in active_orders
                )
                # Trigger a clean batched 128c window when:
                #  - an active order has ONLY 128c left (immediate CSV win), OR
                #  - no sub-full-width MC tasks remain and either an order needs
                #    128c or we have room to admit new orders to keep cores busy.
                free_for_work = total_machine_cores - (mc_used + len(running_1c))
                if any_order_only_needs_fw:
                    drain_for_full_width = True
                elif not any_sub_mc_launchable and any_order_needs_fw:
                    drain_for_full_width = True
                elif (
                    not any_sub_mc_launchable
                    and pending_orders
                    and (
                        len(active_orders) < soft_active_orders
                        or (
                            len(active_orders) < hard_active_orders
                            and free_for_work >= max(2, total_machine_cores // 2)
                        )
                    )
                ):
                    drain_for_full_width = True

            # PHASE 1: Exclusive full-width burst (e.g. 128c on 128 cores)
            if has_full_width and drain_for_full_width:
                if mc_used == 0:
                    # While already at mc_used == 0, refill active window up to
                    # soft_active_orders (or a small wave up to hard_active_orders
                    # if all active orders are only waiting on slow 1c tasks).
                    active_with_sub_mc = sum(
                        1 for o in active_orders
                        if any(1 < c < full_width_cores for c in uncompleted_tasks[o])
                    )
                    if len(active_orders) < soft_active_orders:
                        while len(active_orders) < soft_active_orders and pending_orders:
                            admit_order(pending_orders.pop(0))
                    elif active_with_sub_mc == 0 and len(active_orders) < hard_active_orders and pending_orders:
                        wave_size = min(4, hard_active_orders - len(active_orders), len(pending_orders))
                        for _ in range(wave_size):
                            admit_order(pending_orders.pop(0))

                    # Find highest-priority active order needing full_width_cores
                    fw_candidates = [
                        o for o in active_orders
                        if full_width_cores in uncompleted_tasks[o]
                        and full_width_cores not in running_tasks[o]
                    ]
                    if fw_candidates:
                        fw_candidates.sort(key=order_priority_key)
                        target_oid = fw_candidates[0]
                        if running_1c and supports_sigstop:
                            pause_serial_procs(len(running_1c), running_1c)
                            mc_used, running_1c = get_core_usage()
                        if not running_1c:
                            launch_task(target_oid, full_width_cores)
                            mc_used += full_width_cores
                    else:
                        # All active orders have completed their full-width task!
                        drain_for_full_width = False
                else:
                    # Still draining sub-MC tasks before full-width burst:
                    # keep freed cores busy with paused/ready 1c tasks!
                    free_cores = total_machine_cores - (mc_used + len(running_1c))
                    if free_cores > 0:
                        resumed = resume_serial_procs(free_cores)
                        free_cores -= resumed
                    if free_cores > 0:
                        for oid in sorted(active_orders, key=order_priority_key):
                            if free_cores <= 0:
                                break
                            if 1 in uncompleted_tasks[oid] and 1 not in running_tasks[oid]:
                                launch_task(oid, 1)
                                free_cores -= 1

            # PHASE 2: Normal packing mode (co-schedule 1c + 2..C/2 with 0 oversubscription)
            if not drain_for_full_width:
                # Admit orders up to soft_active_orders if not using full-width bursts
                if not has_full_width:
                    while len(active_orders) < soft_active_orders and pending_orders:
                        admit_order(pending_orders.pop(0))

                mc_used, running_1c = get_core_usage()

                # 2a. Prioritize starting/resuming 1c tasks for active orders
                #     (up to half the machine so large C/2 MC tasks always fit)
                max_1c_budget = max(1, total_machine_cores - (total_machine_cores // 2))
                if not has_full_width:
                    max_1c_budget = total_machine_cores
                free_cores = total_machine_cores - (mc_used + len(running_1c))
                if free_cores > 0 and len(running_1c) < max_1c_budget:
                    can_add_1c = min(free_cores, max_1c_budget - len(running_1c))
                    resumed = resume_serial_procs(can_add_1c)
                    can_add_1c -= resumed
                    if can_add_1c > 0:
                        for oid in sorted(active_orders, key=order_priority_key):
                            if can_add_1c <= 0:
                                break
                            if 1 in uncompleted_tasks[oid] and 1 not in running_tasks[oid]:
                                launch_task(oid, 1)
                                can_add_1c -= 1

                # 2b. Bin-pack sub-full-width MC tasks (largest fitting task for
                #     highest-priority active order)
                mc_used, running_1c = get_core_usage()
                while True:
                    mc_free = total_machine_cores - mc_used
                    if mc_free < 2:
                        break
                    # Choose highest-priority active order with a fitting MC task
                    chosen = None
                    for oid in sorted(active_orders, key=order_priority_key):
                        avail_mc = [
                            c for c in uncompleted_tasks[oid]
                            if c > 1
                            and (not has_full_width or c < full_width_cores)
                            and c not in running_tasks[oid]
                            and c <= mc_free
                        ]
                        if avail_mc:
                            chosen = (oid, max(avail_mc))
                            break
                    if chosen is None:
                        break

                    oid, chosen_c = chosen
                    currently_active = mc_used + len(running_1c)
                    need_to_free = (currently_active + chosen_c) - total_machine_cores
                    if need_to_free > 0:
                        if supports_sigstop and len(running_1c) >= need_to_free:
                            pause_serial_procs(need_to_free, running_1c)
                            mc_used, running_1c = get_core_usage()
                        else:
                            # Cannot fit without oversubscribing -> stop packing
                            break

                    launch_task(oid, chosen_c)
                    mc_used += chosen_c

                # 2c. Fill any still-idle cores with remaining paused/ready 1c tasks
                mc_used, running_1c = get_core_usage()
                free_cores = total_machine_cores - (mc_used + len(running_1c))
                if free_cores > 0:
                    resumed = resume_serial_procs(free_cores)
                    free_cores -= resumed
                if free_cores > 0:
                    for oid in sorted(active_orders, key=order_priority_key):
                        if free_cores <= 0:
                            break
                        if 1 in uncompleted_tasks[oid] and 1 not in running_tasks[oid]:
                            launch_task(oid, 1)
                            free_cores -= 1

                # 2d. If not using full-width bursts and >= 50% of cores are still
                #     idle because active orders only await slow 1c tasks, admit a
                #     small replenishment wave up to hard_active_orders.
                if (
                    not has_full_width
                    and free_cores >= max(2, total_machine_cores // 2)
                    and len(active_orders) < hard_active_orders
                    and pending_orders
                ):
                    admit_order(pending_orders.pop(0))
                    continue

            # --- D. Periodic live status heartbeat ---
            now = time.monotonic()
            if status_interval_seconds and (now - last_status_time) >= status_interval_seconds:
                mc_used, running_1c = get_core_usage()
                used_total = mc_used + len(running_1c)
                mode_label = "FULL_WIDTH" if drain_for_full_width else "PACKING"
                print(
                    f"[STATUS {time.strftime('%H:%M:%S')}] "
                    f"Completed: {completed_session}/{len(seen_oids) - skipped_count} | "
                    f"Active orders: {len(active_orders)} (pending: {len(pending_orders)}) | "
                    f"Cores: {used_total}/{total_machine_cores} "
                    f"(MC:{mc_used} + 1c:{len(running_1c)}, paused:{len(paused_serial_set)}) | "
                    f"Mode: {mode_label}",
                    flush=True,
                )
                last_status_time = now

            # --- E. Event-driven wait until any worker finishes (0% scheduler CPU) ---
            if active_processes:
                wait_conns = [info[0] for info in active_processes.values()]
                mp_wait(wait_conns, timeout=0.5)
            elif pending_orders or active_orders:
                time.sleep(0.05)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPT] Graceful shutdown requested — stopping all worker process groups...", flush=True)
    finally:
        if prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
            except (ValueError, OSError):
                pass
        for proc, (conn, _, _, _) in list(active_processes.items()):
            kill_proc_group(proc, conn)
        active_processes.clear()
        conn_to_proc.clear()
        paused_serial_procs.clear()
        paused_serial_set.clear()

    print("\n" + "=" * 100)
    if interrupted:
        print("Benchmark scheduler stopped cleanly (zero orphan processes, all finished CSVs intact).")
    else:
        print("All requested BnB speed comparisons finished successfully!")
    print(f"Completed this session : {completed_session} orders")
    print(f"Previously completed   : {skipped_count} orders (skipped)")
    print(f"Remaining unfinished   : {len(pending_orders) + len(active_orders)} orders")
    if failed_orders:
        print(f"Failed orders          : {failed_orders}")
    if mismatch_orders:
        print(f"WARNING — Mismatches   : {mismatch_orders}")
    else:
        print("Pallet verification    : 100% matched single-core baseline")
    print(f"Results directory      : {output_dir}")
    print("=" * 100, flush=True)
    return {
        'completed': completed_session,
        'skipped': skipped_count,
        'remaining': len(pending_orders) + len(active_orders),
        'failed': failed_orders,
        'mismatches': mismatch_orders,
        'interrupted': interrupted,
    }


def _find_existing_test_csv(test_name, order_type, order_id, output_dir,
                            topx_min=1, topx_max=15, topx_step=1):
    """Return existing non-empty CSV path for a single (test_name, order_type, order_id), or None."""
    prefix = "O" if order_type == "given" else "T"
    if test_name == "spee":
        return _find_existing_speed_csv(output_dir, prefix, order_id)
    if test_name == "optg":
        candidate = os.path.join(output_dir, f"optg_comparison_{prefix}{order_id}_to_{prefix}{order_id}.csv")
    elif test_name == "algo":
        candidate = os.path.join(output_dir, f"algorithm_comparison_{prefix}{order_id}_to_{prefix}{order_id}.csv")
    elif test_name == "topx":
        candidate = os.path.join(output_dir, f"topx_test_O{order_id}-O{order_id}_L{topx_min}-L{topx_max}-S{topx_step}.csv")
    else:
        raise ValueError(f"Unknown test_name: {test_name}")
    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
        return candidate
    return None


def _master_single_core_worker(worker_id, worker_conn):
    """Persistent 1-core worker process for the unified master test pool.

    Runs each dispatched `(test_name, order_type, order_id, config)` job strictly
    single-core (`use_mp=False`) so node counts, pruning counts, and runtimes
    remain scientifically identical to single-core baselines, while achieving
    100% linear throughput across all 128 cores with zero nested process pools.
    """
    try:
        if hasattr(os, "setpgrp"):
            try:
                os.setpgrp()
            except OSError:
                pass
        os.environ["TQDM_DISABLE"] = "1"
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

        while True:
            msg = worker_conn.recv()
            if msg is None:
                break

            test_name, order_type, order_id, cfg = msg
            order_dict = orders_dict if order_type == "given" else test_orders_dict
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            t0 = time.perf_counter()

            try:
                if test_name == "algo":
                    run_algorithm_comparison_test(
                        start_order=order_id,
                        end_order=order_id,
                        order_dict=order_dict,
                        criterion=cfg["criterion"],
                        metric=cfg["metric"],
                        print_pallets=False,
                        save_pallets=cfg["save_pallets"],
                        random_attempts=cfg["random_attempts"],
                        bnb_topx=cfg["bnb_topx"],
                        output_dir=output_dir,
                        use_mp=False,
                        verbose=False,
                    )
                elif test_name == "topx":
                    run_topx_limiting_test(
                        start_order=order_id,
                        end_order=order_id,
                        order_dict=order_dict,
                        criterion=cfg["criterion"],
                        print_pallets=False,
                        save_pallets=cfg["save_pallets"],
                        topx_min=cfg["topx_min"],
                        topx_max=cfg["topx_max"],
                        topx_step=cfg["topx_step"],
                        topx_test_without=cfg["topx_test_without"],
                        output_dir=output_dir,
                        verbose=False,
                    )
                elif test_name == "optg":
                    run_optimality_guarantee_test(
                        start_order=order_id,
                        end_order=order_id,
                        order_dict=order_dict,
                        criterion=cfg["criterion"],
                        metric=cfg["metric"],
                        print_pallets=False,
                        save_pallets=cfg["save_pallets"],
                        output_dir=output_dir,
                        verbose=False,
                    )
                else:
                    raise ValueError(f"Unsupported single-core test type: {test_name}")

                elapsed = time.perf_counter() - t0
                worker_conn.send(("OK", worker_id, test_name, order_type, order_id, elapsed))
            except Exception as exc:
                worker_conn.send((
                    "ERR",
                    worker_id,
                    test_name,
                    order_type,
                    order_id,
                    str(exc),
                    traceback.format_exc(),
                ))
    finally:
        try:
            worker_conn.close()
        except Exception:
            pass


def _run_unified_single_core_pool(phase_label, jobs, total_machine_cores,
                                  status_interval_seconds=30.0):
    """Run heterogeneous 1-core `{algo, topx, optg}` jobs across all cores.

    Uses persistent worker processes in isolated process groups, event-driven
    dispatch (`mp_wait`), live per-test progress reporting, crash resilience,
    and guaranteed zero-orphan cleanup on `SIGINT`/`SIGTERM`.
    """
    if not jobs:
        return {'completed': 0, 'failed': [], 'by_test': {}, 'interrupted': False}

    worker_count = min(max(1, int(total_machine_cores)), len(jobs))
    mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    # Track per-test totals for clean status output
    planned_by_test = {}
    completed_by_test = {}
    for test_name, order_type, _, _ in jobs:
        key = f"{test_name}/{order_type}"
        planned_by_test[key] = planned_by_test.get(key, 0) + 1
        completed_by_test.setdefault(key, 0)

    print("\n" + "=" * 100)
    print(f"MASTER SUITE — {phase_label}")
    print(f"Workers / Cores : {worker_count} / {total_machine_cores}")
    print(f"Queued Jobs     : {len(jobs):,}  ({', '.join(f'{k}: {v}' for k, v in sorted(planned_by_test.items()))})")
    print("=" * 100, flush=True)

    workers = {}          # wid -> (proc, conn)
    conn_to_wid = {}
    idle_wids = []
    inflight = {}         # wid -> (test_name, order_type, order_id, start_monotonic)
    job_idx = 0
    completed_total = 0
    failed_jobs = []
    interrupted = False
    phase_start = time.monotonic()
    last_status = phase_start

    def kill_worker_group(proc, conn):
        pid = proc.pid
        if pid is not None:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            else:
                try:
                    proc.terminate()
                except Exception:
                    pass
            proc.join(timeout=1.5)
            if proc.is_alive():
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                else:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                proc.join(timeout=1.0)
        try:
            conn.close()
        except Exception:
            pass

    def spawn_worker(wid):
        parent_conn, child_conn = mp_ctx.Pipe(duplex=True)
        proc = mp_ctx.Process(
            target=_master_single_core_worker,
            args=(wid, child_conn),
            daemon=False,
        )
        proc.start()
        child_conn.close()
        workers[wid] = (proc, parent_conn)
        conn_to_wid[parent_conn] = wid
        return proc, parent_conn

    def dispatch_ready():
        nonlocal job_idx
        while idle_wids and job_idx < len(jobs):
            wid = idle_wids.pop()
            job = jobs[job_idx]
            job_idx += 1
            _, conn = workers[wid]
            conn.send(job)
            inflight[wid] = (job[0], job[1], job[2], time.monotonic())

    def print_status(force=False):
        nonlocal last_status
        now = time.monotonic()
        if not force and (not status_interval_seconds or (now - last_status) < status_interval_seconds):
            return
        elapsed = max(0.001, now - phase_start)
        rate_per_min = (completed_total / elapsed) * 60.0
        breakdown = " | ".join(
            f"{k}:{completed_by_test[k]}/{planned_by_test[k]}"
            for k in sorted(planned_by_test)
        )
        print(
            f"[STATUS {time.strftime('%H:%M:%S')}] {phase_label} | "
            f"Done: {completed_total:,}/{len(jobs):,} ({rate_per_min:.1f}/min) | "
            f"Busy cores: {len(inflight)}/{worker_count} | {breakdown}",
            flush=True,
        )
        last_status = now

    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    prev_sigterm = None
    if hasattr(signal, "SIGTERM"):
        try:
            prev_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
        except (ValueError, OSError):
            prev_sigterm = None

    try:
        for wid in range(worker_count):
            spawn_worker(wid)
            idle_wids.append(wid)
        idle_wids.reverse()
        dispatch_ready()

        while inflight or job_idx < len(jobs):
            active_conns = [workers[w][1] for w in inflight]
            ready_conns = mp_wait(active_conns, timeout=0.5)

            for conn in ready_conns:
                wid = conn_to_wid.get(conn)
                if wid is None:
                    continue
                while conn.poll():
                    try:
                        msg = conn.recv()
                    except EOFError:
                        # Worker crashed unexpectedly — log job and respawn worker
                        bad_job = inflight.pop(wid, None)
                        proc, old_conn = workers.pop(wid)
                        conn_to_wid.pop(old_conn, None)
                        kill_worker_group(proc, old_conn)
                        if bad_job is not None:
                            t_name, o_type, oid, _ = bad_job
                            failed_jobs.append((t_name, o_type, oid, "Worker exited unexpectedly"))
                            print(f"[ERROR] Worker {wid} crashed on {t_name}/{o_type} order {oid}.", flush=True)
                        if job_idx < len(jobs):
                            spawn_worker(wid)
                            idle_wids.append(wid)
                        break

                    tag = msg[0]
                    inflight.pop(wid, None)
                    idle_wids.append(wid)
                    if tag == "OK":
                        _, _, t_name, o_type, oid, _ = msg
                        completed_total += 1
                        key = f"{t_name}/{o_type}"
                        completed_by_test[key] = completed_by_test.get(key, 0) + 1
                    elif tag == "ERR":
                        _, _, t_name, o_type, oid, err_str, tb = msg
                        failed_jobs.append((t_name, o_type, oid, err_str))
                        print(f"[ERROR] {t_name}/{o_type} order {oid} failed:\n{tb}", flush=True)

            # Check for dead inflight workers that closed without signalling
            for wid in list(inflight.keys()):
                proc, old_conn = workers[wid]
                if not proc.is_alive() and not old_conn.poll():
                    bad_job = inflight.pop(wid)
                    workers.pop(wid)
                    conn_to_wid.pop(old_conn, None)
                    kill_worker_group(proc, old_conn)
                    t_name, o_type, oid, _ = bad_job
                    failed_jobs.append((t_name, o_type, oid, "Worker died unexpectedly"))
                    print(f"[ERROR] Worker {wid} died on {t_name}/{o_type} order {oid}.", flush=True)
                    if job_idx < len(jobs):
                        spawn_worker(wid)
                        idle_wids.append(wid)

            dispatch_ready()
            print_status(force=False)

        # Graceful shutdown of persistent workers
        for wid, (proc, conn) in list(workers.items()):
            try:
                conn.send(None)
            except Exception:
                pass
        for wid, (proc, conn) in list(workers.items()):
            proc.join(timeout=3.0)
            if proc.is_alive():
                kill_worker_group(proc, conn)
            else:
                try:
                    conn.close()
                except Exception:
                    pass
        workers.clear()
        conn_to_wid.clear()
        print_status(force=True)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERRUPT] Stopping {phase_label} cleanly — terminating all worker process groups...", flush=True)
    finally:
        if prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
            except (ValueError, OSError):
                pass
        for wid, (proc, conn) in list(workers.items()):
            kill_worker_group(proc, conn)
        workers.clear()
        conn_to_wid.clear()

    return {
        'completed': completed_total,
        'failed': failed_jobs,
        'by_test': completed_by_test,
        'interrupted': interrupted,
    }


def run_master_test_suite(
    optg_given=None,
    optg_test=None,
    topx_given=None,
    topx_test=None,
    algo_given=None,
    algo_test=None,
    spee_given=None,
    spee_test=None,
    total_machine_cores=None,
    mp_core_max=None,
    criterion=DEFAULT_CRITERION,
    metric=Metric.MAX_Z,
    random_attempts=DEFAULT_MAX_ATTEMPTS,
    bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
    save_given_pallets=False,
    phase1_topx_max_boxes=22,
    phase1_optg_max_boxes=19,
    run_spee=True,
    run_heavy_phase=True,
    results_root="./results",
    status_interval_seconds=30.0,
):
    """Master overnight test orchestrator for all 4 benchmark suites on both
    given orders (1-40) and type-2 test orders (1000-3999).

    Reads the missing-order lists from the Testing Area by default, skips any
    order whose CSV is already present on disk, writes every CSV atomically
    into the exact `./results/{test}_comparisons/{given,test}_orders/` folder
    expected by `data_processing.ipynb`, and maximizes both 128-core CPU
    utilization and CSV registration rate via a 3-phase pipeline:

      Phase 1 — Unified High-Yield 1-Core Pool (100% of cores):
        Runs all tractable `{algo, topx, optg}` jobs together across all cores,
        starting with the fastest orders first for immediate CSV registration,
        then scheduling the heavier Phase-1 tail orders longest-first so all
        128 cores finish together with minimal drain idle time.

      Phase 2 — Dedicated Multi-Core Speed Comparisons (`spee`):
        Runs `spee` on missing test orders and given orders using the bounded-WIP
        two-phase smart scheduler (`run_smart_concurrent_bnb_mc_speed_comparison`)
        with pristine, uncontended timings across `1, 2, ..., mp_core_max` cores.

      Phase 3 — Heavy / Exponential Unified 1-Core Pool (100% of cores):
        Runs remaining deep `topx` (unlimited `topx=None` at 23+ boxes) and
        exact `optg` (20+ boxes and given orders) strictly easiest-first across
        all cores so exponential orders never block tractable results.
    """
    if optg_given is None:
        optg_given = optg_missing_given_orders
    if optg_test is None:
        optg_test = optg_missing_test_orders
    if topx_given is None:
        topx_given = topx_missing_given_orders
    if topx_test is None:
        topx_test = topx_missing_test_orders
    if algo_given is None:
        algo_given = algo_missing_given_orders
    if algo_test is None:
        algo_test = algo_missing_test_orders
    if spee_given is None:
        spee_given = spee_missing_given_orders
    if spee_test is None:
        spee_test = spee_missing_test_orders

    if total_machine_cores is None:
        total_machine_cores = os.cpu_count() or DEFAULT_MP_CORES
    total_machine_cores = max(1, int(total_machine_cores))
    if mp_core_max is None:
        mp_core_max = total_machine_cores

    # Standard per-suite configs matching data_processing.ipynb expectations
    suite_specs = [
        ("algo", "given", algo_given, orders_dict, {
            "output_dir": os.path.join(results_root, "algo_comparisons", "given_orders"),
            "criterion": criterion, "metric": metric,
            "save_pallets": save_given_pallets,
            "random_attempts": random_attempts, "bnb_topx": bnb_topx,
        }),
        ("algo", "test", algo_test, test_orders_dict, {
            "output_dir": os.path.join(results_root, "algo_comparisons", "test_orders"),
            "criterion": criterion, "metric": metric,
            "save_pallets": False,
            "random_attempts": random_attempts, "bnb_topx": bnb_topx,
        }),
        ("topx", "given", topx_given, orders_dict, {
            "output_dir": os.path.join(results_root, "topx_comparisons", "given_orders"),
            "criterion": criterion,
            "save_pallets": save_given_pallets,
            "topx_min": 1, "topx_max": 5, "topx_step": 1, "topx_test_without": False,
        }),
        ("topx", "test", topx_test, test_orders_dict, {
            "output_dir": os.path.join(results_root, "topx_comparisons", "test_orders"),
            "criterion": criterion,
            "save_pallets": False,
            "topx_min": 1, "topx_max": 15, "topx_step": 1, "topx_test_without": True,
        }),
        ("optg", "given", optg_given, orders_dict, {
            "output_dir": os.path.join(results_root, "optg_comparisons", "given_orders"),
            "criterion": criterion, "metric": metric,
            "save_pallets": save_given_pallets,
        }),
        ("optg", "test", optg_test, test_orders_dict, {
            "output_dir": os.path.join(results_root, "optg_comparisons", "test_orders"),
            "criterion": criterion, "metric": metric,
            "save_pallets": False,
        }),
    ]

    def get_box_count(order_type, oid):
        od = orders_dict if order_type == "given" else test_orders_dict
        return len(get_box_list_from_order(oid, od))

    def difficulty_rank(test_name, order_type, oid):
        n_boxes = get_box_count(order_type, oid)
        if test_name == "algo":
            return (float(n_boxes), 0, oid)
        if test_name == "topx":
            if order_type == "given":
                return (float(n_boxes) + 3.0, 1, oid)
            return (n_boxes * 1.45, 1, oid)
        # optg (exact BnB grows steeply with box count)
        return (n_boxes * 1.75, 2, oid)

    phase1_fast = []
    phase1_tail = []
    phase3_heavy = []
    skipped_summary = {}

    for test_name, order_type, raw_ids, od, cfg in suite_specs:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        seen = set()
        skipped = 0
        for oid in (raw_ids or []):
            if oid in seen or oid not in od:
                continue
            seen.add(oid)
            existing = _find_existing_test_csv(
                test_name,
                order_type,
                oid,
                cfg["output_dir"],
                topx_min=cfg.get("topx_min", 1),
                topx_max=cfg.get("topx_max", 15),
                topx_step=cfg.get("topx_step", 1),
            )
            if existing is not None:
                skipped += 1
                continue

            n_boxes = len(get_box_list_from_order(oid, od))
            job = (test_name, order_type, oid, cfg)

            if test_name == "algo":
                if order_type == "test" and n_boxes <= 29:
                    phase1_fast.append(job)
                else:
                    phase1_tail.append(job)
            elif test_name == "topx":
                if order_type == "given":
                    phase1_tail.append(job)
                elif n_boxes <= phase1_topx_max_boxes - 1:
                    phase1_fast.append(job)
                elif n_boxes <= phase1_topx_max_boxes:
                    phase1_tail.append(job)
                else:
                    phase3_heavy.append(job)
            elif test_name == "optg":
                if order_type == "given":
                    phase3_heavy.append(job)
                elif n_boxes <= phase1_optg_max_boxes - 1:
                    phase1_fast.append(job)
                elif n_boxes <= phase1_optg_max_boxes:
                    phase1_tail.append(job)
                else:
                    phase3_heavy.append(job)

        skipped_summary[f"{test_name}/{order_type}"] = skipped

    # Phase 1 order:
    #   1) Easiest jobs ascending -> rapid initial CSV registration
    #   2) Medium-hard Phase-1 tail jobs descending (LPT) -> heavy orders start
    #      early enough that remaining medium orders fill all 128 cores until
    #      Phase 1 finishes cleanly with minimal idle drain before Phase 2.
    phase1_fast.sort(key=lambda j: difficulty_rank(j[0], j[1], j[2]))
    phase1_tail.sort(key=lambda j: difficulty_rank(j[0], j[1], j[2]), reverse=True)
    phase1_jobs = phase1_fast + phase1_tail

    # Phase 3 order: strictly easiest-first so tractable frontier orders complete
    # before multi-hour exponential orders ever occupy worker slots.
    phase3_heavy.sort(key=lambda j: difficulty_rank(j[0], j[1], j[2]))

    # Prepare Phase 2 (`spee`) lists sorted by box count ascending
    spee_test_dir = os.path.join(results_root, "spee_comparisons", "test_orders")
    spee_given_dir = os.path.join(results_root, "spee_comparisons", "given_orders")
    spee_test_pending = sorted(
        {oid for oid in (spee_test or []) if oid in test_orders_dict
         and _find_existing_speed_csv(spee_test_dir, "T", oid) is None},
        key=lambda oid: (len(get_box_list_from_order(oid, test_orders_dict)), oid),
    )
    spee_given_pending = sorted(
        {oid for oid in (spee_given or []) if oid in orders_dict
         and _find_existing_speed_csv(spee_given_dir, "O", oid) is None},
        key=lambda oid: (len(get_box_list_from_order(oid, orders_dict)), oid),
    )
    skipped_summary["spee/test"] = len(set(spee_test or [])) - len(spee_test_pending)
    skipped_summary["spee/given"] = len(set(spee_given or [])) - len(spee_given_pending)

    master_start = time.monotonic()
    print("=" * 100)
    print("MASTER BENCHMARK SUITE ORCHESTRATOR (128-CORE HIGH-YIELD 3-PHASE EDITION)")
    print(f"Machine Cores          : {total_machine_cores} (spee mp_core_max={mp_core_max})")
    print(f"Phase 1 (Fast 1-core)  : {len(phase1_jobs):,} jobs (fast={len(phase1_fast):,}, LPT-tail={len(phase1_tail):,})")
    print(f"Phase 2 (Speed MC)     : {len(spee_test_pending):,} test orders + {len(spee_given_pending):,} given orders (enabled={run_spee})")
    print(f"Phase 3 (Heavy 1-core) : {len(phase3_heavy):,} jobs (enabled={run_heavy_phase})")
    print(f"Already Completed      : {sum(skipped_summary.values()):,} CSVs skipped on disk")
    print("=" * 100, flush=True)

    summary = {
        'phase1': None,
        'phase2_test': None,
        'phase2_given': None,
        'phase3': None,
        'skipped': skipped_summary,
        'interrupted': False,
    }

    # --- PHASE 1 ---
    if phase1_jobs:
        res1 = _run_unified_single_core_pool(
            "PHASE 1 (Tractable algo + topx + optg)",
            phase1_jobs,
            total_machine_cores=total_machine_cores,
            status_interval_seconds=status_interval_seconds,
        )
        summary['phase1'] = res1
        if res1['interrupted']:
            summary['interrupted'] = True

    # --- PHASE 2 ---
    if run_spee and not summary['interrupted']:
        if spee_test_pending:
            print("\n" + "=" * 100)
            print(f"MASTER SUITE — PHASE 2A (spee/test: {len(spee_test_pending):,} orders)")
            print("=" * 100, flush=True)
            res2t = run_smart_concurrent_bnb_mc_speed_comparison(
                order_list=spee_test_pending,
                mp_core_max=mp_core_max,
                order_dict=test_orders_dict,
                criterion=criterion,
                metric=metric,
                bnb_topx=bnb_topx,
                output_dir=spee_test_dir,
                total_machine_cores=total_machine_cores,
                max_active_orders=None,
                status_interval_seconds=status_interval_seconds,
            )
            summary['phase2_test'] = res2t
            if res2t and res2t.get('interrupted'):
                summary['interrupted'] = True

        if spee_given_pending and not summary['interrupted']:
            print("\n" + "=" * 100)
            print(f"MASTER SUITE — PHASE 2B (spee/given: {len(spee_given_pending):,} orders)")
            print("=" * 100, flush=True)
            res2g = run_smart_concurrent_bnb_mc_speed_comparison(
                order_list=spee_given_pending,
                mp_core_max=mp_core_max,
                order_dict=orders_dict,
                criterion=criterion,
                metric=metric,
                bnb_topx=bnb_topx,
                output_dir=spee_given_dir,
                total_machine_cores=total_machine_cores,
                max_active_orders=None,
                status_interval_seconds=status_interval_seconds,
            )
            summary['phase2_given'] = res2g
            if res2g and res2g.get('interrupted'):
                summary['interrupted'] = True

    # --- PHASE 3 ---
    if run_heavy_phase and phase3_heavy and not summary['interrupted']:
        res3 = _run_unified_single_core_pool(
            "PHASE 3 (Deep topx + exact optg frontier)",
            phase3_heavy,
            total_machine_cores=total_machine_cores,
            status_interval_seconds=status_interval_seconds,
        )
        summary['phase3'] = res3
        if res3['interrupted']:
            summary['interrupted'] = True

    total_elapsed = time.monotonic() - master_start
    total_completed = (
        (summary['phase1']['completed'] if summary['phase1'] else 0)
        + (summary['phase2_test']['completed'] if summary['phase2_test'] else 0)
        + (summary['phase2_given']['completed'] if summary['phase2_given'] else 0)
        + (summary['phase3']['completed'] if summary['phase3'] else 0)
    )

    print("\n" + "=" * 100)
    if summary['interrupted']:
        print("MASTER TEST SUITE STOPPED CLEANLY (resumable — re-run to continue from remaining orders).")
    else:
        print("MASTER TEST SUITE COMPLETED ALL REQUESTED PHASES!")
    print(f"Total CSVs registered this session : {total_completed:,}")
    print(f"Previously saved (skipped)         : {sum(skipped_summary.values()):,}")
    print(f"Total elapsed wall-clock time      : {total_elapsed / 60.0:.2f} min ({total_elapsed:.1f}s)")
    print("=" * 100, flush=True)
    return summary


# %% [markdown]
# #### Testing Area

# %%
current_order_dict = test_orders_dict
current_orderID = 25
current_algo = Algorithm.BNB
current_criterion = Criterion.VOLUME
current_metric = Metric.MAX_Z
current_nett = BNB_TOPX_DEFAULT_LIMIT
use_mp = False

testing_random_fulfillment = False
testing_optg_comparisons = False
testing_topx_comparisons = False
testing_algo_comparisons = False
testing_speed_comparisons = False
testing_smart_speed_comparisons = False
testing_master_suite = True

given_order_list = list(range(1, 41))
type_2_test_order_list = list(range(1000, 4000))
reduced_type_2_test_order_list = list(range(1000, 3000))

optg_missing_given_orders = given_order_list
optg_missing_test_orders = type_2_test_order_list
topx_missing_given_orders = given_order_list
topx_missing_test_orders = type_2_test_order_list
algo_missing_given_orders = given_order_list
algo_missing_test_orders = type_2_test_order_list
spee_missing_given_orders = given_order_list
spee_missing_test_orders = type_2_test_order_list

if __name__ == "__main__":
    if NOTEBOOK_MODE:
        if testing_master_suite:
            run_master_test_suite(
                optg_given=optg_missing_given_orders,
                optg_test=optg_missing_test_orders,
                topx_given=topx_missing_given_orders,
                topx_test=topx_missing_test_orders,
                algo_given=algo_missing_given_orders,
                algo_test=algo_missing_test_orders,
                spee_given=spee_missing_given_orders,
                spee_test=spee_missing_test_orders,
                total_machine_cores=DEFAULT_MP_CORES,
                mp_core_max=DEFAULT_MP_CORES,
                criterion=DEFAULT_CRITERION,
                metric=Metric.MAX_Z,
                bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
            )
        elif testing_speed_comparisons:
            run_bnb_mc_speed_comparison(DEFAULT_MP_CORES, 2000, 2099, test_orders_dict, DEFAULT_CRITERION, Metric.MAX_Z, False)
        elif testing_smart_speed_comparisons:
            run_smart_concurrent_bnb_mc_speed_comparison(
                order_list=spee_missing_test_orders,
                mp_core_max=DEFAULT_MP_CORES,
                order_dict=test_orders_dict,
                criterion=DEFAULT_CRITERION,
                metric=Metric.MAX_Z,
                bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
                output_dir="./results/spee_comparisons/test_orders",
                total_machine_cores=DEFAULT_MP_CORES,
                max_active_orders=None,
            )
        elif current_algo == Algorithm.BNB:
            testpallet, bnb_stats = process_order(current_orderID, algo=current_algo, criterion=current_criterion, order_dict=current_order_dict, metric=current_metric, num_extpts_to_try=current_nett, use_mp=use_mp)
            testpallet.get_pallet_results(current_algo, current_orderID, current_order_dict, print_mode=True, bnb_stats=bnb_stats)
        else:
            testpallet, nodes = process_order(current_orderID, algo=current_algo, criterion=current_criterion, order_dict=current_order_dict, metric=current_metric)
            testpallet.get_pallet_results(current_algo, current_orderID, current_order_dict, print_mode=True, nodes=nodes)

    elif testing_master_suite:
        run_master_test_suite(
            optg_given=optg_missing_given_orders,
            optg_test=optg_missing_test_orders,
            topx_given=topx_missing_given_orders,
            topx_test=topx_missing_test_orders,
            algo_given=algo_missing_given_orders,
            algo_test=algo_missing_test_orders,
            spee_given=spee_missing_given_orders,
            spee_test=spee_missing_test_orders,
            total_machine_cores=DEFAULT_MP_CORES,
            mp_core_max=DEFAULT_MP_CORES,
            criterion=DEFAULT_CRITERION,
            metric=Metric.MAX_Z,
            bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
        )

    elif testing_random_fulfillment:
        run_random_fulfillment_test(1000, 3999, test_orders_dict, 500, 10000, 100, True)

    elif testing_optg_comparisons:
        with concurrent.futures.ProcessPoolExecutor(max_workers=125) as executor:
            if current_order_dict == orders_dict:
                for i in optg_missing_test_orders:
                    executor.submit(
                        run_optimality_guarantee_test, 
                        start_order=i, 
                        end_order=i, 
                        )
            else:
                for i in optg_missing_given_orders:
                    executor.submit(
                        run_optimality_guarantee_test, 
                        start_order=i, 
                        end_order=i, 
                        )

    elif testing_topx_comparisons:
        with concurrent.futures.ProcessPoolExecutor(max_workers=125) as executor:
            if current_order_dict == orders_dict:
                for i in topx_missing_given_orders:
                    executor.submit(
                        run_topx_limiting_test, 
                        start_order=i, 
                        end_order=i, 
                        order_dict=orders_dict, 
                        print_pallets=False, 
                        save_pallets=True,
                        topx_min = 1,
                        topx_max= 5,
                        topx_step= 1,
                        topx_test_without= False
                        )
            else:
                for i in topx_missing_test_orders:
                    executor.submit(
                        run_topx_limiting_test, 
                        start_order=i, 
                        end_order=i, 
                        order_dict=test_orders_dict, 
                        print_pallets=False, 
                        save_pallets=False,
                        topx_min = 1,
                        topx_max= 15,
                        topx_step= 1,
                        topx_test_without= True
                        )
                    
    elif testing_algo_comparisons:
        with concurrent.futures.ProcessPoolExecutor(max_workers=125) as executor:
            if current_order_dict == orders_dict:
                for i in algo_missing_given_orders:
                    executor.submit(
                        run_algorithm_comparison_test, 
                        start_order=i,
                        end_order=i,
                        order_dict=current_order_dict, 
                        print_pallets=False, 
                        save_pallets=True,
                        )
            else:
                for i in algo_missing_test_orders:
                    executor.submit(
                        run_algorithm_comparison_test, 
                        start_order=i,
                        end_order=i,
                        order_dict=current_order_dict, 
                        print_pallets=False, 
                        save_pallets=False,
                        )

    elif testing_speed_comparisons:
        run_bnb_mc_speed_comparison(DEFAULT_MP_CORES, 2500, 2599, test_orders_dict)

    elif testing_smart_speed_comparisons:
        run_smart_concurrent_bnb_mc_speed_comparison(
            order_list=spee_missing_test_orders,
            mp_core_max=DEFAULT_MP_CORES,
            order_dict=test_orders_dict,
            criterion=DEFAULT_CRITERION,
            metric=Metric.MAX_Z,
            bnb_topx=BNB_TOPX_DEFAULT_LIMIT,
            output_dir="./results/spee_comparisons/test_orders",
            total_machine_cores=DEFAULT_MP_CORES,
            max_active_orders=None,
        )
    
    else:
        print("No workload specified. Exiting...")


