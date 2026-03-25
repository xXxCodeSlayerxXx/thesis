#### Old optimality BnB Results

The results in this folder are pre- mar 25 2025, where BnB's symmetry break rule was based on comparison keys rather than heightmap hashing.

OLD:
'''
        # Check if the current box matches the dimensions of the previous box and record its dimensions and placed position as a comparison key for symmetry breaking rule
        if dimension_tuples[box_index] == dimension_tuples[box_index - 1]:
            symbreak_key = current_sequence[-1]   # ((dx, dy, dz), x, y) of the previous identical box
        else:
            symbreak_key = None


                    # Symmetry breaking rule: if a comparison key is registered and the candidate placement is smaller than the key (checked value by value in the tuples ((dx, dy, dz), x, y) ), prune branch as it would lead to an identical resultant pallet in terms of dimensions but with different boxes (of the same or different box IDs, like a type 8 or 10 box, which are dimensionally identical) occupying the same place.
                    if symbreak_key is not None and symbreak_key > (dims, x, y):
                        counter_symbreak.update(1)
                        count_symbreak += 1
                        continue

'''

