def find_monotonic_combinations(arrays):
    """
    Find all combinations of selecting one integer from each subarray
    such that the result is monotonic non-decreasing.
    
    Args:
        arrays: List of lists of integers
        
    Returns:
        List of lists, where each inner list is a valid combination
    """
    if not arrays:
        return []
    
    def backtrack(index, current_combination, last_value):
        """
        Backtracking function to find all valid combinations.
        
        Args:
            index: Current array index to process
            current_combination: Current combination being built
            last_value: Last value added to ensure monotonicity
        """
        # Base case: if we've processed all arrays
        if index == len(arrays):
            result.append(current_combination[:])
            return
        
        # Try each value in the current array
        for value in arrays[index]:
            # Check if this value maintains monotonic non-decreasing property
            if value >= last_value:
                current_combination.append(value)
                backtrack(index + 1, current_combination, value)
                current_combination.pop()
    
    result = []
    backtrack(0, [], float('-inf'))  # Start with negative infinity as the last value
    return result

def round_up_to_multiple_of_4(n):
    """
    Round an integer up to the nearest multiple of 4.
    Args:
        n: integer
    Returns:
        The smallest multiple of 4 greater than or equal to n
    """
    return ((n + 3) // 4) * 4

# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: [[1,2,3],[3,4,5], [2,4,6]]
    test1 = [[1,2,3], [3,4,5], [2,4,6]]
    print("Test 1:", test1)
    combinations1 = find_monotonic_combinations(test1)
    print("Valid combinations:")
    for combo in combinations1:
        print(f"  {combo}")
    print(f"Total: {len(combinations1)} combinations\n")
    
    # Test case 2: [[1,2], [2,3], [1,3]]
    test2 = [[1,2], [2,3], [1,3]]
    print("Test 2:", test2)
    combinations2 = find_monotonic_combinations(test2)
    print("Valid combinations:")
    for combo in combinations2:
        print(f"  {combo}")
    print(f"Total: {len(combinations2)} combinations\n")
    
    # Test case 3: [[1], [2], [3]]
    test3 = [[1], [2], [3]]
    print("Test 3:", test3)
    combinations3 = find_monotonic_combinations(test3)
    print("Valid combinations:")
    for combo in combinations3:
        print(f"  {combo}")
    print(f"Total: {len(combinations3)} combinations\n")
    
    # Test case 4: Empty list
    test4 = []
    print("Test 4:", test4)
    combinations4 = find_monotonic_combinations(test4)
    print("Valid combinations:", combinations4)
    print(f"Total: {len(combinations4)} combinations\n")

    # Test round_up_to_multiple_of_4
    print("Testing round_up_to_multiple_of_4:")
    for n in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 20]:
        print(f"{n} -> {round_up_to_multiple_of_4(n)}") 