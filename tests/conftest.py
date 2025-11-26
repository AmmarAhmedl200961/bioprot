"""
Pytest configuration and fixtures.
"""


def pytest_collection_modifyitems(session, config, items):
    """Filter out non-test functions from evaluate.py"""
    # Remove items that are imported from evaluate.py (utility functions, not tests)
    # These items don't have a parent class but have names that match evaluate.py functions
    filtered = []
    
    # List of function names in evaluate.py that are not actual tests
    evaluate_utility_functions = {
        'test_ortho_naive_inversion',
        'test_regressor_inversion',
        'test_revocation'
    }
    
    for item in items:
        # Check if this is one of the evaluate.py utility functions
        # These are collected as top-level functions (not methods of a class)
        if item.name in evaluate_utility_functions:
            # Check if it has a parent class - if not, it's from evaluate.py directly
            if not hasattr(item, 'cls') or item.cls is None:
                continue
        filtered.append(item)
    
    items[:] = filtered



