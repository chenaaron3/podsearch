import json
import hashlib
import os
import pickle
import time
from functools import wraps
from typing import Any, Callable
from pathlib import Path


def json_cache(cache_dir: str = ".cache", expire_hours: int = None):
    """
    Decorator that caches function results to JSON files based on input parameters.
    
    Args:
        cache_dir: Directory to store cache files (default: ".cache")
        expire_hours: Cache expiration in hours (None = never expire)
    
    Usage:
        @json_cache(cache_dir="./cache", expire_hours=24)
        def my_function(param1, param2):
            return expensive_computation(param1, param2)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache directory if it doesn't exist
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True)
            
            # Create a unique cache key based on function name and arguments
            func_name = func.__name__
            
            # Serialize arguments to create consistent hash
            cache_key_data = {
                'function': func_name,
                'args': args,
                'kwargs': kwargs
            }
            
            # Create hash from serialized data
            cache_key_str = json.dumps(cache_key_data, sort_keys=True, default=str)
            cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
            
            # Cache file path
            cache_file = cache_path / f"{func_name}_{cache_key_hash}.json"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Check expiration if set
                    if expire_hours is not None:
                        cache_time = cache_data.get('timestamp', 0)
                        current_time = time.time()
                        if current_time - cache_time > (expire_hours * 3600):
                            print(f"Cache expired for {func_name}, running function...")
                            os.remove(cache_file)
                        else:
                            print(f"Loading cached result for {func_name}")
                            return cache_data['result']
                    else:
                        print(f"Loading cached result for {func_name}")
                        return cache_data['result']
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Cache file corrupted for {func_name}, running function...")
                    os.remove(cache_file)
            
            # Run the actual function
            print(f"Running {func_name} and caching result...")
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                cache_data = {
                    'result': result,
                    'timestamp': time.time() if expire_hours else None,
                    'function': func_name,
                    'args_hash': cache_key_hash
                }
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, default=str, ensure_ascii=False)
                
                print(f"Result cached to {cache_file}")
                
            except (TypeError, json.JSONEncodeError) as e:
                print(f"Warning: Could not cache result for {func_name}: {e}")
            
            return result
        
        # Add cache management methods
        def clear_cache():
            """Clear all cache files for this function."""
            cache_path = Path(cache_dir)
            if cache_path.exists():
                for cache_file in cache_path.glob(f"{func.__name__}_*.json"):
                    cache_file.unlink()
                print(f"Cleared cache for {func.__name__}")
        
        def cache_info():
            """Get information about cached files."""
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                return {"cache_files": 0, "total_size": 0}
            
            cache_files = list(cache_path.glob(f"{func.__name__}_*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_files": len(cache_files),
                "total_size": total_size,
                "files": [str(f) for f in cache_files]
            }
        
        wrapper.clear_cache = clear_cache
        wrapper.cache_info = cache_info
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    import time
    
    @json_cache(cache_dir="./test_cache", expire_hours=1)
    def expensive_function(x, y):
        """Simulate an expensive operation."""
        time.sleep(2)  # Simulate work
        return {"sum": x + y, "product": x * y, "timestamp": time.time()}
    
    @json_cache(cache_dir="./test_cache")
    def fetch_data(url, params=None):
        """Simulate API call."""
        time.sleep(1)
        return {
            "url": url,
            "params": params,
            "data": f"Result for {url}",
            "fetched_at": time.time()
        }
    
    # Test the caching
    print("=== Testing Caching ===")
    
    # First call - will run function
    result1 = expensive_function(5, 3)
    print(f"Result 1: {result1}")
    
    # Second call - will load from cache
    result2 = expensive_function(5, 3)
    print(f"Result 2: {result2}")
    
    # Different params - will run function again
    result3 = expensive_function(10, 20)
    print(f"Result 3: {result3}")
    
    # Test cache info
    print(f"Cache info: {expensive_function.cache_info()}")
    
    # Test API-like function
    data1 = fetch_data("https://api.example.com", {"page": 1})
    data2 = fetch_data("https://api.example.com", {"page": 1})  # From cache
    
    print(f"Data 1: {data1}")
    print(f"Data 2: {data2}") 