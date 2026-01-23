
import h5py
import numpy as np
import sys

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
        # Print attributes if any
        for key, val in obj.attrs.items():
            print(f"  Attr: {key} = {val}")

if len(sys.argv) < 2:
    print("Usage: python inspect_hdf5.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
try:
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        # Print top-level attributes
        print("Top-level attributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        print("\nStructure:")
        f.visititems(print_structure)
        
        # Print some sample data from the first demo
        if 'data' in f:
            demos = list(f['data'].keys())
            if demos:
                first_demo = demos[0]
                print(f"\nSample data from {first_demo}:")
                demo_group = f['data'][first_demo]
                if 'obs' in demo_group:
                    print("  Observations keys:", list(demo_group['obs'].keys()))
                if 'actions' in demo_group:
                    print("  Actions shape:", demo_group['actions'].shape)
except Exception as e:
    print(f"Error reading file: {e}")
