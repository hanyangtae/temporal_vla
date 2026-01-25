import h5py
import os

# Path to a sample HDF5 file
hdf5_path = "data/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"

if os.path.exists(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        print("Keys in root:", list(f.keys()))
        data = f['data']
        print("Keys in data:", list(data.keys())[:1]) # Print first 2 demos
        demo_key = list(data.keys())[0]
        demo = data[demo_key]
        print(f"Keys in demo {demo_key}:", list(demo.keys()))
        if 'obs' in demo:
            print(f"Keys in obs:", list(demo['obs'].keys()))
            # Check shapes
            for k in demo['obs'].keys():
                print(f"  {k}: {demo['obs'][k].shape}")
                print(demo['obs'][k][:1])
else:
    print(f"File not found: {hdf5_path}")
