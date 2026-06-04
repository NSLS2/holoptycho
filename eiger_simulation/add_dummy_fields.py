import h5py
import os
import numpy as np

if __name__ == "__main__":
    filename = os.environ.get("AS_HDF5_MASTER_FILE")

    dummy_data = [
        ("/entry/instrument/detector/geometry/translation/distances", np.random.rand(3)),
        ("/entry/sample/goniometer/omega", np.zeros(3)),
        ("/entry/instrument/detector/detectorSpecific/ntrigger", 1),
        ("/entry/instrument/detector/detectorSpecific/nimages", 1),
        ("/entry/instrument/detector/threshold_energy", 10)
    ]
    with h5py.File(filename, "r+") as f:
        for (key, value) in dummy_data:
            f.create_dataset(key, data=value)

