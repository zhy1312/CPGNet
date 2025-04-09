import h5py


def save_hdf5(output_path, asset_dict, mode="a"):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():  
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                if key=="filename":
                    data_type = h5py.special_dtype(vlen=str)
                # chunk_shape = (chunk_size,) + data_shape[1:]
                maxshape = (None,None)
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    # chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
            else:
                dset = file[key]
                dset.resize((len(dset) + data_shape[0],len(dset) +data_shape[1]))
                dset[-data_shape[0]:] = val


def load_hdf5(input_path, keys=None):
    with h5py.File(input_path, "r") as file:
        if keys is None:
            keys = list(file.keys())
        asset_dict = {}
        for key in keys:
            asset_dict[key] = file[key][:]
        return asset_dict
