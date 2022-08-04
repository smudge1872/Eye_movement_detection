import h5py
import time


def read(file_path, dataset_name='name', max_attempts=1):
    is_success = False
    num_attempts = 0
    contents = []
    while num_attempts < max_attempts and not is_success:
        try:
            with h5py.File(file_path, 'r') as hf:
                contents = hf[dataset_name][:]
            is_success = True
        except OSError:
            time.sleep(5)

        num_attempts += 1

    if is_success:
        return contents
    else:
        raise H5ReadError("Attempted %d times and failed to read H5 file %s" % (num_attempts, file_path))


class H5ReadError(Exception):
    pass
