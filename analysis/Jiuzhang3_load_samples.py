import numpy as np

data = np.unpackbits(np.fromfile('<replace_with_samples_binary_file>',dtype=np.uint8)).reshape(-1, 1152)

np.save('<replace_with_destination_of_experimental_samples>', data)