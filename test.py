
from model.support_functions_tf import *

import model.L_layer_model_tf
import numpy as np
import h5py



print(sigmoid(20))

train_dataset = h5py.File('data/train_signs.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])

print(train_set_x_orig[1])


