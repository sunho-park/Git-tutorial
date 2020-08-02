couple = {'tom' : 'jerry', 'mario' : 'cooper', 'ironman' : 'captainAmerica'}
print(couple.items())
for key, keys in couple.items():
    print(key, keys)

import numpy as np
shape_np = np.zeros((1, 68, 2), dtype=np.int)
print(shape_np)
print(shape_np.shape)