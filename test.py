import numpy as np 

d = 3
coords = [1, 1, 1]
offsets = np.indices((3,) * d) - 1
reshaped_offsets = np.stack(offsets, axis=d).reshape(-1, d)
offsets_without_middle_point = np.delete(reshaped_offsets, int(d**3 / 2), axis=0)
neighbours = offsets_without_middle_point + coords
neighbours = neighbours.tolist()