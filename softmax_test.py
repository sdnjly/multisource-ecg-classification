import numpy as np

# calculate the softmax of a vector
def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

v = np.array([-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, 0, 0])
print(softmax(v))