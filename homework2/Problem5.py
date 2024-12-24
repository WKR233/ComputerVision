import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
n = 1024
lam = 0.05
L = 1000
N = int(lam*n*n)
positions = n*np.random.rand(N, 2)
statistics = np.zeros((n-L, n-L))
for p in tqdm(positions):
    offsetx_lower_bound = max(0, int(np.ceil(p[0]) - L))
    offsetx_upper_bound = min(n-L-1, int(np.floor(p[0])))
    offsety_lower_bound = max(0, int(np.ceil(p[1]) - L))
    offsety_upper_bound = min(n-L-1, int(np.floor(p[1])))
    for offsetx in range(offsetx_lower_bound, offsetx_upper_bound):
        for offsety in range(offsety_lower_bound, offsety_upper_bound):
            statistics[offsetx, offsety] += 1
hist = np.histogram(statistics, N, (0, N))[0]
R = [i for i in range(N)]
plt.plot(R, hist)
plt.show()