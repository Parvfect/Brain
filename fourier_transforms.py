import scipy
import matplotlib.pyplot as plt
import numpy as np
from read_data import main
from scipy.fft import fft, fftfreq

file_name = "activity.txt"

# sample spacing
T = 1.0 / 250.
y = main(file_name)

# Number of sample points
N = len(y)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.title(file_name)
plt.show()

"""
References 
1) https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
"""