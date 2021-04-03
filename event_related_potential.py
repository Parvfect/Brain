from scipy.io import loadmat
from pylab import *
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import pandas as pd
rcParams['figure.figsize']=(12,3)  # Change the default figure size


data = loadmat('matfiles/02_EEG-1.mat')         # Load the data,
EEGa = data['EEGa']                             # ... and get the EEG from one condition,
EEGb = data['EEGb']
t = data['t'][0]                                # ... and a time axis,

# 1000 rows and 500 columns - 1000 trials with 500 time points each

ntrials, nsamples = EEGa.shape

# So we can plot them all, that's good enough to convert them into a dataframe and run unsupervised learning on them

# Sampling rate or sampling frequency defines the number of samples per second (or per other unit)
#  taken from a continuous signal to make a discrete or digital signal
sampling_frequency = 1 / (t[1] - t[0])

"""print("Sampling frequency = ", sampling_frequency, "Hz")

plot(t, EEGa.mean(0))        # Plot the ERP of condition A
xlabel('Time [s]')           # Label the axes
ylabel('Voltage [$\mu V$]')
title('ERP of condition A')  # ... provide a title
show()                       # ... and show the plot


plot(t, EEGb.mean(0))        # Plot the ERP of condition A
xlabel('Time [s]')           # Label the axes
ylabel('Voltage [$\mu V$]')
title('ERP of condition B')  # ... provide a title
show()                       # ... and show the plot
"""
def filter_data(data, low, high, sf, order=2):
    # Determine Nyquist frequency
    nyq = sf/2

    # Set bands
    low = low/nyq
    high = high/nyq

    # Calculate coefficients
    b, a = butter(order, [low, high], btype='band')

    # Filter signal
    filtered_data = lfilter(b, a, data)
    
    return filtered_data

spike_data = filter_data(EEGa, low=10, high=200, sf=500)

mn = EEGa.mean(0)




# Plot signals
plt.plot(t, spike_data.mean(0))
plt.show()

from sklearn.cluster import KMeans
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Apply min-max scaling
scaler= sk.preprocessing.MinMaxScaler()
dataset_scaled = scaler.fit_transform(spike_data)

# Do PCA
pca = PCA(n_components=12)
pca_result = pca.fit_transform(dataset_scaled)

# Plot the 1st principal component aginst the 2nd and use the 3rd for color
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])

fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

def k_means(data):

    # Convert data to Numpy array
    cluster_data = np.array(data)
    
    cluster = KMeans(random_state = 0, n_clusters=2).fit(cluster_data)
    
    labels = cluster.labels_
    centers = cluster.cluster_centers_
            
    return cluster, centers, labels


scaler = StandardScaler()
pca_result = scaler.fit_transform(pca_result)

cluster, centers, labels = k_means(pca_result)

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
plt.show()


