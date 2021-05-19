import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Datasets\smni_eeg_data.tar.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)

print(data)