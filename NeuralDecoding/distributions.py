
import numpy as np
import matplotlib.pyplot as plt
import math

class Poisson:

    def __init__(self, k):
        self.k = k

    def fz(self, r, T):
        return (math.pow((r*T),self.k) * math.exp((-r*T)))/(math.factorial(int(self.k)))

    def plot(self):
        r = np.arange(0,100,0.01)
        T = np.arange(0,100,0.01) 
        P = []
        
        for i in range(len(r)):
            P.append(self.fz(r[i],T[i]))

        plt.plot(P)
        plt.show()



a = Poisson(2)
a.plot()



