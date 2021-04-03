import matplotlib.pyplot as plt
import numpy as np 
import numerical_integration as ode

class ElasticPendulum:

    y = np.array([0, 0.0, 0.0, 0.0])

    def __init__(self, x1, y1, x2, y2, m, l, k, g):
        
        self.y[0] = x1
        self.y[1] = x2
        self.y[2] = y1
        self.y[3] = y2
        self.m = m 
        self.l = l
        self.k = k 
        self.g = g 

    def fz(self, y, t):
    
        x1, x2, y1, y2 = y[0], y[1], y[2], y[3]

        y12 = (- self.g * np.sin(y[0]) - 2 * y[3] * y[2] ) / (self.l + y[1])
        y22 = (self.l + y[1]) * y[2]**2 - (self.k * (y[1]) / self.m)  +  self.g * np.cos(y[0])  

        return np.array([y1, y2, y12, y22]) 

    def solve(self, steps, dt, type):
        

        y_arr = [[],[],[],[]]
        times = []
        t = 0

        for i in range(steps):

            if type == 1:
                self.y += ode.runge_kutta(self.fz, self.y, t, dt)
            elif type == 2:
                self.y += ode.euler(self.fz, self.y, t, dt)
            elif type == 3:
                val, dt = ode.runge_kutta_adaptive_stepper(self.fz, self.y,t, dt)
                self.y += val
            else:
                val, dt = ode.euler_adaptive_step(self.fz, self.y,t, dt)
                self.y += val
            
            #Appending into arrays
            y_arr[0].append(self.y[0] * 180 / 3.14)
            y_arr[1].append(self.y[1])
            y_arr[2].append(self.y[2])
            y_arr[3].append(self.y[3])
            times.append(t)
            
            #Increasing the time
            t += dt

        plt.plot(times, y_arr[0])
        plt.title("Position vs time for theta")
        plt.xlabel("Time")
        plt.ylabel("Theta")
        plt.grid(True, which='both')
        plt.show()

        plt.plot(times, y_arr[1])
        plt.title("Position vs time for spring extension")
        plt.xlabel("Time")
        plt.ylabel("Spring extension")
        plt.show()
        
        plt.plot(y_arr[0], y_arr[2])
        plt.title("Phase space for theta")
        plt.xlabel("Theta")
        plt.ylabel("Angular Velocity")
        plt.grid(True, which='both')
        plt.show()

        plt.plot(y_arr[1], y_arr[3])
        plt.title("Phase space for spring extension")
        plt.xlabel("Spring extension")
        plt.ylabel("Spring extension velocity")
        plt.grid(True, which='both')
        plt.show()

t = ElasticPendulum(0.02,0.001,0.03,0.002, 5, 5,8.5,9.8)
t.solve(10000,0.01,2)