# Import numpy
from numpy import *

#import pyplot from matplotlib
import matplotlib.pyplot as plt

# Define the option price function
def binomial_option(spot: float, strike: float, rate: float, sigma: float, time:float, steps: int, output: int=0) -> ndarray:
    # params
    # Initialize dt
    ts = time/steps
    # Initial option price up u
    u = 1+sigma*sqrt(ts)
    # Initial option price down v
    v = 1-sigma*sqrt(ts)
    # Initial Neutral probability
    p = 0.5+rate*sqrt(ts)/(2*sigma)
    df = 1/(1+rate*ts)
    # initialize arrays
    px = zeros((steps+1, steps+1))
    cp = zeros((steps+1, steps+1))
    V = zeros((steps+1, steps+1))
    d = zeros((steps+1, steps+1))
    # binomial loop
    # forward loop
    for j in range(steps+1):
        for i in range(j+1):
            px[i,j] = spot*power(v,i)*power(u,j-i)
            cp[i,j] = maximum(px[i,j]-strike, 0)
    # reverse loop
    for j in range(steps+1, 0, -1):
        for i in range(j):
            if (j==steps+1):
                V[i,j-1] = cp[i,j-1]
                d[i,j-1] = 0
            else:
                V[i,j-1] = df*(p*V[i,j]+(1-p)*V[i+1,j])
                d[i,j-1] = (V[i,j]-V[i+1,j])/(px[i,j]-px[i+1,j])
    results = around(px,2), around(cp,2), around(V,2), around(d,4)
    return results[output]
# Define a function of the change in the value of the option over the time step
def val_to_step(step:int):
    return binomial_option(100,100,0.05,0.2,1,step,2)[0,0]
# Define a function of the value of the option as it changes with volatility
def val_to_vol(sigma:float):
    return binomial_option(100,100,0.05,sigma,1,4,2)[0,0]

x1 = linspace(0.05,0.8,100)
y1 = []
for i in range(len(x1)):
    y1.append(val_to_vol(x1[i]))
x2 = range(3,51)
y2 = []
for j in range(len(x2)):
    y2.append(val_to_step(x2[j]))
# Plot option value changes with volatility
plt.xlabel("volatility")
plt.ylabel("value")
plt.plot(x1,y1)

# Plot option value changes with time step
plt.xlabel("time steps")
plt.ylabel("value")
plt.plot(x2,y2)
plt.show()