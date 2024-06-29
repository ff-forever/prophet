# import warnings
import warnings

warnings.filterwarnings('ignore')

# import numpy and math
import numpy as np
import math
from numpy.linalg import multi_dot
from numpy.linalg import inv

# initialize portfolio return m
m = 0.045
# initialize each asset
mu = np.array([[0.02],
      [0.07],
      [0.15],
      [0.20]])
# Initialization correlation coefficient R
R = np.array([[1,0.3,0.3,0.3],
     [0.3,1,0.6,0.6],
     [0.3,0.6,1,0.6],
     [0.3,0.6,0.6,1]])
# Initialize the variance diagonal matrix
S = np.array([[0.05,0,0,0],
     [0,0.12,0,0],
     [0,0,0.17,0],
     [0,0,0,0.25]])
# Calculate the portfolio covariance matrix
cov = multi_dot([S,R,S])
# Initializes the unit vector matrix
one = np.ones([4,1])
# calculate A,B,C,
A = multi_dot([one.T,inv(cov),one])[0,0]
B = multi_dot([mu.T,inv(cov),one])[0,0]
C = multi_dot([mu.T,inv(cov),mu])[0,0]
# calculate constraints a1,a2
a1 = (A*m-B)/(A*C-math.pow(B,2))
a2 = (C-B*m)/(A*C-math.pow(B,2))
# calculate allocations w
w = multi_dot([inv(cov),multi_dot([a1,mu])+multi_dot([a2,one])])
# calculate risk sigma
sigma =np.sqrt(multi_dot([w.T,cov,w]))
print("σΠ = ", sigma, "\nw∗ = ", w)