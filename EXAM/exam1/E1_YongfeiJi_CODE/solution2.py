# import warnings
import warnings

warnings.filterwarnings('ignore')
# import numpy and math
import numpy as np
import math
from numpy.linalg import multi_dot
from numpy.linalg import inv

# Initialize the asset ratio
w = np.array([[0.5],
              [0.2],
              [0.3]])
# Initialization correlation coefficient R
R = np.array([[1,0.8,0.5],
              [0.8,1,0.3],
              [0.5,0.3,1]])
# Initialize the variance diagonal matrix
S = np.array([[0.30,0,0],
              [0,0.20,0],
              [0,0,0.15]])
# Calculate the portfolio covariance matrix
cov = multi_dot([S,R,S])
# calculate risk sigma
sigma = np.sqrt(multi_dot([w.T,cov, w]))[0,0]
# Initialize the confidence level
m = 0.99
# Initialize the ES factor at 99% confidence level
n = -2.33
# Calculate VaR
VaR = n * sigma
# Calculate ES
ES = -sigma*(np.exp(-0.5*np.power(n,2)))/((1-m)*np.sqrt(2*(np.pi)))
print("Var = ", VaR, "\nES = ", ES)

