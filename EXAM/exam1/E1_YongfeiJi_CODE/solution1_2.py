# Import warnings
import warnings

warnings.filterwarnings('ignore')
# import pandas and numpy
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
# Set numpy random seed
np.random.seed(42)
# Import cufflinks
import cufflinks as cf

cf.set_config_file(offline=True, dimensions=((1000, 600)))
# Import plotly express for EF plot
import plotly.express as px

px.defaults.width, px.defaults.height = 1000, 600
# Set precision
pd.set_option('display.precision', 4)

# Initialization return mu
mu = list([0.02,0.07,0.15,0.20])
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

# Define a simulation function for a portfolio
def portfolio_simulation(mu):
    # initialize the lists
    rets = []
    vols = []
    wts = []
    # initialize number of simulations
    numofportfolio = 700
    # simulate 700 portfolio
    for i in range(numofportfolio):

        # generate random 4 weights
        weights = np.random.random(3)
        w1 = np.array([1-weights[0]- weights[1] - weights[2]])
        weights = np.append(weights,w1)

        # portfolio stats
        rets.append(weights.T @ np.array(mu))
        vols.append(
            np.sqrt(multi_dot([weights.T,cov, weights])))
        wts.append(weights)

    # create a datafrme for analysis
    data = {'port_rets': rets, 'port_vols': vols}
    portdf = pd.DataFrame(data)
    return round(portdf, 4)


temp = portfolio_simulation(mu)
# Plot simulated portfolio
fig = px.scatter(temp,
                 x='port_vols',
                 y='port_rets',
                 labels={
                     'port_vols': 'Expected Volatility σΠ',
                     'port_rets': 'Expected Return µΠ',
                 },
                 title="Monte Carlo Simulated Portfolio").update_traces(
                     mode='markers', marker=dict(symbol='cross'))
# Show spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()
