# Taku Ito
# 12/21/2016

# Run Logistic regression given some input matrix (i.e., region x time)

import sklearn
import numpy as np
import scipy.stats as stats

def logMultRegFC(timeseries):
    """
    Takes as input a region x time matrix and provides a region x region FC matrix derived using multiple logistic regression
    """

    nregions = timeseries.shape[0]
    ntime = timeseries.shape[1]

    for roi in range(nregions):
        
