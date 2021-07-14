# Taku Ito
# 05/29/16

#Multiple linear regression for FC approximation in python (templated off Mike's MATLAB script)

import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA

def multregressionconnectivity(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    Assumes all time series are de-meaned
    """

    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use Principle component regression')

    interaction_mat = np.zeros((nregions,nregions))
    rsquared = []
    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = activityMatrix[targetregion,:]
        model = sm.OLS(y, X)
        results = model.fit()
        interaction_mat[otherregions, targetregion] = results.params[1:] # all betas except for constant betas
        # get b0 as a diagonal
        interaction_mat[targetregion, targetregion] = results.params[0]
        # Get r-squared values
        rsquared.append(results.rsquared)
        
    return interaction_mat, rsquared

def tanhFC(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    Uses a tanh fit
    but achieves same idea (i.e., saturation of activity)
    """
    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use Principle component regression')

    interaction_mat = np.zeros((nregions,nregions))
    rsquared = []
    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = activityMatrix[targetregion,:]
        # Transform y variables and fit to an arctanh func
        y2 = y/(np.max(np.abs(y))+.5)
        y2 = np.arctanh(y2)
        # Fit a Generalized Linear Model with a log func (as opposed to identity func)
        model = sm.OLS(y2, X)
        results = model.fit()
        interaction_mat[otherregions, targetregion] = results.params[1:] # all betas except for constant betas
        # get b0 as a diagonal
        interaction_mat[targetregion, targetregion] = results.params[0]
        # Get r-squared values
        rsquared.append(results.rsquared)

    return interaction_mat, rsquared


def squareRootFC(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    FC that uses a square root function as transfer func
    but achieves same idea (i.e., saturation of activity)
    """
    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use Principle component regression')

    interaction_mat = np.zeros((nregions,nregions))
    rsquared = []
    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = activityMatrix[targetregion,:]
        # Transform y variables and fit to an arctanh func
        ytmp = y < 0
        if ytmp.any(): 
            # Subtract the most negative value (if any are negative)
            y2 = y - np.min(y) 
        else:
            y2 = y
        # Square this; inverse is therefore a square root transfer func
        y2 = y**2
        # Fit a Generalized Linear Model with a log func (as opposed to identity func)
        model = sm.OLS(y2, X)
        results = model.fit()
        interaction_mat[otherregions, targetregion] = results.params[1:] # all betas except for constant betas
        # get b0 as a diagonal
        interaction_mat[targetregion, targetregion] = results.params[0]
        # Get r-squared values
        rsquared.append(results.rsquared)

    return interaction_mat, rsquared

def squareFC(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    FC that uses a square root function as transfer func
    but achieves same idea (i.e., saturation of activity)
    """
    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use Principle component regression')

    interaction_mat = np.zeros((nregions,nregions))
    rsquared = []
    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = activityMatrix[targetregion,:]
        # Transform y variables and fit to an arctanh func
        ytmp = y < 0
        if ytmp.any(): 
            # Subtract the most negative value (if any are negative)
            y2 = y - np.min(y) 
        else:
            y2 = y
        # Square this; inverse is therefore a square root transfer func
        y2 = np.sqrt(y2)
        # Fit a Generalized Linear Model with a log func (as opposed to identity func)
        model = sm.OLS(y2, X)
        results = model.fit()
        interaction_mat[otherregions, targetregion] = results.params[1:] # all betas except for constant betas
        # get b0 as a diagonal
        interaction_mat[targetregion, targetregion] = results.params[0]
        # Get r-squared values
        rsquared.append(results.rsquared)

    return interaction_mat, rsquared


def nonlinearGLMFC(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    Uses a log fit (not really a logistic function)
    but achieves same idea (i.e., saturation of activity)
    """
    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use Principle component regression')

    interaction_mat = np.zeros((nregions,nregions))

    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = activityMatrix[targetregion,:]
        # Fit a Generalized Linear Model with a log func (as opposed to identity func)
        gauss_log = sm.GLM(y,X,family=sm.families.Gaussian(sm.families.links.log))
        gauss_log_results = gauss_log.fit()
        interaction_mat[otherregions, targetregion] = gauss_log_results.params[1:] # all betas except for constant betas
        
    return interaction_mat 
    



def pcaRegressionConnectivity(activityMatrix, ncomponents=500, outdir='/projects/ModalityControl/data/results/multregconn_restfc_vertices/',filename='default.csv'):
    """
    Inputs:
    activityMatrix - regions X time matrix
    ncomponents - number components to run regression on

    Returns:
    region X region multiple regression connectivity matrix
    """

    nregions = activityMatrix.shape[0]
    interaction_mat = np.zeros((ncomponents,ncomponents));

    pca = PCA(n_components=ncomponents)
    reduced_mat = pca.fit_transform(activityMatrix.T)
    reduced_mat = reduced_mat.T

    for targetregion in range(ncomponents):
        otherregions = range(ncomponents)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regiosn'
        X = reduced_mat[otherregions,:].T
        # Add 'constant' regressor
        X = sm.add_constant(X)
        y = reduced_mat[targetregion,:]
         ##*** STARTHERE
        model = sm.OLS(y, X)
        results = model.fit()
        interaction_mat[otherregions, targetregion] = results.params[1:] # all betas except for constant betas

    # Now tranpose matrix back to original space, i.e., ncomponents -> nvertices/nregions
    # Since it's not in samples x component space and just run each 'sample' or component separately
    multreg_mat = np.zeros((nregions,nregions))
    for sample in range(ncomponents):
        feat_array = pca.inverse_transform(interaction_mat[sample,:])
        multreg_mat[sample,:] = feat_array

    outfile = outdir + filename
    # Output a csv file
    np.savetxt(outfile, multreg_mat, delimiter=',')
        
    return multreg_mat

