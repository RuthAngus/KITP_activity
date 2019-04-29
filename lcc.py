"""
Code for calculating stellar parameters from light curves.

"""

import numpy as np
from scipy.optimize import minimize
from tqdm import trange


class LCC(object):
    """
    The Light Curve Cannon.

    A series of operations for training the light curve Cannon and using it
    to predict stellar properties.

    Args:
        labels (array): The Nstar x Nlabel array of stellar labels.
        acfs (array): The Nstar x Npixel array of ACFs.

    """

    def __init__(self, labels, acfs):
        self.labels = labels
        self.acfs = acfs


    def fit_to_pixel(self, pixel_acf_values):
        """
        For each pixel in the acf, fit a linear relation to the pixel values

        Args:
            pixel_acf_values (array): An Nstar array of pixel values.

        Returns:
            weights (array): An Nlabel + 1 array of weights, fit to the data.
        """

        A = np.concatenate((np.ones((len(self.labels), 1)), self.labels),
                           axis=1)
        AT = A.T
        ATA = np.dot(AT, A)
        weights = np.linalg.solve(ATA, np.dot(AT, pixel_acf_values))
        return weights


def train(labels, acfs):
    """
    Train the Cannon.

    Args:
        labels (array): The Nstar x Nlabel array of stellar labels.
        acfs (array): The Nstar x Npixel array of ACFs.

    Returns:
        weights (array): The Npixel x (Nlabel + 1) array of weights.
    """

    Nstar, Npixel = np.shape(acfs)
    weights = np.zeros((Npixel, np.shape(labels)[1]+1))
    for pixel in range(Npixel):
        weights[pixel, :] = fit_to_pixel(labels, acfs[:, pixel])
    return weights


def nll(labels_pred, acf, weights):
    """
    The negative-log likelihood of the data (acf) given the model (mod_pred)

    Args:
        labels_pred (array): The stellar parameters (labels) to optimize for.
        acf (array): The y-values, which is the ACF of a star's light curve.
        weights (array): The trained weights used to make a model (predicted)
            ACF.

    Returns:
        The negative log-likelihood.
    """
    A_pred = np.append([1], labels_pred)
    mod_pred = np.dot(weights, A_pred)
    return 0.5 * np.sum((acf - mod_pred)**2)


def predict(labels, acfs, weights):
    """
    Predict the labels of a star from its ACF and trained weights.

    Args:
        labels (array): The Nstar x Nlabel array of stellar labels.
        acfs (array): The Nstar x Npixel array of ACFs.
        weights (array): The Npixel x (Nlabel + 1) trained weights.

    """
    Nstar, Nlabel = np.shape(labels)
    predicted = np.zeros((Nstar, Nlabel))
    chi2 = np.zeros(Nstar)
    inits = np.ones(np.shape(labels)[1])

    for star in trange(Nstar):
        soln = minimize(nll, inits, args=(acfs[star, :], weights));
        predicted[star, :] = soln.x;
        chi2[star] = soln.fun;

    return predicted, chi2
