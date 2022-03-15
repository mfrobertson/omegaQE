import numpy as np


class Maths:


    def __init__(self): pass

    def rectangular_pulse_steps(self, arr, min, max):
        """
        Produces the steps of a rectangular pulse function for given boundaries acting on the input array.

        Parameters
        ----------
        arr : ndarray
            Array for which the steps will be calculated.
        min : int or float
            Minimum boundary.
        max : int or float
            Maximum boundary.

        Returns
        -------
        ndarray
            The steps of same dimensions as the input array, containing the steps resulting from a rectangular pulse given the boundaries.
        """
        shape = np.shape(arr)
        if shape == ():
            shape = (1,)
        step = np.ones(shape)
        step[:] = 1
        step[arr < min] = 0
        step[arr > max] = 0
        return step

    def heaviside_steps(self, arr, x=0, reverse=False):
        """
        Produces the steps of a heaviside function for a given boundary acting on the input array.

        Parameters
        ----------
        arr : ndarray
            Array for which the steps will be calculated.
        x : int or float
            Step point.
        reverse : bool
            Reverse Heaviside function.

        Returns
        -------
        ndarray
            The steps of same dimensions as the input array, containing the steps resulting from the heaviside location of the step.
        """
        steps = np.ones(arr.shape)
        steps[:] = 1
        if reverse:
            steps[arr > x] = 0
        else:
            steps[arr < x] = 0
        return steps

    def cross(self, mag1, mag2, theta):
        """

        Parameters
        ----------
        mag1
        mag2
        theta

        Returns
        -------

        """
        return mag1 * mag2 * np.sin(theta)

    def dot(self, mag1, mag2, theta):
        """

        Parameters
        ----------
        mag1
        mag2
        theta

        Returns
        -------

        """
        return mag1 * mag2 * np.cos(theta)

    def cosine_rule(self, a, b, theta):
        """

        Parameters
        ----------
        a
        b
        theta

        Returns
        -------

        """
        return np.sqrt(a**2 + b**2 - (2 * a * b * np.cos(theta)))

    def sine_rule(self, a, theta_a, b=None, theta_b=None):
        """

        Parameters
        ----------
        a
        theta_a
        b
        theta_b

        Returns
        -------

        """
        if (b is None and theta_b is None) or (b is not None and theta_b is not None):
            print("Must supply either b or theta_b")
            return
        if b is not None:
            return np.arcsin(b*np.sin(theta_a)/a)
        return a*np.sin(theta_b)/np.sin(theta_a)