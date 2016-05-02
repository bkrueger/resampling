"""The module resampling can be used for estimating errors of of non-linear functions of random measurements.

It can be used mainly for the analysis of Markov chain Monte Carlo methods.

"""

import math
import numpy
import numpy.random

# Note:
# How to apply a function over different axis
# numpy.apply_over_axes(numpy.sum, a, [0,2])

def __array_mean_indices(a, indices, func_axis=None, dtype=None):
    """Calculates the mean of an array using the given indices and the optional function axis.
    
    If no function axis is given, the function calculates the mean of the array at the given indices.
    If a function axis is given, the function calculates the mean of the array at the given indices for each entry in the function axis seperatly and returns a tuple of means.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose jackknife mean and error is desired. 
        If `a` is not an array, a conversion is attempted.
    indices : array_like
        One-dimensional list of integer values between 0 and a.size for func_axis = None, or 0 and a.size / a.shape[func_axis] for func_axis != None.
        The mean is calculated using this flat indices.
    dtype : data-type, optional
        Type to use in computing the jackknife mean and error. 
        (For integer inputs, the default is `float64`; for floating point inputs, it is the same as the input dtype.)

    Returns
    -------
    mean_value: tuple
        Tuple containing the means of the dimensions of the function axis.
        If func_axis == None, the tuple contains only one argument.
    
    """
    if func_axis == None:
        return (numpy.mean(a.flat[indices], dtype=dtype), )
    else:
        return tuple(numpy.mean(numpy.reshape(numpy.take(a, [j,], axis=func_axis), -1)[indices]) for j in range(a.shape[func_axis]))

def __number_measurements(a, func_axis=None):
    """ Calculates the number of measurements of an array from the array and the function axis.
    """
    if func_axis == None:
        return a.size
    else:
        return a.size / a.shape[func_axis]

def identity(x):
    """
    Identity function used as default function in the resampling methods.

    """
    return x

def jackknife(a, func=identity, func_axis=None, dtype=None):
    r""" Compute the jackknife mean and error of an array (with :math:`N` values) with respect to a given function :math:`f`.

    Notes
    -----
    The single jackknife values are calculated by

    .. math:: \overline{x_i} = \frac{1}{N} \sum_{j \neq i} x_j,

    and the mean :math:`\overline f` and the error :math:`\sigma_{\overline f}` of the function :math:`f` are calculated using

    .. math:: \overline f &= \frac{1}{N} \sum_i f(\overline{x_i}) \\ \sigma_{\overline f} &= \sqrt{\frac{N - 1}{N}} \cdot \sqrt{\sum_i \left( f(\overline{x_i})^2 - \overline f ^2 \right) }


    Parameters
    ----------
    a : array_like
        Array containing numbers whose jackknife mean and error is desired. 
        If `a` is not an array, a conversion is attempted.
    func: callable function, optional
        (Ususally non-linear) Function that maps a scalar to a scalar (or number to a number), whichs is applied to each jackknife mean value. (The default is the identity function.)
    dtype : data-type, optional
        Type to use in computing the jackknife mean and error. 
        (For integer inputs, the default is `float64`; for floating point inputs, it is the same as the input dtype.)

    Returns
    -------
    jackknife_mean : dtype
        Mean value :math:`\overline f` of the function applied to the data set.
    jackknife_error: dtype
        Error :math:`\sigma_{\overline f}` of the function applied to the data set.

    Examples
    --------
    >>> a = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> jackknife(a)
    (3.0, 0.70710678118654757)
    >>> jackknife(a, func=lambda x: x**2)
    (9.125, 4.247793544888923)
    >>> b = numpy.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 3.0, 5.0, 7.0, 9.0]])
    >>> jackknife(b, func=lambda x,y: x*y, func_axis=0)

    References
    ----------
    .. [1] M. H. Quenouille: Notes on bias in estimation. Biometrika, 43, S.353ff (1956)
    .. [2] J. W. Tukey: Bias and confidence in not quite large samples. Annls. Math. Stat. 29, S. 614 (1958)

    """
    # Calculate the number of measurements
    n = __number_measurements(a, func_axis)
    # Evaluate the function on the jackknife means
    jackknife_values = [func(*(__array_mean_indices(a, range(0,i) + range(i+1, n), func_axis=func_axis, dtype=dtype))) for i in range(n)]

    # Return the average value and the error of this averaged value
    return numpy.mean(jackknife_values), math.sqrt(n - 1)*numpy.std(jackknife_values)


def bootstrap(a, iterations, func=identity, func_axis=None, dtype=None):
    """
    Compute the bootstrap mean and error of an array with respect to a given function.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose jackknife mean and error is desired. 
        If `a` is not an array, a conversion is attempted.
    iterations: int
        Number of bootstrap iterations to execute.
    func: callable function
        (Ususally non-linear) Function that maps a scalar to a scalar (or number to a number), whichs is applied to each jackknife mean value.
    dtype : data-type, optional
        Type to use in computing the jackknife mean and error. 
        For integer inputs, the default is `float64`; for floating point inputs, it is the same as the input dtype.

    Examples
    --------
    >>> a = numpy.random.normal(loc=5.0, scale=2.0, size=1000)
    >>> mean_a, error_a = bootstrap(a, 100)
    >>> (mean_a > 4.9, mean_a < 5.1)
    (True, True)
    >>> (error_a > 2.0/math.sqrt(1000 - 1) - 0.01, error_a < 2.0/math.sqrt(1000 - 1) + 0.01)
    (True, True)

    """
    # Calculate the number of measurements
    n = __number_measurements(a, func_axis)
    # Evaluate the function on the bootstrap means
    bootstrap_values = [func(*(__array_mean_indices(a, numpy.random.randint(0, high=n, size=n), func_axis=func_axis, dtype=dtype))) for i in range(iterations)]

    # Return the average value and the error of this averaged value
    return numpy.mean(bootstrap_values), math.sqrt(float(iterations)/float(iterations - 1))*numpy.std(bootstrap_values)


def subsampling(a, samples, iterations, func=identity, func_axis=None, dtype=None):
    """
    Compute the subsampling mean and error of an array with respect to a given function.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose jackknife mean and error is desired. 
        If `a` is not an array, a conversion is attempted.
    iterations: int
        Number of subsampling iterations to execute.
    samples: int
        Number of samples from the data set to include into the subsampling
    func: callable function
        (Ususally non-linear) Function that maps a scalar to a scalar (or number to a number), whichs is applied to each jackknife mean value.
    dtype : data-type, optional
        Type to use in computing the jackknife mean and error. 
        For integer inputs, the default is `float64`; for floating point inputs, it is the same as the input dtype.

    Examples
    --------
    >>> a = numpy.random.normal(loc=5.0, scale=2.0, size=1000)
    >>> mean_a, error_a = subsampling(a, 100, 50)
    >>> (mean_a > 4.9, mean_a < 5.1)
    (True, True)
    >>> (error_a > 2.0/math.sqrt(1000 - 1) - 0.01, error_a < 2.0/math.sqrt(1000 - 1) + 0.01)
    (True, True)

    """
    # Calculate the number of measurements
    n = __number_measurements(a, func_axis)
    # Evaluate the function on the subsampling means
    subsampling_values = [func(*(__array_mean_indices(a, numpy.random.permutation(range(n))[0:samples], func_axis=func_axis, dtype=dtype))) for i in range(iterations)]

    # Return the average value and the error of this averaged value
    return numpy.mean(subsampling_values), math.sqrt(float(samples)/float(iterations*(iterations - 1)))*numpy.std(subsampling_values)
