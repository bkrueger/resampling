"""The module resampling can be used for estimating errors of of non-linear functions of random measurements.

It can be used mainly for the analysis of Markov chain Monte Carlo methods.

"""

import math
import numpy
import numpy.random

# Note:
# How to apply a function over different axis
# numpy.apply_over_axes(numpy.sum, a, [0,2])

def identity(x):
    """
    Identity function used as default function in the resampling methods.

    """
    return x

def jackknife(a, func=identity, axis=None, dtype=None):
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
    func: callable function. optional
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
    >>> jackknife(a, func=lambda x:x**2)
    (9.125, 4.247793544888923)

    References
    ----------
    .. [1] M. H. Quenouille: Notes on bias in estimation. Biometrika, 43, S.353ff (1956)
    .. [2] J. W. Tukey: Bias and confidence in not quite large samples. Annls. Math. Stat. 29, S. 614 (1958)

    """
    # Jackknife for the whole array
    n = len(a)
    jackknife_values = numpy.fromiter((func(numpy.mean(a[range(0, i) + range(i+1, n)], dtype=dtype)) for i in range(n)), numpy.float)

    # Return the average value and the error of this averaged value
    return numpy.mean(jackknife_values, axis=axis), math.sqrt(n - 1)*numpy.std(jackknife_values, axis=axis)


def bootstrap(a, iterations, func=identity, axis=None, dtype=None):
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
    # Bootstrap for the whole array
    n = len(a)
    bootstrap_values = numpy.fromiter((func(numpy.mean(a[numpy.random.randint(0, high=n, size=n)], dtype=dtype)) for i in range(iterations)), numpy.float)

    # Return the average value and the error of this averaged value
    return numpy.mean(bootstrap_values), math.sqrt(float(iterations)/float(iterations - 1))*numpy.std(bootstrap_values)


def subsampling(a, samples, iterations, func=identity, axis=None, func_axis=None, dtype=None):
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
    # Subsampling for the whole array
    n = len(a)
    subsampling_values = numpy.fromiter((func(numpy.mean(numpy.random.permutation(a)[0:samples], dtype=dtype)) for i in range(iterations)), numpy.float)

    # Return the average value and the error of this averaged value
    return numpy.mean(subsampling_values), math.sqrt(float(samples)/float(iterations*(iterations - 1)))*numpy.std(subsampling_values)
