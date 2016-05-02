# resampling

This is a python module for doing resampling error analysis using the Jackknife, the Bootstrap or the Subsampling method.

## Why to use resampling methods?

Suppose you take N samples X_i from a randomly distributed quantity X, e.g. by actual measurements in the laboratory or by Monte Carlo simulations, and store them in a numpy array `x_measurements`
Good estimators for the mean `mean_x` of the quantity X and its standard error `mean_x_error` can be calculated using the standard numpy routines by
```python
mean_x = x_measurements.mean()
mean_x_error = x_measurements.std() / math.sqrt(x.size() - 1)
```

This is totally enough if you want to calculate the mean of X or any linear function of X. But what if you are interested in a non-linear function `func` of the quantity, e.g. if you are interested in the squared mean <X>**2 and its errors? Then [one can show](http://arxiv.org/abs/cond-mat/0410490v1) that the following estimates are biased and should not be used:
```python
func_mean_x = func(x_measurements).mean()
func_mean_x_error = func(x_measurements).std() / math.sqrt(x.size() - 1)
```

Instead you should use resampling methods as the [Jackknife](https://en.wikipedia.org/wiki/Jackknife_resampling), the [Bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) or the [subsampling](https://normaldeviate.wordpress.com/2013/01/27/bootstrapping-and-subsampling-part-ii/) approach. 


## Installation

Until now there is no installation routine. Just download (or clone) resampling.py into your source directory or a directory that is visible to your python interpreter.


## Usage

The following code snippet shows the usage of the resampling package for estimating the square of the expectation value of a uniform distribution:

```python
import numpy
import resampling

a = numpy.random.random(200)

# Resampling using Jackknife
squared_mean_a, squared_mean_a_error = resampling.jackknife(a, func=lambda x: x**2)

```

## Documentation

The full documentation of all functions can be found [here](http://bkrueger.github.io/resampling/).