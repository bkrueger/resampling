# This file is part of the resampling package that can be found at https://github.com/bkrueger/resampling 
# Copyright (c) 2015 Benedikt Krüger
# Resampling is published under the MIT license, see https://opensource.org/licenses/MIT or the LICENSE file in the root directory of the project

import sys
sys.path.append('../')

import math
import numpy
import resampling

# Number of data points
N = 2000

# Create uniformly and gaussian distributed values
uniform_dist_numbers = numpy.random.random(N)
gaussian_dist_numbers = numpy.random.normal(0.0, 1.0, N)
# Create an array with both uniformly and gaussian distributed values
uniform_gaussian_dist_numbers = numpy.zeros((N,2))
uniform_gaussian_dist_numbers[:,0] = numpy.random.random(N)
uniform_gaussian_dist_numbers[:,1] = numpy.random.normal(1.0, 1.0, N)

# Tests whether the two values have the same shape and are equal up to a given delta
def assert_equal_delta(actual_value, desired_value, delta=0.1):
    assert math.fabs(actual_value - desired_value) <= delta, "Assertion Error:\nValues are not equal up to delta = " + str(delta) + "\n ACTUAL: " + str(actual_value) + "\n DESIRED: " + str(desired_value)

# Class for testing the functionality of the jackknife algorithm
class TestJackknife(numpy.testing.TestCase):
    def test_flat(self):
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.jackknife(uniform_dist_numbers)
        assert_equal_delta(uniform_dist_mean, 0.5, delta = 0.1)
        assert_equal_delta(uniform_dist_error, 1.0 / math.sqrt(12) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.jackknife(gaussian_dist_numbers)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, 1.0 / math.sqrt(N - 1), delta = 0.1)

    def test_function(self):
        # Define the non-linear function to test with
        test_function = lambda x: x**2
        
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.jackknife(uniform_dist_numbers, func=test_function)
        assert_equal_delta(uniform_dist_mean, 0.25, delta = 0.1)
        assert_equal_delta(uniform_dist_error, math.sqrt(4.0 / 45.0) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.jackknife(gaussian_dist_numbers, func=test_function)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, math.sqrt(2) / math.sqrt(N - 1), delta = 0.1)

    def test_func_axis(self):
        # Define the non-linear function with two arguments to test with
        test_function = lambda x,y: x*y

        # Test the results for the combined distribution
        uniform_gaussian_dist_mean, uniform_gaussian_dist_error = resampling.jackknife(uniform_gaussian_dist_numbers, func=test_function, func_axis=1)
        assert_equal_delta(uniform_gaussian_dist_mean, 0.5, delta = 0.1)
        assert_equal_delta(uniform_gaussian_dist_error, math.sqrt(1 + 1.0/12.0) / math.sqrt(N - 1), delta = 0.1)

# Class for testing the functionality of the bootstrap algorithm
class TestBootstrap(numpy.testing.TestCase):
    def test_flat(self):
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.bootstrap(uniform_dist_numbers, 50)
        assert_equal_delta(uniform_dist_mean, 0.5, delta = 0.1)
        assert_equal_delta(uniform_dist_error, 1.0 / math.sqrt(12) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.bootstrap(gaussian_dist_numbers, 50)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, 1.0 / math.sqrt(N - 1), delta = 0.1)

    def test_function(self):
        # Define the non-linear function to test with
        test_function = lambda x: x**2
        
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.bootstrap(uniform_dist_numbers, 50, func=test_function)
        assert_equal_delta(uniform_dist_mean, 0.25, delta = 0.1)
        assert_equal_delta(uniform_dist_error, math.sqrt(4.0 / 45.0) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.bootstrap(gaussian_dist_numbers, 50, func=test_function)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, math.sqrt(2) / math.sqrt(N - 1), delta = 0.1)


# Class for testing the functionality of the subsampling algorithm
class TestSubsampling(numpy.testing.TestCase):
    def test_flat(self):
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.subsampling(uniform_dist_numbers, 20, 50)
        assert_equal_delta(uniform_dist_mean, 0.5, delta = 0.1)
        assert_equal_delta(uniform_dist_error, 1.0 / math.sqrt(12) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.subsampling(gaussian_dist_numbers, 20, 50)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, 1.0 / math.sqrt(N - 1), delta = 0.1)

    def test_function(self):
        # Define the non-linear function to test with
        test_function = lambda x: x**2
        
        # Test the results for the uniform distribution
        uniform_dist_mean, uniform_dist_error = resampling.subsampling(uniform_dist_numbers, 20, 50, func=test_function)
        assert_equal_delta(uniform_dist_mean, 0.25, delta = 0.1)
        assert_equal_delta(uniform_dist_error, math.sqrt(4.0 / 45.0) / math.sqrt(N - 1), delta = 0.1)

        # Test the results for the gaussian distribution
        gaussian_dist_mean, gaussian_dist_error = resampling.subsampling(gaussian_dist_numbers, 20, 50, func=test_function)
        assert_equal_delta(gaussian_dist_mean, 0.0, delta = 0.1)
        assert_equal_delta(gaussian_dist_error, math.sqrt(2) / math.sqrt(N - 1), delta = 0.1)


if __name__ == "__main__":
    numpy.testing.run_module_suite()
