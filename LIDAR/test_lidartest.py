import unittest
import numpy as np
from numpy import pi, cos, sin, sqrt, allclose
import sys
import os

# Add the LIDAR directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidartest import sph_to_cart, estimate_params


class TestLidarFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with known values and tolerances."""
        self.tolerance = 1e-10  # Tolerance for floating point comparisons
    
    def test_sph_to_cart_origin(self):
        """Test spherical to cartesian conversion at origin (r=0)."""
        result = sph_to_cart(0.0, 0.0, 0.0)
        expected = np.array([0.0, 0.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_positive_x_axis(self):
        """Test conversion for point on positive x-axis."""
        # epsilon=0, alpha=0, r=1 should give (1, 0, 0)
        result = sph_to_cart(0.0, 0.0, 1.0)
        expected = np.array([1.0, 0.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_positive_y_axis(self):
        """Test conversion for point on positive y-axis."""
        # epsilon=0, alpha=pi/2, r=1 should give (0, 1, 0)
        result = sph_to_cart(0.0, pi/2, 1.0)
        expected = np.array([0.0, 1.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_positive_z_axis(self):
        """Test conversion for point on positive z-axis."""
        # epsilon=pi/2, alpha=0, r=1 should give (0, 0, 1)
        result = sph_to_cart(pi/2, 0.0, 1.0)
        expected = np.array([0.0, 0.0, 1.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_negative_axes(self):
        """Test conversion for points on negative axes."""
        # Test negative x-axis: epsilon=0, alpha=pi, r=1 should give (-1, 0, 0)
        result = sph_to_cart(0.0, pi, 1.0)
        expected = np.array([-1.0, 0.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
        
        # Test negative y-axis: epsilon=0, alpha=-pi/2, r=1 should give (0, -1, 0)
        result = sph_to_cart(0.0, -pi/2, 1.0)
        expected = np.array([0.0, -1.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
        
        # Test negative z-axis: epsilon=-pi/2, alpha=0, r=1 should give (0, 0, -1)
        result = sph_to_cart(-pi/2, 0.0, 1.0)
        expected = np.array([0.0, 0.0, -1.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_arbitrary_point(self):
        """Test conversion for arbitrary point with known result."""
        # epsilon=pi/4, alpha=pi/4, r=sqrt(2)
        epsilon, alpha, r = pi/4, pi/4, sqrt(2)
        result = sph_to_cart(epsilon, alpha, r)
        
        # Manual calculation:
        # x = sqrt(2) * cos(pi/4) * cos(pi/4) = sqrt(2) * (1/sqrt(2)) * (1/sqrt(2)) = 1/sqrt(2)
        # y = sqrt(2) * sin(pi/4) * cos(pi/4) = sqrt(2) * (1/sqrt(2)) * (1/sqrt(2)) = 1/sqrt(2)
        # z = sqrt(2) * sin(pi/4) = sqrt(2) * (1/sqrt(2)) = 1
        expected = np.array([1/sqrt(2), 1/sqrt(2), 1.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_different_radius(self):
        """Test conversion with different radius values."""
        # Test with r=5
        result = sph_to_cart(0.0, 0.0, 5.0)
        expected = np.array([5.0, 0.0, 0.0])
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_sph_to_cart_return_type(self):
        """Test that function returns numpy array of correct size."""
        result = sph_to_cart(0.0, 0.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
    
    def test_estimate_params_horizontal_plane(self):
        """Test parameter estimation for horizontal plane (z = constant)."""
        # Points on plane z = 5
        P = np.array([
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5],
            [1, 1, 5]
        ])
        result = estimate_params(P)
        expected = np.array([5.0, 0.0, 0.0])  # z = 5 + 0*x + 0*y
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_estimate_params_tilted_plane_x(self):
        """Test parameter estimation for plane tilted in x direction."""
        # Points on plane z = 2*x (passes through origin with slope in x)
        P = np.array([
            [0, 0, 0],
            [1, 0, 2],
            [2, 0, 4],
            [0, 1, 0],
            [1, 1, 2]
        ])
        result = estimate_params(P)
        expected = np.array([0.0, 2.0, 0.0])  # z = 0 + 2*x + 0*y
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_estimate_params_tilted_plane_y(self):
        """Test parameter estimation for plane tilted in y direction."""
        # Points on plane z = 3*y
        P = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 3],
            [0, 2, 6],
            [1, 1, 3]
        ])
        result = estimate_params(P)
        expected = np.array([0.0, 0.0, 3.0])  # z = 0 + 0*x + 3*y
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_estimate_params_general_plane(self):
        """Test parameter estimation for general plane."""
        # Points on plane z = 1 + 2*x + 3*y
        P = np.array([
            [0, 0, 1],    # z = 1 + 2*0 + 3*0 = 1
            [1, 0, 3],    # z = 1 + 2*1 + 3*0 = 3  
            [0, 1, 4],    # z = 1 + 2*0 + 3*1 = 4
            [1, 1, 6],    # z = 1 + 2*1 + 3*1 = 6
            [2, 1, 8]     # z = 1 + 2*2 + 3*1 = 8
        ])
        result = estimate_params(P)
        expected = np.array([1.0, 2.0, 3.0])  # z = 1 + 2*x + 3*y
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_estimate_params_minimum_points(self):
        """Test parameter estimation with minimum number of points (3)."""
        # Three points defining a plane z = 1 + x + y
        P = np.array([
            [0, 0, 1],    # z = 1 + 0 + 0 = 1
            [1, 0, 2],    # z = 1 + 1 + 0 = 2
            [0, 1, 2]     # z = 1 + 0 + 1 = 2
        ])
        result = estimate_params(P)
        expected = np.array([1.0, 1.0, 1.0])  # z = 1 + 1*x + 1*y
        self.assertTrue(allclose(result, expected, atol=self.tolerance))
    
    def test_estimate_params_return_type(self):
        """Test that estimate_params returns numpy array of correct size."""
        P = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        result = estimate_params(P)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
    
    def test_estimate_params_noisy_data(self):
        """Test parameter estimation with slightly noisy data."""
        # Points approximately on plane z = 2 + x + 0.5*y with small noise
        np.random.seed(42)  # For reproducible results
        true_params = np.array([2.0, 1.0, 0.5])
        
        # Generate points on the true plane
        x_coords = np.random.uniform(-2, 2, 20)
        y_coords = np.random.uniform(-2, 2, 20)
        z_coords = true_params[0] + true_params[1] * x_coords + true_params[2] * y_coords
        
        # Add small amount of noise
        z_coords += np.random.normal(0, 0.01, 20)
        
        P = np.column_stack([x_coords, y_coords, z_coords])
        result = estimate_params(P)
        
        # Should be close to true parameters (within noise tolerance)
        self.assertTrue(allclose(result, true_params, atol=0.1))
    
    def test_integration_sph_to_cart_and_estimate_params(self):
        """Integration test: convert spherical to cartesian then estimate plane."""
        # Create spherical coordinates for points on a known plane
        # We'll create points that should lie approximately on z = 1 + 0.5*x
        
        # Spherical coordinates
        epsilons = [0.1, 0.2, 0.15, 0.05]
        alphas = [0, pi/4, pi/2, pi]
        radii = [2, 3, 2.5, 4]
        
        # Convert to cartesian
        cartesian_points = []
        for i in range(len(epsilons)):
            point = sph_to_cart(epsilons[i], alphas[i], radii[i])
            cartesian_points.append(point)
        
        P = np.array(cartesian_points)
        
        # Estimate parameters (this should work without error)
        result = estimate_params(P)
        
        # Just check that we get a valid result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)