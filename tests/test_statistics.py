"""
Unit tests for the Statistical Process Control (SPC) module.
Ensures accuracy of control limit calculations and anomaly detection.
"""

import pytest
import numpy as np
import pandas as pd
from src.analytics.statistical_control import SPCAnalyzer, ControlLimits

@pytest.fixture
def stable_process_data():
    """Generates stable process data with known mean and std dev."""
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=100, scale=5, size=100))

@pytest.fixture
def unstable_process_data():
    """Generates data with a clear outlier (anomaly)."""
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=5, size=100)
    data[50] = 150 # Significant outlier
    return pd.Series(data)

def test_xbar_limit_calculation(stable_process_data):
    """Verifies that X-bar limits are calculated correctly."""
    analyzer = SPCAnalyzer(sigma_level=3.0)
    limits = analyzer.calculate_xbar_limits(stable_process_data)
    
    assert isinstance(limits, ControlLimits)
    assert limits.mean == pytest.approx(stable_process_data.mean(), rel=1e-5)
    assert limits.ucl > limits.mean
    assert limits.lcl < limits.mean
    
    # Expected UCL = Mean + 3 * StdDev
    expected_ucl = stable_process_data.mean() + (3 * stable_process_data.std(ddof=1))
    assert limits.ucl == pytest.approx(expected_ucl, rel=1e-5)

def test_anomaly_detection(unstable_process_data):
    """Verifies that outliers are correctly identified as anomalies."""
    analyzer = SPCAnalyzer(sigma_level=3.0)
    # Calculate limits from first 40 stable points
    base_data = unstable_process_data[:40]
    limits = analyzer.calculate_xbar_limits(base_data)
    
    # Analyze the whole series
    anomalies = analyzer.detect_anomalies(unstable_process_data, limits)
    
    assert isinstance(anomalies, pd.Series)
    assert anomalies.dtype == bool
    # Index 50 should be an anomaly
    assert anomalies[50] == True
    # Most other points should be False (stable)
    assert anomalies.sum() >= 1

def test_process_capability(stable_process_data):
    """Verifies Cp and Cpk calculations."""
    analyzer = SPCAnalyzer()
    usl = 120
    lsl = 80
    
    metrics = analyzer.calculate_process_capability(stable_process_data, usl, lsl)
    
    assert "cp" in metrics
    assert "cpk" in metrics
    assert metrics["cp"] > 0
    assert metrics["cpk"] <= metrics["cp"]
    
    # Manual calculation verification
    mean = stable_process_data.mean()
    std = stable_process_data.std(ddof=1)
    expected_cp = (usl - lsl) / (6 * std)
    assert metrics["cp"] == pytest.approx(expected_cp, rel=1e-5)

def test_empty_data_raises_error():
    """Ensures robust error handling for empty datasets."""
    analyzer = SPCAnalyzer()
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        analyzer.calculate_xbar_limits(pd.Series([]))

if __name__ == "__main__":
    pytest.main([__file__])
