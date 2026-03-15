"""
Statistical Process Control (SPC) and Six Sigma Anomaly Detection Module.
Provides enterprise-grade tools for monitoring process stability and identifying special-cause variation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ControlLimits(BaseModel):
    """Data model for statistical control limits."""
    ucl: float = Field(..., description="Upper Control Limit")
    lcl: float = Field(..., description="Lower Control Limit")
    mean: float = Field(..., description="Process Mean (Center Line)")
    sigma: float = Field(..., description="Process Standard Deviation")


class SPCAnalyzer:
    """
    Implements Six Sigma Statistical Process Control (SPC) methodologies.
    Focuses on X-bar and R-charts for continuous data streams.
    """

    def __init__(self, sigma_level: float = 3.0):
        """
        Initialize the SPC Analyzer.

        Args:
            sigma_level: The number of standard deviations for control limits (default is 3.0 for Six Sigma).
        """
        self.sigma_level = sigma_level

    def calculate_xbar_limits(self, data: Union[pd.Series, np.ndarray]) -> ControlLimits:
        """
        Calculate control limits for an X-bar chart.

        Args:
            data: The process data series.

        Returns:
            ControlLimits object.
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty.")

        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        
        ucl = mean + (self.sigma_level * std_dev)
        lcl = mean - (self.sigma_level * std_dev)

        logger.info(f"Calculated X-bar Limits - Mean: {mean:.4f}, UCL: {ucl:.4f}, LCL: {lcl:.4f}")
        
        return ControlLimits(ucl=ucl, lcl=lcl, mean=mean, sigma=std_dev)

    def detect_anomalies(self, data: pd.Series, limits: ControlLimits) -> pd.Series:
        """
        Identify statistical anomalies based on Western Electric Rules (simplified).
        Points outside control limits are considered 'out of control'.

        Args:
            data: The process data to analyze.
            limits: The control limits to apply.

        Returns:
            Boolean series where True indicates an anomaly.
        """
        out_of_ucl = data > limits.ucl
        out_of_lcl = data < limits.lcl
        
        anomalies = out_of_ucl | out_of_lcl
        
        anomaly_count = anomalies.sum()
        if anomaly_count > 0:
            logger.warning(f"Detected {anomaly_count} statistical anomalies out of {len(data)} points.")
            
        return anomalies

    def calculate_process_capability(self, data: pd.Series, usl: float, lsl: float) -> Dict[str, float]:
        """
        Calculate Cp and Cpk indices to measure process capability.

        Args:
            data: Process data.
            usl: Upper Specification Limit.
            lsl: Lower Specification Limit.

        Returns:
            Dictionary containing Cp and Cpk values.
        """
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        
        cp = (usl - lsl) / (6 * std_dev)
        cpu = (usl - mean) / (3 * std_dev)
        cpl = (mean - lsl) / (3 * std_dev)
        cpk = min(cpu, cpl)
        
        metrics = {
            "cp": float(cp),
            "cpk": float(cpk),
            "mean": float(mean),
            "std_dev": float(std_dev)
        }
        
        logger.info(f"Process Capability - Cp: {cp:.4f}, Cpk: {cpk:.4f}")
        return metrics

def get_spc_report(data: List[float], usl: Optional[float] = None, lsl: Optional[float] = None) -> Dict:
    """
    Utility function to generate a comprehensive SPC report.
    """
    series = pd.Series(data)
    analyzer = SPCAnalyzer()
    limits = analyzer.calculate_xbar_limits(series)
    anomalies = analyzer.detect_anomalies(series, limits)
    
    report = {
        "summary": {
            "mean": limits.mean,
            "ucl": limits.ucl,
            "lcl": limits.lcl,
            "anomaly_count": int(anomalies.sum())
        },
        "anomalies": anomalies.tolist()
    }
    
    if usl is not None and lsl is not None:
        report["capability"] = analyzer.calculate_process_capability(series, usl, lsl)
        
    return report
