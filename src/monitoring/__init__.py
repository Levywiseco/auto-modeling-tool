# -*- coding: utf-8 -*-
"""Feature and model monitoring: drift, stability, and alerting.

Fit binning rules on a benchmark slice, bucket every period with the same
rules, then track PSI / missing rate / bad rate / score trends over time.
"""

from .alerting import AlertConfig, generate_monitoring_alert
from .monitor import Monitor

__all__ = ["Monitor", "AlertConfig", "generate_monitoring_alert"]
