"""
Codes borrowed from Open Catalyst Project (OCP) https://github.com/Open-Catalyst-Project/ocp (MIT license)
"""

__all__ = ["TrainTask", "PredictTask", "ValidateTask", "RelxationTask"]

from .task import PredictTask, RelxationTask, TrainTask, ValidateTask
