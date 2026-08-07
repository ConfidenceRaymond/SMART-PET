"""Historical-versus-modern SMART-PET conformance utilities."""

from .architecture import architecture_report
from .gradients import generator_gradient_attribution
from .legacy_reference import attention_comparison_report

__all__ = [
    "architecture_report",
    "attention_comparison_report",
    "generator_gradient_attribution",
]
