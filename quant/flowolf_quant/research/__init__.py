"""Research loop: propose variants, record every trial, let the gate decide."""

from .loop import ResearchReport, research
from .proposers import GridProposer, LLMProposer, RandomProposer, TrialSummary
from .strategies import FAMILIES, Family, Param

__all__ = ["research", "ResearchReport", "GridProposer", "RandomProposer", "LLMProposer",
           "TrialSummary", "FAMILIES", "Family", "Param"]
