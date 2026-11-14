"""
Agent 工具集合

"""
from src.tools.search import make_search_law
from src.tools.calc import make_calc_severance
from src.tools.check import make_check_contract

__all__ = [
  "make_search_law",
  "make_calc_severance",
  "make_check_contract",
]