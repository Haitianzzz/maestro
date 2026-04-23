import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import double


def test_double_of_three():
    assert double(3) == 6


def test_double_of_ten():
    assert double(10) == 20
