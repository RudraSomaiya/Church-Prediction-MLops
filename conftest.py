"""
conftest.py
-----------
Pytest configuration. Adds the project root to sys.path so that
`from src.data.preprocess import ...` works without installing the package.
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
