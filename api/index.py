import sys
import os

# Add backend to path so imports work as expected
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.src.main import app
