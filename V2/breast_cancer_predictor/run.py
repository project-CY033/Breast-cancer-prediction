#!/usr/bin/env python3
"""
Breast Cancer Mutation Prediction Tool
Main entry point for the application.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.gui.main_app import main

if __name__ == "__main__":
    main()
