#!/usr/bin/env python3

from pathlib import Path
import pytest


def start():
    pytest.main([str(Path(__file__).parent)])
