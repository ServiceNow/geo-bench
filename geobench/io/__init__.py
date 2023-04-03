"""Geo-Bench io."""

from geobench.io.dataset import *
from geobench.io.label import *
from geobench.io.task import *

import geobench.io.task as task
import geobench.io.dataset as dataset
import geobench.io.label as label

import sys
sys.modules["ccb.io.task"] = task
sys.modules["ccb.io.dataset"] = dataset
sys.modules["ccb.io.label"] = label