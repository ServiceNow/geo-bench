"""geobench package."""
__version__ = "0.0.3"


from geobench.dataset import *
from geobench.label import *
from geobench.task import *
from geobench.config import *
from geobench import config

# for backward compatibility of pickled objects
import geobench.task as task
import geobench.dataset as dataset
import geobench.label as label
import geobench

import sys

sys.modules["geobench.io.task"] = task
sys.modules["geobench.io.dataset"] = dataset
sys.modules["geobench.io.label"] = label
sys.modules["geobench.io"] = geobench
sys.modules["ccb.io.task"] = task
sys.modules["ccb.io.dataset"] = dataset
sys.modules["ccb.io.label"] = label
sys.modules["ccb.io"] = geobench
sys.modules["ccb"] = geobench
