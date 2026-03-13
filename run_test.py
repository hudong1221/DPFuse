import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools import test
test.run_batch_fusion()

