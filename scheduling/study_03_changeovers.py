# This script benchmark Changeover as
# Conditional Task Duration V.S. Independent Events

from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

