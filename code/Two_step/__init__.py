import pylab as plt

from . import parallel_processing as pp
from . import utility as ut
from . import RL_agents as rl 
from . import plotting as pl 
from . import model_plotting as mp
from . import model_fitting as mf
from . import group_comparison as gc
from . import logistic_regression as lr 
from . import model_comparison as mc 
from . import human_experiment as he 
from . import human_session as hs
from . import simulation as sm

plt.ion()
plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)