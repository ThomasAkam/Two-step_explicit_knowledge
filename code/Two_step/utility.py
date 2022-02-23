import pickle
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.utils import resample

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest floating point value.

def log_safe(x):
    '''Return log of x protected against giving -inf for very small values of x.'''
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

def exp_mov_ave(data, tau = 8., initValue = 0., alpha = None):
    '''Exponential Moving average for 1d data.  The decay of the exponential can 
    either be specified with a time constant tau or a learning rate alpha.'''
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x 
    return mov_ave[1:]

def resample_subjects(sessions):
    '''Generate a new list of sessions by reampling subjects with replacement.'''
    subjects = set([s.subject_ID for s in sessions])
    resampled_sessions = []
    for subject in resample(list(subjects)):
        resampled_sessions += [s for s in sessions if s.subject_ID == subject]
    return resampled_sessions

def nans(shape, dtype=float):
    '''return array of nans of specified shape.'''
    return np.full(shape, np.nan)

def nansem(x,dim = 0, ddof = 1):
    '''Standard error of the mean ignoring nans along dimension dim.'''
    return np.sqrt(np.nanvar(x,dim)/(np.sum(~np.isnan(x),dim) - ddof))

def nanGaussfilt1D(x, sigma, axis=-1):
    '''1D Gaussian filter which ignores NaNs.'''
    v = x.copy()
    w = np.ones(x.shape)
    v[np.isnan(x)]=0
    w[np.isnan(x)]=0
    vf = gaussian_filter1d(v,sigma,axis)
    wf = gaussian_filter1d(w,sigma,axis)
    return vf/wf

def save_item(item, file_name):
    '''Save an item using pickle.'''
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(item, f)

def load_item(file_name):
    '''Unpickle and return specified item.'''
    with open(file_name+'.pkl', 'rb') as f:
        return pickle.load(f)