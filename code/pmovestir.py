from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import datetime
import scipy.interpolate as interp
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RectBivariateSpline

def interpolate_all_trajectories(step, dat):
    """
    Linearly interpolate all of the deer trajectories to the step scale.

    Parameters
    ----------
    step : int
        Step size on which to linear interpolate pig trajectories
    dat : DataFrame
        All pig data 

    Returns
    -------
    : dict
        Keys are host IDs, values are interpolated movement trajectories
        to the "step" scale.  
    """

    unq_collar = np.unique(dat.individual_ID)

    # Ensure all pigs are aligned when interpolating
    interp_vals = np.arange(dat.unix_time.min(), dat.unix_time.max() + step, step=step)

    all_fitted = {}
    for unq_ind in unq_collar:

        trial_dat = dat[dat.individual_ID == unq_ind]

        # Remove any of the same datetimes
        trial_dat = trial_dat[~trial_dat.datetime.duplicated()].sort_values("datetime").reset_index(drop=True)

        min_time = trial_dat.unix_time.min()
        time = trial_dat.unix_time.values
        xloc = trial_dat.UTMx.values
        yloc = trial_dat.UTMy.values

        fitted = fit_interp_to_movement(time, xloc, yloc, interp_vals=interp_vals)
        all_fitted[unq_ind] = fitted

    return(all_fitted)

def randomize_day(host):
    """
    Randomize locations by day following Speigel et al. 2016

    Parameters
    ----------
    host : DataFrame
        Has columns time
    """
    
    host = host.assign(datetime=lambda x: pd.to_datetime(x.time*60*10**9))
    host = host.assign(month_day = lambda x: x.datetime.dt.month.astype(str) + "_" + x.datetime.dt.day.astype(str))

    unique_days = host.month_day.unique()
    unique_days_rand = unique_days.copy()
    np.random.shuffle(unique_days_rand)
    day_map = pd.DataFrame({'month_day': unique_days, 'month_day_rand': unique_days_rand})
    host = (host.set_index("month_day")
          .join(day_map.set_index("month_day"))
          .reset_index()
          .assign(rand_datetime=lambda x: [datetime.datetime(year, month, day, hour, minute, second)
                                              for year, month, day, hour, minute, second,
                                             in zip(x.datetime.dt.year, 
                                              [int(x[0]) for x in x.month_day_rand.str.split("_")],
                                              [int(x[1]) for x in x.month_day_rand.str.split("_")],
                                               x.datetime.dt.hour, 
                                               x.datetime.dt.minute,
                                               x.datetime.dt.second)])
          .sort_values("rand_datetime")[['x', 'y', 'rand_datetime']]
          .assign(time= lambda x: x.rand_datetime.astype(np.int64) / (60*10**9))[['x', 'y', 'time']])
    return(host)



def align_trajectories(host1, host2):
    """
    Align host movement trajectories to the same time window

    Parameters
    ----------
    host1 : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location
    host2 : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location

    Returns
    -------
    : tuple
        (host1, host2), aligned (truncated) host trajectories
    """

     # Align time stamps. We are only comparing hosts where they overlap in time
    mintime = np.max([np.min(host1.time), np.min(host2.time)])
    maxtime = np.min([np.max(host1.time), np.max(host2.time)])
    host1 = host1[(host1.time >= mintime) & (host1.time <= maxtime)].reset_index(drop=False)
    host2 = host2[(host2.time >= mintime) & (host2.time <= maxtime)].reset_index(drop=False)
    return((host1, host2))


def matrix_correlation(host1, host2, flatX, flatY):
    """
    Compute the correlation surface between host1 and host2 occurences at all
    locations given by the flatX and flatY pairs

    Parameters
    ----------
    host1 : DataFrame 
        With columns x and y that indicate locations
    host2 : DataFrame
        With columns x and y that indicate locations
    flatX : array-like
        All x points to evaluation correlation
    flatY : array-like
        
    """
    
    host1_xlocs = host1.x.values
    host1_ylocs = host1.y.values

    host2_xlocs = host2.x.values
    host2_ylocs = host2.y.values
    
    flatX_lower, flatX_upper = flatX
    flatY_lower, flatY_upper = flatY
    
    # Check host 1 and host 2 in cells
    inX1 = np.bitwise_and((flatX_lower < host1_xlocs[:, np.newaxis]), (flatX_upper >= host1_xlocs[:, np.newaxis]))
    inY1 = np.bitwise_and((flatY_lower < host1_ylocs[:, np.newaxis]), (flatY_upper >= host1_ylocs[:, np.newaxis]))
    incell1 = (inX1 * inY1).astype(np.int64)

    inX2 = np.bitwise_and((flatX_lower < host2_xlocs[:, np.newaxis]), (flatX_upper >= host2_xlocs[:, np.newaxis]))
    inY2 = np.bitwise_and((flatY_lower < host2_ylocs[:, np.newaxis]), (flatY_upper >= host2_ylocs[:, np.newaxis]))
    incell2 = (inX2 * inY2).astype(np.int64)
    
    # Compute the vectorized correlation
    sd1 = np.std(incell1, axis=0)
    sd2 = np.std(incell2, axis=0)
    mean1 = np.mean(incell1, axis=0)
    mean2 = np.mean(incell2, axis=0)
    mean12 = np.mean(incell1 * incell2, axis=0)
    cor12 = (mean12 - mean1*mean2) / (sd1 * sd2)
    #cor12[np.isnan(cor12)] = 0
    return(cor12)


def fit_interp_to_movement(time, xloc, yloc, interp_vals=None, step=None):
    """
    Fit a simple linear interpolator to movement trajectory. Interpolates x and y dimensions separately
    
    Parameters
    -----------
    time : array-like
        Time at which location was recorded. They do not need to be equally spaced
    xloc : array-like
        x-location at time 
    yloc : array-like
        y-location at time
    interp_vals : None or array
        If None, then the interpolated time is specified by the min and max of time and step
        If not None, interp_vals is an array with prespecified times to be interpolated. The range
        of interp_vals should be larger or equal to the range of time. The interpolated times are then
        selected as a subset or interp_vals
    step : None or float
        It interp_vals is None, the time step of the interpolation
    
    Returns
    -------
    : DataFrame of interpolated movements
        x, y, and time
    """

    
    if interp_vals is not None:
        time_pred = interp_vals[np.argmax(interp_vals >= np.min(time)):np.argmax(interp_vals >= np.max(time))]
    else:
        
        if step is not None:
            time_pred = np.arange(np.min(time), np.max(time), step=step)
        else:
            raise KeyError("Step value must be not None")
            
    interp_dat = np.empty((len(time_pred), 3))
    interp_dat[:, -1] = time_pred
    
    for i, loc_dat in enumerate([xloc, yloc]):
        
        loc_z = (loc_dat - loc_dat.mean()) / (loc_dat.std())
        interp_mod = interp.interp1d(time, loc_z)
        loc_mean = interp_mod(time_pred)
        
        # Unstandarize
        interp_dat[:, i] = loc_mean*loc_dat.std() + loc_dat.mean()
    
    return(pd.DataFrame(interp_dat, columns=['x', 'y', 'time']))


def distance(p1, p2):
    """
    Distance between two points. p1 and p2 are (x, y) pairs
    """
    return(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


