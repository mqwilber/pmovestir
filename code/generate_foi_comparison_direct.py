import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import importlib
import pmovestir as pmstir
# import seaborn as sns
import scipy.stats as stats
import itertools
import multiprocessing as mp
import logging
import pmovestir as pmstir
import importlib
import glob
import os
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
import shapely.geometry as geometry
from shapely import intersection
importlib.reload(pmstir)

logging.basicConfig(filename='process_correlation.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Initializing logger")


"""
Script computes the contribution of correlation to FOI for each pair of individuals
across biological seasons and time. 

"""

def trim_movement_data(dat):
    """
    Trim the movement data by the first 15 days after capture and remove the
    last two days
    """

    # Trim the movement 
    host_ids = dat.animal_id.unique()

    sorted_hosts = []
    for hid in host_ids:
        
        # Remove approximately the first 15 days and the last 2 days
        tdat = (dat.query("animal_id == {0}".format(hid))
                        .sort_values(by="t_").iloc[48*15:-48*2, :]
                        .assign(datetime = lambda x: pd.to_datetime(x.t_))
                        .assign(year = lambda x: x.datetime.dt.year)
                        .assign(unix_time = lambda x: x.datetime.astype(np.int64) / (60 * 10**9)) # Convert nanoseconds to mintues
                        .rename(columns={'x_': 'UTMx', 'y_': 'UTMy', 
                                         'animal_id': "individual_ID"}))
        
        sorted_hosts.append(tdat)

    # Removing the first 15 days removes 13 animals
    trun_dat = pd.concat(sorted_hosts)
    trun_dat.individual_ID.unique().shape
    return(trun_dat)


def get_combos_for_correlation(trun_dat, minutes=30):
    """
    Get the host combinations needed to compute the correlations
    """

    # Get combinations that we can loop through
    all_host_ids = trun_dat.individual_ID.unique()
    combos = list(itertools.combinations(all_host_ids, 2))

    deer_dat = trun_dat[['individual_ID', 'datetime', 'unix_time', 'UTMx', 'UTMy', 'year']]
    trajs = pmstir.interpolate_all_trajectories(minutes, deer_dat)

    # Restrict to combinations where we have some temporal and spatial
    # overlap
    keep_combos = []
    for combo in combos:

        h1 = trajs[combo[0]]
        h2 = trajs[combo[1]]

        # Check spatial and temporal overlap
        values = ['time', 'x', 'y']

        overlap_bools = []
        for val in values:
            min1 = h1[val].min()
            max1 = h1[val].max()
            min2 = h2[val].min()
            max2 = h2[val].max()
            no_overlap = (min1 > max2) or (min2 > max1)
            overlap_bools.append(not no_overlap)

        if np.all(overlap_bools):
            keep_combos.append(combo)

    return((keep_combos, trajs))


def get_correlation_surface(host1, host2, X, Y, grid_size, num_per_split=150, shifts=5):
    """
    Pass in aligned trajectories and an X, Y girdded surface and get the
    correlation surface

    Parameters
    ----------
    host1 : dataframe
    host2 : dataframe
    grid_size : float
        The grid size of the landscape
    num_per_split : int
        Default is 150.  Do this in splits to avoid memory issues
    shifts : int
        Number of shifts of the grid cells to calculate the split. Particularly,
        for smaller grid cells, helps eliminate boundary errors

    Returns
    -------
    : array-like
        Matrix corresponding to X, Y with the correlation values. nans indicate
        no shared space use so no correlation could be computed. 
    """
    
    # Depending on where you put the grid bounds, you can get slightly
    # different answer.  Compute this multiple times using a sliding
    # window and then average.
    shifts = np.linspace(0, 1, num=5)
    all_cor = np.empty((len(shifts), len(X.ravel())))
    
    for j, shift in enumerate(shifts):
        
        flatX_lower = X.ravel() - shift*grid_size
        flatX_upper = X.ravel() + (1 - shift)*grid_size
        flatY_lower = Y.ravel() - shift*grid_size
        flatY_upper = Y.ravel() + (1 - shift)*grid_size
    
        host1_xlocs = host1.x.values
        host1_ylocs = host1.y.values
    
        host2_xlocs = host2.x.values
        host2_ylocs = host2.y.values
    
        # Break this into chunks for memory issues
        inds = np.arange(len(flatX_lower))
    
        n = np.ceil(len(flatX_lower) / num_per_split)
        splits = np.array_split(inds, n)    
    
        cor_res = []
        pvalues = []
    
        # Compute across different splits
        for s in splits:
    
            flatX = (flatX_lower[s], flatX_upper[s])
            flatY = (flatY_lower[s], flatY_upper[s])
            tcor = pmstir.matrix_correlation(host1, host2, flatX, flatY)
            df = host1.shape[0] - 2
            tval = tcor / np.sqrt((1 - tcor**2) / df)
            pvalue = 2*(1 - stats.t.cdf(np.abs(tval), df))
            cor_res.append(tcor)
            pvalues.append(pvalue)
    
        cor12 = np.concatenate(cor_res)
        pvalue12 = np.concatenate(pvalues)
    
        # Use a bonferonni correction to remove apparent correlations
        p_correction = 1 #0.001 / len(pvalue12)
        no_sig_cor = pvalue12 > p_correction
        cor12[no_sig_cor] = 0
        all_cor[j, :] = cor12
    
    # Average over different shifts
    cor12 = np.nanmean(all_cor, axis=0)
    
    # Reformat as a matrix
    C12 = cor12.reshape(X.shape)
    
    return(C12)

def area_of_mcp_overlap(host1, host2):
    """
    Given two host trajectories, compute the area of overlap for the 100% MCP

    Parameters
    ----------
    host1 : DataFrame
        Contains x and y values for locations
    host 2 : DataFrame
        Contains x and y values for locations

    """

    polygons = []
    hosts = [host1, host2]
    
    for host in hosts:
    
        locs = host[['x', 'y']].values
        hull = ConvexHull(locs)
        x = locs[hull.vertices, 0]
        y = locs[hull.vertices, 1]    
        poly = geometry.Polygon(zip(x, y))
        polygons.append(poly)

    return(intersection(polygons[0], polygons[1]).area)


if __name__ == '__main__':

    ### 1.  Load and clean data

    ## NOTE: Given ongoing analyses with these data, we only provide the raw movement
    ## data for the four focal
    dat = pd.read_csv("../data/cleaned_movement_data_08292024.csv")
    dat = dat.query("fast_step_ == False and fast_roundtrip_ == False")
    trun_dat = trim_movement_data(dat)


    #### 2. Build combinations list with deer that have some spatial
    #### and temporal overlap
    keep_combos, trajs = get_combos_for_correlation(trun_dat, minutes=10)

    ### 3. Get seasonal information
    seasons_map = {1: "Rut", 2: "Gestation", 3: "Gestation", 4: "Gestation", 5: "Gestation",
                   6: "Fawning", 7: "Fawning", 8: "Lactation",
                   9: "Lactation", 10: "Lactation", 11: "Rut", 12: "Rut"}
    deer_trajs = (pd.concat(trajs)
                    .reset_index()
                    .drop(columns="level_1")
                    .rename(columns={"level_0": "individual_ID"})
                    .assign(datetime=lambda x: pd.to_datetime(x.time*60, unit="s"))
                    .assign(month=lambda x: x.datetime.dt.month, 
                            year=lambda x: x.datetime.dt.year)
                    .assign(season=lambda x: x.month.map(seasons_map))
                    .assign(year_season=lambda x: x.year.astype(np.str_) + "_" + x.season))

    # Adjust wrapping of the Rut season...this works for now...but does not generalize
    deer_trajs.loc[deer_trajs.year_season == "2024_Rut", "year_season"] = "2023_Rut"

    #### 4. Compute correlation surfaces

    # NOTE: This is the grid size to compute the correlation. Technically,
    # it should be on the same scale as a contact. However, simulation shows 
    # that you can do it a slightly larger scale and get a smoother surface, 
    # though you dampen the magnitude of per cell FOI. 
    grid_size = 40 # We scale this up to get a smoother surface
    interval = (5, 95) # Define the area where we will calculate contacts and correlation surfaces
    # randomize = False

    foi_results = []

    # Loop through pairs with overlap
    for nc, combo in enumerate(keep_combos[:]):

        print("{0} of {1}".format(nc + 1, len(keep_combos[:])))
        host1_dat = deer_trajs.query("individual_ID == {0}".format(combo[0]))
        host2_dat = deer_trajs.query("individual_ID == {0}".format(combo[1]))

        # Get year gestation combos that are shared
        unq_year_gestation_combos = np.intersect1d(host1_dat.year_season.unique(), 
                                                   host2_dat.year_season.unique())

        # Loop through unique season and year combinations, recognizing
        # that correlation can be very dynamic!
        for yg in unq_year_gestation_combos:

            d1 = host1_dat.query("year_season == '{0}'".format(yg))
            d2 = host2_dat.query("year_season == '{0}'".format(yg))

            # Align trajectories
            host1, host2 = pmstir.align_trajectories(d1.reset_index(drop=True), 
                                                     d2.reset_index(drop=True))

            # Check on spatial overlap
            try:
                
                area_overlap = area_of_mcp_overlap(host1, host2)

                # Compute direct contacts
                direct_distance = np.array([pmstir.distance(p1, p2) for p1, p2 in 
                                            zip(host1[['x', 'y']].values, 
                                                host2[['x', 'y']].values)])
                dcontact = (direct_distance < grid_size).astype(np.int64)
                contact_df = host1[['x', 'y']].assign(contact=dcontact).query("contact == True")
                num_contacts = np.sum(dcontact == 1)

                # Are there enough contacts to even do an analysis?
                if area_overlap > 0:

                    # Set the bounding box on which to calculate the correlation surface
                    host1_lowerx, host1_upperx = stats.scoreatpercentile(host1.x.values, interval)
                    host1_lowery, host1_uppery = stats.scoreatpercentile(host1.y.values, interval)
                    host2_lowerx, host2_upperx = stats.scoreatpercentile(host2.x.values, interval)
                    host2_lowery, host2_uppery = stats.scoreatpercentile(host2.y.values, interval)

                    xmin = np.min(np.r_[host1_lowerx, host2_lowerx])
                    xmax = np.max(np.r_[host1_upperx, host2_upperx])
                    ymin = np.min(np.r_[host1_lowery, host2_lowery])
                    ymax = np.max(np.r_[host1_uppery, host2_uppery])

                    xvals = np.arange(xmin, xmax + grid_size, step=grid_size)
                    yvals = np.arange(ymin, ymax + grid_size, step=grid_size)
                    X, Y = np.meshgrid(xvals, yvals)

                    # Compute the correlation surface
                    C12 = get_correlation_surface(host1, host2, X, Y, grid_size, num_per_split=150, shifts=5)
                
                    # Calculate UD overlap
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    
                    # UD 1
                    host1_locs = host1[['x', 'y']].values.T
                    kde1 = gaussian_kde(host1_locs[:, ::10])
                    Z1 = np.reshape(kde1(positions).T, X.shape)*grid_size*grid_size
                    
                    # UD 2
                    host2_locs = host2[['x', 'y']].values.T
                    kde2 = gaussian_kde(host2_locs[:, ::10])
                    Z2 = np.reshape(kde2(positions).T, X.shape)*grid_size*grid_size
                    
                    # Overlap metrics
                    spaceuse_component = Z1*Z2
                    bcoef_std = np.sum(np.sqrt((Z1 / np.sum(Z1)) * (Z2 / np.sum(Z2))))
                    sd_component = np.sqrt(Z1*(1 - Z1))*np.sqrt(Z2*(1 - Z2))
                    cor_component = C12
                    
                    # Get per-cell means
                    mask = np.copy(cor_component)
                    mask[~np.isnan(mask)] = 1
                    only_space_use_foi = spaceuse_component*mask
                    mean_space_use_foi = np.nanmean(only_space_use_foi)
                    mean_total_foi = np.nanmean(sd_component*cor_component + only_space_use_foi)
                    mean_cor_foi = np.nanmean(sd_component*cor_component)

                    # Save results
                    foi_results.append([str(combo), yg, bcoef_std, area_overlap, mean_cor_foi, 
                                        mean_space_use_foi, num_contacts, host1_locs.shape[1]])
            except:
                pass


    foi_results_df = pd.DataFrame(foi_results, columns=['pair', 'year_season', 
                                                        'BC', 'area_overlap_m2', 
                                                        'foi_cor', 'foi_space', 
                                                        'num_contacts', 'num_locs'])

    foi_results_df.to_csv("../results/cor_contributions_grid_size_{0}.csv".format(grid_size), index=False)



