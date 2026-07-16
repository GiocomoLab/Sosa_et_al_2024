---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Run dayData module to create multiDayData for analysis

multiDayData is a dictionary where each entry holds the dayData class for a single day,  \
where each dayData class runs calculations such as finding circular distances between  \
reward-relative spatial firing peaks and comparing to a shuffle, for each animal.  \
Most attributes of dayData have an entry for each animal.

Requires `multi_anim_sess` to already be saved for each day, which is a dictionary containing  \
the sess data, dF/F, deconvolved events, and place cell booleans for each animal. 


```python tags=[]
%load_ext autoreload
%autoreload 2

import os
import pickle
import dill
import numpy as np
import warnings
from datetime import datetime

from reward_relative import utilities as ut
from reward_relative import dayData as dd
 
save_figures = False
```

```python
from reward_relative.path_dict_firebird import path_dictionary as path_dict
path_dict
```

# Create multiDayData class for each experiment day

```python tags=[]
## Specify parameters (these are already defaults in dayData class)
bin_size = 10  # for quantifying distribution of place field peak locations
sigma = 1  # for smoothing
smooth = False  # whether to smooth for finding place cell peaks
exclude_int = True  # exclude putative interneurons
int_thresh = 0.5
impute_NaNs = True # whether to impute (interpolate) bins that are NaN in spatially-binned data

## Place cell definitions:
## 'and' = must have significant spatial information 
##        in trial set 0 AND trial set 1 (i.e. before and after the reward switch)
## 'or' = must have signitive spatial information in trial set 0 OR trial set 1
place_cell_logical = 'or' 
ts_key = 'dff' # which timeseries to use for finding peaks
use_speed_thr = True # use a speed threshold to calculate new trial matrices
speed_thr = 2 # speed threshold in cm/s (excludes data at speed less than this)

reward_dist_inclusive = 50 #in cm
reward_dist_exclusive = 50 #in cm
reward_overrep_dist = 50 #in cm

experiment = 'MetaLearn'
year = 'combined'

if experiment == 'MetaLearn':
    # exp_days = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # all days
    exp_days = [3, 5, 7, 8, 10, 12, 14] # switch days
    # exp_days = [1,2,4,6,9,11,13] # "stay" days


# create a tag to label the filename with params
tag = ''
if smooth:
    tag = ('smoothed_sig%d' % sigma)
else:
    tag = 'unsmoothed'

if exclude_int:
    tag = tag + ('_excInt%.1f' % int_thresh)

tag = tag + ('_inc%d' % reward_dist_inclusive)

if use_speed_thr:
    tag = tag + '_useSpeed'

# For loading individual day pickles
day_params={'speed': str(speed_thr),
          'nperms': 100, # shuffles for defining place cells
          'baseline_method': 'maximin', # dF/F method
          'ts_key': 'events' # timeseries used for identifying place cells
          }

multiDayData = dict()

add_all_computations = False


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    for d_i, exp_day in enumerate(exp_days):

        anim_list = dd.define_anim_list(experiment, exp_day, year=year)

        print(anim_list)

        multi_anim_sess = dd.load_multi_anim_sess(path_dict, exp_day, anim_list,
                                                  params=day_params
                                                  )

        # initialize class with basic info
        multiDayData[exp_day] = dd.dayData(anim_list,
                                           multi_anim_sess,
                                           exp_day=exp_day,
                                           experiment=experiment,
                                           # timeseries to use
                                           ts_key=ts_key,  # to use for finding place cell peaks
                                           force_two_sets=True,  # of trials
                                           use_speed_thr=use_speed_thr,
                                           speed_thr=speed_thr,
                                           exclude_int=exclude_int,
                                           int_thresh=int_thresh,
                                           int_method='speed',
                                           reward_dist_exclusive=reward_dist_inclusive,
                                           reward_dist_inclusive=reward_dist_exclusive,
                                           reward_overrep_dist=reward_overrep_dist,
                                           )

        # add things to the class that are computationally intensive/time-consuming
        if add_all_computations:
            multiDayData[exp_day].add_all_the_things(anim_list, 
                                                    multi_anim_sess,
                                                    add_behavior=True,
                                                    add_cell_classes=True,
                                                    add_circ_relative_peaks=True,
                                                    add_field_dict=True,
                                                    bin_size=bin_size,  # for quantifying distribution of place field peak locations
                                                    sigma=sigma,  # for smoothing
                                                    smooth=smooth,  # whether to smooth for finding place cell peaks
                                                    # (activity will be auto smoothed for everything else)
                                                    impute_NaNs=True,
                                                    place_cell_logical=place_cell_logical,
                                                    ts_key=ts_key,
                                                    lick_correction_thr=0.35,
                                                    )

        %reset_selective -f multi_anim_sess
```

```python
# print attributes of dayData class for day 3
multiDayData[3].__dict__.keys()
```

```python tags=[]
max_anim_list = sorted(np.unique(np.concatenate([multiDayData[day].anim_list
                                                     for day in exp_days])), 
                           key=len)
max_anim_list
```

```python
multiDayData.keys()
```

```python
include_ans = multiDayData[exp_days[-1]].circ_rel_stats_across_an['include_ans']
include_ans
```

## Save multiDayData as pickle

```python
from datetime import datetime

pkl_name = "%s_expdays%s_multiDayData_%s_%s_%s.pickle" % (ut.make_anim_tag(max_anim_list),
                                                          ut.make_day_tag(
                                                              exp_days),
                                                          ts_key,
                                                          tag,
                                                          datetime.now().strftime("%Y%m%d-%H%M"))
print(pkl_name)
file_dir = os.path.join(path_dict['preprocessed_root'], 'multiDayData')
ut.write_sess_pickle(multiDayData, file_dir, pkl_name, overwrite=False)
```

# Load previously saved multiDayData and resave using this repo

```python
def max_anim_list_tmp(experiment, exp_days):
    
    def define_anim_list(experiment, exp_day):
        if exp_day in [1, 2]:
            an_list = ['GCAMP2', 'GCAMP3', 'GCAMP4', 'GCAMP5',
                       'GCAMP6', 'GCAMP7',
                       'GCAMP10', 'GCAMP12', 'GCAMP13', 'GCAMP14',
                       'GCAMP15', 'GCAMP17', 'GCAMP18', 'GCAMP19']
        elif exp_day == 3:
            an_list = ['GCAMP2', 'GCAMP3', 'GCAMP4', 'GCAMP5',
                       'GCAMP6', 'GCAMP7',
                       'GCAMP10', 'GCAMP11', 'GCAMP12', 'GCAMP13', 'GCAMP14',
                       'GCAMP15', 'GCAMP17', 'GCAMP18', 'GCAMP19']

        elif (exp_day > 3) and (exp_day <= 14):
            an_list = ['GCAMP2', 'GCAMP3', 'GCAMP4', 'GCAMP5','GCAMP6', 'GCAMP7',
                       'GCAMP10', 'GCAMP11', 'GCAMP12', 'GCAMP13', 'GCAMP14',
                       'GCAMP15', 'GCAMP17', 'GCAMP18', 'GCAMP19']
        elif exp_day == 15:
            an_list = ['GCAMP10', 'GCAMP11', 'GCAMP12', 'GCAMP13', 'GCAMP14',
                       'GCAMP15', 'GCAMP17', 'GCAMP18', 'GCAMP19']
        else:
            raise NotImplementedError(
                "Animal list not defined for this day")
            
        return np.asarray(an_list)
        
    return sorted(np.unique(np.concatenate([define_anim_list(experiment,
                                                             day,
                                                            )
                                            for day in exp_days])),
                  key=len)
    
```

```python
experiment = 'MetaLearn'
year = 'combined'
if experiment == 'MetaLearn':
    exp_days = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # exp_days = [3, 5, 7, 8, 10, 12, 14]

# max_anim_list = dd.max_anim_list(experiment,exp_days, year=year)
max_anim_list = max_anim_list_tmp(experiment,exp_days)

ts_key = 'dff' # used to find place field peaks
exclude_end_cells = False
exclude_reward_cells = False
exclude_track_cells = False
smooth = False  # whether to smooth for finding place cell peaks
sigma=1
exclude_int = True  # exclude putative interneurons
int_thresh = 0.5
activity_criterion = False

place_cell_logical = 'or'

reward_dist_inclusive = 50
reward_dist_exclusive = 50

use_speed_thr = True

circ_tag = ''
if smooth:
    circ_tag = ('smoothed_sig%d' % sigma)
else:
    circ_tag = 'unsmoothed'

if exclude_int:
    circ_tag = circ_tag + ('_excInt%.1f' % int_thresh)
if exclude_reward_cells:
    circ_tag = circ_tag + ('_excRew%d' % reward_dist_exclusive)
if exclude_track_cells:
    circ_tag = circ_tag + ('_excStable%d' % reward_dist_exclusive)
if exclude_end_cells:
    circ_tag = circ_tag + '_excEnd'

circ_tag = circ_tag + ('_inc%d' % reward_dist_inclusive)
if activity_criterion:
    circ_tag = circ_tag + '_actCrit'
if use_speed_thr:
    circ_tag = circ_tag + '_useSpeed'
    
dt = "202504" #"20240821-1205" #"202503" #"20240530-1141"

## combined cohorts: 20240530-1141

pkl_name = "m2-19_expdays1-2-3-4-5-6-7-8-9-10-11-12-13-14_multiDayData_%s_%s.pickle" % (
                                                   
                                                       ts_key,
                                                  
                                                         # )
                                                      dt)
# pkl_name = "%s_expdays%s_multiDayData_%s_%s_%s.pickle" % (ut.make_anim_tag(max_anim_list),
#                                                     ut.make_day_tag(exp_days),
#                                                        ts_key,
#                                                     circ_tag,
#                                                          # )
#                                                       dt)
pkl_path = os.path.join(path_dict['preprocessed_root'],'multiDayData',pkl_name)
print(pkl_path)
multiDayData_old = dill.load(open(pkl_path,"rb"))
```

```python
import pickletools

with open(pkl_path, "rb") as f:
    data = f.read() 

pickletools.dis(data)
```

```python
state = multiDayData_old.to_dict()  # only built-ins

#pickle.dump(state, f)
```

```python
type(multiDayData_new[day])
```

```python
## Convert each class to a dict for pickling
multiDayData_new = multiDayData_old.copy()

for day in exp_days:
    multiDayData_new[day] = multiDayData_new[day].to_dict()
    
```

```python
## Rename outdated field names
# multiDayData_new = dict()

for day in exp_days:

#     for attr in multiDayData_new[day].__dict__.keys():
#         if attr in multiDayData[day].__dict__.keys():
#             if attr != 'pos_bin_centers':
#                 if type(getattr(multiDayData_new[day], attr)) is dict:

#                     for key, value in getattr(multiDayData[day], attr).items():
#                         getattr(multiDayData_new[day], attr).update({key: value})
#                 else:
#                     setattr(multiDayData_new[day], attr, getattr(multiDayData[day], attr))
#         else:
#             print(attr, "does not match")

    multiDayData_new[day].anim_list = multiDayData_old[day].anim_list
    multiDayData_new[day].experiment = multiDayData_old[day].experiment 
    multiDayData_new[day].place_cell_logical = multiDayData_old[day].place_cell_logical
    multiDayData_new[day].force_two_sets = multiDayData_old[day].force_two_sets
    multiDayData_new[day].ts_key = multiDayData_old[day].ts_key
    multiDayData_new[day].use_speed_thr = multiDayData_old[day].use_speed_thr
    multiDayData_new[day].speed_thr = multiDayData_old[day].speed_thr
    multiDayData_new[day].exclude_int = multiDayData_old[day].exclude_int
    multiDayData_new[day].int_thresh = multiDayData_old[day].int_thresh
    multiDayData_new[day].int_method = multiDayData_old[day].int_method
    multiDayData_new[day].reward_dist_exclusive = multiDayData_old[day].reward_dist_exclusive
    multiDayData_new[day].reward_dist_inclusive = multiDayData_old[day].reward_dist_inclusive
    multiDayData_new[day].reward_overrep_dist = multiDayData_old[day].reward_cell_dist #multiDayData_old[day].reward_overrep_dist
    multiDayData_new[day].activity_criterion = multiDayData_old[day].activity_criterion
    multiDayData_new[day].bin_size = multiDayData_old[day].bin_size
    multiDayData_new[day].sigma = multiDayData_old[day].sigma
    multiDayData_new[day].smooth = multiDayData_old[day].smooth
    multiDayData_new[day].impute_NaNs = multiDayData_old[day].impute_NaNs
    multiDayData_new[day].sim_method = multiDayData_old[day].sim_method
    multiDayData_new[day].lick_correction_thr = multiDayData_old[day].lick_correction_thr
    multiDayData_new[day].exp_day = multiDayData_old[day].exp_day
    multiDayData_new[day].is_switch = multiDayData_old[day].is_switch
    multiDayData_new[day].anim_tag = multiDayData_old[day].anim_tag
    multiDayData_new[day].trial_dict = multiDayData_old[day].trial_dict
    multiDayData_new[day].rzone_pos = multiDayData_old[day].rzone_pos
    multiDayData_new[day].rzone_by_trial = multiDayData_old[day].rzone_by_trial
    multiDayData_new[day].rzone_label = multiDayData_old[day].rzone_label
    multiDayData_new[day].blocks = multiDayData_old[day].blocks
    multiDayData_new[day].activity_matrix = multiDayData_old[day].activity_matrix
    multiDayData_new[day].events = multiDayData_old[day].events
    multiDayData_new[day].place_cell_masks = multiDayData_old[day].place_cell_masks
    multiDayData_new[day].SI = multiDayData_old[day].SI
    multiDayData_new[day].overall_place_cell_masks = multiDayData_old[day].overall_place_cell_masks
    multiDayData_new[day].peaks = multiDayData_old[day].peaks
    multiDayData_new[day].field_dict = multiDayData_old[day].field_dict
    multiDayData_new[day].plane_per_cell = multiDayData_old[day].plane_per_cell
    multiDayData_new[day].is_int = multiDayData_old[day].is_int
    multiDayData_new[day].is_reward_cell = multiDayData_old[day].is_reward_cell
    multiDayData_new[day].is_end_cell = multiDayData_old[day].is_end_cell
    multiDayData_new[day].is_track_cell = multiDayData_old[day].is_stable_cell #multiDayData_old[day].is_track_cell
    multiDayData_new[day].pc_distr = multiDayData_old[day].pc_distr
    multiDayData_new[day].rew_frac = multiDayData_old[day].rew_frac
    multiDayData_new[day].rate_map = multiDayData_old[day].rate_map
    multiDayData_new[day].pv_sim_mean = multiDayData_old[day].pv_sim_mean
    multiDayData_new[day].sim_to_set0 = multiDayData_old[day].sim_to_set0
    multiDayData_new[day].sim_mat = multiDayData_old[day].sim_mat
    multiDayData_new[day].curr_zone_lickrate = multiDayData_old[day].curr_zone_lickrate
    multiDayData_new[day].other_zone_lickrate = multiDayData_old[day].other_zone_lickrate
    multiDayData_new[day].curr_vs_other_lickratio = multiDayData_old[day].curr_vs_other_lickratio
    multiDayData_new[day].in_vs_out_lickratio = multiDayData_old[day].in_vs_out_lickratio
    multiDayData_new[day].lickpos_std = multiDayData_old[day].lickpos_std
    multiDayData_new[day].lickpos_com = multiDayData_old[day].lickpos_com
    multiDayData_new[day].lick_mat = multiDayData_old[day].lick_mat
    multiDayData_new[day].def_block_by = multiDayData_old[day].def_block_by
    multiDayData_new[day].cell_class = multiDayData_old[day].cell_class
    multiDayData_new[day].pos_bin_centers = multiDayData_old[day].pos_bin_centers
    multiDayData_new[day].dist_btwn_rel_null = multiDayData_old[day].dist_btwn_rel_null
    multiDayData_new[day].dist_btwn_rel_peaks = multiDayData_old[day].dist_btwn_rel_peaks
    multiDayData_new[day].reward_rel_cell_ids = multiDayData_old[day].reward_rel_cell_ids
    multiDayData_new[day].xcorr_above_shuf = multiDayData_old[day].xcorr_above_shuf
    multiDayData_new[day].reward_rel_dist_along_unity = multiDayData_old[day].reward_rel_dist_along_unity
    multiDayData_new[day].rel_peaks = multiDayData_old[day].rel_peaks
    multiDayData_new[day].rel_null = multiDayData_old[day].rel_null
    multiDayData_new[day].circ_licks = multiDayData_old[day].circ_licks
    multiDayData_new[day].circ_speed = multiDayData_old[day].circ_speed
    multiDayData_new[day].circ_map = multiDayData_old[day].circ_map
    multiDayData_new[day].circ_trial_matrix = multiDayData_old[day].circ_trial_matrix
    multiDayData_new[day].circ_rel_stats_across_an = multiDayData_old[day].circ_rel_stats_across_an

```

```python
multiDayData_new[3].cell_class['GCAMP3']['masks']
```

```python
# only if we need to convert "stable" to "track"

for day in exp_days:
    for an in multiDayData_new[day].anim_list:
        multiDayData_new[day].cell_class[an]['masks']['track'] = multiDayData_new[day].cell_class[an]['masks'].pop('stable')
        multiDayData_new[day].cell_class[an]['fractions_placeor']['track'] = multiDayData_new[day].cell_class[an]['fractions_placeor'].pop('stable')
        multiDayData_new[day].cell_class[an]['fractions_total']['track'] = multiDayData_new[day].cell_class[an]['fractions_total'].pop('stable')

```

```python
multiDayData_new[day].cell_class[an]['masks']
```

```python
from datetime import datetime

# pkl_name = "%s_expdays%s_multiDayData_%s_%s_%s.pickle" % (ut.make_anim_tag(max_anim_list),
#                                                     ut.make_day_tag(exp_days),
#                                                        ts_key,
#                                                     circ_tag,
#                                                       "202504")

pkl_name = "%s_expdays%s_multiDayData_%s_%s.pickle" % ('m2-19', #ut.make_anim_tag(max_anim_list),
                                                    ut.make_day_tag(exp_days),
                                                       ts_key,
                                                    
                                                      "202504")
print(pkl_name)
file_dir = os.path.join(path_dict['preprocessed_root'],'multiDayData')
ut.write_sess_pickle(multiDayData_new, file_dir, pkl_name, overwrite=True)
```

```python
!pip uninstall -y InVivoDA_analyses
```

```python
pkl_name = "%s_expdays%s_multiDayData_%s_%s_%s.pickle" % (ut.make_anim_tag(max_anim_list),
                                                    ut.make_day_tag(exp_days),
                                                       ts_key,
                                                    circ_tag,
                                                         # )
                                                      dt)
pkl_path = os.path.join(path_dict['preprocessed_root'],'multiDayData',pkl_name)
print(pkl_path)
multiDayData_old = dill.load(open(pkl_path,"rb"))
```
