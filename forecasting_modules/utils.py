import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy, json
from datetime import timedelta, datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


from typing import Callable

# def compute_timeseries_split_cutoffs(
#         full_index: pd.DatetimeIndex, # full time-sereis dataframe index columns
#         horizon: int, # number of hours (should be divisible by 24 hours)
#         folds: int, # number of splits to do
#         min_train_size: int, # minimum allowed train_idx
# ) -> list[pd.Timestamp]:
#     '''
#     The function takes a timesereis dataframe index column (full_index) and computes a list of cutoffs
#     such that each cutoff splits that train and test part of the dataframe as follows:
#     train_idx computed as full_index < cutoff
#     test_idx computed as so that (full_index >= cutoff) & (full_index < cutoff + horizon).
#
#     Important: train_idx and test_idx both should end at 23 hours (end of the day)
#     and start at 00.00 hours (start of the day). Length of the test_idx should always 'delta' number of hours,
#     where delta is divisible by 24 hours. Length of the train_idx should always be divisible by test_idx.
#     The function checks this and returns a list of cutoffs and a list of train and test indexes.
#
#
#     Args:
#         full_index: DatetimeIndex containing timestamps for the entire dataset
#         horizon: Forecast horizon in hours
#         folds: Number of folds for cross-validation
#         min_train_size: Minimum required training size in hours
#
#     Returns:
#         list[pd.Timestamp]: List of cutoff timestamps for the folds. (Last one is the latest one)
#
#     Raises:
#         ValueError: If index length isn't divisible by horizon or other validation failures
#     '''
#     # Input validation
#     if not isinstance(full_index, pd.DatetimeIndex):
#         raise ValueError("full_index must be a pandas DatetimeIndex")
#
#     if len(full_index) == 0:
#         raise ValueError("full_index cannot be empty")
#
#     # Check if index length is divisible by horizon
#     if len(full_index) % horizon != 0:
#         raise ValueError(f"Index length ({len(full_index)}) must be divisible by horizon ({horizon})")
#
#     if folds == 0:
#         return []
#
#     # Basic parameter validation
#     if horizon <= 0 or delta <= 0 or folds < 0 or min_train_size <= 0:
#         raise ValueError("horizon, delta, folds, and min_train_size must be positive")
#
#
#     if not len(full_index) % horizon == 0:
#         print(f"Train: {full_index[0]} to {full_index[-1]} ({len(full_index)/7/24} weeks, "
#               f"{len(full_index)/horizon} horizons) Horizon={horizon/7/24} weeks")
#         raise ValueError("Train set size should be divisible by the test size")
#
#     # Number of timesteps to forecast
#     horizon_duration = pd.Timedelta(hours=horizon)
#     max_date = full_index.max()
#     min_date = full_index.min()
#     last_cutoff = max_date - horizon_duration
#
#     # Calculate the time delta between cutoffs
#     delta = pd.Timedelta(hours=delta if delta else horizon)  # if delta == horizon: non-overlapping windows
#
#     # Generate cutoffs starting from the end of the time series
#     cutoffs = [last_cutoff - i * delta + pd.Timedelta(hours=1) for i in range(folds)]
#
#     # Validate minimum training size
#     if (cutoffs[-1] - min_date) < timedelta(hours=min_train_size):
#         raise ValueError(
#             f"Not enough train data for {len(cutoffs)}-fold cross-validation. "
#             f"(Need {min_train_size} hours at least). "
#             f"Last cutoff = {cutoffs[-1]}, min_date={min_date}"
#         )
#
#     # Reverse so that the last one is the latest one
#     cutoffs = cutoffs[::-1]
#     delta = timedelta(hours=horizon)
#
#     # check if split does not introduce segments that are not divisible by horizon length
#     for idx, cutoff in enumerate(cutoffs):
#         # Train matrix should have negth devisible for the length of the forecasting horizon,
#         # ane be composed of N segments each of which start at 00 hour and ends at 23 hour
#         train_mask = full_index < cutoff
#         # test mask should start at 00 hour and end on 23 hour (several full days)
#         test_mask = (full_index >= cutoff) & (full_index < cutoff + delta)
#
#         train_idx = full_index[train_mask]
#         test_idx = full_index[test_mask]
#
#         if not len(train_idx) % len(test_idx) == 0:
#             print(f"Train: {train_idx[0]} to {train_idx[-1]} ({len(train_idx)/7/24} weeks, "
#                   f"{len(train_idx)/len(test_idx)} horizons) Horizon={len(test_idx)/7/24} weeks | delta={delta}")
#             print(f"Test: {test_idx[0]} to {test_idx[-1]}")
#             raise ValueError(f"Fold {idx}/{len(cutoffs)} individual train set size is not divisible by the test size")
#
#     return cutoffs

# def compute_timeseries_split_cutoffs(
#         full_index: pd.DatetimeIndex,
#         horizon: int,         # number of hours (must be divisible by 24)
#         folds: int,           # number of splits (folds) to generate
#         min_train_size: int,  # minimum required train length in hours
# ) -> tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
#     """
#     The function takes a timeseries dataframe index column (full_index) and computes a list of cutoffs
#     such that each cutoff splits the train and test parts of the dataframe as follows:
#         - train_idx = all timestamps < cutoff
#         - test_idx  = all timestamps >= cutoff AND < cutoff + horizon
#
#     Important:
#       - train_idx and test_idx both should align with full-day boundaries
#         (i.e., start at 00:00 and end at 23:00).
#       - horizon must be divisible by 24 hours (so the test set spans full days).
#       - min_train_size must be a multiple of horizon (so that the train length
#         is always an integer multiple of the test length).
#       - For each split, we check that len(train_idx) is divisible by len(test_idx).
#
#     Returns
#     -------
#     cutoffs : list of pd.Timestamp
#         The list of cutoff timestamps used for each fold.
#     splits : list of (train_index, test_index)
#         For each cutoff, a tuple of (train_idx, test_idx), which are sub-indexes
#         of the original full_index.
#     """
#
#     # 1) Basic validations
#     # Horizon must be multiple of 24
#     if horizon % 24 != 0:
#         raise ValueError("`horizon` must be a multiple of 24 hours.")
#
#     # min_train_size must be a multiple of horizon
#     if min_train_size % horizon != 0:
#         raise ValueError(
#             "`min_train_size` must be an integer multiple of `horizon` "
#             "so that the train length is divisible by the test length."
#         )
#
#     # 2) Align full_index to full days only
#     #    We only consider midnight boundaries (00:00) for potential cutoffs
#     #    because we want train and test to start/end on full-day boundaries.
#     #    Filter out any timestamps that are not at midnight.
#     full_days = full_index[full_index.indexer_between_time("00:00", "00:00")]
#     if len(full_days) < 1:
#         raise ValueError("No full-day boundaries (midnight) found in the provided index.")
#
#     # 3) Determine the allowable range for cutoffs
#     #    We need at least `min_train_size` hours of training data before the cutoff,
#     #    and we need at least `horizon` hours after the cutoff for testing.
#     #    Hence, the first possible cutoff day is the earliest day + min_train_size.
#     #    The last possible cutoff day is the latest day - horizon.
#
#     # earliest possible midnight in the data
#     first_day = full_days.min()
#     # latest possible midnight in the data
#     last_day = full_days.max()
#
#     # Convert to Timestamps for arithmetic
#     first_day_ts = pd.Timestamp(first_day)
#     last_day_ts = pd.Timestamp(last_day)
#
#     # The earliest cutoff must be after we accumulate min_train_size
#     earliest_cutoff = first_day_ts + pd.Timedelta(hours=min_train_size)
#     # The latest cutoff must allow horizon hours for test
#     latest_cutoff = last_day_ts - pd.Timedelta(hours=horizon)
#
#     if earliest_cutoff >= latest_cutoff:
#         raise ValueError(
#             "Not enough data: `earliest_cutoff` (start + min_train_size) is "
#             "not before `latest_cutoff` (end - horizon)."
#         )
#
#     # Snap earliest_cutoff and latest_cutoff to midnight boundaries within full_days
#     # .searchsorted(...) helps us find the positions
#     idx_start = full_days.searchsorted(earliest_cutoff, side="left")
#     idx_end   = full_days.searchsorted(latest_cutoff, side="right") - 1
#
#     # valid_midnights are all potential cutoff midnights between earliest_cutoff and latest_cutoff
#     valid_midnights = full_days[idx_start : idx_end + 1]
#
#     # If we do not have enough midnight points to create `folds` distinct cutoffs, raise error
#     if len(valid_midnights) < folds:
#         raise ValueError(
#             f"Cannot generate {folds} folds between "
#             f"{earliest_cutoff} and {latest_cutoff} (only {len(valid_midnights)} midnights available)."
#         )
#
#     # 4) Choose the cutoffs
#     #    Below, we simply pick `folds` evenly spaced points within valid_midnights.
#     cutoffs_idx = np.linspace(0, len(valid_midnights) - 1, folds, dtype=int)
#     chosen_cutoffs = valid_midnights[cutoffs_idx]
#
#     # 5) Build the actual splits (train/test DatetimeIndexes)
#     splits = []
#     for cutoff in chosen_cutoffs:
#         cutoff_ts = pd.Timestamp(cutoff)
#
#         # Train = all timestamps < cutoff_ts
#         train_mask = (full_index < cutoff_ts)
#         train_idx = full_index[train_mask]
#
#         # Test = cutoff_ts <= timestamps < cutoff_ts + horizon
#         test_mask = (full_index >= cutoff_ts) & (full_index < cutoff_ts + pd.Timedelta(hours=horizon))
#         test_idx = full_index[test_mask]
#
#         # Additional checks:
#         # (a) ensure train and test are not empty
#         if len(train_idx) == 0:
#             raise ValueError(f"Train set is empty for cutoff {cutoff_ts}.")
#         if len(test_idx) == 0:
#             raise ValueError(f"Test set is empty for cutoff {cutoff_ts}.")
#
#         # (b) check that each ends at 23:00 and starts at 00:00
#         #     This ensures we're dealing with complete days (assuming data is hourly).
#         #     We do a simple check: the number of hours from min -> max + 1 == length
#         #     and min is at hour=0, max is at hour=23.
#         def check_full_day_range(idx: pd.DatetimeIndex, label: str):
#             if not len(idx):
#                 return  # skip if empty
#             start_h = idx[0].hour
#             end_h   = idx[-1].hour
#             total_hours = (idx[-1] - idx[0]).total_seconds()/3600 + 1  # inclusive
#             if start_h != 0:
#                 raise ValueError(f"{label} does not start at 00:00 for cutoff {cutoff_ts}")
#             if end_h != 23:
#                 raise ValueError(f"{label} does not end at 23:00 for cutoff {cutoff_ts}")
#             # Could also check that total_hours % 24 == 0, just to be thorough
#             if total_hours % 24 != 0:
#                 raise ValueError(f"{label} does not span complete days for cutoff {cutoff_ts}")
#
#         check_full_day_range(train_idx, "train_idx")
#         check_full_day_range(test_idx, "test_idx")
#
#         # (c) verify that the length of test is horizon hours
#         test_hours = (test_idx[-1] - test_idx[0]).total_seconds() / 3600 + 1
#         if test_hours != horizon:
#             raise ValueError(
#                 f"Test set length is {test_hours} hours, expected {horizon} for cutoff {cutoff_ts}."
#             )
#
#         # (d) check that len(train_idx) is divisible by len(test_idx)
#         #     Because test_idx is horizon hours, we can equivalently check
#         #     train_hours % horizon == 0.
#         train_hours = (train_idx[-1] - train_idx[0]).total_seconds() / 3600 + 1
#         if train_hours % horizon != 0:
#             raise ValueError(
#                 f"For cutoff {cutoff_ts}, train length is not divisible by test length "
#                 f"(train_hours={train_hours}, horizon={horizon})."
#             )
#
#         splits.append((train_idx, test_idx))
#
#     # Return the chosen cutoffs and the train/test splits
#     return list(chosen_cutoffs), splits



# def compute_timeseries_split_cutoffs(
#         full_index: pd.DatetimeIndex,
#         horizon: int,         # number of hours (must be divisible by 24)
#         folds: int,           # number of splits (folds) to generate
#         min_train_size: int,  # minimum required train length in hours (must be multiple of 24)
# ) -> tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
#     """
#     The function takes a time series DatetimeIndex (with hourly data) and computes a list of cutoffs
#     for rolling/anchored cross-validation from the *latest* possible date to earlier dates.
#
#     For each fold i:
#        - cutoff_i is at 00:00 (midnight).
#        - test set = [cutoff_i, cutoff_i + horizon)   # also ends at 23:00 of the last test day
#        - train set = [anchor, cutoff_i)             # must end at 23:00 (which it does if cutoff_i is 00:00)
#
#     We ensure:
#       1) horizon % 24 == 0  (whole days)
#       2) min_train_size % 24 == 0
#       3) full_index[-1].hour == 23  (dataset ends at 23:00)
#       4) Each train length is a multiple of horizon and >= min_train_size.
#       5) We generate folds from the latest possible date to the earliest possible date,
#          raising an exception if there's not enough data.
#
#     Returns:
#        cutoffs, train_test_splits
#     where
#        cutoffs is a list of length `folds` with the cutoff timestamps (descending in time)
#        train_test_splits is a list of (train_idx, test_idx) for each fold in the same order.
#     """
#     # 1. Basic checks
#     if horizon % 24 != 0:
#         raise ValueError(f"horizon must be divisible by 24. Received: {horizon}")
#     if min_train_size % 24 != 0:
#         raise ValueError(f"min_train_size must be divisible by 24. Received: {min_train_size}")
#     if full_index[-1].hour != 23:
#         raise ValueError(
#             f"The dataset must end at 23:00, but the last timestamp is {full_index[-1]}"
#         )
#
#     # For convenience
#     dataset_start = full_index[0]
#     dataset_end   = full_index[-1]  # we already checked dataset_end is 23:00
#
#     # 2. Compute the 'last' cutoff (the one used for the most recent fold).
#     #    The test set will be from last_cutoff to last_cutoff + horizon - 1H (which should be dataset_end).
#     #    So last_cutoff + horizon - 1 hour = dataset_end
#     #    => last_cutoff = dataset_end - horizon + 1 hour
#     last_cutoff = dataset_end - pd.Timedelta(hours=horizon - 1)
#
#     # By definition, if horizon is multiple of 24, last_cutoff.hour should be 0.
#     # But let's still ensure it's at 00:00. If it's not, we raise an error or "snap" it.
#     if last_cutoff.hour != 0:
#         raise ValueError(
#             f"Calculated last cutoff {last_cutoff} is not at 00:00; check your data/frequency."
#         )
#
#     # 3. Now we build cutoffs backward in increments of horizon, so we have:
#     #    c_{folds-1} = last_cutoff
#     #    c_{folds-2} = last_cutoff - horizon hours
#     #    ...
#     #    c_0         = last_cutoff - (folds-1)*horizon hours
#     #    We'll store them in ascending order first [c_0, ..., c_{folds-1}],
#     #    but you said you'd like to compute from latest to earliest.
#     #    Ultimately, we can just reverse them if you want the final list from latest to earliest.
#     cutoffs_ascending = [
#         last_cutoff - pd.Timedelta(hours=horizon * (folds - 1 - i))
#         for i in range(folds)
#     ]
#     # So cutoffs_ascending[0] = last_cutoff - (folds-1)*horizon
#     # and cutoffs_ascending[-1] = last_cutoff
#
#     # 4. Figure out how to "anchor" the training start so that:
#     #    (cutoffs_ascending[0] - anchor) is a multiple of horizon,
#     #    and also for each i, the train length >= min_train_size.
#     c0 = cutoffs_ascending[0]
#
#     # If c0 is earlier than dataset_start, that already won't work,
#     # but let's handle the possibility that we skip some data at the front
#     # to ensure the train length is a multiple of horizon.
#     #
#     # We'll define anchor = dataset_start + "some offset" such that (c0 - anchor) is multiple of horizon.
#     # Then we check if anchor <= c0 and if we can still get a train length >= min_train_size.
#
#     # Calculate how many hours from dataset_start to c0
#     total_hours_c0 = int((c0 - dataset_start) / pd.Timedelta(hours=1))
#     if total_hours_c0 < 0:
#         raise ValueError(
#             f"Even the earliest cutoff {c0} is before the start of data {dataset_start}. "
#             "Not enough data to make the required folds."
#         )
#
#     # remainder if we start exactly from dataset_start
#     remainder = total_hours_c0 % horizon
#     # if remainder=0, anchor = dataset_start
#     # if remainder!=0, we skip (horizon - remainder) hours from dataset_start
#     skip_hours = (horizon - remainder) if remainder != 0 else 0
#     anchor = dataset_start + pd.Timedelta(hours=skip_hours)
#
#     if anchor > c0:
#         raise ValueError(
#             "Not enough data to align the training window so that its length is a multiple of the horizon. "
#             f"Computed anchor={anchor} is beyond c0={c0}."
#         )
#
#     # Now check min_train_size for the smallest fold, which is fold #0 (the earliest cutoff):
#     # train_length_0 = (c0 - anchor)
#     train_length_0 = int((c0 - anchor) / pd.Timedelta(hours=1))
#     if train_length_0 < min_train_size:
#         raise ValueError(
#             f"Train length for earliest cutoff is only {train_length_0} hours, "
#             f"which is < min_train_size={min_train_size}. Not enough data."
#         )
#
#     # 5. Now that anchor is fixed, we can verify that for each cutoff c_i,
#     #    the train length is (c_i - anchor), which should be multiple of horizon
#     #    if the offset (c0 - anchor) was made multiple of horizon AND we only
#     #    step in increments of horizon. We also check >= min_train_size.
#     for c_i in cutoffs_ascending:
#         # ensure c_i is not before anchor:
#         if c_i < anchor:
#             raise ValueError(
#                 f"For cutoff={c_i}, we have c_i < anchor={anchor}, cannot form a valid train set."
#             )
#         train_length_i = int((c_i - anchor) / pd.Timedelta(hours=1))
#         # Must be multiple of horizon:
#         if train_length_i % horizon != 0:
#             raise ValueError(
#                 f"Train length for cutoff={c_i} is {train_length_i} hours, not divisible by horizon={horizon}."
#             )
#         # Must be >= min_train_size:
#         if train_length_i < min_train_size:
#             raise ValueError(
#                 f"Train length for cutoff={c_i} is {train_length_i} hours, "
#                 f"which is < min_train_size={min_train_size}."
#             )
#
#         # Also check we have enough room for the test set:
#         # c_i + horizon - 1 hour must be <= dataset_end
#         test_end_i = c_i + pd.Timedelta(hours=horizon) - pd.Timedelta(hours=1)
#         if test_end_i > dataset_end:
#             raise ValueError(
#                 f"For cutoff={c_i}, test_end={test_end_i} exceeds dataset_end={dataset_end}."
#             )
#
#     # 6. Everything is good, so let's build the final data structures.
#     #    We'll produce them in "descending" order (latest fold first) if you wish,
#     #    since you said “compute from the latest to earliest.”
#     cutoffs_descending = list(reversed(cutoffs_ascending))
#
#     train_test_splits = []
#     for cutoff in cutoffs_descending:
#         # train = [anchor, cutoff)
#         train_mask = (full_index >= anchor) & (full_index < cutoff)
#         train_idx = full_index[train_mask]
#
#         # test = [cutoff, cutoff + horizon)
#         test_mask = (full_index >= cutoff) & (full_index < cutoff + pd.Timedelta(hours=horizon))
#         test_idx = full_index[test_mask]
#
#         train_test_splits.append((train_idx, test_idx))
#
#     return cutoffs_descending, train_test_splits

# def compute_timeseries_split_cutoffs(
#         full_index: pd.DatetimeIndex,
#         horizon: int,         # number of hours (must be divisible by 24)
#         folds: int,           # number of splits (folds) to generate
#         min_train_size: int,  # minimum required train length in hours (must be multiple of 24)
# ) -> tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
#     """
#     The function takes a time series DatetimeIndex (with hourly data) and computes a list of cutoffs
#     for rolling/anchored cross-validation from the *latest* possible date to earlier dates.
#
#     For each fold i:
#        - cutoff_i is at 00:00 (midnight).
#        - test set = [cutoff_i, cutoff_i + horizon)   # also ends at 23:00 of the last test day
#        - train set = [anchor, cutoff_i)             # must end at 23:00 (which it does if cutoff_i is 00:00)
#
#     We ensure:
#       1) horizon % 24 == 0  (whole days)
#       2) min_train_size % 24 == 0
#       3) full_index[-1].hour == 23  (dataset ends at 23:00)
#       4) Each train length is a multiple of horizon and >= min_train_size.
#       5) We generate folds from the latest possible date to the earliest possible date,
#          raising an exception if there's not enough data.
#
#     Returns:
#        cutoffs, train_test_splits
#     where
#        cutoffs is a list of length `folds` with the cutoff timestamps (descending in time)
#        train_test_splits is a list of (train_idx, test_idx) for each fold in the same order.
#     """
#     # 1. Basic checks
#     if horizon % 24 != 0:
#         raise ValueError(f"horizon must be divisible by 24. Received: {horizon}")
#     if min_train_size % 24 != 0:
#         raise ValueError(f"min_train_size must be divisible by 24. Received: {min_train_size}")
#     if full_index[-1].hour != 23:
#         raise ValueError(
#             f"The dataset must end at 23:00, but the last timestamp is {full_index[-1]}"
#         )
#
#     # For convenience
#     dataset_start = full_index[0]
#     dataset_end   = full_index[-1]  # we already checked dataset_end is 23:00
#
#     # 2. Compute the 'last' cutoff (the one used for the most recent fold).
#     #    The test set will be from last_cutoff to last_cutoff + horizon - 1H (which should be dataset_end).
#     #    So last_cutoff + horizon - 1 hour = dataset_end
#     #    => last_cutoff = dataset_end - horizon + 1 hour
#     last_cutoff = dataset_end - pd.Timedelta(hours=horizon - 1)
#
#     # By definition, if horizon is multiple of 24, last_cutoff.hour should be 0.
#     # But let's still ensure it's at 00:00. If it's not, we raise an error or "snap" it.
#     if last_cutoff.hour != 0:
#         raise ValueError(
#             f"Calculated last cutoff {last_cutoff} is not at 00:00; check your data/frequency."
#         )
#
#     # 3. Now we build cutoffs backward in increments of horizon, so we have:
#     #    c_{folds-1} = last_cutoff
#     #    c_{folds-2} = last_cutoff - horizon hours
#     #    ...
#     #    c_0         = last_cutoff - (folds-1)*horizon hours
#     #    We'll store them in ascending order first [c_0, ..., c_{folds-1}],
#     #    but you said you'd like to compute from latest to earliest.
#     #    Ultimately, we can just reverse them if you want the final list from latest to earliest.
#     cutoffs_ascending = [
#         last_cutoff - pd.Timedelta(hours=horizon * (folds - 1 - i))
#         for i in range(folds)
#     ]
#     # So cutoffs_ascending[0] = last_cutoff - (folds-1)*horizon
#     # and cutoffs_ascending[-1] = last_cutoff
#
#     # 4. Figure out how to "anchor" the training start so that:
#     #    (cutoffs_ascending[0] - anchor) is a multiple of horizon,
#     #    and also for each i, the train length >= min_train_size.
#     c0 = cutoffs_ascending[0]
#
#     # If c0 is earlier than dataset_start, that already won't work,
#     # but let's handle the possibility that we skip some data at the front
#     # to ensure the train length is a multiple of horizon.
#     #
#     # We'll define anchor = dataset_start + "some offset" such that (c0 - anchor) is multiple of horizon.
#     # Then we check if anchor <= c0 and if we can still get a train length >= min_train_size.
#
#     # Calculate how many hours from dataset_start to c0
#     total_hours_c0 = int((c0 - dataset_start) / pd.Timedelta(hours=1))
#     if total_hours_c0 < 0:
#         raise ValueError(
#             f"Even the earliest cutoff {c0} is before the start of data {dataset_start}. "
#             "Not enough data to make the required folds."
#         )
#
#     # remainder if we start exactly from dataset_start
#     remainder = total_hours_c0 % horizon
#     # if remainder=0, anchor = dataset_start
#     # if remainder!=0, we skip (horizon - remainder) hours from dataset_start
#     skip_hours = (horizon - remainder) if remainder != 0 else 0
#     anchor = dataset_start + pd.Timedelta(hours=skip_hours)
#
#     if anchor > c0:
#         raise ValueError(
#             "Not enough data to align the training window so that its length is a multiple of the horizon. "
#             f"Computed anchor={anchor} is beyond c0={c0}."
#         )
#
#     # Now check min_train_size for the smallest fold, which is fold #0 (the earliest cutoff):
#     # train_length_0 = (c0 - anchor)
#     train_length_0 = int((c0 - anchor) / pd.Timedelta(hours=1))
#     if train_length_0 < min_train_size:
#         raise ValueError(
#             f"Train length for earliest cutoff is only {train_length_0} hours, "
#             f"which is < min_train_size={min_train_size}. Not enough data."
#         )
#
#     # 5. Now that anchor is fixed, we can verify that for each cutoff c_i,
#     #    the train length is (c_i - anchor), which should be multiple of horizon
#     #    if the offset (c0 - anchor) was made multiple of horizon AND we only
#     #    step in increments of horizon. We also check >= min_train_size.
#     for c_i in cutoffs_ascending:
#         # ensure c_i is not before anchor:
#         if c_i < anchor:
#             raise ValueError(
#                 f"For cutoff={c_i}, we have c_i < anchor={anchor}, cannot form a valid train set."
#             )
#         train_length_i = int((c_i - anchor) / pd.Timedelta(hours=1))
#         # Must be multiple of horizon:
#         if train_length_i % horizon != 0:
#             raise ValueError(
#                 f"Train length for cutoff={c_i} is {train_length_i} hours, not divisible by horizon={horizon}."
#             )
#         # Must be >= min_train_size:
#         if train_length_i < min_train_size:
#             raise ValueError(
#                 f"Train length for cutoff={c_i} is {train_length_i} hours, "
#                 f"which is < min_train_size={min_train_size}."
#             )
#
#         # Also check we have enough room for the test set:
#         # c_i + horizon - 1 hour must be <= dataset_end
#         test_end_i = c_i + pd.Timedelta(hours=horizon) - pd.Timedelta(hours=1)
#         if test_end_i > dataset_end:
#             raise ValueError(
#                 f"For cutoff={c_i}, test_end={test_end_i} exceeds dataset_end={dataset_end}."
#             )
#
#     # 6. Everything is good, so let's build the final data structures.
#     #    We'll produce them in "descending" order (latest fold first) if you wish,
#     #    since you said “compute from the latest to earliest.”
#     cutoffs_descending = list(reversed(cutoffs_ascending))
#
#     train_test_splits = []
#     for cutoff in cutoffs_descending:
#         # train = [anchor, cutoff)
#         train_mask = (full_index >= anchor) & (full_index < cutoff)
#         train_idx = full_index[train_mask]
#
#         # test = [cutoff, cutoff + horizon)
#         test_mask = (full_index >= cutoff) & (full_index < cutoff + pd.Timedelta(hours=horizon))
#         test_idx = full_index[test_mask]
#
#         train_test_splits.append((train_idx, test_idx))
#
#     return cutoffs_descending, train_test_splits
#


def visualize_splits(
        full_index: pd.DatetimeIndex,
        cutoffs: list[pd.Timestamp],
        train_test_splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
):
    """
    Visualizes the train and test splits for time series cross-validation.

    Args:
        full_index: DatetimeIndex containing timestamps for the entire dataset
        cutoffs: List of cutoff timestamps used for splitting
        train_test_splits: List of tuples containing train and test indices
    """
    plt.figure(figsize=(15, 8))

    for i, (train_idx, test_idx) in enumerate(train_test_splits):
        # Plot training indices
        plt.plot(train_idx, [i + 1] * len(train_idx), '|', label=f'Train {i + 1}' if i == 0 else "", color='blue')
        # Plot testing indices
        plt.plot(test_idx, [i + 1] * len(test_idx), '|', label=f'Test {i + 1}' if i == 0 else "", color='orange')

    # Plot cutoffs
    for cutoff in cutoffs:
        plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff' if cutoff == cutoffs[0] else "")

    plt.title("Train-Test Splits Visualization")
    plt.xlabel("Time")
    plt.ylabel("Fold")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_timeseries_split_cutoffs(
        full_index: pd.DatetimeIndex,
        horizon: int,
        folds: int
        # min_train_size: int
) -> tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
    """
    Computes cutoffs and corresponding train and test indices for time series cross-validation.

    Args:
        full_index: DatetimeIndex containing timestamps for the entire dataset
        horizon: Forecast horizon in hours
        folds: Number of folds for cross-validation
        min_train_size: Minimum required training size in hours

    Returns:
        Tuple containing a list of cutoffs and a list of tuples with train and test indices
    """

    min_train_size = len(full_index) - folds * horizon

    print(f"| folds={folds} min_train_size={min_train_size} full_index={len(full_index)} |")

    if horizon % 24 != 0:
        raise ValueError("Horizon must be divisible by 24 (whole days).")
    if min_train_size % 24 != 0 or min_train_size < 1:
        raise ValueError("Minimum train size must be divisible by 24 (whole days).")
    # Check if full_index is continuous (hourly data)
    expected_range = pd.date_range(start=full_index.min(), end=full_index.max(), freq='h')
    if not full_index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")

    cutoffs = []
    train_test_splits = []

    step = horizon
    current_index = len(full_index) - 1



    while len(cutoffs) < folds and current_index >= 0:
        cutoff = full_index[current_index]

        # Check if test period fits within the full index
        test_end = cutoff + pd.Timedelta(hours=horizon - 1)
        if test_end not in full_index:
            current_index -= 1
            continue

        # Determine train and test indices
        train_end = cutoff - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=min_train_size - 1)
        test_start = cutoff

        if train_start not in full_index or train_end not in full_index:
            current_index -= 1
            continue

        train_idx = full_index[(full_index >= train_start) & (full_index <= train_end)]
        test_idx = full_index[(full_index >= test_start) & (full_index <= test_end)]

        if len(train_idx) < 1:
            raise ValueError(f"For cutoff={cutoff}, there are no train indices for "
                             f"train_start = {train_start} train_end = {train_end} test_start = {test_start}; "
                             f"full_index = {len(full_index)} len(cutoffs)= {len(cutoffs)} min_train_size={min_train_size}")
        if len(test_idx) < 1:
            raise ValueError(f"For cutoff={cutoff}, there are no train indices for "
                             f"train_start = {train_start} train_end = {train_end} test_start = {test_start}; "
                             f"full_index = {len(full_index)} len(cutoffs)= {len(cutoffs)} min_train_size={min_train_size}")

        # Ensure train and test indices end at 23:00 and start at 00:00
        if train_idx[-1].hour != 23 or train_idx[0].hour != 0:
            current_index -= 1
            continue
        if test_idx[-1].hour != 23 or test_idx[0].hour != 0:
            current_index -= 1
            continue

        # Check divisibility conditions
        if len(train_idx) % len(test_idx) != 0:
            current_index -= 1
            continue

        assert len(test_idx) == horizon
        assert len(train_idx) % horizon == 0

        # Append valid cutoff and splits
        cutoffs.append(cutoff)
        train_test_splits.append((train_idx, test_idx))

        # Move to the next potential cutoff point
        current_index -= step

        print(f"\tFor cutoff={cutoff} | "
              f"(folds={folds}) len(cutoffs)={len(cutoffs)} min_train_size={min_train_size} "
              f"full_index={len(full_index)} | train={len(train_idx)} test={len(test_idx)} | "
              f"train_start = {train_start} | train_end = {train_end} | test_start = {test_start}")

    if len(cutoffs) < folds:
        raise ValueError("Unable to generate the required number of folds with the given constraints. ")

    # invert so that the last fold is the latest fold
    cutoffs = cutoffs[::-1]
    train_test_splits = train_test_splits[::-1]

    # visualize_splits(full_index, cutoffs, train_test_splits)

    return cutoffs, train_test_splits



def compute_error_metrics(target:str,result:pd.DataFrame)->dict:

    res = copy.deepcopy(result)

    def smape(actual, predicted):
        """
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

        Parameters:
        actual (array-like): Array of actual values.
        predicted (array-like): Array of predicted values.

        Returns:
        float: sMAPE value as a percentage.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Avoid division by zero using (|actual| + |predicted|) in the denominator
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        smape_value = np.mean(2 * np.abs(predicted - actual) / denominator) * 100

        return smape_value


    # extract arrays
    y_true = res[f'{target}_actual'].values
    y_pred = res[f'{target}_fitted'].values
    y_lower = res[f'{target}_lower'].values
    y_upper = res[f'{target}_upper'].values
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    if not np.all(np.isfinite(y_true)):
        print ("WARNIGN! y_true contains NaN, infinity, or values too large for dtype('float64').")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.all(np.isfinite(y_pred)):
        print ("WARNING! y_pred contains NaN, infinity, or values too large for dtype('float64').")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)

    # compute metrics
    res_dict = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true+1e-10, y_pred) * 100,
        'smape': smape(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'variance': np.var(y_pred - y_true),
        'std': np.std(y_pred - y_true),
        'r2':r2_score(y_true, y_pred),
        'prediction_interval_coverage':coverage,
        'prediction_interval_width':np.mean(y_upper - y_lower)
    }

    return res_dict

def compute_error_metrics_aggregate_over_horizon(
        target:str, cv_result:list[pd.DataFrame], unscaler:Callable[[pd.Series], pd.Series] = None)->dict:
    ''' compute error metrics for each forecasted hour using forecasted horizon to aggregate over
        and compute mean and std of the result (aggregating over cv_runs) '''
    cv_metrics = []
    for i in range(len(cv_result)):
        if unscaler is None:
            cv_metrics.append(compute_error_metrics(target, cv_result[i]))
        else:
            # apply function that takes pd.Seris to invert scale the target column
            cv_metrics.append(compute_error_metrics(target, cv_result[i].apply(unscaler)))

    res = {'mean':{}, 'std':{}}
    for metric in cv_metrics[0].keys():
        res['mean'][metric] = np.mean([cv_metrics[i][metric] for i in range(len(cv_metrics))])
        res['std'][metric] = np.std([cv_metrics[i][metric] for i in range(len(cv_metrics))])
    return res

def compute_error_metrics_aggregate_over_cv_runs(
        target:str, cv_result: list[pd.DataFrame], unscaler:Callable[[pd.Series], pd.Series] or None) -> list[dict]:
    ''' Compute error metrics for each forecasted hour using cross-validation runs to aggregate over '''
    n_hours_forecasted = len(cv_result[0].iloc[:, 0])  # Access first column with .iloc
    folds = len(cv_result)
    entries = list(cv_result[0].columns)

    cv_results = copy.deepcopy(cv_result)
    if not unscaler is None:
        for i in range(len(cv_results)):
            cv_results[i] = cv_results[i].apply(unscaler)

    # Reshape the data so that each DataFrame for each hour contains values for all CV runs
    tmp_list = [pd.DataFrame() for _ in range(n_hours_forecasted)]
    for i_hour in range(n_hours_forecasted):
        tmp_dict = {key: [] for key in entries}
        for key in entries:
            for i_cv in range(folds):
                tmp_dict[key].append(float(cv_results[i_cv][key].iloc[i_hour]))  # Use .iloc[i_hour] here
        tmp_list[i_hour] = pd.DataFrame(tmp_dict, columns=entries, index=[i_cv for i_cv in range(folds)])

    # Compute error metrics for each hour aggregating over CV runs
    res = [{} for _ in range(n_hours_forecasted)]
    for i_hour in range(n_hours_forecasted):
        res[i_hour] = compute_error_metrics(target, tmp_list[i_hour])

    return res

def save_datetime_now(outdir:str):
    # save when fine-tuning was done
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    with open(f'{outdir}datetime.json', "w") as file:
        json.dump({"datetime": today.isoformat()}, file)