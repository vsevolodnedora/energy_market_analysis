from typing import Optional, Union

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

def get_pslp_category( date: Union[str, pd.Timestamp], weekday: Optional[int] = None, holiday: Optional[bool] = None, country_code: Optional[str] = "DE") -> int:
    """
    Get PSLP category from date, weekday information, and holiday information.

    0 : weekday, 1 : Saturday, 2 : Sunday and holiday

    Parameters
    ----------
    date : Union[str, pd.Timestamp]
        The date in "YYYYMMDD" format or a pd.Timestamp object.
    weekday : int, optional
        The corresponding weekday (0: Mon, 1: Tue, ..., 6: Sun).
    holiday : bool, optional
        True if public holiday, False if not.
    country_code : str, optional
        The country code. Default is Germany.

    Returns
    -------
    int
        The PSLP category.
    """
    if isinstance(
            date, str
    ):  # Convert string-type date to datetime object if necessary.
        date = pd.to_datetime(date)

    if weekday is None:  # Assign weekday if not given.
        weekday = date.weekday()

    if holiday is None:  # Assign holiday category if not given.
        holiday = date in holidays.country_holidays(country_code)
        # TODO: Make mapping of ENTSO-E to holidays country codes.

    # Special treatment for Christmas Eve and New Year's Eve as Saturdays.
    if date.day in {24, 31} and date.month == 12 and weekday != 6:
        return 1
    elif weekday < 5 and holiday is False:  # Weekdays
        return 0
    elif weekday == 5 and holiday is False:  # Saturdays
        return 1
    elif weekday == 6 or holiday is True:  # Sundays and holidays
        return 2
    else:
        raise ValueError("Invalid combination of weekday and holiday feature!")

def _assign_pslp_categories( df: pd.DataFrame, country_code: Optional[str] = "DE" ) -> pd.DataFrame:
    """
    Assign PSLP categories to dates in dataframe's datetime index.

    0 is weekday, 1 is Saturday, 2 is Sunday or holiday.
    Special treatment for Christmas Eve and New Year's Eve (as Saturdays).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    country_code : str, optional
        The country to determine holidays for. Default is Germany.

    Returns
    -------
    pd.DataFrame
            Integer PSLP categories for each point datetime index.
    """
    # Return PSLP categories calculated using list comprehension.
    df_pslp_categories = pd.DataFrame.from_dict(
        {
            "pslp_category": [
                get_pslp_category(
                    date, weekday, date in holidays.country_holidays(country_code)
                )
                for date, weekday in zip(df.index.date, df.index.weekday)
            ]
        },
    ).astype("category")
    df_pslp_categories.index = df.index
    return df_pslp_categories

def get_nearest_future_pslp_date( date_str: str, pslp_category: Optional[Union[int, None]] = None) -> datetime.date:
    """
    For a given date, get nearest day in future for its PSLP category.

    Parameters
    ----------
    date_str : str
        The considered date.
    pslp_category : int, optional
        The corresponding PSLP category.

    Returns
    -------
    datetime.date
        The next future date of the same PSLP category.
    """
    if pslp_category is None:  # Determine PSLP category if not provided.
        pslp_category = get_pslp_category(date_str)
    start = pd.to_datetime(date_str).date() + pd.Timedelta(days=1)
    end = pd.to_datetime(date_str).date() + pd.Timedelta(weeks=1)
    future_dates = pd.date_range(
        start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")
    )
    pslp_categories = np.array([get_pslp_category(d) for d in future_dates])
    return future_dates[np.where(pslp_categories == pslp_category)][0].date()

def calculate_single_pslp( df: pd.DataFrame, date_str: str, df_pslp_categories: Optional[pd.DataFrame] = None,
                           lookback: Optional[int] = 3, country_code: Optional[str] = "DE" ) -> pd.DataFrame:
    """
    Calculate PSLP for given date from given time-series data.

    The data is categorized into weekdays, Saturdays, and Sundays/holidays.
    The `lookback` most recent days from the specified date's category are used to
    calculate the corresponding PSLP as the average.

    Note that the given time series dataframe `df` must contain full days for each
    date. This needs to be considered in particular w.r.t. winter / summer time shifts.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the time series data.
    date_str : str
        The date "YYYYMMDD" to calculate PSLP for.
    df_pslp_categories : pd.DataFrame, optional
        The corresponding PSLP categories.
    lookback : int, optional
        The number of days to consider in each category for calculating PSLP. Default is 3.
    country_code : str, optional
        The considered country (for holidays). Default is Germany.
    """
    # Get unique dates in datetime index of original time series.
    unique_dates = df.index.to_series().dt.date.drop_duplicates().tolist()

    # Get datetime object from date string of interest.
    date = pd.to_datetime(date_str)
    # Get PSLP category for date of interest.
    pslp_category = get_pslp_category(date_str)
    print(f"PSLP category of {date.date()} is {pslp_category}.")

    # If not provided, assign PSLP categories in original time series.
    if df_pslp_categories is None:
        df_pslp_categories = _assign_pslp_categories(df, country_code)

    last_date_of_same_category = (
        df[df_pslp_categories == pslp_category]
        .index.to_series()
        .dt.date.drop_duplicates()
        .tolist()[-1]
    )
    print(f"Last day of same category is {last_date_of_same_category}.")
    # Check whether the PSLP of the date of interest can be calculated from the given dataframe.
    if date.date() < unique_dates[0]:  # Date of interest is in the past.
        raise IndexError(f"PSLP cannot be calculated. Date {date_str} is in the past.")

    if date.date() > unique_dates[-1] and date.date() != get_nearest_future_pslp_date(
            last_date_of_same_category, pslp_category
    ):
        raise IndexError(f"PSLP cannot be calculated. Date {date_str} is too far in the future.")

    date_in_future = False
    if date.date() == get_nearest_future_pslp_date(last_date_of_same_category, pslp_category):
        date_in_future = True

    print(f"Date of interest in future? {date_in_future}")
    # Get unique dates with same PSLP category as date of interest from original time series datetime index.
    unique_dates_pslp = (
        df[df_pslp_categories["pslp_category"] == pslp_category]
        .index.to_series()
        .dt.date.drop_duplicates()
        .tolist()
    )
    print(f"Dates with same PSLP category: N={len(unique_dates_pslp)}")

    if date_in_future:
        lookback_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in unique_dates_pslp[-lookback:] ]
        if len(lookback_dates) < lookback:
            raise ValueError(
                f"PSLP cannot be calculated. Less than {lookback} samples in PSLP category for date {date_str}."
            )
    else:
        # Find the index of the date of interest within the list of unique dates in the time series.
        idx_pslp = unique_dates_pslp.index(date.date())
        print(f"Index in unique days of PSLP category is {idx_pslp}.")

        # Check whether number of samples is sufficient to calculate PSLP for date of interest.
        if idx_pslp - lookback < 0:
            raise IndexError(f"PSLP cannot be calculated. Less than {lookback} "
                             f"samples in PSLP category for date {date_str}.")

        # Get lookback dates.
        lookback_dates = [ pd.to_datetime(d).strftime("%Y-%m-%d")
                           for d in unique_dates_pslp[idx_pslp - lookback : idx_pslp] ]

    freq = 'h'
    periods = 24 if freq == "h" else 96

    for d in lookback_dates:
        if df.loc[d].shape[0] != periods:
            raise IndexError(
                f"Date {d} does not correspond to a full day in the given time series data. "
                f"shape={df.loc[d].shape[0]} periods={periods}"
            )
    # print(f"Dates to consider for calculating PSLP: {lookback_dates}")

    print(f"Calculating PSLPs for date {date.date()}...")

    # Set up dictionary to save PSLPs for all categories.
    pslp_dict = {}
    for header in df.columns:
        vals = []
        for d in lookback_dates:
            arr = df[header].at[d].reset_index(drop=True).to_numpy()
            arr[~np.isfinite(arr)] = 0.
            vals.append(arr)
        vals = np.array(vals)
        # vals = vals[np.isfinite(vals)]
        pslp_dict[header] = np.mean(vals,axis=0)

    # pslp_dict = { header: np.mean(
    #     np.array([df[header].at[d].reset_index(drop=True).to_numpy()
    #     for d in lookback_dates]),axis=0) for header in df.columns
    # }
    df_pslp = pd.DataFrame.from_dict(pslp_dict)
    df_pslp.index = pd.date_range(date_str, periods=periods, freq=freq, tz=df.index.tz)
    return df_pslp

def calculate_pslps( df: pd.DataFrame, start_date:pd.Timestamp,
                     lookback: Optional[int] = 3, country_code: Optional[str] = "DE" ) -> pd.DataFrame:
    """
    Calculate PSLPs for all dates in dataframe.

    The data is categorized into weekdays, Saturdays, and Sundays/holidays.
    The `lookback` most recent days from the specified date's category are used to
    calculate the corresponding PSLP as the average.

    Parameters
    ----------
    df : pd.Data
        The time series data to calculate PSLPs for.
    lookback : int, optional
        The number of days to consider in each category for calculating PSLP.
        Default is 3.
    country_code : str, optional
        The considered country (for holidays). Default is Germany.

    Returns
    -------
    pd.DataFrame
        The calculated PSLP(s).
    """
    if not df.index.name == 'date':
        df.set_index('date', inplace=True)
    df_pslp = pd.DataFrame(columns=df.columns, index=df.index[df.index >= start_date])
    df_pslp_categories = _assign_pslp_categories(df)
    print("Calculating PSLPs for all dates in dataframe...")
    unique_dates = (df.index.to_series().dt.date.drop_duplicates().tolist())  # Get unique dates in data index.
    unique_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in unique_dates]

    for i_date, date in enumerate(unique_dates):
        if pd.Timestamp(date,tz='UTC') >= start_date:
            try:
                df_pslp.loc[date] = calculate_single_pslp(
                    df,
                    date_str=date,
                    df_pslp_categories=df_pslp_categories,
                    lookback=lookback,
                    country_code=country_code,
                )
            except IndexError as e:
                print(f"Failed {i_date}/{len(unique_dates)} with {e}")
    df_pslp.reset_index('date',inplace=True)
    return df_pslp

def crop_dataframe_to_last_full_day(df: pd.DataFrame) -> pd.DataFrame:
    # Identify the last date
    last_datetime = df['date'].iloc[-1]

    # Check if the last entry is not at 23:00
    if last_datetime.hour != 23:
        last_date = last_datetime.normalize()  # This sets the time to 00:00 of the last day
        # Set all values for the last day to NaN
        df.loc[df['date'].dt.date == last_date.date(), 'value'] = np.nan
    return df