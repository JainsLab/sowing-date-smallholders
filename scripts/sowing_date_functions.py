# scripts/sowing_date_functions.py

import pandas as pd
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


## define data formats
def rs_date_preparation(df, init_date, year, date_column, doy, woy, week, dop_day, dop_week, wop):
    #check if the year is leap
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    # Survey has a different week, we are keeping the pattern
    def generate_week_ranges(year):
        feb_end = '29' if is_leap_year(year) else '28'
        week_ranges = [
            (f'{year}-01-01', f'{year}-01-07'),
            (f'{year}-01-08', f'{year}-01-14'),
            (f'{year}-01-15', f'{year}-01-21'),
            (f'{year}-01-22', f'{year}-01-31'),
            (f'{year}-02-01', f'{year}-02-07'),
            (f'{year}-02-08', f'{year}-02-14'),
            (f'{year}-02-15', f'{year}-02-21'),
            (f'{year}-02-22', f'{year}-02-{feb_end}'),
            (f'{year}-03-01', f'{year}-03-07'),
            (f'{year}-03-08', f'{year}-03-14'),
            (f'{year}-03-15', f'{year}-03-21'),
            (f'{year}-03-22', f'{year}-03-31'),
            (f'{year}-04-01', f'{year}-04-07'),
            (f'{year}-04-08', f'{year}-04-14'),
            (f'{year}-04-15', f'{year}-04-21'),
            (f'{year}-04-22', f'{year}-04-30'),
            (f'{year}-05-01', f'{year}-05-07'),
            (f'{year}-05-08', f'{year}-05-14'),
            (f'{year}-05-15', f'{year}-05-21'),
            (f'{year}-05-22', f'{year}-05-31'),
            (f'{year}-06-01', f'{year}-06-07'),
            (f'{year}-06-08', f'{year}-06-14'),
            (f'{year}-06-15', f'{year}-06-21'),
            (f'{year}-06-22', f'{year}-06-30'),
            (f'{year}-07-01', f'{year}-07-07'),
            (f'{year}-07-08', f'{year}-07-14'),
            (f'{year}-07-15', f'{year}-07-21'),
            (f'{year}-07-22', f'{year}-07-31'),
            (f'{year}-08-01', f'{year}-08-07'),
            (f'{year}-08-08', f'{year}-08-14'),
            (f'{year}-08-15', f'{year}-08-21'),
            (f'{year}-08-22', f'{year}-08-31'),
            (f'{year}-09-01', f'{year}-09-07'),
            (f'{year}-09-08', f'{year}-09-14'),
            (f'{year}-09-15', f'{year}-09-21'),
            (f'{year}-09-22', f'{year}-09-30'),
            (f'{year}-10-01', f'{year}-10-07'),
            (f'{year}-10-08', f'{year}-10-14'),
            (f'{year}-10-15', f'{year}-10-21'),
            (f'{year}-10-22', f'{year}-10-31'),
            (f'{year}-11-01', f'{year}-11-07'),
            (f'{year}-11-08', f'{year}-11-14'),
            (f'{year}-11-15', f'{year}-11-21'),
            (f'{year}-11-22', f'{year}-11-30'),
            (f'{year}-12-01', f'{year}-12-07'),
            (f'{year}-12-08', f'{year}-12-14'),
            (f'{year}-12-15', f'{year}-12-21'),
            (f'{year}-12-22', f'{year}-12-31'),
        ]
        return [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in week_ranges]
    #save the week 
    def create_week_lookup(start_year, end_year, filename):
        week_data = []
        for year in range(start_year, end_year + 1):
            week_ranges = generate_week_ranges(year)
            for week_num, (start_date, end_date) in enumerate(week_ranges, start=1):
                for date in pd.date_range(start=start_date, end=end_date):
                    week_data.append({
                        'date': date,
                        'year': year,
                        'week_num': week_num
                    })
        week_df = pd.DataFrame(week_data)
        week_df.to_csv(filename, index=False)
    #check if the file exist. If not, created and load
    filename = r'../data/week_lookup.csv'
    if os.path.exists(filename):
        # Load the precomputed week lookup table
        week_lookup = pd.read_csv(filename, parse_dates=['date'])
    else:
        # Generate week lookup table for years 2017 to 2025
        create_week_lookup(2017, 2025, filename)
        # Load the precomputed week lookup table
        week_lookup = pd.read_csv(filename, parse_dates=['date'])

    # get week according with the date
    def get_custom_week(date):
        # Find the corresponding week number from the precomputed table
        result = week_lookup.loc[week_lookup['date'] == date, 'week_num']
        return result.values[0] if not result.empty else None
    
    #  Setup the datetime format for different columns in our dataset.
    df[date_column] = pd.to_datetime(df[date_column].astype(str).str[:10], format='%Y-%m-%d')

    # Day Of Year
    df[doy] = df[date_column].dt.dayofyear
    # Week Of Year
    df[woy] = df[date_column].dt.isocalendar().week
    # Custom week
    df[week] = df[date_column].apply(get_custom_week)

    # Calculating the difference to init_date. Day of Period
    df[dop_day] = (df[date_column] - init_date).dt.days

    # Calculate week number based on custom week ranges
    init_week = get_custom_week(init_date)
    init_year = init_date.year
    
    def calculate_relative_week(week1, date):
        year_diff = date.year - init_year
        if year_diff == 0:
            return week1 - init_week
        elif year_diff > 0:
            return (year_diff * 48) + (week1 - init_week)
        else:
            return -((abs(year_diff) * 48) - (week1 - init_week))
        
    df[wop] = df.apply(lambda row: calculate_relative_week(row[week], row[date_column]), axis=1)
    df[dop_week] = df[dop_day] /7
    return df








def filter_lower_values_interpolation(df, col,col_day, thres):
    """This function replaces values lower than the threshold value by the weighted mean of the nearest neighbors using linear interpolation.

    Args:
        df (dataframe): The input dataframe.
        col (string): Name of the column to be used in the replace process.
        thres (numeric): Threshold (value) of the col column that will need to be replaced.
        col_day: column with the datetime formaty
    Returns:
        dataframe: Returns the same dataframe as df, but with the column modified.
    """
    # Reset index
    df = df.reset_index(drop=True)
    # Create a copy of the column to avoid modifying the original dataframe
    df_copy = df.copy()
    # Find indices where values are below the threshold
    indices = df[df[col] < thres].index

    for idx in indices:
        if idx > 0 and idx < len(df) - 1:
            # Get the neighboring NDVI values and their corresponding dates
            prev_value = df.at[idx - 1, col]
            next_value = df.at[idx + 1, col]

            prev_date = df.at[idx - 1, col_day]
            next_date = df.at[idx + 1, col_day]
            current_date = df.at[idx, col_day]

            # Calculate the weights based on the time differences
            weight_prev = abs((next_date - current_date).days)
            weight_next = abs((current_date - prev_date).days)

            total_weight = weight_prev + weight_next
            interpolated_value = (prev_value * weight_prev + next_value * weight_next) / total_weight
            # If total weight is zero, use the average of the previous and next values
            # interpolated_value = np.mean([prev_value, next_value])
            
            # Update the value in the dataframe
            df_copy.at[idx, col] = interpolated_value

        elif idx == 0:
            # If the first element is below the threshold, use the next element
            df_copy.at[idx, col] = df.at[idx + 1, col]
        elif idx == len(df) - 1:
            # If the last element is below the threshold, use the previous element
            df_copy.at[idx, col] = df.at[idx - 1, col]

    return df_copy

    
#detect drops
def detect_abrupt_drops(df, col_of_interest,col_day, threshold):
    """
    Detects abrupt drops in a time series of NDVI values in a specified column of a dataframe.

    Args:
        df (DataFrame): The dataframe containing NDVI time series data.
        col_of_interest (str): The name of the column containing the NDVI time series data.
        threshold (float): Threshold for allowed variation in NDVI values.
        col_day: column with the datetime formaty

    Returns:
        DataFrame: Return the input dataframe with interpolated values where abrupt drops are detected.
    """
    # Create a copy of the dataframe
    dataset = df.copy()
    dataset = dataset.reset_index(drop=True)

    # Iterate over the rows of the dataframe
    for index, row in dataset.iterrows():
        # Get the NDVI value for the current row and column of interest
        ndvi_value = row[col_of_interest]

        # Check if the change between previous and next row exceeds the negative threshold
        if index > 0 and index < len(dataset) - 1:
            prev_ndvi_value = dataset.at[index - 1, col_of_interest]
            next_ndvi_value = dataset.at[index + 1, col_of_interest]
            
            prev_date = dataset.at[index - 1, col_day]
            next_date = dataset.at[index + 1,col_day] 
            current_date = dataset.at[index, col_day]
            
            
            dif_prev = ndvi_value - prev_ndvi_value
            dif_next = ndvi_value - next_ndvi_value

            # Check for abrupt drops (negative changes)
            if dif_prev < -threshold and dif_next < -threshold:
                
                
                # Calculate the weights based on the time differences
                weight_prev = abs((next_date - current_date).days)#days until next image
                weight_next = abs((current_date - prev_date).days)#days until the previous image
                total_weight = weight_prev + weight_next #total days
                # Interpolate the value using weighted average
                interpolated_value = (prev_ndvi_value * weight_prev + next_ndvi_value * weight_next) / total_weight
                
                # Interpolate the value
                # interpolated_value = np.mean([prev_ndvi_value, next_ndvi_value])
                
                # Update the value in the dataframe
                dataset.at[index, col_of_interest] = interpolated_value

    return dataset

#detect spikes
def detect_abrupt_spikes(df, col_of_interest,col_day, threshold):
    """
    Detects spikes in a time series of NDVI values in a specified column of a dataframe.

    Args:
        df (DataFrame): The dataframe containing NDVI time series data.
        col_of_interest (str): The name of the column containing the NDVI time series data.
        threshold (float): Threshold for allowed variation in NDVI values.
        col_day: column with the datetime formaty

    Returns:
        DataFrame: Return the input dataframe with interpolated values where spikes are detected.
    """
    # Create a copy of the dataframe
    dataset = df.copy()
    dataset = dataset.reset_index(drop=True)

    # Iterate over the rows of the dataframe
    for index, row in dataset.iterrows():
        # Get the NDVI value for the current row and column of interest
        ndvi_value = row[col_of_interest]

        # Check if the change between previous and next row exceeds the positive threshold
        if index > 0 and index < len(dataset) - 1:
            
            prev_ndvi_value = dataset.at[index - 1, col_of_interest]
            next_ndvi_value = dataset.at[index + 1, col_of_interest]

            prev_date = dataset.at[index - 1, col_day]
            next_date = dataset.at[index + 1,col_day] 
            current_date = dataset.at[index, col_day] #

            dif_prev = ndvi_value - prev_ndvi_value
            dif_next = ndvi_value - next_ndvi_value

            # Check for spikes (positive changes)
            if dif_prev > threshold and dif_next > threshold:
                
                # Calculate the weights based on the time differences
                weight_prev = abs((next_date - current_date).days)#days until next image
                weight_next = abs((current_date - prev_date).days)#days until the previous image
                total_weight = weight_prev + weight_next #total days
                # Interpolate the value using weighted average
                interpolated_value = (prev_ndvi_value * weight_prev + next_ndvi_value * weight_next) / total_weight
                
                # # Interpolate the value
                # interpolated_value = np.mean([prev_ndvi_value, next_ndvi_value])
                
                # Update the value in the dataframe
                dataset.at[index, col_of_interest] = interpolated_value

    return dataset

def interpolate_timeseries(df, vi_col='NDVI', col_day="day", thres_lower=0, thres_change=0.1):
    df = filter_lower_values_interpolation(df, vi_col,col_day, thres_lower)
    df = detect_abrupt_drops(df, vi_col,col_day, thres_change)
    df =detect_abrupt_spikes(df, vi_col,col_day, thres_change)
    """Fills gaps in time series with linear interpolation."""
    to_resample = df[["day","NDVI"]].copy()
    # We keep the 'first' occurrence for each day.
    to_resample.drop_duplicates(subset=[col_day], keep='first', inplace=True)
    to_resample.set_index('day', inplace=True)
    # Resample to daily frequency and interpolate
    
    df_resampled = to_resample.resample('D').asfreq()

    df_interpolated = df_resampled[vi_col].interpolate(method='linear')
    # df_interpolated.reset_index(inplace = True, drop=False)
    df_interpolated = df_interpolated.reset_index()
    return df_interpolated

def apply_savgol_filter(vi_series, window_length=20, polyorder=2):
    """
    Applies a Savitzky-Golay filter to smooth the data.
    Window length must be an odd integer.
    """
    if len(vi_series) < window_length:
        return vi_series # Not enough data to filter
    return signal.savgol_filter(vi_series, window_length, polyorder)

def apply_spline_filter(vi_series,day_series, s=0.2, k=4):
    spline_coefficients = interpolate.splrep(day_series, vi_series, s=s, k=k)
    ndvi_spline = interpolate.splev(day_series, spline_coefficients)
    return ndvi_spline


def extract_phenology_metrics(df_smoothed, season_info, config):
    """
    Extracts phenology metrics (SOS) from a smoothed time-series using the
    'minimum NDVI at inflection' derivative approach.

    Args:
        df_smoothed (DataFrame): DataFrame with daily smoothed VI values.
        season_info (dict): Dictionary with 'start' and 'end' dates for the season.
        config (dict): Configuration dictionary with column names and parameters.

    Returns:
        DataFrame: A DataFrame with the calculated phenology metrics.
    """
    # Unpack configuration
    vi_col, date_col, delta_days, sow_date_survey, sow_dop_survey, init_date = config['vi_col'], config['date_col'], config['delta_days_sowing_sos'], config['sow_date_survey'],config['sow_dop_survey'], config['init_date']
    sensor = df_smoothed.sensor.unique()[0]
    fkey = df_smoothed.fkey.unique()[0]
    # --- 1. Find Max and Min NDVI within the Season ---
    start_date_phe, end_date_phe = season_info["start"], season_info["end"]
    
    t0_max = start_date_phe + pd.DateOffset(days=30) #max ndvi sohuld be at least 30 days after start_date (phenology)

    #grab the max from the same colum to be use in the sow date analysis.
    field_max = df_smoothed[(df_smoothed[date_col].dt.date > t0_max.date()) & (df_smoothed[date_col].dt.date < end_date_phe)].reset_index(drop = True)
    
    if field_max.empty or len(field_max[vi_col].unique()) < 2:
        return None

    df_max_vi = field_max.loc[field_max[vi_col].idxmax()]
    day_max = df_max_vi[date_col].date()
    
    # Define window for finding the minimum based on the day of the maximum and the start of the phenological cycle
    t1_min = pd.Timestamp(start_date_phe)
    t2_min = pd.Timestamp(day_max)
    
    # --- 2. Calculate Derivatives to Find Inflection Points (SOS) ---
    df_derivative = df_smoothed[(df_smoothed[date_col] >= t1_min) & (df_smoothed[date_col] <= t2_min)].reset_index(drop = True)
    if len(df_derivative) ==0:
        return None
        
    df_derivative.sort_values(by=date_col, inplace=True)
    df_derivative['first_dif_rec'] = np.where(df_derivative[vi_col].diff() > 0, 1, 0)
    df_derivative['second_dif_rec'] = np.where(df_derivative['first_dif_rec'].diff() > 0, 1, 0)

    inflection_points = df_derivative[df_derivative['second_dif_rec'] == 1]
    if inflection_points.empty:
        return None

    # --- 3. Apply the "Minimum" Approach ---
    # Find the inflection point with the lowest NDVI value.
    sos_point = inflection_points.loc[inflection_points[vi_col].idxmin()]
    sos_detected_date = sos_point[date_col]
    sos_detected_dop = (sos_detected_date - init_date).days
    
    # Calculate the estimated sowing date by subtracting the delta
    estimated_sowing_date = sos_detected_date - pd.Timedelta(days=delta_days)
    estimated_sowing_dop = (estimated_sowing_date - init_date).days
    # difference from Y-X
    Di = estimated_sowing_date-sow_date_survey
    # --- 4. Compile and Return Results ---
    result_df = pd.DataFrame([{
        'fkey': fkey,
        'sos_detected_date': sos_detected_date,
        'predicted_sowing_date': estimated_sowing_date,
        'sos_detected_dop': sos_detected_dop,
        'predicted_sowing_dop': estimated_sowing_dop,
        'sow_date_survey': sow_date_survey,
        'sow_dop_survey': sow_dop_survey,
        'Di_estimated_observed': Di,
        'approach': 'minimum', # Label the approach used
        'sensor': sensor
    }])
    
    return result_df




def create_phenology_plot(daily_vi, survey_date, sg_sow_date, spline_sow_date, field_id, sensor, output_dir):
    """
    Generates and saves a plot comparing raw, smoothed, and predicted phenology.

    Args:
        daily_vi (DataFrame): DataFrame containing daily NDVI and smoothed values.
        survey_date (datetime): The ground-truth sowing date.
        sg_sow_date (datetime): The predicted sowing date from the SG-smoothed data.
        spline_sow_date (datetime): The predicted sowing date from the spline-smoothed data.
        field_id (int): The ID of the field being plotted.
        sensor (str): The name of the sensor.
        output_dir (str): The directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 1. Plot the raw NDVI points (after interpolation)
    ax.plot(daily_vi['day'], daily_vi['NDVI'], 'o', color='gray', label='Daily Interpolated NDVI', markersize=3, alpha=0.6)
    
    # 2. Plot the smoothed lines
    ax.plot(daily_vi['day'], daily_vi['ndvi_sav'], '-', color='cornflowerblue', label='SG Smoothed', linewidth=2)
    ax.plot(daily_vi['day'], daily_vi['ndvi_spline'], '-', color='darkorange', label='Spline Smoothed', linewidth=2)
    
    # 3. Plot the vertical lines for sowing dates
    survey_ts = pd.to_datetime(survey_date)
    ax.axvline(x=survey_ts, color='black', linestyle='--', linewidth=2.5, label=f'Survey Sowing Date: {survey_ts.strftime("%Y-%m-%d")}')
    if sg_sow_date:
        ax.axvline(x=sg_sow_date, color='blue', linestyle=':', linewidth=2, label=f'Predicted (SG): {sg_sow_date.strftime("%Y-%m-%d")}')
    if spline_sow_date:
        ax.axvline(x=spline_sow_date, color='red', linestyle=':', linewidth=2, label=f'Predicted (Spline): {spline_sow_date.strftime("%Y-%m-%d")}')
        
    # 4. Customize the plot
    ax.set_title(f'NDVI Phenology Profile | Field ID: {field_id} | Sensor: {sensor}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NDVI', fontsize=12)
    ax.set_ylim(0, 1.0) # Set a standard y-axis range for NDVI
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    
    # Format the date axis
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # 5. Save and close the plot
    plot_filename = f"phenology_profile_{sensor}_{field_id}_smoothed.png"
    full_plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(full_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory