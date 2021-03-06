"""
    Module for parsing meteorological observations
    from text files

    Raul Valenzuela
    August, 2015
"""


from datetime import datetime
import pandas as pd
import numpy as np
import Thermodyn as tm
import itertools
import os
# from scipy.interpolate import UnivariateSpline
from scipy.interpolate import Rbf
from os import path


def parse_sounding(file_sound):

    col_names = get_var_names(file_sound)
    # col_units = get_var_units(file_sound)

    ''' read tabular file '''
    raw_sounding = pd.read_table(file_sound, skiprows=36, header=None)
    raw_sounding.drop(19, axis=1, inplace=True)
    raw_sounding.columns = col_names
    sounding = raw_sounding[
        ['Height', 'TE', 'TD', 'RH', 'u', 'v', 'P', 'MR', 'DD']]
    sounding.units = {'Height': 'm', 'TE': 'K', 'TD': 'K',
                      'RH': '%', 'u': 'm s-1', 'v': 'm s-1',
                      'P': 'hPa', 'MR': 'g kg-1'}

    ''' replace nan values '''
    nan_value = -32768.00
    sounding = sounding.applymap(lambda x: np.nan if x == nan_value else x)

    ''' QC soundings that include descening trayectories;
        criteria is 3 consecutive values descending
    '''
    sign = np.sign(np.diff(sounding['Height']))
    rep = find_repeats(sign.tolist(), -1, 3)
    try:
        lastgood = np.where(rep)[0][0]-1
        sounding = sounding.ix[0:lastgood]
    except IndexError:
        ''' all good'''
        pass

    ''' set index '''
    sounding = sounding.set_index('Height')

    ''' QC  '''
    sounding = sounding.groupby(sounding.index).first()
    sounding.dropna(how='all', inplace=True)
    sounding.RH = sounding.RH.apply(lambda x: 100 if x > 100 else x)
    u_nans = nan_fraction(sounding.u)
    v_nans = nan_fraction(sounding.v)
    if u_nans > 0. or v_nans > 0.:
        sounding.u.interpolate(method='linear', inplace=True)
        sounding.v.interpolate(method='linear', inplace=True)
    rh_nans = nan_fraction(sounding.RH)
    td_nans = nan_fraction(sounding.TD)
    mr_nans = nan_fraction(sounding.MR)
    if rh_nans < 5. and td_nans > 50. and mr_nans > 50.:
        sat_mixr = tm.sat_mix_ratio(K=sounding.TE, hPa=sounding.P)
        mixr = (sounding.RH/100)*sat_mixr*1000
        sounding.loc[:, 'MR'] = mixr  # [g kg-1]

    ''' add potential temperature '''
    theta = tm.theta2(
        K=sounding.TE, hPa=sounding.P, mixing_ratio=sounding.MR/1000)
    thetaeq = tm.theta_equiv2(K=sounding.TE, hPa=sounding.P,
                              relh=sounding.RH, mixing_ratio=sounding.MR/1000)
    sounding.loc[:, 'theta'] = pd.Series(theta, index=sounding.index)
    sounding.loc[:, 'thetaeq'] = pd.Series(thetaeq, index=sounding.index)

    ''' add Brunt-Vaisala frequency '''
    hgt = sounding.index.values
    bvf_dry = tm.bv_freq_dry(theta=sounding.theta, agl_m=hgt,
                             depth_m=100, centered=True)
    bvf_moist = tm.bv_freq_moist(K=sounding.TE, hPa=sounding.P,
                                 mixing_ratio=sounding.MR/1000,
                                 agl_m=hgt, depth_m=100, centered=True)

    sounding = pd.merge(
        sounding, bvf_dry, left_index=True, right_index=True, how='outer')
    sounding = pd.merge(
        sounding, bvf_moist, left_index=True, right_index=True, how='outer')

    ''' interpolate between layer-averaged values '''
    sounding.bvf_dry.interpolate(method='linear', inplace=True)
    sounding.bvf_moist.interpolate(method='linear', inplace=True)
    sounding.loc[sounding.MR.isnull(), 'bvf_dry'] = np.nan
    sounding.loc[sounding.MR.isnull(), 'bvf_moist'] = np.nan

    '''     NOTE: if sounding hgt jumps from 12 to 53m then 12m
        bvf values are NaN since there are no data between 12
        and 53m to calculate a layer-based value.
    '''

    return sounding


def parse_sounding2(file_sound):
    '''
    This version make a resample of vertical gates
    and uses make_layer2 in bvf calculations
    (Thermodyn module)
    '''

    col_names = get_var_names(file_sound)
    # col_units = get_var_units(file_sound)

    ''' read tabular file '''
    raw_sounding = pd.read_table(file_sound, skiprows=36, header=None)
    raw_sounding.drop(19, axis=1, inplace=True)
    raw_sounding.columns = col_names
    sounding = raw_sounding[
        ['Height', 'TE', 'TD', 'RH', 'u', 'v', 'P', 'MR', 'DD']]
    sounding.units = {'Height': 'm', 'TE': 'K', 'TD': 'K',
                      'RH': '%', 'u': 'm s-1', 'v': 'm s-1',
                      'P': 'hPa', 'MR': 'g kg-1'}

    ''' replace nan values '''
    nan_value = -32768.00
    sounding = sounding.applymap(lambda x: np.nan if x == nan_value else x)

    ''' QC soundings that include descening trayectories;
        criteria is 3 consecutive values descending
    '''
    sign = np.sign(np.diff(sounding['Height']))
    rep = find_repeats(sign.tolist(), -1, 3)
    try:
        lastgood = np.where(rep)[0][0]-1
        sounding = sounding.ix[0:lastgood]
    except IndexError:
        ''' all good'''
        pass
#    print sounding
    ''' resample to constant height values '''
    resamp = 2  # [m]
    sounding = sounding.set_index('Height')
    hgt = sounding.index.values[-1]
    resample = np.arange(10, hgt, resamp)
    soundres = sounding.loc[resample]
    soundres.iloc[0] = sounding.iloc[0]  # copy first value
    soundres.iloc[-1] = sounding.iloc[-1]  # copy last value
    soundres.interpolate(
        method='linear', inplace=True, limit=80, limit_direction='backward')
    soundres = soundres.reset_index().drop_duplicates(
        subset='Height', keep='first')
    soundres = soundres.set_index('Height')

    ''' QC  '''
    soundres.RH = soundres.RH.apply(lambda x: 100 if x > 100 else x)
    rh_nans = nan_fraction(soundres.RH)  # [%]
    td_nans = nan_fraction(soundres.TD)  # [%]
    mr_nans = nan_fraction(soundres.MR)  # [%]
    if rh_nans < 5. and td_nans > 50. and mr_nans > 50.:
        sat_mixr = tm.sat_mix_ratio(K=soundres.TE, hPa=soundres.P)
        mixr = (soundres.RH/100.)*sat_mixr*1000
        soundres.loc[:, 'MR'] = mixr  # [g kg-1]

    ''' add potential temperature '''
    theta = tm.theta2(K=soundres.TE, hPa=soundres.P, mixing_ratio=soundres.MR/1000)
    thetaeq = tm.theta_equiv2(K=soundres.TE, hPa=soundres.P,
                              relh=soundres.RH, mixing_ratio=soundres.MR/1000)
    soundres.loc[:, 'theta'] = pd.Series(theta, index=soundres.index)
    soundres.loc[:, 'thetaeq'] = pd.Series(thetaeq, index=soundres.index)

    ''' add Brunt-Vaisala frequency '''
    hgt = soundres.index.values
    depth = 100  # [m]
    bvf_dry = tm.bv_freq_dry(theta=soundres.theta, agl_m=hgt, depth_m=depth)
    bvf_moist = tm.bv_freq_moist(K=soundres.TE,
                                 hPa=soundres.P,
                                 mixing_ratio=soundres.MR/1000,
                                 agl_m=hgt, depth_m=depth)

    soundres = pd.merge(
        soundres, bvf_dry, left_index=True, right_index=True, how='outer')
    soundres = pd.merge(
        soundres, bvf_moist, left_index=True, right_index=True, how='outer')

    ''' interpolate between layer-averaged values '''
    soundres.bvf_dry.interpolate(method='cubic', inplace=True)
    soundres.bvf_moist.interpolate(method='cubic', inplace=True)

    return soundres


def parse_surface(file_met):

    file_extension = file_met.split('/')[-1][-3:]

    if file_extension == 'met':
        meteo =  parse_surface_met(file_met)

    else:
        meteo = parse_surface_surf(file_met)

    return meteo

def parse_surface_surf(file_met):

    from scipy import stats

    ''' 
        Process surface meteorological files 
        with .surf extension downloaded from
        http://www.esrl.noaa.gov/psd/data/obs/datadisplay/
        
        File does not have mixing ratio.

        Output dataframe assumes:
        - Each file is a 24h period observation
        - Frequency of observation is constant
    '''

    station_id = file_met.split('/')[-1][:3]

    index_field = {'bby': (4, 5, 6, 8, 9, 14)}
    index_field = index_field[station_id]

    name_field = ['press', 'temp', 'rh', 'wspd', 'wdir', 'precip']

    elev = {'bby': 15, 'czc': 462, 'frs': 45}
    locelevation = elev[station_id]

    ''' read the csv file '''
    raw_dframe = pd.read_csv(file_met, header=None)

    ''' parse date columns into a single date col '''
    dates_col = [1, 2, 3]
    raw_dates = raw_dframe.ix[:, dates_col]
    raw_dates.columns = ['Y', 'j', 'HHMM']
    raw_dates['HHMM'] = raw_dates['HHMM'].apply(lambda x: '{0:0>4}'.format(x))
    raw_dates = raw_dates.apply(
                                lambda x: '%s %s %s' % (x['Y'], x['j'], x['HHMM']),
                                axis=1)
    dates_fmt = '%Y %j %H%M'
    dates = raw_dates.apply(lambda x: datetime.strptime(x, dates_fmt))

    ''' make meteo df, assign datetime index, and name columns '''
    meteo = raw_dframe.ix[:, index_field]
    meteo.index = dates
    meteo.columns = name_field

    ''' create a dataframe with regular and continuous 24h period time index
        (this works as a QC for time gaps)
    '''
    nano = 1000000000
    time_diff = np.diff(meteo.index)
    sample_freq = (stats.mode(time_diff)[0][0]/nano).astype(int)  # [seconds]
    fidx = meteo.index[0]  # (start date for dataframe)
    fidx_str = pd.to_datetime(
        fidx.year*10000 + fidx.month*100 + fidx.day, format='%Y%m%d')
    periods = (24*60*60)/sample_freq
    ts = pd.date_range(fidx_str, periods=periods, freq=str(sample_freq)+'s')
    nanarray = np.empty((periods, 1))
    nanarray[:] = np.nan
    df = pd.DataFrame(nanarray, index=ts)
    meteo = df.join(meteo, how='outer')
    meteo.drop(0, axis=1, inplace=True)

    ''' correct pressure by offset '''
    meteo.press = meteo.press + 400

    ''' make field with hourly acum precip '''
    hour = pd.TimeGrouper('H')
    preciph = meteo.precip.groupby(hour).sum()
    meteo = meteo.join(preciph, how='outer', rsuffix='h')

    ''' add thermodynamics '''
    theta = tm.theta1(C=meteo.temp, hPa=meteo.press)
    thetaeq = tm.theta_equiv1(C=meteo.temp, hPa=meteo.press)
    meteo.loc[:, 'theta'] = pd.Series(theta, index=meteo.index)
    meteo.loc[:, 'thetaeq'] = pd.Series(thetaeq, index=meteo.index)

    ''' add sea level pressure '''
    # Tv = tm.virtual_temperature(C=meteo.temp, mixing_ratio=meteo.mixr/1000.)
    # slp = tm.sea_level_press(K=Tv+273.15, Pa=meteo.press*100, m=locelevation)
    # meteo.loc[:, 'sea_levp'] = slp

    ''' assign metadata (prototype, not really used) '''
    units = {'press': 'mb', 'temp': 'C', 'rh': '%', 'wspd': 'm s-1',
             'wdir': 'deg', 'precip': 'mm'}
    agl = {'press': 'NaN', 'temp': '10 m', 'rh': '10 m',
           'wspd': 'NaN', 'wdir': 'NaN', 'precip': 'NaN'}
    for n in name_field:
        meteo[n].units = units[n]
        meteo[n].agl = agl[n]
        meteo[n].nan = -9999.999
        meteo[n].sampling_freq = '2 minute'

    return meteo

def parse_surface_met(file_met, index_field=None,
                  name_field=None, locelevation=None):

    from scipy import stats

    ''' 
        Process surface meteorological files 
        with .met extension

        Output dataframe assumes:
        - Each file is a 24h period observation
        - Frequency of observation is constant
    '''

    casename = file_met.split('/')[-2]
    station_id = file_met.split('/')[-1][:3]
    usr_case = str(int(casename[-2:]))

    if usr_case in ['1', '2']:
        index_field = {'bby': [3, 4, 10, 5, 6, 11, 13],
                       'czc': [3, 4, 10, 5, 6, 11, 13]}
    elif usr_case in ['3', '4', '5', '6', '7']:
        index_field = {'bby': [3, 5, 8, 10, 12, 17, 26],
                       'czc': [3, 4, 5, 6, 8, 13, 22]}
    else:
        index_field = {'bby': [3, 4, 5, 6, 8, 13, 15],
                       'czc': [3, 4, 5, 6, 8, 13, 15],
                       'frs': [3, 4, 5, 6, 8, 13, 15]}

    elev = {'bby': 15, 'czc': 462, 'frs': 45}
    name_field = ['press', 'temp', 'rh', 'wspd', 'wdir', 'precip', 'mixr']

    index_field = index_field[station_id]
    locelevation = elev[station_id]

    ''' read the csv file '''
    raw_dframe = pd.read_csv(file_met, header=None)

    ''' parse date columns into a single date col '''
    dates_col = [0, 1, 2]
    raw_dates = raw_dframe.ix[:, dates_col]
    raw_dates.columns = ['Y', 'j', 'HHMM']
    raw_dates['HHMM'] = raw_dates['HHMM'].apply(lambda x: '{0:0>4}'.format(x))
    raw_dates = raw_dates.apply(
        lambda x: '%s %s %s' % (x['Y'], x['j'], x['HHMM']), axis=1)
    dates_fmt = '%Y %j %H%M'
    dates = raw_dates.apply(lambda x: datetime.strptime(x, dates_fmt))

    ''' make meteo df, assign datetime index, and name columns '''
    meteo = raw_dframe.ix[:, index_field]
    meteo.index = dates
    meteo.columns = name_field

    ''' create a dataframe with regular and continuous 24h period time index
        (this works as a QC for time gaps)
    '''
    nano = 1000000000
    time_diff = np.diff(meteo.index)
    sample_freq = (stats.mode(time_diff)[0][0]/nano).astype(int)  # [seconds]
    fidx = meteo.index[0]  # (start date for dataframe)
    fidx_str = pd.to_datetime(
        fidx.year*10000 + fidx.month*100 + fidx.day, format='%Y%m%d')
    periods = (24*60*60)/sample_freq
    ts = pd.date_range(fidx_str, periods=periods, freq=str(sample_freq)+'s')
    nanarray = np.empty((periods, 1))
    nanarray[:] = np.nan
    df = pd.DataFrame(nanarray, index=ts)
    meteo = df.join(meteo, how='outer')
    meteo.drop(0, axis=1, inplace=True)

    ''' make field with hourly acum precip '''
    hour = pd.TimeGrouper('H')
    preciph = meteo.precip.groupby(hour).sum()
    meteo = meteo.join(preciph, how='outer', rsuffix='h')

    ''' add thermodynamics '''
    theta = tm.theta1(C=meteo.temp, hPa=meteo.press)
    thetaeq = tm.theta_equiv1(C=meteo.temp, hPa=meteo.press)
    meteo.loc[:, 'theta'] = pd.Series(theta, index=meteo.index)
    meteo.loc[:, 'thetaeq'] = pd.Series(thetaeq, index=meteo.index)

    ''' add sea level pressure '''
    Tv = tm.virtual_temperature(C=meteo.temp, mixing_ratio=meteo.mixr/1000.)
    slp = tm.sea_level_press(K=Tv+273.15, Pa=meteo.press*100, m=locelevation)
    meteo.loc[:, 'sea_levp'] = slp

    ''' assign metadata (prototype, not really used) '''
    units = {'press': 'mb', 'temp': 'C', 'rh': '%', 'wspd': 'm s-1',
             'wdir': 'deg', 'precip': 'mm', 'mixr': 'g kg-1'}
    agl = {'press': 'NaN', 'temp': '10 m', 'rh': '10 m',
           'wspd': 'NaN', 'wdir': 'NaN', 'precip': 'NaN', 'mixr': 'NaN'}
    for n in name_field:
        meteo[n].units = units[n]
        meteo[n].agl = agl[n]
        meteo[n].nan = -9999.999
        meteo[n].sampling_freq = '2 minute'

    return meteo


def parse_windprof(windprof_file, mode):
    """ NOAA HHw files are one per hour
    """

    ' fix for files with coarse resolution starting at row 61'
    year = path.basename(windprof_file)[3:5]
    if year in ['98']:
        skfoot = 50
        skrow = 51
    elif year in ['03']:
        skfoot = 51
        skrow = 60
    else:
        skfoot = 50
        skrow = 59

    if mode == 'fine':
        raw = pd.read_table(
            windprof_file, skiprows=10, skipfooter=skfoot,
            engine='python', delimiter='\s*')
    elif mode == 'coarse':
        raw = pd.read_table(
            windprof_file, skiprows=skrow, skipfooter=1,
            engine='python', delimiter='\s*')

    ''' get timestamp '''
    raw_timestamp = pd.read_table(
        windprof_file, skiprows=4, skipfooter=94, engine='python')
    raw_timestamp = raw_timestamp.columns
    date_fmt = '%y %m %d %H %M %S'
    timestamp = datetime.strptime(raw_timestamp[0][:-1].strip(), date_fmt)

    ''' replace nan values '''
    nan_value = 999999
    wp = raw.applymap(lambda x: np.nan if x == nan_value else x)
    wp.timestamp = timestamp

    return wp


def parse_rass(rass_file):
    ''' input file is cns rev 4.1 and rev 5.0'''
    from datetime import datetime, timedelta

    with open(rass_file) as f:
        lines = f.readlines()

    hgt = []
    t = []
    tc = []
    timestamp = []
    is_chunk = False
    first_chunk = True
    for line in lines:
        split = line[:-2].split()
        is_chunk = (len(split) == 10)
        date_line = (len(split) == 7)

        if is_chunk:
            try:
                hgt.append(float(line[:-2].split()[0]))
                t.append(float(line[:-2].split()[1]))
                tc.append(float(line[:-2].split()[2]))
            except ValueError:
                'found header (rev5.0 files)'
                continue
        elif len(hgt) > 0:
            if first_chunk:
                HGT = np.array(hgt)
                T = np.array(t)
                Tc = np.array(tc)
                first_chunk = False
            else:
                HGT = np.vstack((HGT, hgt))
                T = np.vstack((T, t))
                Tc = np.vstack((Tc, tc))
            hgt = []
            t = []
            tc = []
        elif date_line:
            fmt = '  %y %m %d %H %M %S   0'
            ts = datetime.strptime(line[:-2], fmt)
            timestamp.append(ts)

    nan_value = 9999.
    T[T == nan_value] = np.nan
    Tc[Tc == nan_value] = np.nan

    ''' QC outliers '''
    mu = np.nanmean(T)
    sigma = np.nanstd(T)
    bot = mu-3*sigma
    top = mu+3*sigma
    T[T <= bot] = np.nan
    T[T >= top] = np.nan

    if np.mod(len(timestamp), 24) > 0:
        ''' there is one or more time gaps
        assuming 24 hr file '''
        hg, ti = T.shape
        newDim = (24, hg)
        newT = np.zeros(newDim) + nan_value
        newTc = np.zeros(newDim) + nan_value
        newHGT = np.zeros(newDim) + nan_value
        newTs = [nan_value]*23

        for i in range(ti):
            h = timestamp[i].hour
            newT[h, :] = T[i, :]
            newTc[h, :] = Tc[i, :]
            newHGT[h, :] = HGT[i, :]
            newTs.insert(h, timestamp[i])

        newT[newT == nan_value] = np.nan
        newTc[newTc == nan_value] = np.nan
        newHGT[newHGT == nan_value] = np.nan
        newTs = np.array(newTs[:24])
        ts_nan = np.where(newTs == nan_value)[0]
        ''' this works only for one gap (1 or more hrs)'''
        filldates = [
            newTs[ts_nan[0]-1]+timedelta(hours=n+1)
            for n in range(len(ts_nan))]
        newTs[ts_nan] = filldates
        print('time gap resolved (check if there is more than 1 gap)')
        return newT.T, newTc.T, newHGT.T, newTs
    else:
        # T[T == nan_value] = np.nan
        # Tc[Tc == nan_value] = np.nan
        return T.T, Tc.T, HGT.T, timestamp


def parse_buoy(buoy_file, start=None, end=None):

    # obs_per_hour = 6  # obs every 10 minutes
    # hrs_per_day = 24
    # ndays = 23
    raw = pd.read_table(buoy_file,
                        parse_dates=[[0, 1, 2, 3, 4]],
                        engine='python',
                        delimiter='\s*')
    # raw = raw.rename(columns = {'YYYY_MM_DD_hh_mm':'timestamp'})
    raw['Datetime'] = pd.to_datetime(
        raw['YYYY_MM_DD_hh_mm'], format='%Y %m %d %H %M')
    raw = raw.set_index(raw['Datetime'])
    raw.drop('YYYY_MM_DD_hh_mm', axis=1, inplace=True)
    raw.drop('Datetime', axis=1, inplace=True)

    ''' replace nan values '''
    nan_value = 999
    bu = raw.applymap(lambda x: np.nan if x == nan_value else x)

    return bu


def parse_mesowest_excel(mesowest_file):
    """ URL: http://mesowest.utah.edu/
    """
    filename = os.path.basename(mesowest_file)
    station_id = 'ID = ' + filename[:4]
    raw = pd.read_excel(mesowest_file, skip_footer=3)
    # raw['Datetime'] = pd.to_datetime(raw[station_id],
    # format='%m-%d-%Y %H:%M GMT')
    raw.index = pd.to_datetime(raw[station_id], format='%m-%d-%Y %H:%M GMT')
    # meso = raw.set_index(raw['Datetime'])
    # meso.drop(station_id, axis=1, inplace=True)
    # meso.drop('Datetime', axis=1, inplace=True)
    meso = raw.drop(station_id, axis=1)

    return meso


def parse_gps_iwv(gps_file):

    raw = pd.read_table(gps_file, engine='python', delimiter='\s*')
    # raw = pd.read_table(gps_file, parse_dates=[[1,2,3]],engine='python',delimiter='\s*')
    raw.drop(raw.index[0], inplace=True)  # unnecesary row with some units
    year = raw['YEAR']
    julian_day = raw['JJJ.dddd'].apply(lambda x: str(int(float(x))))
    time = raw['HH:MM:SS']
    timestamp = year+'-'+julian_day+'-'+time
    raw.index = pd.to_datetime(timestamp, format='%Y-%j-%H:%M:%S')
    raw.drop(['SITE', 'YEAR', 'JJJ.dddd', 'HH:MM:SS'], axis=1, inplace=True)

    return raw


def parse_acft_sounding(flight_level_file, req_ini, req_end, return_interp):

    raw = pd.read_table(flight_level_file, engine='python', delimiter='\s*')

    ini_raw = raw['TIME'].iloc[0]
    end_raw = raw['TIME'].iloc[-1]
    timestamp = pd.date_range(
        '2001-01-23 '+ini_raw, '2001-01-24 '+end_raw, freq='s')
    raw.index = timestamp
    raw.drop(['TIME'], axis=1, inplace=True)

    ''' req_ini and req_end are python datatime variables '''
    st = raw.index.searchsorted(req_ini)
    en = raw.index.searchsorted(req_end)
    data = raw[['PRES_ALT', 'AIR_PRESS', 'AIR_TEMP', 'DEW_POINT',
                'WIND_SPD', 'WIND_DIR', 'LAT', 'LON']].ix[st:en]

    x = data['PRES_ALT'].values
    xnew = np.linspace(min(x), max(x), x.size)

    if x[0] > x[-1]:
        descening = True
        data = data.iloc[::-1]

    if return_interp:
        ''' flight sounding might include constant height levels
        or descening trayectories, so we interpolate data to a
        common vertical grid '''
        data2 = data.drop_duplicates(subset='PRES_ALT')
        x2 = data2['PRES_ALT'].values

        ''' interpolate each field '''
        for n in ['AIR_PRESS', 'AIR_TEMP', 'DEW_POINT', 'WIND_SPD', 'WIND_DIR']:
            y = data2[n].values
            # spl = UnivariateSpline(x2,y,k=4)
            # data[n]= spl(xnew)
            rbf = Rbf(x2, y, smooth=1.0)
            ynew = rbf(xnew)
            data[n] = ynew

        '''' update values '''
        data['PRES_ALT'] = xnew

        ''' set index '''
        data = data.set_index('PRES_ALT')

        ''' compute thermo '''
        hgt = data.index.values
        Tc = data['AIR_TEMP']
        Tk = Tc+273.15
        press = data['AIR_PRESS']
        dewp = data['DEW_POINT']
        Rh = tm.relative_humidity(C=Tc, Dewp=dewp)
        Sat_mixr = tm.sat_mix_ratio(C=Tc, hPa=press)
        Mr = Rh*Sat_mixr/100.
        theta = tm.theta2(K=Tk, hPa=press, mixing_ratio=Mr)
        thetaeq = tm.theta_equiv2(K=Tk, hPa=press, relh=Rh, mixing_ratio=Mr)
        data['theta'] = theta.values
        data['thetaeq'] = thetaeq.values
        bvf_dry = tm.bv_freq_dry(
            theta=theta, agl_m=hgt, depth_m=100, centered=True)
        bvf_moist = tm.bv_freq_moist(K=Tk, hPa=press, mixing_ratio=Mr,
                                     agl_m=hgt, depth_m=100, centered=True)

        data = pd.merge(
            data, bvf_dry, left_index=True, right_index=True, how='outer')
        data = pd.merge(
            data, bvf_moist, left_index=True, right_index=True, how='outer')
        data.bvf_dry.interpolate(method='linear', inplace=True)
        data.bvf_moist.interpolate(method='linear', inplace=True)

        ''' drop last incorrect decreasing (increasing) altitude (pressure) value '''
        data2 = data.ix[:xnew[-2]]

        ''' return dataframe '''
        return data2
    else:
        hgt = data['PRES_ALT']
        Tc = data['AIR_TEMP']
        Tk = Tc+273.15
        press = data['AIR_PRESS']
        dewp = data['DEW_POINT']
        Rh = tm.relative_humidity(C=Tc, Dewp=dewp)
        Sat_mixr = tm.sat_mix_ratio(C=Tc, hPa=press)
        Mr = Rh*Sat_mixr/100.
        theta = tm.theta2(K=Tk, hPa=press, mixing_ratio=Mr)
        thetaeq = tm.theta_equiv2(K=Tk, hPa=press, relh=Rh, mixing_ratio=Mr)
        data['theta'] = theta.values
        data['thetaeq'] = thetaeq.values
        data['bvf_dry'] = np.nan
        data['bvf_moist'] = np.nan

        return data


"""
    supporting functions
"""


def get_var_names(file_sound):

    names = []
    with open(file_sound, 'r') as f:
        for line in itertools.islice(f, 15, 34):
            foo = line.split()
            if foo[0] == 'T':
                '''pandas has a T property so
                needs to be replaced'''
                names.append('TE')
            else:
                names.append(foo[0])
    return names


def get_var_units(file_sound):

    units = []
    with open(file_sound, 'r') as f:
        for line in itertools.islice(f, 15, 34):
            foo = line.split()
            units.append(foo[1])
    return units


def nan_fraction(series):

    nans = float(series.isnull().sum())
    total = float(len(series.index))
    return (nans/total)*100


def find_repeats(L, required_number, num_repeats, stop_after_match=False):
    ''' John La Rooy solution in stackoverflow '''
    idx = 0
    while idx < len(L):
        if [required_number]*num_repeats == L[idx:idx+num_repeats]:
            L[idx:idx+num_repeats] = [True]*num_repeats
            idx += num_repeats
            if stop_after_match:
                break
        else:
            L[idx] = False
            idx += 1
    return L
