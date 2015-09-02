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
import sys

def parse_sounding(file_sound):

	col_names=get_var_names(file_sound)
	col_units=get_var_units(file_sound)

	''' read tabular file '''
	raw_sounding = pd.read_table(file_sound,skiprows=36,header=None)
	raw_sounding.drop(19 , axis=1, inplace=True)	
	raw_sounding.columns=col_names
	sounding=raw_sounding[['Height','TE','TD','RH','u','v','P','MR']]
	sounding.units={'Height':'m','TE':'K', 'TD':'K', 'RH':'%' ,'u':'m s-1','v':'m s-1','P':'hPa','MR':'g kg-1'}

	''' replace nan values '''
	nan_value = -32768.00
	sounding = sounding.applymap(lambda x: np.nan if x == nan_value else x)
	
	''' set index '''
	sounding = sounding.set_index('Height')

	''' QC  '''
	sounding= sounding.groupby(sounding.index).first()
	sounding.dropna(how='all',inplace=True)
	sounding.RH = sounding.RH.apply(lambda x: 100 if x>100 else x)
	u_nans = nan_fraction(sounding.u)
	v_nans = nan_fraction(sounding.v)
	if u_nans>0. or v_nans>0.:
		sounding.u.interpolate(method='linear',inplace=True)
		sounding.v.interpolate(method='linear',inplace=True)
	rh_nans = nan_fraction(sounding.RH)
	td_nans = nan_fraction(sounding.TD)
	mr_nans = nan_fraction(sounding.MR)
	if rh_nans<5. and td_nans>50. and mr_nans>50.:
		sat_mixr = tm.sat_mix_ratio(K= sounding.TE,hPa=sounding.P)
		mixr=(sounding.RH/100)*sat_mixr*1000
		sounding.loc[:,'MR']=mixr #[g kg-1]

	''' add potential temperature '''
	theta = tm.theta2(K=sounding.TE, hPa=sounding.P,mixing_ratio=sounding.MR/1000)	
	thetaeq = tm.theta_equiv2(K=sounding.TE, hPa=sounding.P,
										relh=sounding.RH,mixing_ratio=sounding.MR/1000)	
	sounding.loc[:,'theta'] = pd.Series(theta, index=sounding.index)	
	sounding.loc[:,'thetaeq'] = pd.Series(thetaeq,index=sounding.index)

	''' add Brunt-Vaisala frequency '''
	hgt=sounding.index.values
	bvf_dry= tm.bv_freq_dry(theta=sounding.theta, agl_m=hgt, depth_m=100,centered=True)
	bvf_moist= tm.bv_freq_moist(K=sounding.TE, hPa=sounding.P, mixing_ratio=sounding.MR/1000,
										agl_m=hgt, depth_m=100,centered=True)

	sounding = pd.merge(sounding,bvf_dry,left_index=True,right_index=True,how='outer')
	sounding = pd.merge(sounding,bvf_moist,left_index=True,right_index=True,how='outer')
	sounding.bvf_dry.interpolate(method='linear',inplace=True)
	sounding.bvf_moist.interpolate(method='linear',inplace=True)
	sounding.loc[sounding.MR.isnull(),'bvf_dry']=np.nan
	sounding.loc[sounding.MR.isnull(),'bvf_moist']=np.nan
	
	return sounding

def parse_surface(file_met,index_field,name_field,locelevation):

	dates_col=[0,1,2]
	dates_fmt='%Y %j %H%M'

	''' read the csv file '''
	dframe = pd.read_csv(file_met,header=None)

	''' parse date columns into a single date col '''
	raw_dates=dframe.ix[:,dates_col]
	raw_dates.columns=['Y','j','HHMM']
	raw_dates['HHMM'] = raw_dates['HHMM'].apply(lambda x:'{0:0>4}'.format(x))
	raw_dates=raw_dates.apply(lambda x: '%s %s %s' % (x['Y'],x['j'],x['HHMM']), axis=1)
	dates=raw_dates.apply(lambda x: datetime.strptime(x, dates_fmt))

	''' make meteo df, assign datetime index, and name columns '''
	meteo=dframe.ix[:,index_field]
	meteo.index=dates
	meteo.columns=name_field

	''' make field with hourly acum precip '''
	hour=pd.TimeGrouper('H')
	preciph = meteo.precip.groupby(hour).sum()
	meteo = meteo.join(preciph, how='outer', rsuffix='h')

	''' add thermodynamics '''
	theta = tm.theta1(C=meteo.temp,hPa=meteo.press)
	thetaeq = tm.theta_equiv1(C=meteo.temp,hPa=meteo.press)
	meteo.loc[:,'theta'] = pd.Series(theta,index=meteo.index)	
	meteo.loc[:,'thetaeq'] = pd.Series(thetaeq,index=meteo.index)	

	''' add sea level pressure '''
	Tv = tm.virtual_temperature(C=meteo.temp,mixing_ratio=meteo.mixr/1000.)
	slp = tm.sea_level_press(K=Tv+273.15, Pa=meteo.press*100, m=locelevation)
	meteo.loc[:,'sea_levp']=slp

	''' assign metadata (prototype, not really used) '''
	units = {'press':'mb', 'temp':'C', 'rh':'%', 'wspd':'m s-1', 'wdir':'deg', 'precip':'mm', 'mixr': 'g kg-1'}
	agl = {'press':'NaN', 'temp':'10 m', 'rh':'10 m', 'wspd':'NaN', 'wdir':'NaN', 'precip':'NaN', 'mixr': 'NaN'}
	for n in name_field:
		meteo[n].units=units[n]
		meteo[n].agl=agl[n]
		meteo[n].nan=-9999.999
		meteo[n].sampling_freq='1 minute'	

	return meteo

def parse_windprof(windprof_file,mode):

	if mode == 'fine':
		raw = pd.read_table(windprof_file,skiprows=10,skipfooter=50,engine='python',delimiter='\s*')
	elif mode == 'coarse':
		raw = pd.read_table(windprof_file,skiprows=59,skipfooter=1,engine='python',delimiter='\s*')

	''' get timestamp '''
	raw_timestamp = pd.read_table(windprof_file,skiprows=4,skipfooter=94,engine='python')
	raw_timestamp = raw_timestamp.columns
	date_fmt = '%y %m %d %H %M %S'
	timestamp = datetime.strptime(raw_timestamp[0][:-1].strip(),date_fmt)

	''' replace nan values '''
	nan_value = 999999
	wp = raw.applymap(lambda x: np.nan if x == nan_value else x)	
	wp.timestamp = timestamp

	return wp



""" 
	supporting functions
"""
def get_var_names(file_sound):

	names=[]
	with open(file_sound,'r') as f:
		for line in itertools.islice(f, 15, 34):
			foo = line.split()
			if foo[0]=='T':
				'''pandas has a T property so
				needs to be replaced'''
				names.append('TE')
			else:
				names.append(foo[0])
	return names

def get_var_units(file_sound):

	units=[]
	with open(file_sound,'r') as f:
		for line in itertools.islice(f, 15, 34):
			foo = line.split()
			units.append(foo[1])
	return units	

def nan_fraction(series):

	nans = float(series.isnull().sum())
	total = float(len(series.index))
	return (nans/total)*100		