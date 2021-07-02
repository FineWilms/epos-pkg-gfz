import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
import fortranformat as ff
from itertools import zip_longest
from scipy.signal import argrelextrema, argrelmin
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from ast import literal_eval as make_tuple
import pyshtools
from scipy.io import loadmat
from pathlib import Path
from scipy.special import lpmn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import copy
import cartopy.feature as cfeature

"""
author: J.M. Wilms 
contact: jowilms@gfz-potsdam.de
description: A scipt containing tools to post-process the orbits, obtained from a forward simulation (and also recovery), from epos-oc.
 """


def create_element_lines(ffp):
	#input: Unformatted file that contains orbit elements needed to start each of the runs for GRACE-FO simulation
	#output: Orbit elements that can be used as input for prepare_EPOSIN_4_orbit_integration.sh (located at
	#/GFZ/project/f_grace/NGGM_SIM/SIM_FORWARD )
	with open(ffp) as f:
		lines = f.read().splitlines()
		splits = [i for i in lines if i.startswith('TPM')]
	print(splits)
	n = 2  # group size
	m = 1  # overlap size
	splits_grouped = [splits[i:i + n] for i in range(0, len(splits)+1, n - m)]


	# # print(lines)
	# split = [i for i in lines if i.startswith('PP')]
	for i in splits_grouped:
		if len(i) > 1:
			start = i[0]
			end = i[1]
			out = '%s_ELEMENT_lines.txt' % (start.strip())
		with open(ffp) as infile, open(out, 'w') as outfile:
			copy = False
			titlewritten0 = False
			titlewritten1 = False
			firsttime6 = False
			linesread = 0
			outfile.write("\n")

			for line in infile:
				if line.strip() == start.strip():
					copy = True
					continue
				elif line.strip() == end.strip():
					copy = False
					continue
				elif copy:
					linesread += 1

					if not titlewritten0:
						outfile.write('        --- Begin initial elements GRACE-C\n')
						titlewritten0 = True
					if line.startswith(
							'ELEMENT') and titlewritten0:  # if line starts with ELEMENT and the first title has been written
						val = list(filter(None, line.strip().split(' ')))[0:-3]
						format_float = ff.FortranRecordWriter('(E19.13)')
						val5 = str(format_float.write([np.float(val[5])]))
						val6 = str(format_float.write([np.float(val[6])]))

						val5 = val5.replace('E', 'e')
						val6 = val6.replace('E', 'e')


						if val[7] == '0201201': val[7] = '1804701'
						if val[7] == '0201202': val[7] = '1804702'
						str_new2 = ('%7.7s' '%4.3s' '%2.1i' '%2.1i' '%2.1i' '%20.19s' '%20.19s' '%8.7s') % (val[0], val[1], int(val[2]), int(val[3]), int(val[4]), val5, val6, val[7])


						# outfile.write("\n")
						if int(val[2]) < 6:
							outfile.write(str_new2)
							outfile.write("\n")

						if int(val[
								   2]) == 6 and not titlewritten1:  # if element six has been reached and no 'end1' has been written yet:
							if not firsttime6:
								titlewritten1 = True
								# titlewritten2 = True
								outfile.write(str_new2)
								outfile.write("\n")
								outfile.write('        --- End initial elements GRACE-C\n\n')
								outfile.write('        --- Begin initial elements GRACE-D\n')

						if int(val[2]) == 6:
							print(linesread)
							if linesread > 7:
								outfile.write(str_new2)
								outfile.write("\n")

			outfile.write('        --- End initial elements GRACE-D')
			outfile.write("\n")
			outfile.write('\n')
		outfile.close()
		infile.close()


def files(path):
	#input: path to a directory
	#output: files within the directory (omitting nested directories)
	for file in os.listdir(path):
		if os.path.isfile(os.path.join(path, file)):
			yield file


def create_case_directories(fp, fp_out):
	#function to prepare the case directories for each of the simulations specified for the GRACE-FO project.
	#It will
	element_files = []
	# current_dir = os.path.dirname(__file__)
	for file in files(fp):
		element_files.append(file)

	IDs = ['PP.1', 'PP.2']
	altitudes = [490, 490]
	extens = [0, 0]
	angles = [89, 89]
	seperations = [200, 100]
	repeats = [30, 30]
	simdirs = ['FD', 'FD']
	df = pd.DataFrame(columns=['id', 'altitude', 'extens', 'seperation', 'repeatvals', 'sim_direction'])
	df['id'] = IDs
	df['altitude'] = altitudes
	df['angles'] = angles
	df['extens'] = extens
	df['seperation'] = seperations
	df['repeatvals'] = repeats
	df['sim_direction'] = simdirs
	df.set_index('id', inplace=True)

	for idx in df.index:
		

		dirname = '%s_%i_%i_%i_%i_%id_%s' % (idx, df.loc[idx].altitude,
											 df.loc[idx].angles,
											 df.loc[idx].z,
											 df.loc[idx].seperation,
											 df.loc[idx].repeatvals,
											 df.loc[idx].sim_direction
											 )
		if not os.path.exists(dirname):
			os.mkdir(dirname)
		ef = [f for f in element_files if f.startswith(idx)][0]

		dst = os.path.abspath(fp, dirname, 'ELEMENT_lines')

		src = os.path.abspath(os.path.join(os.path.dirname(__file__), ef))

		copyfile(src, dst)
def serial_date_to_string(srl_no):
    new_date = datetime.datetime(2000,1,1) + datetime.timedelta(srl_no+1)
    return new_date.strftime("%Y-%m-%d")


def cart_2_kep_matrix(R, V, mu, Re):

	# step1
	h_bar = np.cross(R, V)
	h = np.linalg.norm(h_bar, axis=1)
	# step2
	r = np.linalg.norm(R, axis=1)
	v = np.linalg.norm(V, axis=1)
	# step3
	E = 0.5 * (v ** 2) - mu / r
	# step4
	a = -mu / (2 * E)

	return (a-Re)/1000.0

def cart_2_kep(r_vec, v_vec, t, mu, Re):

	# step1
	h_bar = np.cross(r_vec, v_vec)
	h = np.linalg.norm(h_bar)
	# step2
	r = np.linalg.norm(r_vec)
	v = np.linalg.norm(v_vec)
	# step3
	E = 0.5 * (v ** 2) - mu / r
	# step4
	a = -mu / (2 * E)
	# step5
	e = np.sqrt(1 - (h ** 2) / (a * mu))
	# step6
	i = np.arccos(h_bar[2] / h)
	# step7
	omega_LAN = np.arctan2(h_bar[0], -h_bar[1])
	# step8
	# beware of division by zero here
	lat = np.arctan2(np.divide(r_vec[2], (np.sin(i))), \
					 (r_vec[0] * np.cos(omega_LAN) + r_vec[1] * np.sin(omega_LAN)))
	# step9
	p = a * (1 - e ** 2)
	nu = np.arctan2(np.sqrt(p / mu) * np.dot(r_vec, v_vec), p - r)
	# step10
	omega_AP = lat - nu
	# step11
	EA = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
	# step12
	n = np.sqrt(mu / (a ** 3))
	T = t - (1 / n) * (EA - e * np.sin(EA))

	return a, e, i, omega_AP, omega_LAN, T, EA



def orbit_altitude(satfile):
	mu = G.value * M_earth.value
	Re = R_earth.value
	with open(satfile) as infile:
		"""read all lines from CIS files"""
		lines = infile.readlines()
		"""set the start and end characters for splitting the lines into X,Y,Z, U,V,W coordinates"""
		start0, start1, start2, start3, start4, start5, end = 23, 41, 59, 77, 95, 113, 131
		X = np.array([np.float(i[start0:start1]) for i in lines])
		Y = np.array([np.float(i[start1:start2]) for i in lines])
		Z = np.array([np.float(i[start2:start3]) for i in lines])

		X = X.reshape(X.shape[0], 1)
		Y = Y.reshape(Y.shape[0], 1)
		Z = Z.reshape(Z.shape[0], 1)

		R = np.concatenate((X, Y, Z), axis=1)



		U = np.array([np.float(i[start3:start4]) for i in lines])
		V = np.array([np.float(i[start4:start5]) for i in lines])
		W = np.array([np.float(i[start5:end]) for i in lines])

		U = U.reshape(U.shape[0], 1)
		V = V.reshape(V.shape[0], 1)
		W = W.reshape(W.shape[0], 1)

		V = np.concatenate((U, V, W), axis=1)



		"""calculate orbit altitude and convert to km"""

		ALTITUDE_sec = cart_2_kep_matrix(R, V, mu, Re)

		"""read the days and seconds """
		daysStart, daysEnd, secondsStart, secondsEnd = 4, 11, 11, 23
		seconds = np.array([np.float(i[secondsStart:secondsEnd]) for i in lines])
		days = np.array([np.float(i[daysStart:daysEnd]) for i in lines])
		"""calculate the decimal format for the days and subtract 51.184 to convert to correct time format"""
		decimalDays = days + (seconds-51.184)/(24.0*60.*60.)



		"""convert decimal days to a date"""
		YMDHMS = [datetime.datetime(2000, 1, 1, 12) + datetime.timedelta(day) for day in decimalDays]




		"""create an empty Pandas dataframe, called df"""
		df = pd.DataFrame()
		"""add the dates, decimal days and separation to the dataframe"""
		df['date'] = pd.to_datetime(YMDHMS)
		df['decimalDays'] = decimalDays
		df['altitude_raw'] = ALTITUDE_sec
		"""set the index of the dataframe to the date values"""
		df.set_index('date', inplace=True)
		df.drop(df.tail(1).index, inplace=True)





		"""calculate the daily average"""
		# df_daily = df.resample('D').mean()
		df_daily = df.resample('24H', base=0, closed='left').mean()
		print(df_daily.tail(10))
		# print(df_daily.tail(1000))
		# df_daily = df.resample('D').mean(min_count=200)
		# print(df_daily.tail(10))
		"""calculate extremas"""
		m = argrelextrema(df_daily['altitude_raw'].values, np.greater)[0]
		n = argrelextrema(df_daily['altitude_raw'].values, np.less)[0]
		rawsignal = df_daily['altitude_raw'].values
		extrema = np.empty(rawsignal.shape)
		extrema[:] = np.nan

		extrema1 = np.empty(rawsignal.shape)
		extrema1[:] = np.nan

		df_daily['extrema'] = extrema
		df_daily['extrema1'] = extrema1

		for ind in m:
			df_daily['extrema'].iloc[ind] = rawsignal[ind]

		for ind in n:
			df_daily['extrema1'].iloc[ind] = rawsignal[ind]

		"""interpolate extremas where they do not exist"""

		df_daily.interpolate(inplace=True)
		"""calculate the average of the two extrema columns"""
		df_daily['altitude'] = df_daily[['extrema', 'extrema1']].mean(axis=1)
		df_daily.to_csv('%s.csv' %satfile)
		df_daily['altitude'].plot(figsize=(10,4))
		# df_daily.drop(df_daily.tail(1).index, inplace=True)








		"""set the x,y labels and x, y ticks to a readable fontsize and make layout tight"""
		plt.xlabel('Date', fontsize=18)
		plt.ylabel('Altitude [km]', fontsize=18)
		plt.xticks(fontsize=12, rotation=70)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=14)
		plt.tight_layout()

		# plt.show()

		"""save plot to file"""
		plt.savefig('altitude_%s.png' % (satfile))
		plt.figure()

		ax = df_daily['altitude_raw'].plot(marker='.', figsize=(10, 3))

		"""set the x,y labels and x, y ticks to a readable fontsize and make layout tight"""
		plt.xlabel('Date', fontsize=18)
		plt.ylabel('Altitude [km]', fontsize=18)
		plt.xticks(fontsize=12, rotation=70)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=14)
		ax.legend(['altitude'])  # replacing legend entry 'altitude_raw' with 'altitude'
		plt.tight_layout()

		plt.savefig('altitude_raw%s.png' % (satfile))
		# plt.show()




def sat_separation(satA, satB):

	with open(satA) as infileA, open(satB) as infileB:
		"""read all lines from CIS files"""
		linesA = infileA.readlines()
		linesB = infileB.readlines()
		"""set the start and end characters for splitting the lines into X,Y,Z coordinates"""
		start0, start1, start2, end = 23, 41, 59, 77
		Xa = np.array([np.float(i[start0:start1]) for i in linesA])
		Ya = np.array([np.float(i[start1:start2]) for i in linesA])
		Za = np.array([np.float(i[start2:end]) for i in linesA])

		Xb = np.array([np.float(i[start0:start1]) for i in linesB])
		Yb = np.array([np.float(i[start1:start2]) for i in linesB])
		Zb = np.array([np.float(i[start2:end]) for i in linesB])

		"""calculate distance between satellites and convert to km"""

		SEP_AB = np.sqrt((Xa - Xb)**2. + (Ya - Yb)**2. + (Za - Zb)**2.)/1000.


		"""read the days and seconds """
		daysStart, daysEnd, secondsStart, secondsEnd = 4, 11, 11, 23
		seconds = np.array([np.float(i[secondsStart:secondsEnd]) for i in linesA])
		days = np.array([np.float(i[daysStart:daysEnd]) for i in linesA])
		"""calculate the decimal format for the days and subtract 51.184 to convert to correct time format"""
		decimalDays = days + (seconds-51.184)/(24.0*60.*60.)
		"""convert decimal days to a date"""
		YMDHMS = [datetime.datetime(2000, 1, 1) + datetime.timedelta(day+1) for day in decimalDays]

		"""create an empty Pandas dataframe, called df"""
		df = pd.DataFrame()
		"""add the dates, decimal days and separation to the dataframe"""
		df['date'] = pd.to_datetime(YMDHMS)
		df['decimalDays'] = decimalDays
		df['separation'] = SEP_AB
		"""set the index of the dataframe to the date values"""
		df.set_index('date', inplace=True)
		"""calculate the daily average"""
		df_daily = df.resample('D').mean()
		"""save the dataframe to file"""
		df_daily.to_csv('separation_AB.csv')
		"""For plotting purposes, drop the decimalDays from the dataframe"""
		df_daily.drop(['decimalDays'], axis=1, inplace=True)
		"""plot the dataframe"""
		df_daily.plot(figsize=(10,4))

		"""set the x,y labels and x, y ticks to a readable fontsize and make layout tight"""
		plt.xlabel('Date', fontsize=18)
		plt.ylabel('Separation [km]', fontsize=18)
		plt.xticks(fontsize=12, rotation=70)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=14)
		plt.tight_layout()

		"""save plot to file"""
		plt.savefig('separation_%s_%s.png' %(satA, satB))




def kep_2_cart(a, e, i, omega_AP, omega_LAN, T, EA):
	# step1
	n = np.sqrt(mu / (a ** 3))
	M = n * (t - T)
	# step2
	MA = EA - e * np.sin(EA)
	# step3
	#
	nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(EA / 2))
	# step4
	r = a * (1 - e * np.cos(EA))
	# step5
	h = np.sqrt(mu * a * (1 - e ** 2))
	# step6
	Om = omega_LAN
	w = omega_AP

	X = r * (np.cos(Om) * np.cos(w + nu) - np.sin(Om) * np.sin(w + nu) * np.cos(i))
	Y = r * (np.sin(Om) * np.cos(w + nu) + np.cos(Om) * np.sin(w + nu) * np.cos(i))
	Z = r * (np.sin(i) * np.sin(w + nu))

	# step7
	p = a * (1 - e ** 2)

	V_X = (X * h * e / (r * p)) * np.sin(nu) - (h / r) * (np.cos(Om) * np.sin(w + nu) + \
														  np.sin(Om) * np.cos(w + nu) * np.cos(i))
	V_Y = (Y * h * e / (r * p)) * np.sin(nu) - (h / r) * (np.sin(Om) * np.sin(w + nu) - \
														  np.cos(Om) * np.cos(w + nu) * np.cos(i))
	V_Z = (Z * h * e / (r * p)) * np.sin(nu) + (h / r) * (np.cos(w + nu) * np.sin(i))

	return [X, Y, Z], [V_X, V_Y, V_Z]

def grouper(n, iterable, padvalue=None):
	"""grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

	return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def process_chunk(d):
	mu = G.value * M_earth.value
	Re = R_earth.value
	vals = [np.float(i) for i in list(filter(None,d.strip().split(' ')))]

	try:
		r_vec = np.array([vals[2], vals[3], vals[4]])
		v_vec = np.array([vals[5], vals[6], vals[7]])
		a, e, i, omega_AP, omega_LAN, T, EA = cart_2_kep(r_vec, v_vec, 0, mu, Re)


		return vals[0], vals[1], a-Re
	except:
		return


def ACCPAR2Newton(dirname,date_start, number_of_days, sat):

	"""extract the year, month, and day from the user input and convert to datetime object"""
	base = datetime.datetime(int(date_start[0:4]), int(date_start[4:6]), int(date_start[6:8]))
	"""create a list of datetimes for which the acceleration parameters will be read"""
	date_list = [base + datetime.timedelta(days=x) for x in range(int(number_of_days))]
	"""convert the dates back to strings"""
	date_list_strings = [d.date().strftime('%Y%m%d') for d in date_list]
	"""create a list of all the files within the user provided directory"""
	all_files = os.listdir(dirname)
	"""select only the files with dates that are also present in the date_list_strings list object"""
	selected_files = [f for f in all_files if f.strip().split('.')[2] in date_list_strings]
	"""now only select the satellite of interest"""
	selected_files = [s for s in selected_files if s.strip().split('.')[1]==sat]
	"""sort the files according to date (just incase this doesn't happen automatically)"""
	def last_4chars(x):
		return(x.strip().split('.')[2])

	sorted_files = sorted(selected_files, key = last_4chars)
	"""read each of the selected files into a list object"""
	print(len(sorted_files), 'number of files')
	m = 704.
	List_of_transverse_forces=[]
	for f in sorted_files:
		fp = os.path.join(dirname,f)
		with open(fp, "r") as fi:
			lines = fi.readlines()
		"""extract only those elements in the list that start with acl (only interested in acceleration)"""
		lines = [l for l in lines if l[0:3]=='acl']
		vals = [line.strip().split() for line in lines]
		a_y_list = [np.float(v[2]) for v in vals]
		List_of_transverse_forces.append(a_y_list)
	List_of_transverse_forces = [item for sublist in List_of_transverse_forces for item in sublist]

	Acc_Y = np.array(List_of_transverse_forces).reshape(len(List_of_transverse_forces),1)

	FORCE_Y = np.absolute((Acc_Y/10.**3)*10.**6*m)


	#
	"""We are interested in correcting the force in the transversal direction (i.e. y-direction)"""

	F_Y_min = np.ones_like(FORCE_Y)*FORCE_Y.min()
	F_Y_max = np.ones_like(FORCE_Y)* FORCE_Y.max()
	min_compensation = np.ones_like(FORCE_Y)*200.0
	plt.figure(figsize=(10,3))
	p1, = plt.plot(FORCE_Y,'k-', label='F transvers force')
	p2, = plt.plot(F_Y_min, 'b--', label='F transvers min')
	p3, = plt.plot(F_Y_max, 'm--', label='F transvers max')
	p4, = plt.plot(min_compensation, 'r-', label='Minimal thrust of propulsion system')
	plt.legend()
	plt.ylabel('Transverse force [$\mu$ N]')
	plt.xlabel('Time [s]')
	plt.title('Transverse force %s days. Start at %s. New geometry' % (number_of_days, date_start))
	plt.legend(handles=[p1, p2, p3, p4], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
	plt.tight_layout()
	plt.savefig('forces_%s_day_NewShape' %number_of_days)


def ACCPAR2Newton_daily_mean(dirname, date_start, number_of_days, sat):
	"""extract the year, month, and day from the user input and convert to datetime object"""
	base = datetime.datetime(int(date_start[0:4]), int(date_start[4:6]), int(date_start[6:8]))
	"""create a list of datetimes for which the acceleration parameters will be read"""
	date_list = [base + datetime.timedelta(days=x) for x in range(int(number_of_days))]
	"""convert the dates back to strings"""
	date_list_strings = [d.date().strftime('%Y%m%d') for d in date_list]
	"""create a list of all the files within the user provided directory"""
	all_files = os.listdir(dirname)
	"""select only the files with dates that are also present in the date_list_strings list object"""
	selected_files = [f for f in all_files if f.strip().split('.')[2] in date_list_strings]
	"""now only select the satellite of interest"""
	selected_files = [s for s in selected_files if s.strip().split('.')[1] == sat]
	"""sort the files according to date (just incase this doesn't happen automatically)"""

	def last_4chars(x):
		return (x.strip().split('.')[2])

	sorted_files = sorted(selected_files, key=last_4chars)
	"""read each of the selected files into a list object"""
	print(len(sorted_files), 'number of files')
	m = 704.
	List_of_transverse_forces = []

	List_of_timestamps=[]
	df = pd.DataFrame()

	for f in sorted_files:
		fp = os.path.join(dirname, f)
		with open(fp, "r") as fi:
			lines_all = fi.readlines()

		"""extract only those elements in the list that start with acl (only interested in acceleration)"""
		lines = [l for l in lines_all if l[0:3] == 'acl']
		lines_t = [l for l in lines_all if l[0:3] == 'tim']

		vals = [line.strip().split() for line in lines]
		vals_t = [list(map(float, line.strip().split()[1::])) for line in lines_t]
		vals_t = [list(map(int,v)) for v in vals_t]

		years = [v[0] for v in vals_t]
		months = [v[1] for v in vals_t]
		minutes = [v[4] for v in vals_t]
		seconds = [v[5] for v in vals_t]
		hours = [v[3] for v in vals_t]
		days = [v[2] for v in vals_t]
		for iter, hval in enumerate(hours):

			if hval>23:
				days[iter] = days[iter+1]
				hours[iter] = 0



		timestamps = []
		for i in range(len(years)):
			timestamps.append(datetime.datetime(years[i], months[i], days[i],hours[i],minutes[i], seconds[i]))

		a_y_list = [np.float(v[2]) for v in vals]




		List_of_transverse_forces.append(a_y_list)
		List_of_timestamps.append(timestamps)



	List_of_transverse_forces = [item for sublist in List_of_transverse_forces for item in sublist]

	List_of_timestamps = [item for sublist in List_of_timestamps for item in sublist]


	Acc_Y = np.array(List_of_transverse_forces).reshape(len(List_of_transverse_forces), 1)


	FORCE_Y = np.absolute((Acc_Y / 10. ** 3) * 10. ** 6 * m)

	df['date'] = List_of_timestamps
	df['date']=pd.to_datetime(df['date'])
	df['FORCE_Y'] = FORCE_Y
	# F_Y_min = np.ones_like(FORCE_Y) * FORCE_Y.min()
	# F_Y_max = np.ones_like(FORCE_Y) * FORCE_Y.max()
	min_compensation = np.ones_like(FORCE_Y) * 200.0
	df['drag compensation threshhold'] = min_compensation
	df.set_index('date', inplace=True)

	df = df[~df.index.duplicated(keep='first')]
	# df['rolling mean 1d'] = df['FORCE_Y'].rolling('24H', min_periods=100).mean()
	df['rolling mean 92.7min'] = df['FORCE_Y'].rolling('5562s', min_periods=1100).mean()
	df.plot(figsize=(15,5), color = ['k', 'y','c'])
	plt.ylabel('Transverse force [$\mu$ N]', fontsize=18)
	plt.xlabel('Date', fontsize=18)

	plt.title('Transverse force over %s days from %s. Follow-on macro model.' % (number_of_days, date_start), fontsize=18)
	plt.legend(fontsize=14)
	plt.tight_layout()
	# plt.show()
	plt.savefig('Follow-on_geometry_drag_transverse_sat_%s.png' %sat)









def read_gravity_field_to_dict(fp_full, fp_static):
	fn = os.path.basename(fp_full)

	string_date = fn.strip().split('_')[1]
	start = string_date.split('-')[0]
	end = string_date.split('-')[1]
	year = np.int(start[0:4])
	day = np.int(start[4::])
	date_start = datetime.datetime.strptime('{} {}'.format(day, year), '%j %Y')
	year = np.int(end[0:4])
	day = np.int(end[4::])
	date_end = datetime.datetime.strptime('{} {}'.format(day, year), '%j %Y')
	print('reading gravity coefficients for period %s to %s' % (date_start, date_end))
	string_eoY = '# End of YAML header'
	string_gm = 'earth_gravity_param'
	string_r = 'mean_equator_radius'
	linecount = 0
	GM_r = []
	with open(fp_full, 'r') as f:
		for i, line in enumerate(f):
			if string_gm in line:
				GM_r.append(i + 3)
			if string_r in line:
				GM_r.append(i + 3)

	with open(fp_full, 'r') as f:
		gm_value = np.float(f.readlines()[GM_r[0]].strip().split(':')[1])
	with open(fp_full, 'r') as f:
		r_value = np.float(f.readlines()[GM_r[1]].strip().split(':')[1])

	with open(fp_full, 'r') as f:
		for line in f:
			if string_eoY in line:
				header2skip = linecount
			linecount += 1
	df = pd.read_csv(fp_full, skiprows=header2skip, delim_whitespace=True, usecols=[1, 2, 3, 4], index_col=False,
					 names=['degree_l', 'order_m', 'C_lm', 'S_lm'], header=0)
	L = df['degree_l'].values
	M = df['order_m'].values
	LM = list(zip(L, M))


	C_dict_full = {}
	S_dict_full = {}
	for v in LM:
		C = df.loc[(df['degree_l'] == v[0]) & (df['order_m'] == v[1])]['C_lm'].values[0]
		C_dict_full[v] = C

		S = df.loc[(df['degree_l'] == v[0]) & (df['order_m'] == v[1])]['S_lm'].values[0]
		S_dict_full[v] = S

	fp = fp_static
	rows2skip = list(range(45)) + [46]
	df_static = pd.read_csv(fp, header=0, skiprows=rows2skip, delim_whitespace=True)

	L = df_static['L'].values
	M = df_static['M'].values

	LM = list(zip(L, M))
	LMFiltered = [(x, y) for x, y in LM if (x <= 60)]
	LMFiltered = [(x, y) for x, y in LMFiltered if (y <= 60)]


	C_dict_static = {}
	S_dict_static = {}
	for v in LMFiltered:
		C = df_static.loc[(df_static['L'] == v[0]) & (df_static['M'] == v[1])]['C'].values[0].replace('D', 'E')
		# C = np.log10(np.float(C))
		# print(C, np.log10(C))
		# print('This is C')

		C_dict_static[v] = C
		S = df_static.loc[(df_static['L'] == v[0]) & (df_static['M'] == v[1])]['S'].values[0].replace('D', 'E')
		S_dict_static[v] = S

	keys_values = C_dict_full.items()
	C_dict_full =  {str(key): value for key, value in keys_values}
	delta_C_dict = {key: C_dict_full[key] - C_dict_static.get(key,0) for key in C_dict_full.keys()}
	keys_values = delta_C_dict.items()
	delta_C_dict = {make_tuple(key): value for key, value in keys_values }

	keys_values = S_dict_full.items()
	S_dict_full = {str(key): value for key, value in keys_values}
	delta_S_dict = {key: S_dict_full[key] - S_dict_static.get(key, 0) for key in S_dict_full.keys()}
	keys_values = delta_S_dict.items()
	delta_S_dict = {make_tuple(key): value for key, value in keys_values}




	return delta_C_dict, delta_S_dict, gm_value, r_value, date_start, date_end



def calc_mean_his(fp,lmax):
	import time
	starttime = time.time()
	list_of_SHC_files = os.listdir(fp)

	his_C_all = np.empty((len(list_of_SHC_files), lmax+1, lmax+1))
	his_S_all = np.empty((len(list_of_SHC_files), lmax+1, lmax+1))
	his_C_all[:], his_S_all[:] = 0.,0.

	for day,fn in enumerate(list_of_SHC_files):

		ffp = os.path.join(fp,fn)
		with open(ffp,"r") as fi:
			lines = fi.readlines()[12:]
			split_at_list = [itm for itm in lines if itm.startswith('DATA SET')]
			index_list = []
			for s in split_at_list:
				index_list.append(lines.index(s))
			list_0 = lines[0:index_list[0]]
			list_1 = lines[index_list[0]+1:index_list[1]]
			list_2 = lines[index_list[1]+1:index_list[2]]
			list_3 = lines[index_list[2]+1::]


			his_C, his_S = np.empty((4, lmax + 1, lmax + 1)), np.empty((4, lmax + 1, lmax + 1))

			his_C[:], his_S[:] = 0., 0.

			listnames = [list_0, list_1, list_2, list_3]
			count = 0
			for listname in listnames:
				vals = [line.strip().split() for line in listname]
				vals = [v for v in vals if (int(v[0])<lmax+1 and int(v[1])<lmax+1)]

				n_list = [np.int(val[0]) for val in vals]
				m_list = [np.int(val[1]) for val in vals]
				his_C_list = [np.float(val[2].replace('D', 'E')) for val in vals]
				his_S_list = [np.float(val[3].replace('D', 'E')) for val in vals]

				for i,n in enumerate(n_list):
					m = m_list[i]
					his_C[count,n,m] = his_C_list[i]
					his_S[count,n,m] = his_S_list[i]
				count+=1
			his_C_day = his_C.mean(axis=0)
			his_S_day = his_S.mean(axis=0)
		his_C_all[day,:,:] = his_C_day
		his_S_all[day,:,:] = his_S_day



	his_C_mean = his_C_all.mean(axis=0)
	his_S_mean = his_S_all.mean(axis=0)



	print('time taken %f seconds' %(time.time()-starttime))

	return his_C_mean, his_S_mean

def get_meanhis(fp, lmax):
	"""get mean his from .mat file"""

	mean_his = loadmat(fp)['mean_his']

	C_his_mean = np.tril(mean_his)

	S_his_mean_temp = np.triu(mean_his, 1).transpose()
	S_his_mean = np.empty((lmax + 1, lmax + 1)) * 0.
	S_his_mean[:, 1::] = S_his_mean_temp[:, 0:-1]

	return C_his_mean,S_his_mean
def get_CS(fp, subtractMeanHis=True):

	SHC_list = ['GCN', 'GSN']
	for i, SHCname in enumerate(SHC_list):
		with open(fp, "r") as fi:
			lines = fi.readlines()
		"""extract only those elements in the list that start with acl (only interested in acceleration)"""
		lines = [l for l in lines if l[0:3] == SHCname]

		vals = [line.strip().split() for line in lines]

		lmax = np.int((vals[-1][1]))  # max deg order

		SHC, SHC_sigma, SHC_static = np.empty((lmax + 1, lmax + 1)), np.empty((lmax + 1, lmax + 1)), np.empty(
			(lmax + 1, lmax + 1))

		SHC[:], SHC_sigma[:], SHC_static[:] = 0., 0., 0.

		SHC_n = [np.int(val[1]) for val in vals]


		SHC_m = [np.int(val[2]) for val in vals]

		SHC_vals = [val[4][4::] for val in vals]

		SHC_list = [np.float(SHC_val[0:20].replace('D', 'E')) for SHC_val in SHC_vals]
		SHC_sigma_list = [np.float(SHC_val[20:40].replace('D', 'E')) for SHC_val in SHC_vals]
		SHC_static_list = [np.float(SHC_val[40::].replace('D', 'E')) for SHC_val in SHC_vals]



		for iter, n in enumerate(SHC_n):
			m = SHC_m[iter]
			SHC[n, m] = SHC_list[iter]
			SHC_sigma[n, m] = SHC_sigma_list[iter]
			SHC_static[n, m] = SHC_static_list[iter]


		if SHCname == 'GCN':
			C, C_sigma, C_static = SHC, SHC_sigma, SHC_static


		if SHCname == 'GSN':
			S, S_sigma, S_static = SHC, SHC_sigma, SHC_static

	if subtractMeanHis:
		"""get mean his from .mat file"""
		if lmax > 100:
			mean_his = loadmat('/home/fine/epos-python-processing/matlab_functions/mean_his_jan_2002_120x120.mat')['mean_his']
		else: mean_his = loadmat('/home/fine/epos-python-processing/matlab_functions/mean_his_jan_2002.mat')['mean_his']

		C_his_mean = np.tril(mean_his)

		S_his_mean_temp = np.triu(mean_his,1).transpose()
		S_his_mean = np.empty((lmax+1,lmax+1))*0.
		S_his_mean[:,1::] = S_his_mean_temp[:,0:-1]

		#subtract mean HIS field:
		C_min_static_min_HIS = C - C_static - C_his_mean
		S_min_static_min_HIS = S - S_static - S_his_mean
		C_return = C_min_static_min_HIS
		S_return = S_min_static_min_HIS


	else:
		C_return = C - C_static
		S_return = S - S_static

	return C_return, S_return, C_sigma, S_sigma


def degree_variance(fplist):

	path_loadlove=os.path.abspath(os.path.join(os.path.dirname('__file__'), 'input_data', 'LoadLove_PG_CF_oct.dat'))
	df_ll = pd.read_csv(path_loadlove, delim_whitespace=True,
					  names=['n', 'h', 'nl', 'nk'])
	loadlove = df_ll['nk'].values
	a = 0.6378136460 * 1E7
	rho_e, rho_w = 5517., 1025.
	K = a * rho_e / (3 * rho_w)

	df = pd.DataFrame()


	for fp in fplist:

		SHC_list = ['GCN', 'GSN']
		for i, SHCname in enumerate(SHC_list):
			with open(fp, "r") as fi:
				lines = fi.readlines()
			"""extract only those elements in the list that start with acl (only interested in acceleration)"""
			lines = [l for l in lines if l[0:3] == SHCname]

			vals = [line.strip().split() for line in lines]

			lmax = np.int((vals[-1][1]))  # max deg order

			SHC, SHC_sigma, SHC_static = np.empty((lmax + 1, lmax + 1)), np.empty((lmax + 1, lmax + 1)), np.empty(
				(lmax + 1, lmax + 1))

			SHC[:], SHC_sigma[:], SHC_static[:] = 0., 0., 0.

			SHC_n = [np.int(val[1]) for val in vals]


			SHC_m = [np.int(val[2]) for val in vals]

			SHC_vals = [val[4][4::] for val in vals]

			SHC_list = [np.float(SHC_val[0:20].replace('D', 'E')) for SHC_val in SHC_vals]
			SHC_sigma_list = [np.float(SHC_val[20:40].replace('D', 'E')) for SHC_val in SHC_vals]
			SHC_static_list = [np.float(SHC_val[40::].replace('D', 'E')) for SHC_val in SHC_vals]



			for iter, n in enumerate(SHC_n):
				m = SHC_m[iter]
				SHC[n, m] = SHC_list[iter]
				SHC_sigma[n, m] = SHC_sigma_list[iter]
				SHC_static[n, m] = SHC_static_list[iter]


			if SHCname == 'GCN':
				C, C_sigma, C_static = SHC, SHC_sigma, SHC_static


			if SHCname == 'GSN':
				S, S_sigma, S_static = SHC, SHC_sigma, SHC_static


		"""get mean his from .mat file"""
		if lmax > 100:
			mean_his = loadmat('/home/fine/epos-python-processing/matlab_functions/mean_his_jan_2002_120x120.mat')['mean_his']
		else: mean_his = loadmat('/home/fine/epos-python-processing/matlab_functions/mean_his_jan_2002.mat')['mean_his']

		C_his_mean = np.tril(mean_his)

		S_his_mean_temp = np.triu(mean_his,1).transpose()
		S_his_mean = np.empty((lmax+1,lmax+1))*0.
		S_his_mean[:,1::] = S_his_mean_temp[:,0:-1]
		arr_C_delta = C_his_mean
		arr_S_delta = S_his_mean

		sq = arr_C_delta ** 2. + arr_S_delta ** 2.
		his = []
		for ss in range(0,97):
			his.append(np.sum(sq[ss,0:ss+1]))


		sigma_ground_truth = np.sqrt(np.array(his))
		EWH_true = []

		for l in range(0, lmax+1):
			EWH_true.append(K * (2 * l + 1) / (1 + loadlove[l]) * sigma_ground_truth[l]*1000.)

		df['EWH truth'] = EWH_true

		#subtract mean HIS field:
		C_min_static_min_HIS = C - C_static - C_his_mean

		S_min_static_min_HIS = S - S_static - S_his_mean

		####################################################################################################################
		# Degree amplitudes
		arr_C_delta = C_min_static_min_HIS
		arr_S_delta = S_min_static_min_HIS
		maxdeg = lmax

		sq = arr_C_delta ** 2. + arr_S_delta ** 2.
		hisest = []
		for ss in range(0,lmax+1):
			hisest.append(np.sum(sq[ss, 0:ss+1]))
		sigma_array = np.sqrt(np.array(hisest))

		EWH_est = []

		for l in range(0, lmax + 1):
			EWH_est.append(K * (2 * l + 1) / (1 + loadlove[l]) * sigma_array[l]*1000.)

		name = Path(fp).parents[1].name


		df[name] = EWH_est

		####################################################################################################################

	df = df.iloc[2:]

	ax = df.plot(figsize=(12,6), linewidth=1.5, marker='.')

	ax.set_yscale('log')


	plt.xlabel('SH degree [-]',fontsize=18);
	plt.ylabel('SH degree amplitudes  [mm EWH]', fontsize=18)


	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	title = Path(fp).parents[0].name.split('.')[2]
	plt.title('Gravity field retrieval error for %s' %title, fontsize= 18)
	plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.3)
	plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
	ax.set_axisbelow(True)
	plt.legend(fontsize=16)
	plt.tight_layout()


	plt.savefig('EWH.png')


	####################################################################################################################
	df_cum_sum = pd.DataFrame()
	df_cum_sum['EWH'] = df['EWH truth']

	for sim_name in list(df)[1::]:



		df_cum_sum[sim_name] = np.cumsum(df[sim_name].values)




	ax = df_cum_sum.plot(figsize=(12,6), linewidth=1.5, marker='.')

	ax.set_yscale('log')


	plt.xlabel('SH degree [-]',fontsize=18);
	plt.ylabel('SH degree amplitudes  [mm EWH]', fontsize=18)


	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	title = Path(fp).parents[0].name.split('.')[2]
	plt.title('Cumulitive gravity field retrieval error for %s' %title, fontsize= 18)
	plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.3)
	plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
	ax.set_axisbelow(True)
	plt.legend(fontsize=16)
	plt.tight_layout()


	plt.savefig('EWH_cumulitive_error.png')
def smooth_GF(C,S,avg_rad, start_deg):
	"""from Wahr et al: Time-variable gravity recovery from space eq. 34.
	This is Jekeli's [1981] smoothing method."""
	C_smooth = C
	S_smooth = S
	Re = 6378.1363; # Radius of Earth in km
	b = np.log(2) / (1 - np.cos(avg_rad / Re))
	W=[]
	W.append(1 / (2 * np.pi))
	W.append(1 / (2 * np.pi) * ((1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1 / b))


	for j in range(start_deg,C.shape[0]):
		w = (-(2*(j-1)+1)/b*W[j-1]) + W[j-2]
		W.append(w)
		if W[j] < 0.: W[j] = 0.
		if W[j-1] == 0.: W[j] = 0.


	for i in range(start_deg-1,C.shape[0]):
		C_smooth[i,:]=C[i,:]*W[i]*2.*np.pi
		S_smooth[i,:] = S[i,:]*W[i]*2.*np.pi


	return C_smooth, S_smooth

def legnorm(l,m,col):

	col = np.radians(col)
	lmax = np.max(l)
	t = np.cos(col)

	y = np.sqrt(1-t**2)
	p00=np.ones(t.shape[0])
	p = np.zeros((len(t), lmax - m + 1))

	if m ==0.:
		plm3=np.sqrt(3)*t
		plm2=np.sqrt(5)*(3*t**2-1)/2

		p[:,0]=p00
		p[:,1] = plm3
		p[:,2] = plm2

	if m==1.:
		plm3=np.sqrt(3)*y
		plm2=np.sqrt(15.)*t*y
		p[:,0] = plm3
		p[:,1] = plm2


	if m>1.:
		p = np.zeros((len(t), lmax - m + 1))
		plm3=np.zeros(t.shape[0])

		plm2=np.sqrt(15.)*y**2/2.

		if m>2:
			for n in range(4,m+2):
				plm2=np.sqrt((2*n-1)/(2*n-2))*y*plm2
		p[:,0] = plm2



	if lmax>max(2,m):

		for n in range(max(2,m)+2,lmax+2):

			plm1=np.sqrt((2*n-1)/(n-m-1)*(2*n-3)/(n+m-1))*t*plm2- \
				np.sqrt((2*n-1)/(2*n-5)*(n+m-2)/(n+m-1)*(n-m-2)/(n-m-1))*plm3

			p[:,n-m-1] = plm1
			plm3=plm2
			plm2=plm1
	p = p[:, 0:lmax-m+1]

	return p


def DDK_filter(w, C, S, sigC, sigS): #,Cnm,Snm,sigCnm,sigSnm
	"""Input:
		w       ... binary file with filter coefficients
        Cnm     ... (n,m)-matrix of potential C-coefficients
        Snm     ... (n,m)-matrix of potential S-coefficients
        sigCnm  ... (n,m)-matrix of sigmas of potential C-coefficients
        sigSnm  ... (n,m)-matrix of sigmas of potential S-coefficients

    	Output:
    	wCnm    ... weigthed C-coefficients (nmax+1,min(nmax+1,m)) - matrix
        wSnm    ... weigthed S-coefficients (nmax+1,min(nmax+1,m)) - matrix
        wsigCnm ... sigmas of weigthed C-coefficients (simple error propagation applied)
        wsigSnm ... sigmas of weigthed S-coefficients (simple error propagation applied)

		References: J. Kusche, "Approximate decorrelation and non-isotropic smoothing
		 of time-variable GRACE-type gravity field models", 2007

	# """

	import struct

	with open(w, "rb") as f:
		fileContent = f.read()

	chunk1=[]
	for i in range(14637):
		startval = 356 + i*24
		endval = startval+24
		a=fileContent[startval:endval]

		bbb = struct.unpack("c"*len(a), a)

		chunk1.append(bytes.join(b'', bbb).decode('ascii'))
	break_list = []
	step_list = []
	for ind in range(0,121):
		ind = str(ind)
		break_list.append('GCN 120%s' %(ind.zfill(3)))
		break_list.append('GSN 120%s' % (ind.zfill(3)))

	for ind in range(2, 121):
		ind = str(ind)
		step_list.append('GSN 120%s' % (ind.zfill(3)))

	break_indices = [j+1 for j,i in enumerate(chunk1) if i[0:10] in break_list]
	step_indices = [j + 1 for j, i in enumerate(chunk1) if i[0:10] in step_list]

	aaaaa = np.fromfile(w)
	V = np.empty((14637, 119))
	V[:] = np.nan

	a = aaaaa[(np.abs(aaaaa) > 1E-50) & (np.abs(aaaaa) < 1E+30)]
	step = 119
	start = 0

	sizelist=[]
	wCnm = np.empty_like(C)
	wCnm[:] = 0

	wsigCnm = np.empty_like(C)
	wsigCnm[:] = 0

	wSnm = np.empty_like(S)
	wSnm[:] = 0

	wsigSnm = np.empty_like(S)
	wsigSnm[:] = 0

	nmin = 2



	for ind in range(len(break_indices)):#len(break_indices)-48




		if ind==0:
			m = 0
			n = nmin
			end = break_indices[ind]*step
			w = a[start:end]
			w = w.reshape(break_indices[ind],step,order='F')


			nblocks = len(break_indices)-2*(w.shape[0]+1 - (C.shape[0]-1))




			V[0:w.shape[0],0:w.shape[1]] = w
			# wCnm[2::,0] = np.matmul(w[0:C.shape[0]-2,0:C.shape[0]-2], C[2::,m])


			wCnm[nmin::, m] = np.matmul(w[0:w.shape[0] - (120 - C.shape[0])-1, 0:w.shape[0] - (120 - C.shape[0])-1],
									   C[n::, m])

			wsigCnm[nmin::, m] = np.sqrt(np.matmul((w[0:w.shape[0] - (120 - C.shape[0]) - 1, 0:w.shape[0] - (120 - C.shape[0]) - 1])**2.,
										   (sigC[n::, m])**2.))

		if ind>0:


			end = start+(break_indices[ind]- break_indices[ind-1])*step
			w = a[start:end].reshape((break_indices[ind]- break_indices[ind-1],step), order='F')

			sizelist.append(w.shape[0])

			V[break_indices[ind-1]:break_indices[ind], 0:w.shape[1]] = w

			if ind % 2 != 0:
				m+=1
				n = max(nmin, m)

				wCnm[n::, m] = np.matmul(
					w[0:w.shape[0] - (120 - C.shape[0]) - 1, 0:w.shape[0] - (120 - C.shape[0]) - 1],
					C[n::, m])

				wsigCnm[n::, m] = np.sqrt(np.matmul(
					(w[0:w.shape[0] - (120 - C.shape[0]) - 1, 0:w.shape[0] - (120 - C.shape[0]) - 1])**2.,
					sigC[n::, m]**2.))

			if ind%2 == 0:

				n = max(nmin, m)

				wSnm[n::, m] = np.matmul(
					w[0:w.shape[0] - (120 - S.shape[0]) - 1, 0:w.shape[0] - (120 - S.shape[0]) - 1],
					S[n::, m])

				wsigSnm[n::, m] = np.sqrt(np.matmul(
					(w[0:w.shape[0] - (120 - S.shape[0]) - 1, 0:w.shape[0] - (120 - S.shape[0]) - 1]) ** 2.,
					sigS[n::, m] ** 2.))



		start = end
		if break_indices[ind] in step_indices:

			step -= 1

		if ind==nblocks-1:
			break

	df = pd.DataFrame(V)
	df.index = chunk1
	return(wCnm, wSnm, wsigCnm, wsigSnm)

def shs(C,S, option, max_degree, smooth=False, mask=False):

	#constants:
	ae = 6.378137e6
	GM = 3.986005e14
	G = 6.67259e-11


	col = np.arange(0.5, 180.5, 1.0)
	lon = np.arange(-179.5, 180.5, 1.0)
	degrees = np.arange(0, max_degree + 1)
	path_loadlove = os.path.abspath(os.path.join(os.path.dirname('__file__'), 'input_data', 'LoadLove_PG_CF_oct.dat'))

	df_ll = pd.read_csv(path_loadlove, delim_whitespace=True,
						names=['n', 'h', 'nl', 'nk'])
	m_vec = np.arange(0, max_degree + 1).reshape(max_degree + 1, 1)
	lon = lon.reshape(1, lon.shape[0]) * np.pi / 180
	cos_mlon = np.cos(np.matmul(m_vec, lon))
	sin_mlon = np.sin(np.matmul(m_vec, lon))

	if mask:

		lwmask = loadmat('/home/fine/epos-python-processing/input_data/lwmask_open500.mat')['lwmask_open']


	if smooth:
		print('smoothing')
		C, S = smooth_GF(C, S, 350., 2)

	"""*******************************************************************************************************************"""
	"""											geoid undulation [m]													"""
	"""*******************************************************************************************************************"""

	if option == 1:
		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			p_fun = legnorm(degrees, mvecval, col)
			Am[:, mvecval] = np.matmul(p_fun, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun, S[mvecval:max_degree + 1, mvecval])

		c_fun = ae
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))

	"""*******************************************************************************************************************"""
	"""											geoid undulation [mGal]														"""
	"""*******************************************************************************************************************"""
	if option == 2:
		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append(l - 1)

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun =1e5*GM/ae**2.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))

	"""*******************************************************************************************************************"""
	"""											vertical gravity gradient [E]													"""
	"""*******************************************************************************************************************"""
	if option == 3:

		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append((l + 1.)*(l+2.))

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun = 1e9 * GM / ae ** 3.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))
	"""*******************************************************************************************************************"""
	"""													Compute EWH on grid												"""
	"""*******************************************************************************************************************"""
	if option == 4:
		"""
		
		--The result is in [kg/m^2] which is equivalent to waterheight in mm:
		1 kg of water is equivalent to 0.001 m^3 of water. This implies: 1 kg/m^3 equiv 0.001 m (i.e. 1 mm). 
		
		"""
		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append((2 * l + 1) / (1 + LLN[l]))

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun = GM / G / 4. / np.pi / ae ** 2.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))

	"""*******************************************************************************************************************"""
	"""											no dimension [-]													"""
	"""*******************************************************************************************************************"""
	if option == 5:
		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append(1.)

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun = 1.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))



	"""*******************************************************************************************************************"""
	"""											gravity disturbance [mGal]													"""
	"""*******************************************************************************************************************"""
	if option == 6:
		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append(l + 1)

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun = 1e5*GM/ae**2.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))


	"""*******************************************************************************************************************"""
	"""											pressure [Pa]=[kg/m/s^2]												"""
	"""*******************************************************************************************************************"""
	if option == 7:
		g0 = 9.81

		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append((2 * l + 1) / (1 + LLN[l]))

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun = g0*GM/G/4/np.pi/ae**2.
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))

	"""*******************************************************************************************************************"""
	"""											vertical crustal deformation [m]										"""
	"""*******************************************************************************************************************"""
	if option == 8:
		g0 = 9.81
		LLN_H = df_ll['h'].values[0:C.shape[0]]

		LLN = df_ll['nk'].values[0:C.shape[0]]
		LLN[1] = 0.

		Am = np.zeros((col.shape[0], max_degree + 1))
		Bm = np.zeros((col.shape[0], max_degree + 1))
		for mvecval in range(0, max_degree + 1):
			l_fun = []
			for l in range(mvecval, max_degree + 1):
				l_fun.append(LLN_H[l]/ (1 + LLN[l]))

			LFUN = np.empty((col.shape[0], max_degree + 1 - mvecval))

			LFUN[...] = l_fun

			p_fun = legnorm(degrees, mvecval, col)

			Am[:, mvecval] = np.matmul(p_fun * LFUN, C[mvecval:max_degree + 1, mvecval])
			if mvecval != 0.:
				Bm[:, mvecval] = np.matmul(p_fun * LFUN, S[mvecval:max_degree + 1, mvecval])

		c_fun =  ae
		global_grid = c_fun * (np.matmul(Am, cos_mlon) + np.matmul(Bm, sin_mlon))

	if mask:
		lwmask_inverse = np.where((lwmask == 0) | (lwmask == 1), lwmask ^ 1, lwmask)
		lwmask_inverse = lwmask_inverse.astype(float)
		lwmask_inverse[lwmask_inverse == 0] = np.nan
		gridded_data = global_grid * lwmask_inverse
	else:
		gridded_data = global_grid


	return gridded_data



def plot_spatial(data,titleval, min, max, cbartitle):

	'''plot on map with cartopy'''
	import cartopy.crs as ccrs
	from matplotlib import cm

	col = np.arange(0.5, 180.5, 1.0)
	lat = 90.0 - col
	lon = np.arange(-179.5, 180.5, 1.0)

	x,y = np.meshgrid(lon,lat)
	use_proj = ccrs.EckertV()

	out_xyz = use_proj.transform_points(ccrs.PlateCarree(), x,y)

	x_array = out_xyz[:,:,0]
	y_array = out_xyz[:,:,1]



	list_of_colors_hexRGB = list(reversed(['#68011d','#b5172f','#d75f4e','#f7a580','#fff6f1','#f5f9f3','#eaf1f8','#93c5dc','#4295c1','#2265ad','#062e61']))
	RGB = []
	for color in list_of_colors_hexRGB:
		h = color.lstrip('#')
		c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
		value = map(lambda x: x / 255., c)
		tval=tuple(list(value))
		RGB.append(tval)
	colors = RGB

	cmap_name = 'my_list'
	cmap = LinearSegmentedColormap.from_list(cmap_name, colors, 100)
	# fig = plt.figure(figsize=(8, 5))

	ax = plt.axes(projection=ccrs.EckertV())  # ccrs.PlateCarree()

	levels = np.linspace(min, max, 100)
	step = 10.
	cbarticks = np.arange(min, max + step, step)


	im = plt.contourf(x_array, y_array, data, levels=levels,
				 transform=ccrs.EckertV(), vmin=min, vmax=max,cmap=cmap)

	# im = plt.contourf(x_array, y_array, data/10.,
	# 				  transform=ccrs.EckertV())




	lakes = cfeature.LAKES
	ax.add_feature(lakes, edgecolor='black', linewidth = 0.4)

	ax.coastlines(resolution='50m', color='black', linewidth=0.4)


	cb=plt.colorbar(ax=ax, orientation='horizontal', ticks=cbarticks, aspect=50)
	cb.ax.tick_params(labelsize=14)
	cb.set_label(cbartitle, labelpad=-1, fontsize=18)


	ttl = plt.title(titleval, fontsize=18,fontweight='bold')
	ttl.set_position([.5, 1.05])
	plt.tight_layout()

	plt.show()

