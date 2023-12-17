import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import matplotlib.cm as cm
plt.style.use('tableau-colorblind10')
# color arrays for plotting
color_array = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.size'] = 13

# constants 
c = 2.9979e10
G = 6.674e-8
Msun = 1.989e33
Rsun = 6.957e10
sigma_SB = 5.67e-5

# model parameters (common)
xi = 2.0 
s = 0.0
x_ion_floor = 1e-5 # floor value of ionization

# other free parameters
CO = 'BH' # BH or NS
vary_eps_wind_with_BB = True # Wind efficiency. If True, obtain from outer and inner disk radii using ADIOS model (Blandford & Begelman 99)
eps_wind_const = 1e-2 # if vary_eps_wind_with_BB = False, this constant value is used


# plot data points from Matsumoto & Metzger (2022), Fransson+ (2022), Strotjohann+ (2021), Foley+ (2007), Pastorello+ (2007)
# II: 2009ip, 2010mc, 2015bh, 2016bdu, 2020tlf, 2019zrk
# Ibn: 2006jc, 2019uo
def plot_prec_obs_data(ax_lc_IIn, ax_vel_IIn, ax_lc_Ibn, ax_vel_Ibn, prec_file):
	data = np.genfromtxt(prec_file, skip_header=1, dtype=None)
	t_09ip, t_10mc, t_15bh, t_16bdu, t_20tlf, t_19zrk = [], [], [], [], [], []
	L_09ip, L_10mc, L_15bh, L_16bdu, L_20tlf, L_19zrk = [], [], [], [], [], []
	t_06jc, t_19uo = [], []
	L_06jc, L_19uo = [], []
	for i in range(len(data)): 
		if data[i][0] == b'SN2009ip':
			t_09ip.append(data[i][1]+250)
			L_09ip.append(data[i][2])
		elif data[i][0] == b'SN2010mc':
			t_10mc.append(data[i][1]+250)
			L_10mc.append(data[i][2])
		elif data[i][0] == b'SN2015bh':
			t_15bh.append(data[i][1]+250)
			L_15bh.append(data[i][2])
		elif data[i][0] == b'SN2016bdu':
			t_16bdu.append(data[i][1]+250)
			L_16bdu.append(data[i][2])
		elif data[i][0] == b'SN2020tlf':
			t_20tlf.append(data[i][1]+500)
			L_20tlf.append(data[i][2])
		elif data[i][0] == b'SN2019zrk':
			t_19zrk.append(data[i][1]+250)
			L_19zrk.append(data[i][2])
		elif data[i][0] == b'SN2006jc':
			t_06jc.append(data[i][1]+10)
			L_06jc.append(data[i][2])
		elif data[i][0] == b'SN2019uo':
			t_19uo.append(data[i][1]+360)
			L_19uo.append(data[i][2])
	pk_09ip, pk_10mc, pk_15bh, pk_16bdu, pk_20tlf, pk_19zrk  = np.argmax(L_09ip), np.argmax(L_10mc), np.argmax(L_15bh), np.argmax(L_16bdu), np.argmax(L_20tlf), np.argmax(L_19zrk)
	# LC
	ax_lc_IIn.scatter(t_09ip[:pk_09ip], L_09ip[:pk_09ip], color='limegreen', label='2009ip', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_09ip[pk_09ip:], L_09ip[pk_09ip:], color='limegreen', facecolors='None', s=plt.rcParams['lines.markersize'])
	ax_lc_IIn.scatter(t_10mc[:pk_10mc], L_10mc[:pk_10mc], color='magenta', label='2010mc', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_10mc[pk_10mc:], L_10mc[pk_10mc:], color='magenta', facecolors='None', s=plt.rcParams['lines.markersize'])
	ax_lc_IIn.scatter(t_15bh[:pk_15bh], L_15bh[:pk_15bh], color='gold', label='2015bh', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_15bh[pk_15bh:], L_15bh[pk_15bh:], color='gold', facecolors='None', s=plt.rcParams['lines.markersize'])
	ax_lc_IIn.scatter(t_16bdu[:pk_16bdu], L_16bdu[:pk_16bdu], color='cyan', label='2016bdu', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_16bdu[pk_16bdu:], L_16bdu[pk_16bdu:], color='cyan', facecolors='None', s=plt.rcParams['lines.markersize'])
	ax_lc_IIn.scatter(t_19zrk[:pk_19zrk], L_19zrk[:pk_19zrk], color='black', label='2019zrk', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_19zrk[pk_19zrk:], L_19zrk[pk_19zrk:], color='black',facecolors='None', s=plt.rcParams['lines.markersize'])
	ax_lc_IIn.scatter(t_20tlf[:pk_20tlf], L_20tlf[:pk_20tlf], color='red', label='2020tlf', s=0.5*plt.rcParams['lines.markersize']**2)
	ax_lc_IIn.scatter(t_20tlf[pk_20tlf:], L_20tlf[pk_20tlf:], color='red', facecolors='None', s=plt.rcParams['lines.markersize'])
	for ax in ax_lc_Ibn:
		ax.plot(t_06jc, L_06jc, color='brown', marker='o', ms=7, mec='brown', mfc='brown', label='2006jc')	
		ax.plot(t_19uo, L_19uo, color='purple', marker='o', ms=7,  mec='purple', mfc='purple', label='2019uo')	
	# velocity
	ax_vel_IIn.errorbar(0.10, 1100, 300, color='limegreen', linewidth=2, capsize=6)
	ax_vel_IIn.errorbar(0.26, 2000, 1000, color='magenta', linewidth=2, capsize=6)
	ax_vel_IIn.errorbar(0.42, 800, 200, color='gold', linewidth=2, capsize=6)
	ax_vel_IIn.errorbar(0.58, 400, 0, color='cyan', linewidth=2, capsize=6)
	ax_vel_IIn.errorbar(0.74, 350, 0, color='black', linewidth=2, capsize=6)
	ax_vel_IIn.errorbar(0.90, 125, 75, color='red', linewidth=2, capsize=6)
	for ax in ax_vel_Ibn:
		ax.errorbar(0.25, 2500, 500, color='brown', linewidth=2, capsize=6)
		ax.errorbar(0.75, 775, 225, color='purple', linewidth=2, capsize=6)


# 2D plot of observables as a function of M_CSM and a_bin
# (obs_type, obs_array) : (luminosity, L_prec), (duration, t_prec), (final velocity, v_prec)
def plot_MCSM_abin(obs_type, obs_array, model_name, a_bin_arr, MCSM_arr, Rstar):
	fig, ax = plt.subplots()
	ax.set_xlim(min(a_bin_arr)/Rstar, max(a_bin_arr)/Rstar)
	ax.set_ylim(min(MCSM_arr)/Msun, max(MCSM_arr)/Msun)
	ax.set_xscale('log')
	ax.set_yscale('log')
	# suppress scientific notation for x-axis
	ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
	ax.xaxis.set_minor_formatter(mticker.StrMethodFormatter("{x:g}"))
	plt.title(r'Prec. %s (%s)' % (obs_type, model_name))
	ax.set_xlabel(r'binary separation $a_{\rm bin}$ ($R_*$)')
	ax.set_ylabel(r'$M_{\rm CSM}$ [$M_\odot$]')
	if obs_type == 'luminosity':
		con = plt.contourf(a_bin_arr/Rstar, MCSM_arr/Msun, np.log10(obs_array),levels=np.linspace(38, 43, 20), cmap=cm.viridis)
		bar = plt.colorbar(con)
		bar.ax.set_ylabel(r'$\log_{10}L_{\rm prec}$ [erg s$^{-1}$]')
		bar.ax.set_yticks([38,39,40,41,42,43]) 
		bar.ax.set_yticklabels([38,39,40,41,42,43]) 
	elif obs_type == 'duration':
		con = plt.contourf(a_bin_arr/Rstar, MCSM_arr/Msun, np.log10(obs_array), levels=np.linspace(0., 3, 12), cmap=cm.viridis)
		bar = plt.colorbar(con)
		bar.ax.set_ylabel(r'$\log_{10}t_{\rm prec}$ [day]')
		bar.ax.set_yticks([1,2,3])
		bar.ax.set_yticklabels([1,2,3])
	elif obs_type == 'final velocity':
		con = plt.contourf(a_bin_arr/Rstar, MCSM_arr/Msun, np.log10(obs_array), levels=np.linspace(1.5, 4., 11), cmap=cm.viridis)
		bar = plt.colorbar(con)
		bar.ax.set_ylabel(r'$\log_{10}v_{\rm prec}$ [km s$^{-1}$]')
		bar.ax.set_yticks([1.5,2,2.5,3,3.5,4.0])
		bar.ax.set_yticklabels([1.5,2,2.5,3,3.5,4.0])
	else:
		raise ValueError('invalid obs_type')
	fig.tight_layout()
	plt.savefig('%s_grid_%s.pdf' % (obs_type.replace(' ', '_'), model_name))


# calculate light curve properties, in the region where 10-90 % of energy is radiated
# this assumes time array is evenly sampled
def calc_LC_Lt(time, lum, t_BH):
	# checks
	assert min(lum) >= 0.0, "Negative luminosity"
	#print("dt_start, dt_end:", time[1]-time[0], time[-1]-time[-2])
	assert abs((time[1]-time[0])-(time[-1]-time[-2]))/(time[-1]-time[-2]) < 1e-6, "uneven timestep"
	# extract only when t > tBH
	lum = lum[time > t_BH]
	time = time[time > t_BH]
	# get cumulative sum
	cum_lum = np.cumsum(lum)
	index_10 = np.argmin(np.abs(cum_lum-0.1*cum_lum[-1]))
	index_90 = np.argmin(np.abs(cum_lum-0.9*cum_lum[-1]))
	# compute characteristic timescale and luminosity
	char_time = (time[index_90]-time[index_10])/86400
	char_lum = (cum_lum[index_90] - cum_lum[index_10]) * (time[1]-time[0])/(time[index_90]-time[index_10])
	return char_time, char_lum 


# rho_CSM at (a_bin, t)
def rho_CSM(MCSM, a_bin, t, v_esc, xi, chi):
	return (3.-s)*MCSM*t**(-3)*(a_bin/t)**(-s)/(4.*math.pi*v_esc**(3.-s)*(xi**(3.-s)-chi**(3.-s))) 

def evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion):
	chi = (1.-Rstar/a_bin)**0.5
	v_esc = (2.*G*Mstar/Rstar)**0.5
	v_orb = (2.*G*(Mstar+MCO)/a_bin)**(0.5)
	t_BH = a_bin/v_esc/xi
	BH_flag = 0
	print("MCSM=%g Msun, a_bin=%g Rstar, t_BH=%g day" % (MCSM/Msun, a_bin/Rstar, t_BH/86400))
	# initialize arrays
	if Rstar > 30.*Rsun:
		t_arr = np.linspace(0., min(30.*t_BH/86400., 3000), 100000)*86400
	else:
		t_arr = np.linspace(0., 100, 100000)*86400
	r_arr = np.zeros(len(t_arr)) 
	v_arr = np.zeros(len(t_arr)) 
	Eint_arr = np.zeros(len(t_arr)) 
	L_arr = np.zeros(len(t_arr)) 
	Lwind_arr = np.zeros(len(t_arr)) 
	x_arr = np.zeros(len(t_arr))

	# initial condition
	i = 0
	t = 0.0e0
	t_rec = 1e10 # time when recombination starts
	r = Rstar
	x = 1.0e0
	x_sat = 1.0e0
	# 0.5*MCSM*v^2 = Ekin = Eint = 0.15*MCSM*(xi*v_esc)^2 # 0.5*(0.5*MCSM*(xi*v_esc)^2)
	v = xi*v_esc*math.sqrt(0.3)
	Eint = 0.15*MCSM*(xi*v_esc)**2

	while t<t_arr[-1]:
		if (t > t_BH and t < a_bin/v_esc/chi): 
			Rbondi = G * MCO / ((a_bin/t)**2 + v_orb**2)
			Mdotacc = 4.*math.pi*Rbondi**2 * ((a_bin/t)**2 + v_orb**2)**0.5 * rho_CSM(MCSM, a_bin, t, v_esc, xi, chi)
			if vary_eps_wind_with_BB and Rbondi > 0:
				# obtain average vwind, and then efficiency is 0.5*(vwind/c)^2.
				r_disk_out = min(Rbondi, (Rbondi**2/t)**2/(G*MCO))
				if r_disk_out > r_disk_in:
					vwind = math.sqrt(p_BB/(1.-p_BB) * G*MCO/r_disk_in * ((r_disk_in/r_disk_out)**p_BB - (r_disk_in/r_disk_out))) 
				else: # disk does not form outside ISCO, so no wind
					vwind = 0.0
				# get eps_wind from equation: Lwind = 0.5 * Mdotacc * vwind**2 = eps_wind * Mdotacc * c^2
				eps_wind = 0.5*(vwind/c)**2
			else:
				eps_wind = eps_wind_const

			Lwind = eps_wind * Mdotacc * c**2
		else:
			Lwind = 0.0
		# diffusion time when fully ionized 
		tdiff = 3.*kappa*MCSM/4./math.pi/c/r
		# timestep when fully ionized
		dt = 0.01*min(tdiff, r/v, 10.*(t_arr[1]-t_arr[0]))

		#if x == 1.0e0:
		if abs(x-1.0e0) < 1e-3:
			pdV = v*Eint/r
			r += dt * v
			v += dt * Eint/MCSM/r / 0.6 # 0.6 comes from Ekin=0.3MCSM v^2 
			Lrad = Eint/tdiff/x**2
			Eint += dt * (-pdV + Lwind - Lrad)
			x = min(1.0, math.sqrt(Lrad/4./math.pi/r**2/sigma_SB/T_ion**4))
			if t > 0.0:
				t_rec = t
			if BH_flag==0 and Lwind > 0.0:
				BH_flag = 1
		#elif x < 1.0e0 and x > 0.0e0:
		elif x > 0.0e0:
			if BH_flag==0 and Lwind > 0.0:
				BH_flag = 1
				param = r/(4.*v*tdiff)
				x_sat = -param+math.sqrt(param**2 + 2.*param*Lwind/4./math.pi/r**2/sigma_SB/T_ion**4)
				x_sat = max(x_ion_floor, x_sat)
				print('ionization saturates at x=', min(1.,x_sat), '. Wind lum=', Lwind)

			# limit timestep by not making x_i rise too fast, when wind is turned on
			#print(t, Lwind, dt, 0.01*tdiff, 0.01*r/v)
			if Lwind > 0.0:
				# keep update x_sat
				param = r/(4.*v*tdiff)
				x_sat = -param+math.sqrt(param**2 + 2.*param*Lwind/4./math.pi/r**2/sigma_SB/T_ion**4)
				x_sat = max(x_ion_floor, x_sat)
				#dt = min(dt, 0.05*min(1.,x_sat)*x**3*tdiff*4.*math.pi*r**2*sigma_SB*T_ion**4/(Lwind+1.e0))
				dt = min(dt, 0.05*min(1.,x_sat)*x*tdiff / abs(Lwind/(4.*math.pi*x**2*r**2*sigma_SB*T_ion**4) - 1.))
			xr_old = x*r
			x -= (dt/5.)*(2.*v/r + (1.0 - Lwind/4./math.pi/xr_old**2/sigma_SB/T_ion**4)/x/tdiff)
			# as ionizations goes as runaway, set a floor value
			x = min(max(x_ion_floor, x),1.0)

			r += dt * v
			xr = x*r
			eps_int = 3.*x*tdiff/r * sigma_SB*T_ion**4
			v += 4.*math.pi/3.*(xr**3-xr_old**3)*eps_int/(3.*MCSM*v) / 0.6 # 0.6 comes from Ekin=0.3MCSM v^2 
			Eint = eps_int*(4.*math.pi/3.)*(xr)**3
			Lrad = 4.*math.pi*(xr)**2*sigma_SB*T_ion**4 
		else:
			raise ValueError("Invalid ionization locaton: x=%g at t=%g day" % (x, t/86400.))
		
		# FOR DEBUGGING
		#if Rstar == 50.*Rsun:
		#	print("t=%3g day: x=%4g, r=%4g, v=%4g, Eint=%3g, Lwind=%3g, Lrad=%3g" % (t/86400., x, r, v, Eint, Lwind, Lrad))
		if t > t_arr[i]:
			# append all parameters
			r_arr[i] = r
			v_arr[i] = v
			Eint_arr[i] = Eint
			L_arr[i] = Lrad
			Lwind_arr[i] = Lwind
			x_arr[i] = x
			i += 1
		t += dt
	# find time"step" when recombination started
	i_rec = np.argmin(abs(t_arr - t_rec))
	#print(t_rec, i_rec, max(x_arr))
	return t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH


###############################################
#                                             # 
#         Representative light curves         #
#                                             # 
###############################################

for CO in ['NS']:
	# parameters regarding CO and accretion wind
	print('Compact object: %s' % CO)
	if CO == 'BH':
		MCO = 10.*Msun
	elif CO == 'NS':
		MCO = 1.4*Msun
	else:
		raise ValueError('undefined compact object')
	if vary_eps_wind_with_BB:
		p_BB = 0.5 # power-law index of accretion rate in ADIOS solution (0.3-0.8; Yuan & Narayan 14)
		r_disk_in = 6.*G*MCO/c**2 # ISCO


	fig, [(ax1, ax2, ax7), (ax3, ax4, ax8), (ax5, ax6, ax9)] = plt.subplots(3,3, figsize=(12,12), gridspec_kw={'width_ratios': [4, 4, 1], 'wspace': 0.03}, constrained_layout=True)
	for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
		ax.set_xlabel('day from eruption')
		ax.set_yscale('log')
		ax.grid(linestyle=':')
	for ax in [ax1,ax2]:
		ax.set_title(r'$(10\ M_\odot,\ 500\ R_\odot)$ RSG-%s model' % CO, fontsize=15)
		ax.set_xlim(0, 600.)
	for ax in [ax3,ax4]:
		ax.set_title(r'$(5\ M_\odot, \ 5\ R_\odot)$ HeHighMass-%s model' % CO, fontsize=15)
		ax.set_xlim(0.01, 100.)
		ax.set_xscale('log')
	for ax in [ax5,ax6]:
		ax.set_title(r'$(3\ M_\odot, \ 50\ R_\odot)$ HeLowMass-%s model' % CO, fontsize=15)
		ax.set_xlim(0, 100.)
	for ax in [ax1, ax3, ax5]:
		ax.set_ylim(1e38,1e46)
		ax.set_ylabel('luminosity [erg s$^{-1}$]')
	ax1.set_ylim(1e38,1e45)
	for ax in [ax2, ax4, ax6]:
		ax.set_ylabel('CSM velocity [km s$^{-1}$]')

	# luminosity limit for single stars (Sec 2)
	long_range = np.linspace(0, 1e10)
	ax1.plot(long_range, xi**2/3. * 4.*math.pi*G*10.*Msun*c/0.3*np.ones(len(long_range)), linestyle='dashdot', color='black')
	ax3.plot(long_range, xi**2/3. * 4.*math.pi*G*5.*Msun*c/0.1*np.ones(len(long_range)), linestyle='dashdot', color='black')
	ax5.plot(long_range, xi**2/3. * 4.*math.pi*G*3.*Msun*c/0.1*np.ones(len(long_range)), linestyle='dashdot', color='black')

	ax2.set_ylim(3e1, 5e3)
	ax7.set_ylim(3e1, 5e3)
	ax4.set_ylim(1e2, 1e4)
	ax8.set_ylim(1e2, 1e4)
	ax6.set_ylim(1e2, 5e3)
	ax9.set_ylim(1e2, 5e3)
	for ax in [ax7,ax8,ax9]:
		ax.set_xlim(0,1)
		ax.set_yscale('log')
		ax.set_ylabel('observed CSM velocity [km s$^{-1}$]')
		ax.yaxis.set_label_position("right")
		ax.yaxis.tick_right()
		ax.set(xticklabels=[])  # remove the x tick labels
		ax.tick_params(bottom=False)  # remove the x ticks

	# RSG model
	print("RSG model")
	Mstar = 10.*Msun
	Rstar = 500.*Rsun
	kappa = 0.3
	T_ion = 6e3
	# binary
	for (i, (MCSM, a_bin)) in enumerate([(0.1*Msun, 5*Rstar), (0.1*Msun, 15*Rstar), (1*Msun, 5*Rstar)]):
		t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
		ax1.plot(t_arr/86400., L_arr, color=color_array[i])
		ax1.plot(t_arr/86400., Lwind_arr, linestyle='dotted', color=color_array[i])
		ax2.plot(t_arr/86400., v_arr/1e5, label=r'$%g R_*$, $%g M_\odot$' % (a_bin/Rstar, MCSM/Msun), color=color_array[i])
		np.savetxt('prec_RSG_%s_R%g_M%g_xi%g.txt' % (CO, a_bin/Rstar, MCSM/Msun, xi), np.c_[t_arr,L_arr,v_arr])
	# single star
	t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, 0.0, Rstar, kappa, 10*Rstar, 0.1*Msun, T_ion)
	ax1.plot(t_arr/86400., L_arr, linestyle='dashed')
	ax2.plot(t_arr/86400., v_arr/1e5, linestyle='dashed', label=r'single, $0.1M_\odot$') 


	# HeHigh model
	print("High mass He model")
	Mstar = 5.*Msun
	Rstar = 5.*Rsun
	kappa = 0.1
	T_ion = 1e4
	# binary
	for (i, (MCSM, a_bin)) in enumerate([(0.1*Msun, 5*Rstar), (0.1*Msun, 15*Rstar), (1*Msun, 5*Rstar)]):
		t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
		ax3.plot(t_arr/86400., L_arr, color=color_array[i])
		ax3.plot(t_arr/86400., Lwind_arr, linestyle='dotted', color=color_array[i])
		ax4.plot(t_arr/86400., v_arr/1e5, label=r'$%g R_*$, $%g M_\odot$' % (a_bin/Rstar, MCSM/Msun), color=color_array[i])
		np.savetxt('prec_HeHigh_%s_R%g_M%g_xi%g.txt' % (CO, a_bin/Rstar, MCSM/Msun, xi), np.c_[t_arr,L_arr,v_arr])
	# single star
	t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, 0.0, Rstar, kappa, 10*Rstar, 0.1*Msun, T_ion)
	ax3.plot(t_arr/86400., L_arr, linestyle='dashed')
	ax4.plot(t_arr/86400., v_arr/1e5, linestyle='dashed', label=r'single, $0.1M_\odot$') 



	# HeLow model
	print("Low mass He model")
	Mstar = 3.*Msun
	Rstar = 50.*Rsun
	kappa = 0.1
	T_ion = 1e4
	# binary
	for (i, (MCSM, a_bin)) in enumerate([(0.1*Msun, 5*Rstar), (0.1*Msun, 15*Rstar), (1*Msun, 5*Rstar)]):
		t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
		ax5.plot(t_arr/86400., L_arr, color=color_array[i])
		ax5.plot(t_arr/86400., Lwind_arr, linestyle='dotted', color=color_array[i])
		ax6.plot(t_arr/86400., v_arr/1e5, label=r'$%g R_*$, $%g M_\odot$' % (a_bin/Rstar, MCSM/Msun), color=color_array[i])
		np.savetxt('prec_HeLow_%s_R%g_M%g_xi%g.txt' % (CO, a_bin/Rstar, MCSM/Msun, xi), np.c_[t_arr,L_arr,v_arr])
	# single star
	t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, 0.0, Rstar, kappa, 10*Rstar, 0.1*Msun, T_ion)
	ax5.plot(t_arr/86400., L_arr, linestyle='dashed')
	ax6.plot(t_arr/86400., v_arr/1e5, linestyle='dashed', label=r'single, $0.1M_\odot$') 


	# plot observation
	plot_prec_obs_data(ax1, ax7, [ax3, ax5], [ax8, ax9], 'data_lumi_obs.txt')

	ax1.legend(loc='upper right', fontsize=9.5, ncol=3)
	ax2.legend(loc='upper right', fontsize=11, ncol=2)
	ax3.legend(loc='upper right', fontsize=11, ncol=2)
	ax5.legend(loc='upper right', fontsize=11, ncol=2)
	plt.tight_layout()
	plt.savefig('precursorLC-%s.pdf' % CO)


	###############################################
	#                                             # 
	#        2D contour of t_arr and L_arr        #
	#                                             # 
	###############################################

	# RSG model
	print("RSG model (2D)")
	Mstar = 10.*Msun
	Rstar = 500.*Rsun
	kappa = 0.3
	T_ion = 6e3

	MCSM_arr = np.logspace(-1, 0.5, 10) * Msun 
	a_bin_arr = np.logspace(math.log10(3.), math.log10(30.), 10) * Rstar 
	t_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	L_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	v_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))

	for i,MCSM in enumerate(MCSM_arr):
		for j,a_bin in enumerate(a_bin_arr):
			t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
			t_prec[i,j], L_prec[i,j] = calc_LC_Lt(t_arr, L_arr, t_BH)
			v_prec[i,j] = max(v_arr)/1e5
			print("time=%.3e day, lum=%.3e erg/s, vel=%.3e km/s" % (t_prec[i,j], L_prec[i,j], v_prec[i,j])) 

	# 2D plot of luminosity
	plot_MCSM_abin('luminosity', L_prec, 'RSG-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of duration 
	plot_MCSM_abin('duration', t_prec, 'RSG-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of final velocity
	plot_MCSM_abin('final velocity', v_prec, 'RSG-%s' % CO, a_bin_arr, MCSM_arr, Rstar)

	np.savetxt('prec_RSG_%s.txt' % CO, np.transpose([t_prec.flatten(), L_prec.flatten()]))

	# HeHighMass model
	print("HeHighMass model (2D)")
	Mstar = 5.*Msun
	Rstar = 5.*Rsun
	kappa = 0.1
	T_ion = 1e4

	MCSM_arr = np.logspace(-2, 0, 10) * Msun
	a_bin_arr = np.logspace(math.log10(3.), math.log10(30.), 10) * Rstar 
	t_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	L_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	v_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))

	for i,MCSM in enumerate(MCSM_arr):
		for j,a_bin in enumerate(a_bin_arr):
			t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
			t_prec[i,j], L_prec[i,j] = calc_LC_Lt(t_arr, L_arr, t_BH)
			v_prec[i,j] = max(v_arr)/1e5
			print("time=%.3e day, lum=%.3e erg/s, vel=%.3e km/s" % (t_prec[i,j], L_prec[i,j], v_prec[i,j])) 

	# 2D plot of luminosity
	plot_MCSM_abin('luminosity', L_prec, 'HeHighMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of duration 
	plot_MCSM_abin('duration', t_prec, 'HeHighMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of final velocity
	plot_MCSM_abin('final velocity', v_prec, 'HeHighMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)

	np.savetxt('prec_HeHighMass_%s.txt' % CO, np.transpose([t_prec.flatten(), L_prec.flatten()]))

	# HeLowMass-NS model
	print("HeLowMass model (2D)")
	Mstar = 3.*Msun
	Rstar = 50.*Rsun
	kappa = 0.1
	T_ion = 1e4

	MCSM_arr = np.logspace(-2, 0, 10) * Msun
	a_bin_arr = np.logspace(math.log10(3.), math.log10(30.), 10) * Rstar 
	t_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	L_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
	v_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))

	for i,MCSM in enumerate(MCSM_arr):
		for j,a_bin in enumerate(a_bin_arr):
			t_arr, L_arr, Lwind_arr, v_arr, x_arr, i_rec, t_BH = evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion)
			t_prec[i,j], L_prec[i,j] = calc_LC_Lt(t_arr, L_arr, t_BH)
			v_prec[i,j] = max(v_arr)/1e5
			print("time=%.3e day, lum=%.3e erg/s, vel=%.3e km/s" % (t_prec[i,j], L_prec[i,j], v_prec[i,j])) 

	# 2D plot of luminosity
	plot_MCSM_abin('luminosity', L_prec, 'HeLowMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of duration 
	plot_MCSM_abin('duration', t_prec, 'HeLowMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)
	# 2D plot of final velocity
	plot_MCSM_abin('final velocity', v_prec, 'HeLowMass-%s' % CO, a_bin_arr, MCSM_arr, Rstar)

	np.savetxt('prec_HeLowMass_%s.txt' % CO, np.transpose([t_prec.flatten(), L_prec.flatten()]))
