
from process_files import *
from func_flats import *
config = configobj.ConfigObj('config.ini')
# line = config['line']
line = 'Fe_I_6302'
ffdir = config['flats']['directory']
settings = [f for f in os.listdir(ffdir) if 'settings' in f]
settings = ffdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)

# Other numbers
iline = get_line_num(settings, line)
linestr = 'Line_' + str(iline)
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
wavscan_range = float(settings[linestr]['WaveScanRange'])
wavelength = float(settings[linestr]['LineWavelength'])
wavestep = wavscan_range/(nwav-1)
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
ncyc = len(os.listdir(ffdir+os.sep+line))//3

#
def get_fts_spectra(file_name, wavelength=630.2e-9, wave_range=0.3e-9):
    """
    Initialize the oject by reading spectrum from the file
    Input:  spectrum file name with intensity vs. wavelength data
    """
    wave_step=1e-14
    spectra = np.loadtxt(file_name, delimiter=',')
    waves_full = 1.0e-10*spectra[:,0]
    intens_full = 1.0e-2*spectra[:,1]
    wave_beg, wave_end = wavelength-wave_range/2.0, wavelength+wave_range/2.0
    waves = np.arange(wave_beg, wave_end, wave_step)
    iwave_line = np.argwhere((waves_full>waves[0]) & (waves_full<waves[-1]))
    waves_line = waves_full[iwave_line]
    intens_line = intens_full[iwave_line]
    intens = zoom(intens_line.flatten(), len(waves)/len(waves_line))
    nwaves = len(waves)
    return waves, intens

ffname = config['flats'][line+'/pol1']
ff = np.float64(pf.open(ffname)[0].data)[:,:,0]
obspec = ff[640,640]
plt.plot(obspec/obspec.max())

# broadening = 0.5
# line_shift = 1
# broadening_ = broadening*wavestep*1e4
# line_shift_ = int(line_shift*wavestep*1e4)
# wavscan_ = int(1e4*wavscan_range)
waves, intens = get_fts_spectra('solar_spectrum_fts.csv', wavelength=wavelength*1e-10, wave_range=1.2*wavscan_range*1e-10)

def fts_spectral_line(x, intens, broad, linesh, wavscan_range):
    factor = 1e4
    nwav = len(x)
    wavstep = wavscan_range*factor/(nwav-1)
    indices = x*wavstep + (len(intens)-wavscan_range*factor)//2
    #
    intens_ = shift(intens, linesh*wavstep, mode='nearest')
    intens_ = gaussian_filter(intens_, broad*wavstep)
    intens_ = intens_[np.int32(indices)]
    return intens_
#
nknots = 5
xdata = np.arange(nwav)
model_line = Model(fts_spectral_line, independent_vars=['x', 'intens'])
model_cont = SplineModel(np.linspace(0,nwav,nknots))
# model_cont = LorentzianModel()
params = model_cont.guess(obspec, x=xdata)
params.add('wavscan_range', value=wavscan_range, vary=False)
params.add('broad', value=0.5, min=0, max=2)
params.add('linesh', value=0, min=-25, max=25)
model = model_cont*model_line
res = model.fit(obspec, params, x=xdata, intens=intens)
fitline = model_line.eval(res.params,x=xdata,intens=intens)
fitcont = model_cont.eval(res.params,x=xdata)
#
# Y, X = ff.shape[0:2]
# ff_cont = 0.0*ff
# ff_broad = np.zeros([Y,X])
# ff_linesh = np.zeros([Y,X])
# for i in tqdm.tqdm(range(Y)):
#     for j in range(X):
#         obspec = ff[i,j]
#         res = model.fit(obspec, params, x=xdata, intens=intens)
#         ff_cont[i,j] = obspec/model_line.eval(res.params, x=xdata, intens=intens)
#         ff_broad[i,j] = res.best_values['broad']
#         ff_linesh[i,j] = res.best_values['linesh']
# # #
plt.figure()
plt.plot(obspec, 'k')
# plt.plot(res.init_fit, 'c')
plt.plot(res.best_fit, 'm')
plt.plot(fitcont, 'g')
plt.plot(obspec/fitline, 'b')

res