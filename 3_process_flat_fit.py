# %%
%matplotlib qt5
from process_files import *
from func_flats import *

# %%
#
def process_flat_fit(line, channel, mod):
    #
    config = configobj.ConfigObj('config.ini')
    ffdir = config['flats']['directory']
    settings = [f for f in os.listdir(ffdir) if 'settings' in f]
    settings = ffdir + os.sep + settings[0]
    settings = configobj.ConfigObj(settings)
    # Other numbers
    iline = get_line_num(settings, line)
    linestr = 'Line_' + str(iline)
    nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
    nwav = int(settings[linestr]['NWavePoints'])
    wavescan_range = float(settings[linestr]['WaveScanRange'])
    wavelength = float(settings[linestr]['LineWavelength'])
    wavestep = wavescan_range/(nwav-1)
    filtstr = settings[linestr]['Filter']
    modstr = settings[linestr]['Polarimeter\\Modulation']
    nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
    nfpc = nmod*nacc*nwav
    nfpw = nmod*nacc
    ncyc = len(os.listdir(ffdir+os.sep+line))//3
    #
    ffname = config['flats'][line+'/'+channel]
    ff = np.float64(pf.open(ffname)[0].data)[:,:,mod]
    print(time_now(), 'reading average flat files: ', '\n', ffname)
    spectrum = get_fts_spectra('solar_spectrum_fts.csv', wavelength=wavelength*1e-10, wave_range=2*wavescan_range*1e-10)
    factor_avg = spectrum[1].mean()
    factor_max = spectrum[1][len(spectrum[1])//4:-len(spectrum[1])//4].max()
    #
    nknots = (nwav-1)//3+1
    xdata = np.arange(nwav)
    model_line = Model(real_spectral_line, independent_vars=['x', 'spectrum'])
    model_cont = SplineModel(np.linspace(0,nwav,nknots))
    params = model_cont.guess(np.zeros(nwav), x=xdata)
    params.add('wavelength', value=wavelength, vary=False)
    params.add('wavescan_range', value=wavescan_range, vary=False)
    params.add('broad', value=1, min=0, max=2, vary=True)
    params.add('linesh', value=0, min=-nwav, max=nwav)
    model = model_cont*model_line
    #
    Y, X = ff.shape[0:2]
    ff_line = 0.0*ff
    ff_cont = 0.0*ff
    ff_broad = np.zeros([Y,X])
    ff_linesh = np.zeros([Y,X])
    for i in tqdm.tqdm(range(Y)):
        for j in range(X):
            try:
                for s in range(nknots):
                    params.add('s'+str(s), value=ff[i,j].max()/factor_avg)
                res = model.fit(ff[i,j], params, x=xdata, spectrum=spectrum)
                ff_line[i,j] = model_line.eval(res.params, x=xdata, spectrum=spectrum)
                ff_cont[i,j] = model_cont.eval(res.params, x=xdata)
                ff_broad[i,j] = res.best_values['broad']
                ff_linesh[i,j] = res.best_values['linesh']
            except:
                for s in range(nknots):
                    params.add('s'+str(s), value=ff[i,j].max()/factor_max)
                res = model.fit(ff[i,j], params, x=xdata, spectrum=spectrum)
                ff_line[i,j] = model_line.eval(res.params, x=xdata, spectrum=spectrum)
                ff_cont[i,j] = model_cont.eval(res.params, x=xdata)
                ff_broad[i,j] = res.best_values['broad']
                ff_linesh[i,j] = res.best_values['linesh']
    #
    ffit_name = ffname.replace('_fa', '_mod'+str(mod)+'_ft')
    hdu1 = pf.PrimaryHDU(ff)
    hdu2 = pf.ImageHDU(ff_cont)
    hdu3 = pf.ImageHDU(ff_line)
    hdu4 = pf.ImageHDU(ff_linesh)    
    hdu5 = pf.ImageHDU(ff_broad)
    hdul = pf.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5])
    hdul.writeto(ffit_name, overwrite=True)
    hdul.close()
    config = configobj.ConfigObj('config.ini')
    config['flats'][line+'/'+channel+'_mod'+str(mod)+'_fit'] = ffit_name
    print(line+'/'+channel+'_mod'+str(mod)+'_fit', ffit_name)
    config.write()
    print(time_now(), 'fit data written to: ', '\n', ffit_name)
# # #
# res = model.fit(obspec, params, x=xdata, spectrum=spectrum)
# fitline = model_line.eval(res.params,x=xdata,spectrum=spectrum)
# fitcont = model_cont.eval(res.params,x=xdata)
# plt.figure()
# plt.plot(obspec, 'k')
# plt.plot(res.init_fit, 'c')
# plt.plot(res.best_fit, 'm')
# plt.plot(fitcont, 'g')
# plt.plot(obspec/fitline, 'b')
# res

# %%
config = configobj.ConfigObj('config.ini')
line = config['line']
pool = mp.Pool(8)
mp_args = []
mp_args.append([line, 'pol1', 0])
mp_args.append([line, 'pol1', 1])
mp_args.append([line, 'pol1', 2])
mp_args.append([line, 'pol1', 3])
mp_args.append([line, 'pol2', 0])
mp_args.append([line, 'pol2', 1])
mp_args.append([line, 'pol2', 2])
mp_args.append([line, 'pol2', 3])
pool.starmap(process_flat_fit, mp_args)

# %%
line, channel, mod = 'Ca_II_8542', 'pol1', 0
config = configobj.ConfigObj('config.ini')
ffdir = config['flats']['directory']
settings = [f for f in os.listdir(ffdir) if 'settings' in f]
settings = ffdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
# Other numbers
iline = get_line_num(settings, line)
linestr = 'Line_' + str(iline)
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
wavescan_range = float(settings[linestr]['WaveScanRange'])
wavelength = float(settings[linestr]['LineWavelength'])
wavestep = wavescan_range/(nwav-1)
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
ncyc = len(os.listdir(ffdir+os.sep+line))//3
#
ffname = config['flats'][line+'/'+channel]
ff = np.float64(pf.open(ffname)[0].data)[:,:,mod]
print(time_now(), 'reading average flat files: ', '\n', ffname)
spectrum = get_fts_spectra('solar_spectrum_fts.csv', wavelength=wavelength*1e-10, wave_range=2*wavescan_range*1e-10)
factor_avg = spectrum[1].mean()
factor_max = spectrum[1][len(spectrum[1])//4:-len(spectrum[1])//4].max()
#
Y, X = ff.shape[0:2]
nknots = (nwav-1)//3+1
xdata = np.arange(nwav)
model_line = Model(real_spectral_line, independent_vars=['x', 'spectrum'])
model_cont = SplineModel(np.linspace(0,nwav,nknots))
params = model_cont.guess(np.zeros(nwav), x=xdata)
params.add('wavelength', value=wavelength, vary=False)
params.add('wavescan_range', value=wavescan_range, vary=False)
params.add('broad', value=1, min=0, max=2, vary=True)
params.add('linesh', value=0, min=-25, max=25)
model = model_cont*model_line
#
ff_line = 0.0*ff
ff_cont = 0.0*ff
ff_broad = np.zeros([Y,X])
ff_linesh = np.zeros([Y,X])
for i in tqdm.tqdm(range(Y)):
    for j in range(X):
    # i, j = 31, 516
        try:
            for s in range(nknots):
                params.add('s'+str(s), value=ff[i,j].max()/factor_avg)
            res = model.fit(ff[i,j], params, x=xdata, spectrum=spectrum)
            ff_line[i,j] = model_line.eval(res.params, x=xdata, spectrum=spectrum)
            ff_cont[i,j] = model_cont.eval(res.params, x=xdata)
            ff_broad[i,j] = res.best_values['broad']
            ff_linesh[i,j] = res.best_values['linesh']
        except:
            print(i,j)
            for s in range(nknots):
                params.add('s'+str(s), value=ff[i,j].max()/factor_max)
            res = model.fit(ff[i,j], params, x=xdata, spectrum=spectrum)
            ff_line[i,j] = model_line.eval(res.params, x=xdata, spectrum=spectrum)
            ff_cont[i,j] = model_cont.eval(res.params, x=xdata)
            ff_broad[i,j] = res.best_values['broad']
            ff_linesh[i,j] = res.best_values['linesh']
        # #
# ffit_name = ffname.replace('_fa', '_mod'+str(mod)+'_ft')
# hdu1 = pf.PrimaryHDU(ff)
# hdu2 = pf.ImageHDU(ff_cont)
# hdu3 = pf.ImageHDU(ff_line)
# hdu4 = pf.ImageHDU(ff_linesh)    
# hdu5 = pf.ImageHDU(ff_broad)
# hdul = pf.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5])
# hdul.writeto(ffit_name, overwrite=True)
# hdul.close()
# config = configobj.ConfigObj('config.ini')
# config['flats'][line+'/'+channel+'_mod'+str(mod)+'_fit'] = ffit_name
# print(line+'/'+channel+'_mod'+str(mod)+'_fit', ffit_name)
# config.write()
# print(time_now(), 'fit data written to: ', '\n', ffit_name)

# %%
plt.plot(model.eval(params, x=xdata, spectrum=spectrum))
plt.plot(ff[i,j])
plt.plot(res.best_fit)
# plt.plot(res.init_fit)


