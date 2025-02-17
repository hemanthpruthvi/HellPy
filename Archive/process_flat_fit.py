from process_files import *
from func_flats import *

def process_flat_fit_parallel(line, channel, mod):
    # Read the settings
    config = configobj.ConfigObj('config.ini')
    ffdir = config['flats']['directory']
    settings = [f for f in os.listdir(ffdir) if 'settings' in f]
    settings = ffdir + os.sep + settings[0]
    settings = configobj.ConfigObj(settings)

    # Other numbers
    iline = 0
    linestr = 'Line_' + str(get_line_num(settings, line, iline))
    nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
    nwav = int(settings[linestr]['NWavePoints'])
    filtstr = settings[linestr]['Filter']
    modstr = settings[linestr]['Polarimeter\\Modulation']
    nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
    nfpc = nmod*nacc*nwav
    nfpw = nmod*nacc
    ncyc = len(os.listdir(ffdir+os.sep+line))//3
    # Read the data
    ffname = config['flats'][line+'/'+channel]
    ff = np.float64(pf.open(ffname)[0].data)
    print(time_now(), 'reading average flat files: ', '\n', ffname)
    Y, X = ff.shape[0:2]
    ffm = ff[:,:,mod,:]
    
    # Fit the data
    fit_params = fit_lines_3d_parallel_poly2(ffm, nparallel=16)
    
    # Save the data
    ffit_name = ffname.replace('_fa', '_mod'+str(mod)+'_ft')
    hdu1 = pf.PrimaryHDU(fit_params)
    hdul = pf.HDUList([hdu1])
    hdul.writeto(ffit_name, overwrite=True)
    hdul.close()
    config['flats'][line+'/'+channel+str(mod)+'_fit'] = ffit_name
    config.write()
    print(time_now(), 'fit data written to: ', '\n', ffit_name)
    return

def process_flat_fit(line, channel, mod):
    # Read the settings
    config = configobj.ConfigObj('config.ini')
    ffdir = config['flats']['directory']
    settings = [f for f in os.listdir(ffdir) if 'settings' in f]
    settings = ffdir + os.sep + settings[0]
    settings = configobj.ConfigObj(settings)

    # Other numbers
    iline = 0
    linestr = 'Line_' + str(get_line_num(settings, line, iline))
    nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
    nwav = int(settings[linestr]['NWavePoints'])
    filtstr = settings[linestr]['Filter']
    modstr = settings[linestr]['Polarimeter\\Modulation']
    nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
    nfpc = nmod*nacc*nwav
    nfpw = nmod*nacc
    ncyc = len(os.listdir(ffdir+os.sep+line))//3
    # Read the data
    ffname = config['flats'][line+'/'+channel]
    ff = np.float64(pf.open(ffname)[0].data)
    print(time_now(), 'reading average flat files: ', '\n', ffname)
    Y, X = ff.shape[0:2]
    ffm = ff[:,:,mod,:]
    
    # Fit the data
    guess_params, fit_params = fit_lines_3d(ffm)
    
    # Save the data
    ffit_name = ffname.replace('_fa', '_mod'+str(mod)+'_ft')
    hdu1 = pf.PrimaryHDU(fit_params)
    hdul = pf.HDUList([hdu1])
    hdul.writeto(ffit_name, overwrite=True)
    hdul.close()
    config['flats'][line+'/'+channel+'_mod'str(mod)+'_fit'] = ffit_name
    config.write()
    print(time_now(), 'fit data written to: ', '\n', ffit_name)
    return

# line = 'Fe_I_6173'
# config = configobj.ConfigObj('config.ini')
# ffdir = config['flats']['directory']
# settings = [f for f in os.listdir(ffdir) if 'settings' in f]
# settings = ffdir + os.sep + settings[0]
# settings = configobj.ConfigObj(settings)
# # process_flat_fit(line, 'pol1', 0)
# process_flat_fit(line, 'pol1', 1)
# process_flat_fit(line, 'pol1', 2)
# process_flat_fit(line, 'pol1', 31)
line = 'Fe_I_6173'
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