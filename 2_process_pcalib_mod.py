# %%
from process_files import *
from func_pcalib import *
from func_flats import *

# %%
config = configobj.ConfigObj('config.ini')
pcdir = config['pcalibration']['directory']
line = config['line']
settings = [f for f in os.listdir(pcdir) if 'settings' in f]
settings = pcdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
#
dk0 = np.float64(pf.open(config['darks'][line+'/bbi'])[0].data)
dk1 = np.float64(pf.open(config['darks'][line+'/pol1'])[0].data)
dk2 = np.float64(pf.open(config['darks'][line+'/pol2'])[0].data)
# flats
# ff_bbi = np.float64(pf.open(config['flats'][line+'/bbi'])[0].data)
# ff_bbi = ff_bbi[:,:,np.newaxis]/np.mean(ff_bbi)
# ff_pol = np.float64(pf.open(config['flats'][line+'/pol'])[0].data)
# ff_pol1 = ff_pol[:,:,0:4]
# ff_pol2 = ff_pol[:,:,4::]
#
dir_tree = pcdir.split(os.sep)
l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'
if not (os.path.exists(l0dir)): os.mkdir(l0dir)
l0dir += os.sep+dir_tree[-1]
if not (os.path.exists(l0dir)): os.mkdir(l0dir)
l0dir += os.sep+line
if not (os.path.exists(l0dir)): os.mkdir(l0dir)
print('Calibration data will be saved to ', l0dir)
config['pcalibration'][line+'/l0dir'] = l0dir
config.write()

# %%
npol = int(settings['VTTControls']['ICUPolarizer\\NPositions'])
nret = int(settings['VTTControls']['ICURetarder\\NPositions'])
pang0 = float(settings['VTTControls']['ICUPolarizer\\Start'])
pang1 = float(settings['VTTControls']['ICUPolarizer\\Stop'])
poffset = float(settings['VTTControls']['ICUPolarizer\\Offset'])
pzero = float(settings['VTTControls']['ICUPolarizer\\Zero'])
pfactor = float(settings['VTTControls']['ICUPolarizer\\Factor'])
rang0 = float(settings['VTTControls']['ICURetarder\\Start'])
rang1 = float(settings['VTTControls']['ICURetarder\\Stop'])
roffset = float(settings['VTTControls']['ICURetarder\\Offset'])
rzero = float(settings['VTTControls']['ICURetarder\\Zero'])
rfactor = float(settings['VTTControls']['ICURetarder\\Factor'])
ncyc = npol*nret
ncycobs = len(os.listdir(pcdir+os.sep+line))//3
print('Intended and actual recordings of polcal data: ', ncyc, ', ', ncycobs)

# Analyze log file
logfile = [f for f in os.listdir(pcdir) if 'log' in f]
logfile = pcdir + os.sep + logfile[0]
with open(logfile) as f:
    log = f.readlines()
poltemp = [int(l.split()[-1][4:-1]) for l in log if 'X10A' in l][0:ncycobs]
rettemp = [int(l.split()[-1][4:-1]) for l in log if 'X11A' in l][0:ncycobs]
pang = np.array(poltemp)/pfactor - pzero
rang = np.array(rettemp)/rfactor - rzero
rnpos = sum(pang==pang0)
pnpos = sum(rang==rang1)
rnpos = 9
pnpos = 1
print('Retarder and polarizer positions as per logs: ', rnpos, pnpos)

# %%
# Other numbers
iline = get_line_num(settings, line, 0)
linestr = 'Line_' + str(get_line_num(settings, line))
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
# Time stamps
tsfile = [f for f in os.listdir(pcdir) if 'timestamp' in f]
tsfile = pcdir + os.sep + tsfile[0]
ts = np.loadtxt(tsfile, delimiter=',')
ts_line = ts[np.where(ts[:,0]==iline)]
#
im0ind_ = ts_line[:,3].reshape([ncyc,nfpc])
im1ind_ = ts_line[:,6].reshape([ncyc,nfpc])
im2ind_ = ts_line[:,9].reshape([ncyc,nfpc])
#
im0ind_ = np.int64(im0ind_ - im0ind_[:,0:1])
im1ind_ = np.int64(im1ind_ - im1ind_[:,0:1])
im2ind_ = np.int64(im2ind_ - im2ind_[:,0:1])
#
im0ind = im0ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im1ind = im1ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im2ind = im2ind_.reshape([ncyc, nwav, nacc, nmod])%nmod

# %%
pc0_, pc1_, pc2_ = [], [], []
for i in tqdm.tqdm(range((pnpos*rnpos))):
    dc0 = data_cube(pcdir, line, 0, i)
    Y, X, Z = dc0.data.shape
    pc0 = dc0.data.reshape([Y, X, nmod, nacc, nwav], order='F')
    pc0 = coadd_del_accumulations(pc0, im0ind[i])
    pc0 = np.mean(pc0[400:800,400:800], axis=(0,1))
    pc0_.append(pc0)
    #
    dc1 = data_cube(pcdir, line, 1, i)
    Y, X, Z = dc1.data.shape
    pc1 = dc1.data.reshape([Y, X, nmod, nacc, nwav], order='F')
    pc1 = np.flipud(pc1)
    pc1 = coadd_del_accumulations(pc1, im1ind[i])
    pc1 = np.mean(pc1[400:800,400:800], axis=(0,1))
    pc1_.append(pc1)
    #
    dc2 = data_cube(pcdir, line, 2, i)
    Y, X, Z = dc2.data.shape
    pc2 = dc2.data.reshape([Y, X, nmod, nacc, nwav], order='F')
    pc2 = coadd_del_accumulations(pc2, im2ind[i])
    pc2 = np.mean(pc2[400:800,400:800], axis=(0,1))
    pc2_.append(pc2)
pc0_, pc1_, pc2_ = np.array(pc0_), np.array(pc1_), np.array(pc2_)

# %%
beam0_ = np.sum(pc0_, axis=2).T
beam1_ = np.sum(pc1_, axis=2).T
beam2_ = np.sum(pc2_, axis=2).T
intcorr = np.average(beam1_.T+beam2_.T, axis=1)
intcorr /= intcorr.mean()
# np.mean(beam1.T+beam2.T, axis=1)
beam0 = beam0_/intcorr[np.newaxis,:]
beam1 = beam1_/intcorr[np.newaxis,:]
beam2 = beam2_/intcorr[np.newaxis,:]
# beam0, beam1, beam2 = beam0_, beam1_, beam2_
fig, axs = plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(intcorr)
axs[0,1].plot(beam0.T)
axs[1,0].plot(beam1.T)
axs[1,1].plot(beam2.T)
axs[0,1].plot(beam0_.T)
axs[1,0].plot(beam1_.T)
axs[1,1].plot(beam2_.T)
beam = np.concatenate([beam1, beam2], axis=0).flatten()

# %%
params = Parameters()
params.add('polang', value=0, min=-360, max=360, vary=False)
params.add('retang', value=-5, min=-360, max=360, vary=True)
params.add('wpret', value=90, min=-360, max=360, vary=True)
params.add('intens1', value=beam1.mean())
params.add('intens2', value=beam2.mean() )

mod_model = Model(fit_beam_mods)
result = mod_model.fit(beam, params, xdata=beam)
#
beam_ = result.best_fit.flatten()
beam_ = np.reshape(beam_, [8,len(beam_)//8])
beam1_ = beam_[0:4,:]
beam2_ = beam_[4::,:]
#
#
plt.figure()
plt.plot(beam1_.T,'c')
plt.plot(beam1.T,'k:')
plt.figure()
plt.plot(beam2_.T, 'c')
plt.plot(beam2.T, 'k:')
#
result

# %%
polang = result.best_values['polang']
retang = result.best_values['retang']
wpret = result.best_values['wpret']
#
s_in = compute_input_stokes(1, 9, polang, retang, wpret)
#
mod_mat1 = np.matrix(beam1)*np.linalg.pinv(s_in)
mod_mat1 = np.matrix(np.array(mod_mat1)/np.array(mod_mat1[:,0:1]))
beam1_fit = mod_mat1*np.matrix(s_in)
demod_mat1 = np.linalg.inv(mod_mat1)
s_fit1 = demod_mat1*beam1
s_fit1 = s_fit1/s_fit1[0,:].mean()
#
mod_mat2 = np.matrix(beam2)*np.linalg.pinv(s_in)
mod_mat2 = np.matrix(np.array(mod_mat2)/np.array(mod_mat2[:,0:1]))
beam2_fit = mod_mat2*np.matrix(s_in)
demod_mat2 = np.linalg.inv(mod_mat2)
s_fit2 = demod_mat2*beam2
s_fit2 = s_fit2/s_fit2[0,:].mean()
#
plt.figure()
plt.plot(s_fit1.T, 'c')
plt.plot(s_in.T, 'k:')
plt.figure()
plt.plot(s_fit2.T, 'c')
plt.plot(s_in.T, 'k:')
#
print('Beam 1 efficiency: ', compute_modulation_efficiency(demod_mat1))
print('Beam 2 efficiency: ', compute_modulation_efficiency(demod_mat2))


# %%
#
topdir =  os.sep.join(pcdir.split(os.sep)[0:-1])
topdir += os.sep + 'L0' 
if not (os.path.exists(topdir)): os.mkdir(topdir)
dirtree = pcdir.split(os.sep)[-2::]
pc1name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol1'] + dirtree + [line, 'pc.FITS'])
pc1name = pc1name.replace('PCalibration_', '')
pc2name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol2'] + dirtree + [line, 'pc.FITS'])
pc2name = pc2name.replace('PCalibration_', '')
print('Modulation matrix 1 saved as: ', pc1name)
print('Modulation matrix 2 saved as: ', pc2name)
#
hdu1 = pf.PrimaryHDU(beam1)
hdu2 = pf.ImageHDU(mod_mat1)
hdul = pf.HDUList([hdu1, hdu2])
hdul.writeto(pc1name, overwrite=True)
hdul.close()
#
hdu1 = pf.PrimaryHDU(beam2)
hdu2 = pf.ImageHDU(mod_mat2)
hdul = pf.HDUList([hdu1, hdu2])
hdul.writeto(pc2name, overwrite=True)
hdul.close()
#
config['pcalibration'][line+'/pol1'] = pc1name
config['pcalibration'][line+'/pol2'] = pc2name
config.write()


