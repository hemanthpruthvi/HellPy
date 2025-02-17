# %%
from process_files import *
from func_target import *

# %%
iline = 0
# calib data
config = configobj.ConfigObj('config.ini')
line = config['line']
dkdir = config['darks']['directory']
tgdir = config['targetplate']['directory']
settings = [f for f in os.listdir(tgdir) if 'settings' in f]
settings = tgdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
#
linestr = 'Line_' + str(get_line_num(settings, line, iline))
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
ncyc = len(os.listdir(tgdir+os.sep+line))//3
#
dir_tree = tgdir.split(os.sep)
l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'
if not (os.path.exists(l0dir)): os.makedirs(l0dir)
print('Targetplate data will be saved to ', l0dir)
config['targetplate'][line+'/l0dir'] = l0dir
config.write()

# %%
# darks
dk0 = np.float64(pf.open(config['darks'][line+'/bbi'])[0].data)
dk1 = np.float64(pf.open(config['darks'][line+'/pol1'])[0].data)
dk2 = np.float64(pf.open(config['darks'][line+'/pol2'])[0].data)
# flats
ff_ = pf.open(config['flats'][line+'/pol'])
ff_bbi = ff_[0].data
ff_pol1 = np.mean(ff_[1].data[:,:,:,0], axis=2)[:,:,np.newaxis]
ff_pol2 = np.mean(ff_[2].data[:,:,:,0], axis=2)[:,:,np.newaxis]

# %%
config['flip0_lr'] = 1
config['flip0_ud'] = 1
config['flip1_lr'] = 1
config['flip1_ud'] = 0
config['flip2_lr'] = 1
config['flip2_ud'] = 1
config.write()
# Get target
dc0 = data_cube(tgdir, line, 0, 0)
print('Reading BBI target image from ', dc0.file)
tg0 = np.uint16(16*(dc0.data-dk0)/ff_bbi)
tg0 = np.flipud(np.fliplr(tg0))
#
dc1 = data_cube(tgdir, line, 1, 0)
print('Reading POL1 target image from ', dc1.file)
tg1 = np.uint16(16*(dc1.data-dk1)/ff_pol1)
tg1 = np.fliplr(tg1)
#
dc2 = data_cube(tgdir, line, 2, 0)
print('Reading POL2 target image from ', dc2.file)
tg2 = np.uint16(16*(dc2.data-dk2)/ff_pol2)
tg2 = np.flipud(np.fliplr(tg2))

# %%
tg0_ = np.mean(tg0, axis=2)
tg1_ = np.mean(tg1, axis=2)
tg2_ = np.mean(tg2, axis=2)
blink_frames([tg0_, tg1_, tg2_], repeat=1)

# %%
affine10, params10, pts100, pts101 = compute_align_params(tg1_, tg0_, npts=5)

# %%
affine10_ = 1.0*affine10
affine10_[0,1] *= -1
affine10_[1,0] *= -1
tg0_corr_ = scipy.ndimage.affine_transform(tg0_, affine10_)
blink_frames([tg1_, tg0_corr_], repeat=10)

# %%
affine12, params12, pts120, pts121 = compute_align_params(tg1_, tg2_, npts=5)

# %%
affine12_ = 1.0*affine12
affine12_[0,1] *= -1
affine12_[1,0] *= -1
tg2_corr_ = scipy.ndimage.affine_transform(tg2_, affine12_)
blink_frames([tg1_, tg2_corr_], repeat=10)

# %%
tg_name = l0dir+os.sep+os.path.split(dc0.file)[-1]
tg_name = tg_name.replace('.DAT', '.fits').replace('bbi', line)
print('Saved as: ', tg_name)
hdu1 = pf.PrimaryHDU(tg0)
hdu2 = pf.ImageHDU(tg1)
hdu3 = pf.ImageHDU(tg2)
hdu4 = pf.ImageHDU(np.array([affine10_, affine12_]))
hdu5 = pf.ImageHDU(np.array([params10, params12]))
hdu6 = pf.ImageHDU(np.array([pts100, pts101]))
hdu7 = pf.ImageHDU(np.array([pts120, pts121]))
hdul = pf.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
hdul.writeto(tg_name, overwrite=True)
#
config['targetplate'][line+'/l0data'] = tg_name
config.write()


