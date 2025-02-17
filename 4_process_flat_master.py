# %%
from process_files import *
from func_flats import *

# %%
config = configobj.ConfigObj('config.ini')
line = config['line']
# line = 'Ca_II_8542'
ffdir = config['flats']['directory']
settings = [f for f in os.listdir(ffdir) if 'settings' in f]
settings = ffdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
linestr = 'Line_' + str(get_line_num(settings, line, 0))
nwav = int(settings[linestr]['NWavePoints'])
wavescan_range = float(settings[linestr]['WaveScanRange'])
wavelength = float(settings[linestr]['LineWavelength'])
wavestep = wavescan_range/(nwav-1)

# %%
ffname0 = config['flats'][line+'/bbi']
ff0 = np.float64(pf.open(ffname0)[0].data)
ff0 = ff0[:,:,np.newaxis]/np.mean(ff0)
#
ff10_name = config['flats'][line+'/pol1_mod0_fit']
ff10 = read_flat_fit(ff10_name)
ff11_name = config['flats'][line+'/pol1_mod1_fit']
ff11 = read_flat_fit(ff11_name)
ff12_name = config['flats'][line+'/pol1_mod2_fit']
ff12 = read_flat_fit(ff12_name)
ff13_name = config['flats'][line+'/pol1_mod3_fit']
ff13 = read_flat_fit(ff13_name)
ff20_name = config['flats'][line+'/pol2_mod0_fit']
ff20 = read_flat_fit(ff20_name)
ff21_name = config['flats'][line+'/pol2_mod1_fit']
ff21 = read_flat_fit(ff21_name)
ff22_name = config['flats'][line+'/pol2_mod2_fit']
ff22 = read_flat_fit(ff22_name)
ff23_name = config['flats'][line+'/pol2_mod3_fit']
ff23 = read_flat_fit(ff23_name)

# %%
pix = 500
plt.plot(ff10[1][pix,pix]*ff10[2][pix,pix])
plt.plot(ff10[0][pix,pix])

# %%
# master flat for 2xchannels x 4xmodulations
cont1 = np.array([ff10[1], ff11[1], ff12[1], ff13[1]])
cont1 = np.moveaxis(cont1, 0, 2)
cont1_mean = np.mean(cont1, axis=(0,1,3))
cont1 /= cont1_mean[np.newaxis,np.newaxis,:,np.newaxis]
#
cont2 = np.array([ff20[1], ff21[1], ff22[1], ff23[1]])
cont2 = np.moveaxis(cont2, 0, 2)
cont2_mean = np.mean(cont2, axis=(0,1,3))
cont2 /= cont2_mean[np.newaxis,np.newaxis,:,np.newaxis]
#

# %%
show_img_series(cont1[:,:,3], fps=5)

# %%
# Compute wavelength map for the data cube
sh1 = 0.25*(ff10[3]+ff11[3]+ff12[3]+ff13[3])
sh1_fit, sh1_val = fit_et_blue_shifts(sh1)
#
sh2 = 0.25*(ff20[3]+ff21[3]+ff22[3]+ff23[3])
sh2_fit, sh2_val = fit_et_blue_shifts(sh2)
#
wavemap = (np.arange(nwav)-nwav//2)*wavestep+wavelength
wavemap = wavemap[np.newaxis,np.newaxis,:]
wavemap1 = wavemap + wavestep*sh1_fit[:,:,np.newaxis]
wavemap2 = wavemap + wavestep*sh2_fit[:,:,np.newaxis]
#
print(sh1_val)
print(sh2_val)
config['flats'][line+'/pol1_a'] = sh1_val[0]
config['flats'][line+'/pol1_x0'] = sh1_val[1]
config['flats'][line+'/pol1_y0'] = sh1_val[2]
config['flats'][line+'/pol1_w0'] = sh1_val[3]
config['flats'][line+'/pol2_a'] = sh2_val[0]
config['flats'][line+'/pol2_x0'] = sh2_val[1]
config['flats'][line+'/pol2_y0'] = sh2_val[2]
config['flats'][line+'/pol2_w0'] = sh2_val[3]

# %%
# Extarct reference template for the spectral lines
# pol1 calib
pc1 = pf.open(config['pcalibration'][line+'/pol1'])
modmat1 = pc1[1].data
pc1.close()
demodmat1 = np.linalg.pinv(modmat1)
# pol2 calib
pc2 = pf.open(config['pcalibration'][line+'/pol2'])
modmat2 = pc2[1].data
pc2.close()
demodmat2 = np.linalg.pinv(modmat2)
#
if (line=='Ca_II_8542'):
    demodmat1[0,:] = 0.25
    demodmat1[1::,:] = 0
    demodmat2[0,:] = 0.25
    demodmat2[1::,:] = 0
#
ff1 = np.array([ff10[0], ff11[0], ff12[0], ff13[0]])
ff1 = np.moveaxis(ff1, 0, 2)
ff1_ = ff1/cont1
ff1_demod = np.einsum('ijkl,mk->ijml', ff1_, demodmat1)
template1 = ff1_demod[:,:,0]
#
ff2 = np.array([ff20[0], ff21[0], ff22[0], ff23[0]])
ff2 = np.moveaxis(ff2, 0, 2)
ff2_ = ff2/cont2
ff2_demod = np.einsum('ijkl,mk->ijml', ff2_, demodmat1)
template2 = ff2_demod[:,:,0]


# %%
# Compute average broadening
broad1 = 0.25*(ff10[4]+ff11[4]+ff12[4]+ff13[4])
broad2 = 0.25*(ff20[4]+ff21[4]+ff22[4]+ff23[4])
plt.figure()
plt.imshow(broad1)
plt.colorbar()
plt.figure()
plt.imshow(broad2)
plt.colorbar()

# %%
ffname = ffname0.replace('bbi','pol').replace('_fa','_fm')
hdu1 = pf.PrimaryHDU(ff0)
hdu2 = pf.ImageHDU(cont1)
hdu3 = pf.ImageHDU(cont2)
hdu4 = pf.ImageHDU(sh1)
hdu5 = pf.ImageHDU(sh2)
hdu6 = pf.ImageHDU(wavemap1)
hdu7 = pf.ImageHDU(wavemap2)
hdu8 = pf.ImageHDU(template1)
hdu9 = pf.ImageHDU(template2)
hdu10 = pf.ImageHDU(broad1)
hdu11 = pf.ImageHDU(broad2)
hdul = pf.HDUList([hdu1,hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8,hdu9,hdu10,hdu11])
hdul.writeto(ffname, overwrite=True)
hdul.close()
#
config['flats'][line+'/pol'] = ffname
config.write()
print('Master flat written to ', ffname)

# %%
show_img_series(cont2[:,:,0], fps=5)

# %%
cont1.shape


