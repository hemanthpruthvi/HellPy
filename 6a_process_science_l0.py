# %%
from process_files import *
from func_science import *

# %% [markdown]
# #### 6a. Alternatively, reduce the raw science data without image reconstruction.

# %%
# REad the metadata
config = configobj.ConfigObj('config.ini')
line = config['line']
# line = 'Ca_II_8542'
dkdir = config['darks']['directory']
scdir = config['science']['directory']
settings = [f for f in os.listdir(scdir) if 'settings' in f]
settings = scdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
# Other numbers
iline = get_line_num(settings, line, 0)
linestr = 'Line_' + str(iline)
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
ncyc = len(os.listdir(scdir+os.sep+line))//3
wave_range = np.float64(settings[linestr]['WaveScanRange'])
wave_step = wave_range/(np.float64(settings[linestr]['NWavePoints'])-1)
#
dir_tree = scdir.split(os.sep)
l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'+os.sep+dir_tree[-1]+os.sep+line+os.sep+'stokes_'
os.makedirs(l0dir, exist_ok=True)
print('Science data will be saved to ', l0dir)
config['science'][line+'/l0dir_'] = l0dir
config.write()


# %%
# Time stamps and calibration data
im0ind, im1ind, im2ind = read_time_stamps_obs(scdir, iline)
im0ind_ = im0ind.reshape([ncyc, nfpc])
im1ind_ = im1ind.reshape([ncyc, nfpc])
im2ind_ = im2ind.reshape([ncyc, nfpc])
im0ind_ -= im0ind_[:,0:1]
im1ind_ -= im1ind_[:,0:1]
im2ind_ -= im2ind_[:,0:1]
im0ind = im0ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im1ind = im1ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im2ind = im2ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
# darks
dk0 = np.float64(pf.open(config['darks'][line+'/bbi'])[0].data)
dk1 = np.float64(pf.open(config['darks'][line+'/pol1'])[0].data)
dk2 = np.float64(pf.open(config['darks'][line+'/pol2'])[0].data)
# flats
ff_ = pf.open(config['flats'][line+'/pol'])
ff_bbi = np.float64(ff_[0].data)
ff_pol1 = np.float64(ff_[1].data)
ff_pol2 = np.float64(ff_[2].data)
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

# %%
# Average the accumulations, demodulate and flip the images
for i in tqdm.tqdm(range(1)):
    # bbi data save
    dc0 = data_cube(scdir, line, 0, i)
    sc0 = np.mean(dc0.data[:,:,:,np.newaxis], axis=2)
    sc0 = np.uint16(16*(sc0 - dk0)/ff_bbi)
    sc0 = np.flipud(np.fliplr(sc0))

    # pol1 data save
    dc1 = data_cube(scdir, line, 1, i)
    Y, X, Z = dc1.data.shape
    sc1 = dc1.data.reshape([Y,X,nacc*nmod*nwav], order='F') - dk1
    sc1_add = coadd_modulated_imgs (sc1, im1ind[i], nmod, nacc, nwav)
    sc1_ff = sc1_add/ff_pol1
    sc1_demod = np.einsum('ijkl,mk->ijml', sc1_ff, demodmat1)
    # sc1_demod[:,:,1::] /= sc1_demod[:,:,0:1] 
    sc1_demod = np.fliplr(sc1_demod)
    #
    # pol2 data save
    dc2 = data_cube(scdir, line, 2, i)
    Y, X, Z = dc2.data.shape
    sc2 = dc2.data.reshape([Y,X,nacc*nmod*nwav], order='F') - dk2
    sc2_add = coadd_modulated_imgs (sc2, im2ind[i], nmod, nacc, nwav)
    sc2_ff = sc2_add/ff_pol2
    sc2_demod = np.einsum('ijkl,mk->ijml', sc2_ff, demodmat2)
    # sc2_demod[:,:,1::] /= sc2_demod[:,:,0:1] 
    sc2_demod = np.flipud(np.fliplr(sc2_demod))
    #
    scname = l0dir+os.sep+f'{i:04}_'+'_'.join(os.path.split(dc0.file)[-1].split('_')[3::]).replace('.DAT', '.fits')
    hdu1 = pf.PrimaryHDU(sc0)
    hdu2 = pf.ImageHDU(sc1_demod)
    hdu3 = pf.ImageHDU(sc2_demod)
    hdul = pf.HDUList([hdu1, hdu2, hdu3])
    hdul.writeto(scname, overwrite=True)

# %%
show_img_series(ff_pol2[:,:,2], fps=5)

# %%
config = configobj.ConfigObj('config.ini')
ffname = config['flats'][line+'/pol1']
ff1 = pf.open(ffname)[0].data

# dc1 = data_cube(scdir, line, 1, 0)
# Y, X, Z = dc1.data.shape
# sc1 = dc1.data.reshape([Y,X,nacc*nmod*nwav], order='F')
pix = 200
# plt.figure()
# plt.plot(sc1_add[pix,pix,0]/sc1_add[pix,pix,0].mean())
plt.plot(ff1[pix,pix,0]/ff1[pix,pix,0].mean())
# plt.plot(sc1_add[pix,pix,0,:])
plt.plot(sc1_ff[pix,pix,0,:])
plt.plot(sc1_add[pix,pix,0,:])

# plt.figure()
# plt.plot(ff_pol1[pix,pix,0,:])
# plt.plot(ff_pol1[680,484,0,:])

# %%
# Display for checking
scnames =[l0dir+os.sep+f for f in sorted(os.listdir(l0dir)) if '.fits' in f]
for i, f in enumerate(scnames[1:2]):
    scd = pf.open(f)[1].data
    show_img_series(scd[:,:,1], fps=2)
    plt.close()

# %%
# Load the alignment parameners from the target plate data
tpname = config['targetplate'][line+'/l0data']
tpaffines = pf.open(tpname)[3].data
affine10, affine12 = tpaffines[0], tpaffines[1]
#
scnames =[l0dir+os.sep+f for f in sorted(os.listdir(l0dir)) if '.fits' in f]
l0aligndir = l0dir.replace('stokes_', 'stokes_align_')
if not os.path.exists(l0aligndir): os.makedirs(l0aligndir)
config['science'][line+'/l0aligndir_'] = l0aligndir
config.write()

# %%
# Align the channels with respect to POL1 channel; align the time series with respect to the first observations using BBI channel
sc0_ref = pf.open(scnames[0])[0].data[:,:,0]
sc0_ref = affine_transform(sc0_ref, affine10)
#
imshifts = []
for i, f in tqdm.tqdm(enumerate(scnames)):
    f_ = pf.open(f)
    sc0 = np.float64(f_[0].data[:,:,0])
    sc1 = np.float64(f_[1].data)
    sc2 = np.float64(f_[2].data)
    # Channel align
    sc1_, sc2_ = 1.0*sc1, 1.0*sc2
    sc0_ = affine_transform(sc0, affine10, mode='nearest')
    for i in range(nwav*nmod):
        sc2_[:,:,i%nmod,i//nmod] = affine_transform(sc2[:,:,i%nmod,i//nmod], affine12, mode='nearest')
    # Channel merge
    sc1_[:,:,1::] /= sc1_[:,:,0:1]
    sc2_[:,:,1::] /= sc2_[:,:,0:1]
    sc_ = 0.5*(sc1_+sc2_)
    sc_[:,:,1::] *= sc_[:,:,0:1]
    # Time series align
    ts_shift = compute_image_shift(sc0_ref, sc0_)
    imshifts.append(ts_shift)
    sc0_align = shift(sc0_, ts_shift, mode='nearest')
    sc_align = 1.0*sc_
    for i in range(nwav*nmod):
        sc_align[:,:,i%nmod,i//nmod] = shift(sc_[:,:,i%nmod,i//nmod], ts_shift, mode='nearest')
    # Save
    newname = f.replace('stokes_', 'stokes_align_')
    hdu1 = pf.PrimaryHDU(sc0_align)
    hdu2 = pf.ImageHDU(sc_align)
    hdul = pf.HDUList([hdu1, hdu2])
    hdul.writeto(newname, overwrite=True)
imshifts = np.array(imshifts)

# %%
plt.figure(); plt.imshow(sc_[:,:,0,0], cmap='gray')
# plt.figure(); plt.imshow(sc1_[:,:,0,0])
# plt.figure(); plt.imshow(sc2_[:,:,0,0])

# %%
temp = pf.open(newname)
temp0, temp1 = temp[0].data, temp[1].data
blink_frames([temp0, temp1[:,:,0,0]], repeat=5)


