# %%
from process_files import *
from func_science import *

# %% [markdown]
# #### 9. Create time series maps of physical parameters

# %%
config = configobj.ConfigObj('config.ini')
# line = config['line']
line = 'Ca_II_8542'
if (line == 'Fe_I_6301'): line_ = 'Fe_I_6302'
else: line_ = line
#
suffix_dq = '_'
invdir = config['science']['invdir']
l0dir = config['science'][line_+'/l0dir']
l1dir = l0dir.replace(os.sep+'L0'+os.sep, os.sep+'L1'+os.sep)
l1supdir = os.sep.join(l1dir.split(os.sep)[0:-2])
if not os.path.exists(l1supdir): os.makedirs(l1supdir)
l1dir = l1supdir+os.sep+line+os.sep+'stokes'+suffix_dq
if not os.path.exists(l1dir): os.makedirs(l1dir)
pol_files = [l0dir+os.sep+f for f in sorted(os.listdir(l0dir))]
invoutdir = l1supdir+os.sep+line+os.sep+'tempdir_out'+suffix_dq
nwav = int(config['science'][line+'/nwav'])
#
config['science']['l1dir'] = l1supdir
config['science'][line+'/l1dir'+suffix_dq] = l1dir
config.write()

# %%
# Read the results from the inversion of the template stokes profiles (derived from the flats)
sfile_ff = invoutdir + os.sep + 'pol_flat_stokes.dat' 
pfile_ff = invoutdir + os.sep + 'pol_flat_params.dat' 
#
stokes_ff = np.fromfile(sfile_ff, dtype=np.dtype('f').newbyteorder('<'))
stokes_ff = np.reshape(stokes_ff[0:-16], newshape=(1280,1280,nwav,8), order='F')
stokes_ff = np.swapaxes(stokes_ff, 0, 1)
params_ff = np.fromfile(pfile_ff, dtype=np.dtype('f').newbyteorder('<'))
params_ff = np.reshape(params_ff[0:-2], newshape=(1280,1280,16), order='F')
params_ff = np.swapaxes(params_ff, 0, 1)
#
# angg_ff = params_ff[:,:,1]
# angt_ff = params_ff[:,:,2]
# bmag_ff = params_ff[:,:,5]
vlos_ff = params_ff[:,:,6]
plt.figure()
plt.imshow(vlos_ff, cmap='RdBu')

# %%
plt.figure()
ix, iy, istks = 300, 1000, 0
plt.plot(stokes_ff[iy,ix,:,istks])
plt.plot(stokes_ff[iy,ix,:,istks+4])
ix, iy, istks = 750, 550, 0
plt.plot(stokes_ff[iy,ix,:,istks])
plt.plot(stokes_ff[iy,ix,:,istks+4])
#

# %%
# Read the velocity and magnetic field data and create the time series
stokes_files = [invoutdir+os.sep+f for f in sorted(os.listdir(invoutdir)) if 'sc_stokes' in f]
params_files = [invoutdir+os.sep+f for f in sorted(os.listdir(invoutdir)) if 'sc_params' in f]
l0_files =  [l0dir+os.sep+f for f in sorted(os.listdir(l0dir))]
bmag_tser = []
angg_tser = []
angp_tser = []
vlos_tser = []
for sfile, pfile, l0file in tqdm.tqdm(zip(stokes_files[0:1], params_files[0:1], l0_files[0:1])):
    stokes = np.fromfile(sfile, dtype=np.dtype('f').newbyteorder('<'))
    stokes = np.reshape(stokes[0:-16], newshape=(1280,1280,nwav,8), order='F')
    stokes = np.swapaxes(stokes, 0, 1)
    params = np.fromfile(pfile, dtype=np.dtype('f').newbyteorder('<'))
    params = np.reshape(params[0:-2], newshape=(1280,1280,16), order='F')
    params = np.swapaxes(params, 0, 1)
    #
    angg_tser.append(params[:,:,1])
    angp_tser.append(params[:,:,2])
    bmag_tser.append(params[:,:,5])
    vlos_tser.append(params[:,:,6])
    #
    stks_name = l1dir + os.sep + os.path.split(sfile)[-1].replace('sc_stokes.dat', 'stokes.fits')
    hdu1 = pf.open(l0file)[0]
    hdu2 = pf.ImageHDU(stokes[:,:,:,0:3])
    hdu3 = pf.ImageHDU(stokes[:,:,:,4::])
    hdul = pf.HDUList([hdu1, hdu2, hdu3])
    hdul.writeto(stks_name, overwrite=True)
angg_tser = np.moveaxis(np.array(angg_tser), 0, 2)
angp_tser = np.moveaxis(np.array(angp_tser), 0, 2)
bmag_tser = np.moveaxis(np.array(bmag_tser), 0, 2)
vlos_tser = np.moveaxis(np.array(vlos_tser), 0, 2)

# %%
vlos_corr = vlos_tser-vlos_ff[:,:,np.newaxis]
bz = bmag_tser*np.cos(np.radians(angg_tser))
by = bmag_tser*np.sin(np.radians(angg_tser))*np.sin(np.radians(angp_tser))
bx = bmag_tser*np.sin(np.radians(angg_tser))*np.cos(np.radians(angp_tser))

# %%
plt.figure(); plt.imshow(vlos_corr, cmap='RdBu')
plt.figure(); plt.imshow(stokes[:,:,0,0], cmap='RdBu')

# %%
show_img_series(stokes[:,:,:,4])

# %%
# Save the time stamps for the observations
timestamps = [os.path.split(f)[-1].split('_')[-3] for f in stokes_files]
timestamps = [dt.datetime.strptime(t, '%H%M%S%f').timestamp() for t in timestamps]
timestamps = np.array(timestamps)
timestamps_ = np.array([np.arange(len(timestamps)), timestamps])
plt.plot(timestamps[1::]-timestamps[0:-1])
plt.xlabel('Frame #')
plt.ylabel('Cadence in s')
ts_name = l1dir + os.sep + 'timestamps.csv'
np.savetxt(ts_name, timestamps_.T)

# %%
# Save the physical parameters
prefix = ''.join(l1supdir.split(os.sep)[-3::]).replace('L1Science','')
vlos_name = l1supdir + os.sep + prefix + '_' + line + '_vlos.fits'
bpol_name = l1supdir + os.sep + prefix + '_' + line + '_bpol.fits'
bcar_name = l1supdir + os.sep + prefix + '_' + line + '_bcar.fits'
ts_name = l1supdir + os.sep + prefix + '_' + line + '_timestamps.csv'
#
np.savetxt(ts_name, timestamps_.T)
#
hdu = pf.PrimaryHDU(vlos_corr)
hdu.writeto(vlos_name, overwrite=True)
#
hdu1 = pf.PrimaryHDU(bmag_tser)
hdu2 = pf.ImageHDU(angg_tser)
hdu3 = pf.ImageHDU(angp_tser)
hdul = pf.HDUList([hdu1, hdu2, hdu3])
hdul.writeto(bpol_name, overwrite=True)
#
hdu1 = pf.PrimaryHDU(bz)
hdu2 = pf.ImageHDU(by)
hdu3 = pf.ImageHDU(bx)
hdul = pf.HDUList([hdu1, hdu2, hdu3])
hdul.writeto(bcar_name, overwrite=True)
