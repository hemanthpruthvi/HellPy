# %%
from process_files import *
from func_flats import *

# %% [markdown]
# #### 1. Create master dark and average flat profile

# %%
# calib data
config = configobj.ConfigObj('config.ini')
line = config['line']
dkdir = config['darks']['directory']
ffdir = config['flats']['directory']
settings = [f for f in os.listdir(ffdir) if 'settings' in f]
settings = ffdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)

# %%
# Process dark frames
dk0 = data_cube(dkdir, line, 0, 0 )
dk0.data = median_filter(np.average(dk0.data, axis=2), size=3)
dk1 = data_cube(dkdir, line, 1, 0)
dk1.data = median_filter(np.average(dk1.data, axis=2), size=3)
dk2 = data_cube(dkdir, line, 2, 0)
dk2.data = median_filter(np.average(dk2.data, axis=2), size=3)
dk0m = np.int16(np.rint(dk0.data[:,:,np.newaxis]))
dk1m = np.int16(np.rint(dk1.data[:,:,np.newaxis]))
dk2m = np.int16(np.rint(dk2.data[:,:,np.newaxis]))

# %%
# Save the master dark
topdir =  os.sep.join(dkdir.split(os.sep)[0:-1])
topdir += os.sep + 'L0' 
if not (os.path.exists(topdir)): os.mkdir(topdir)
dirtree = dkdir.split(os.sep)[-2::]
dk0name = topdir + os.sep + '_'.join(['HELLRIDE', 'bbi'] + dirtree + [line, 'da.FITS'])
dk1name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol1'] + dirtree + [line, 'da.FITS'])
dk2name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol2'] + dirtree + [line, 'da.FITS'])
print('Processed files are saved as ', '\n', dk0name, '\n', dk1name, '\n', dk2name)
#
hdu1 = pf.PrimaryHDU(dk0m)
hdul = pf.HDUList([hdu1])
hdul.writeto(dk0name, overwrite=True)
hdul.close()
#
hdu1 = pf.PrimaryHDU(dk1m)
hdul = pf.HDUList([hdu1])
hdul.writeto(dk1name, overwrite=True)
hdul.close()
#
hdu1 = pf.PrimaryHDU(dk2m)
hdul = pf.HDUList([hdu1])
hdul.writeto(dk2name, overwrite=True)
hdul.close()
#
config['darks'][line+'/bbi'] = dk0name
config['darks'][line+'/pol1'] = dk1name
config['darks'][line+'/pol2'] = dk2name
config.write()

# %%
# More details of the flat data
iline = get_line_num(settings, line, 0)
linestr = 'Line_' + str(iline)
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
ncyc = len(os.listdir(ffdir+os.sep+line))//3
ff_ncyc =  ncyc
# Time stamps
im0ind, im1ind, im2ind = read_time_stamps_cal(ffdir, iline)
im0ind_ = im0ind.reshape([ncyc, nfpc])
im1ind_ = im1ind.reshape([ncyc, nfpc])
im2ind_ = im2ind.reshape([ncyc, nfpc])
im0ind_ -= im0ind_[:,0:1]
im1ind_ -= im1ind_[:,0:1]
im2ind_ -= im2ind_[:,0:1]
im0ind = im0ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im1ind = im1ind_.reshape([ncyc, nwav, nacc, nmod])%nmod
im2ind = im2ind_.reshape([ncyc, nwav, nacc, nmod])%nmod

# %%
# Process flat frames
ff0, ff1, ff2 = 0.0, 0.0, 0.0
print('Processing BBI flat files')
for i in tqdm.tqdm(range(ff_ncyc)):
    dc0 = data_cube(ffdir, line, 0, i)
    ff0 += dc0.data
print('Data loaded from ', dc0.file)
ff0 /= ff_ncyc
ff0m = np.int16(np.rint(np.average(ff0, axis=2) - dk0.data))
#
if (line =='Ca_II_8542'): ff_ncyc = 2
print('Processing POL1 flat files')
ff1 = 0
for i in tqdm.tqdm(range(ff_ncyc)):
    dc1 = data_cube(ffdir, line, 1, i)
    Y, X, Z = dc1.data.shape
    ff1_temp = dc1.data.reshape([Y, X, nmod, nacc, nwav], order='F')
    ff1_temp = coadd_del_accumulations(ff1_temp, im1ind[i])
    ff1 += ff1_temp
print('Data loaded from ', dc1.file)
ff1 /= ff_ncyc
ff1m = np.int16(np.rint(ff1 - dk1.data[:,:,np.newaxis,np.newaxis]))
#
print('Processing POL2 flat files')
ff2 = 0
for i in tqdm.tqdm(range(ff_ncyc)):
    dc2 = data_cube(ffdir, line, 2, i)
    Y, X, Z = dc2.data.shape
    ff2_temp = dc2.data.reshape([Y, X, nmod, nacc, nwav], order='F')
    ff2_temp = coadd_del_accumulations(ff2_temp, im2ind[i])
    ff2 += ff2_temp
print('Data loaded from ', dc2.file)
ff2 /= ff_ncyc
ff2m = np.int16(np.rint(ff2 - dk2.data[:,:,np.newaxis,np.newaxis]))

# %%
for i in [0,1]:
    dc1 = data_cube(ffdir, line, 1, i)
    ff1_temp = dc1.data.reshape([Y, X, nmod*nacc*nwav], order='F')
    pix = 200
    plt.plot(ff1_temp[pix, pix])

# %%
show_img_series(ff2m[:,:,0], fps=2)

# %%
# Save the mean flat
topdir =  os.sep.join(ffdir.split(os.sep)[0:-1])
topdir += os.sep + 'L0' 
if not (os.path.exists(topdir)): os.mkdir(topdir)
dirtree = ffdir.split(os.sep)[-2::]
ff0name = topdir + os.sep + '_'.join(['HELLRIDE', 'bbi'] + dirtree + [line, 'fa.FITS'])
ff1name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol1'] + dirtree + [line, 'fa.FITS'])
ff2name = topdir + os.sep + '_'.join(['HELLRIDE', 'pol2'] + dirtree + [line, 'fa.FITS'])
print('Processed flat files are saved as', '\n', ff0name, '\n', ff1name, '\n', ff2name)
#
hdu1 = pf.PrimaryHDU(ff0m)
hdul = pf.HDUList([hdu1])
hdul.writeto(ff0name, overwrite=True)
hdul.close()
#
hdu1 = pf.PrimaryHDU(ff1m)
hdul = pf.HDUList([hdu1])
hdul.writeto(ff1name, overwrite=True)
hdul.close()
#
hdu1 = pf.PrimaryHDU(ff2m)
hdul = pf.HDUList([hdu1])
hdul.writeto(ff2name, overwrite=True)
hdul.close()
#
config['flats'][line+'/bbi'] = ff0name
config['flats'][line+'/pol1'] = ff1name
config['flats'][line+'/pol2'] = ff2name
config.write()


