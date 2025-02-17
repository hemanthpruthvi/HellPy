from process_files import *
from func_flats import *

# Load general config settings
iline = 0
line = 'Fe_I_6173'
config = configobj.ConfigObj('config.ini')
dkdir = config['darks']['directory']
ffdir = config['flats']['directory']
settings = [f for f in os.listdir(ffdir) if 'settings' in f]
settings = ffdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
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
# Name the dark files and save them
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
# Write the details to config file
config['darks'][line+'/bbi'] = dk0name
config['darks'][line+'/pol1'] = dk1name
config['darks'][line+'/pol2'] = dk2name
config.write()
# Process flat frames
ff_ncyc = len(os.listdir(ffdir+os.sep+line))//3
ff0, ff1, ff2 = 0.0, 0.0, 0.0 
for i in tqdm.tqdm(range(ff_ncyc)):
    dc0 = data_cube(ffdir, line, 0, i)
    ff0 += dc0.data
    dc1 = data_cube(ffdir, line, 1, i)
    ff1 += dc1.data
    dc2 = data_cube(ffdir, line, 2, i)
    ff2 += dc2.data
ff0 /= ff_ncyc
ff1 /= ff_ncyc
ff2 /= ff_ncyc
ff0m = np.int16(np.rint(np.average(ff0, axis=2) - dk0.data))
ff1m = np.int16(np.rint(ff1 - dk1.data[:,:,np.newaxis]))
ff2m = np.int16(np.rint(ff2 - dk2.data[:,:,np.newaxis]))
#
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
# Write the details to config file
config['flats'][line+'/bbi'] = ff0name
config['flats'][line+'/pol1'] = ff1name
config['flats'][line+'/pol2'] = ff2name
config.write()