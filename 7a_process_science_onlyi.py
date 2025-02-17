# %%
from process_files import *
from func_science import *

# %%
config = configobj.ConfigObj('config.ini')
pcdir = config['pcalibration']['directory']
# line = config['line']
line = 'Ca_II_8542'
cwavind = int(config['science'][line+'/cwavind'])
settings = [f for f in os.listdir(pcdir) if 'settings' in f]
settings = pcdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
l0aligndir = config['science'][line+'/l0aligndir_']
scnames =[l0aligndir+os.sep+f for f in sorted(os.listdir(l0aligndir)) if '.fits' in f]
l0corrdir = l0aligndir.replace('stokes_align', 'stokes_corr')
config['science'][line+'/l0corrdir_'] = l0corrdir
config.write()
if not os.path.exists(l0corrdir): os.makedirs(l0corrdir)
#

# %%
for scname in tqdm.tqdm((scnames)):
    sc0 = pf.open(scname)[0].data
    sc = pf.open(scname)[1].data
    sc[:,:,1] = sc[:,:,0]*0
    sc[:,:,2] = sc[:,:,0]*0
    sc[:,:,3] = sc[:,:,0]*0
    #
    newname = scname.replace('stokes_align', 'stokes_corr')
    hdu1 = pf.PrimaryHDU(sc0)
    hdu2 = pf.ImageHDU(sc)
    hdul = pf.HDUList([hdu1, hdu2])
    hdul.writeto(newname, overwrite=True)

# %%
show_img_series(sc[:,:,0], fps=2)


