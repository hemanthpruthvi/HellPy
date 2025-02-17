# %%
from process_files import *
from func_science import *

# %% [markdown]
# ## Reduce the data and write individual frames for MOMFBD

# %%
#
config = configobj.ConfigObj('config.ini')
line = config['line']
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
# Time stamps
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
#

# %%
# darks
dk0 = np.float64(pf.open(config['darks'][line+'/bbi'])[0].data)
dk1 = np.float64(pf.open(config['darks'][line+'/pol1'])[0].data)
dk2 = np.float64(pf.open(config['darks'][line+'/pol2'])[0].data)
# flats
# ff_bbi = np.float64(pf.open(config['flats'][line+'/bbi'])[0].data)
# ff_bbi = ff_bbi[:,:,np.newaxis]/np.mean(ff_bbi)
# ff_pol = np.float64(pf.open(config['flats'][line+'/pol'])[0].data)
# ff_pol1 = ff_pol[:,:,0:4]
# ff_pol2 = ff_pol[:,:,4::]
ff_ = pf.open(config['flats'][line+'/pol'])
ff_bbi = ff_[0].data
ff_pol1 = ff_[1].data
ff_pol2 = ff_[2].data
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
dir_tree = scdir.split(os.sep)
l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'+os.sep+dir_tree[-1]+os.sep+line
if not (os.path.exists(l0dir)): os.makedirs(l0dir)
print('Science data will be saved to ', l0dir)
config['science'][line+'/l0dir'] = l0dir
config.write()

# %%
l0subdir_bbi = l0dir+os.sep+'bbi'
if not (os.path.exists(l0subdir_bbi)): os.mkdir(l0subdir_bbi)
l0subdir_pol1 = l0dir+os.sep+'pol1'
if not (os.path.exists(l0subdir_pol1)): os.mkdir(l0subdir_pol1)
l0subdir_pol2 = l0dir+os.sep+'pol2'
if not (os.path.exists(l0subdir_pol2)): os.mkdir(l0subdir_pol2)
print('Data will be written to', l0dir)
for i in tqdm.tqdm(range(ncyc)):
    # print('Processing', i+1, 'of', ncyc, 'frames...')
    # Load the data from all the channels
    dc0 = data_cube(scdir, line, 0, i)
    sc0 = (dc0.data - dk0)/ff_bbi
    dc1 = data_cube(scdir, line, 1, i)
    sc1 = dc1.data - dk1
    dc2 = data_cube(scdir, line, 2, i)
    sc2 = dc2.data - dk2
    #
    rel_ind = np.arange(nwav*nacc*nmod)
    wav_ind = np.int16(rel_ind//int(nacc*nmod))
    mod1_ind = np.int16(im1ind_[i]%nmod)
    mod2_ind = np.int16(im2ind_[i]%nmod)
    #
    for i_ in rel_ind:
        sc0[:,:,i_] = 16*sc0[:,:,i_]
        sc0[:,:,i_] = np.flipud(np.fliplr(sc0[:,:,i_]))
        #
        sc1[:,:,i_] = 16*sc1[:,:,i_]/ff_pol1[:,:,mod1_ind[i_],wav_ind[i_]]
        sc1[:,:,i_] = np.fliplr(sc1[:,:,i_])
        # sc1[:,:,i_] = rotate(sc1[:,:,i_], rotang1, mode='nearest', reshape=False)
        # sc1[:,:,i_] = zoom_clipped(sc1[:,:,i_], mag)
        # sc1[:,:,i_] = shift(sc1[:,:,i_], pol1_shifts[0:2], mode='nearest')        
        #
        sc2[:,:,i_] = 16*sc2[:,:,i_]/ff_pol2[:,:,mod2_ind[i_],wav_ind[i_]]
        sc2[:,:,i_] = np.flipud(np.fliplr(sc2[:,:,i_]))
        # sc2[:,:,i_] = rotate(sc2[:,:,i_], rotang2, mode='nearest', reshape=False)
        # sc2[:,:,i_] = zoom_clipped(sc2[:,:,i_], mag)
        # sc2[:,:,i_] = shift(sc2[:,:,i_], pol2_shifts[0:2], mode='nearest')   
        #
        sc0_name = l0subdir_bbi+os.sep+'bbi_'+f'{i:04}'+f'{im0ind_[i][i_]:04}'+'.fits'
        hdu = pf.PrimaryHDU(np.uint16(sc0[:,:,i_]))
        hdu.writeto(sc0_name, overwrite=True)
        sc1_name = l0subdir_pol1+os.sep+'pol1_'+f'{i:04}'+f'{im1ind_[i][i_]:04}'+'.fits'
        hdu = pf.PrimaryHDU(np.uint16(sc1[:,:,i_]))
        hdu.writeto(sc1_name, overwrite=True)
        sc2_name = l0subdir_pol2+os.sep+'pol2_'+f'{i:04}'+f'{im2ind_[i][i_]:04}'+'.fits'
        hdu = pf.PrimaryHDU(np.uint16(sc2[:,:,i_]))
        hdu.writeto(sc2_name, overwrite=True)

# %%
bbi_files = [l0dir+os.sep+f for f in sorted(os.listdir(l0dir)) if (('bbi' in f) and ('.fits' in f))]
pol1_files = [l0dir+os.sep+f for f in sorted(os.listdir(l0dir)) if (('pol1' in f) and ('.fits' in f))]
pol2_files = [l0dir+os.sep+f for f in sorted(os.listdir(l0dir)) if (('pol2' in f) and ('.fits' in f))]

# %%
for i, f in enumerate(pol1_files):
    print(i)
    sc0 = pf.open(f)[0].data
    show_img_series(sc0[:,:,], fps=2)
    plt.close()

# %% [markdown]
# ## MOMFBD wrapper for image reconstruction

# %% [markdown]
# ## Demodulate the reconstructed data 

# %%
#
config = configobj.ConfigObj('config.ini')
# line =config['line']
line = 'Fe_I_6173'
scdir = config['science']['directory']
imdir = config['science'][line+'/imdir']
settings = [f for f in os.listdir(scdir) if 'settings' in f]
settings = scdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
# Other numbers
linestr = 'Line_' + str(get_line_num(settings, line, 0))
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

# %%
sc_files = sorted(os.listdir(scdir+os.sep+line))[0:ncyc]
# flats
ff_pol1 = np.float64(pf.open(config['flats'][line+'/pol1'])[0].data)
ff_pol2 = np.float64(pf.open(config['flats'][line+'/pol2'])[0].data)
ff_pol = np.float64(pf.open(config['flats'][line+'/pol'])[0].data)
ff_pol1 = ff_pol[:,:,0:4]
ff_pol2 = ff_pol[:,:,4::]
# pol1 calib
pc1 = pf.open(config['pcalibration'][line+'/pol1'])
modmat1 = pc1[1].data
pc1.close()
demodmat1 = np.linalg.inv(modmat1)
# pol2 calib
pc2 = pf.open(config['pcalibration'][line+'/pol2'])
modmat2 = pc2[1].data
pc2.close()
demodmat2 = np.linalg.inv(modmat2)
#
dir_tree = scdir.split(os.sep)
# l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'
# if not (os.path.exists(l0dir)): os.mkdir(l0dir)
# l0dir += os.sep+dir_tree[-1]
# if not (os.path.exists(l0dir)): os.mkdir(l0dir)
# l0dir += os.sep+line
# if not (os.path.exists(l0dir)): os.mkdir(l0dir)

l0dir = os.sep.join(dir_tree[0:-1])+os.sep+'L0'+os.sep+dir_tree[-1]+os.sep+line
if not (os.path.exists(l0dir)): os.mkdirs(l0dir)
config['science'][line+'/l0dir'] = l0dir
config.write()
#
l0subdir = l0dir+os.sep+'stokes_align'
if not (os.path.exists(l0subdir)): os.mkdir(l0subdir)
config['science'][line+'/l0aligndir'] = l0subdir
config.write()
print('Science data will be saved to ', l0subdir)

# %%
for i in tqdm.tqdm(range(120,160)):
    imsubdir = imdir + os.sep +f'{i:04}'
    imlist = [imsubdir+os.sep+f for f in sorted(os.listdir(imsubdir)) if '.fits' in f]
    #
    sc0 = pf.open(imlist[0])[0].data
    sc0 = np.pad(sc0, ((6,6),(7,7)), mode='constant')
    sc1, sc2 = [], []
    for m in range(nmod*nwav):
        sc1_ = pf.open(imlist[m+1])[0].data
        sc1.append(np.pad(sc1_,((6,6),(7,7)),mode='constant'))
        sc2_ = pf.open(imlist[m+1+nmod*nwav])[0].data
        sc2.append(np.pad(sc2_,((6,6),(7,7)),mode='constant'))
    sc1, sc2 = np.array(sc1), np.array(sc2)
    #
    Y, X = sc0.shape
    sc1 = np.reshape(np.moveaxis(sc1,0,2), [Y,X,nmod,nwav], order='F')
    sc2 = np.reshape(np.moveaxis(sc2,0,2), [Y,X,nmod,nwav], order='F')
    sc0[sc0==0] = sc0.mean()
    sc1[sc1==0] = sc1.mean()
    sc2[sc2==0] = sc2.mean()
    # 
    sc1_demod = np.einsum('ijkl,mk->ijml', sc1, demodmat1)
    # sc1_demod[:,:,1::] /= sc1_demod[:,:,0:1] 
    sc2_demod = np.einsum('ijkl,mk->ijml', sc2, demodmat2)
    # sc2_demod[:,:,1::] /= sc2_demod[:,:,0:1] 
    # #
    sc_name = l0subdir + os.sep + f'{i:04}_'+sc_files[i].split('_')[-2]+'_sc.fits'
    hdu1 = pf.PrimaryHDU(sc0)
    hdu2 = pf.ImageHDU(sc1_demod)
    hdu3 = pf.ImageHDU(sc2_demod)
    hdul = pf.HDUList([hdu1,hdu2,hdu3])
    hdul.writeto(sc_name, overwrite=True)

# %%
sc_name

# %%
def save_mono_video(vid_data, name, fps=30, cmap='gray'):
    """
    |   Generate and save a series of mono images as a video
    |   Input:  3d data of size X*Y*N
    |           name of the video file
    |   Output: None
    """
    i, vid_size = 0, 10
    N = vid_data.shape[2]
    # DPI = vid_data.shape[0]/vid_size
    #
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    disp = ax.imshow(vid_data[:,:,0], cmap=cmap, animated=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    def update_frame(i):
        disp.set_array(vid_data[:,:,i])
        return disp,
    writer = animation.writers['ffmpeg'](fps=fps)
    ani = animation.FuncAnimation(fig, update_frame, frames=N, interval=1000.0/fps, blit=True)
    ani.save(name, writer=writer, dpi=120)
    plt.show()


# %%
save_mono_video(sc1_demod[:,:,0], 'fei6173_140_i.mp4', fps=2)

# %%
clip = 20
img = sc1_demod[clip:-clip,clip:-clip,0,10]
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(img, cmap='gray', vmin=1200, vmax=6000)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
plt.savefig(line+'_'+str(i)+'_sample.png')

# %%
img.min()


