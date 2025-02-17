# %%
from process_files import *
from func_science import *

# %%
config = configobj.ConfigObj('config.ini')
pcdir = config['pcalibration']['directory']
# line = config['line']
line = 'Fe_I_6173'
suffix_dq = ''
cwavind = int(config['science'][line+'/cwavind'])
settings = [f for f in os.listdir(pcdir) if 'settings' in f]
settings = pcdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
l0aligndir = config['science'][line+'/l0aligndir'+suffix_dq]
scnames =[l0aligndir+os.sep+f for f in sorted(os.listdir(l0aligndir)) if '.fits' in f]
l0corrdir = l0aligndir.replace('stokes_align', 'stokes_corr')
config['science'][line+'/l0corrdir'+suffix_dq] = l0corrdir
config.write()
if not os.path.exists(l0corrdir): os.makedirs(l0corrdir)

# %%
i2q, i2u, i2v = [], [], []
for f in tqdm.tqdm(scnames):
    sc = pf.open(f)[1].data
    qi = sc[:,:,1,cwavind]/sc[:,:,0,cwavind]
    ui = sc[:,:,2,cwavind]/sc[:,:,0,cwavind]
    vi = sc[:,:,3,cwavind]/sc[:,:,0,cwavind]
    i2q.append(np.median(qi))
    i2u.append(np.median(ui))
    i2v.append(np.median(vi))
i2q, i2u, i2v = np.array(i2q), np.array(i2u), np.array(i2v)

# %%
i2q_fit = fit_quadratic(i2q)
i2u_fit = fit_quadratic(i2u)
i2v_fit = fit_quadratic(i2v)
#
fig, ax = plt.subplots(1,3,figsize=(18,6))
ax[0].plot(i2q)
ax[0].plot(i2q_fit)
ax[1].plot(i2u)
ax[1].plot(i2u_fit)
ax[2].plot(i2v)
ax[2].plot(i2v_fit)
fig.tight_layout()

# %%
v2q, v2u = [], []
for i in tqdm.tqdm(range(len(scnames))):
    sc = pf.open(scnames[i])[1].data
    qi = sc[:,:,1]/sc[:,:,0]-i2q_fit[i]
    ui = sc[:,:,2]/sc[:,:,0]-i2u_fit[i]
    vi = sc[:,:,3]/sc[:,:,0]-i2v_fit[i]
    inds = np.abs(vi)>0.05
    qi = qi[inds]
    ui = ui[inds]
    vi = vi[inds]
    v2q.append(fit_line_slope(qi.flatten(),vi.flatten()))
    v2u.append(fit_line_slope(ui.flatten(),vi.flatten()))
v2q, v2u = np.array(v2q), np.array(v2u)

# %%
v2q_fit = fit_quadratic(v2q)
v2u_fit = fit_quadratic(v2u)
#
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(v2q)
ax[0].plot(v2q_fit)
ax[1].plot(v2u)
ax[1].plot(v2u_fit)
fig.tight_layout()

# %%
for i in tqdm.tqdm(range(len(scnames))):
    sc0 = pf.open(scnames[i])[0].data
    sc = pf.open(scnames[i])[1].data
    vi = sc[:,:,3]/sc[:,:,0]-i2v_fit[i]
    ui = sc[:,:,2]/sc[:,:,0]-i2u_fit[i]-v2u_fit[i]*vi
    qi = sc[:,:,1]/sc[:,:,0]-i2q_fit[i]-v2q_fit[i]*vi
    sc[:,:,1] = sc[:,:,0]*qi
    sc[:,:,2] = sc[:,:,0]*ui
    sc[:,:,3] = sc[:,:,0]*vi
    #
    newname = scnames[i].replace('stokes_align', 'stokes_corr')
    hdu1 = pf.PrimaryHDU(sc0)
    hdu2 = pf.ImageHDU(sc)
    hdul = pf.HDUList([hdu1, hdu2])
    hdul.writeto(newname, overwrite=True)


