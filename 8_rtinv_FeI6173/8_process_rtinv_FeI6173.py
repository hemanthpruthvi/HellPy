# %%
import sys
sys.path.append('../')
from process_files import *
from func_science import *

# %% [markdown]
# #### 8. Invert the Stokes profiles using VFISV for FPI based instruments

# %%
# Load the metadata
config = configobj.ConfigObj('../config.ini')
line = 'Fe_I_6173'
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
wavelength = np.float128(settings[linestr]['LineWavelength'])
wave_range = np.float128(settings[linestr]['WaveScanRange'])
wave_step = wave_range/(np.int16(settings[linestr]['NWavePoints'])-1)
geff = np.float128(config['science'][line+'/geff'])
#
suffix_dq = '' # data quality suffix for image reconstructed vs non-reconstructed data ('' : reconstructed, '_' : not reconstructed)
l0corrdir = config['science'][line+'/l0corrdir'+suffix_dq]
pol_files = [l0corrdir+os.sep+f for f in sorted(os.listdir(l0corrdir)) if '_sc.fits' in f]
invdir = config['science']['invdir']
tempdir = os.sep.join(l0corrdir.split(os.sep)[0:-1])
tempdir_in = tempdir.replace(os.sep+'L0'+os.sep, os.sep+'L1'+os.sep)+os.sep+'tempdir_in'+suffix_dq
tempdir_out = tempdir.replace(os.sep+'L0'+os.sep, os.sep+'L1'+os.sep)+os.sep+'tempdir_out'+suffix_dq
if not os.path.exists(tempdir_in): os.makedirs(tempdir_in)
if not os.path.exists(tempdir_out): os.makedirs(tempdir_out)
#
ff_file = config['flats'][line+'/pol']
tp_file = config['targetplate'][line+'/l0data']

# %%
# Read the template spectrum from the master flat 
lines_ff = pf.open(ff_file)
align_pars = pf.open(tp_file)
affine12 = align_pars[3].data[1]
#
lines_pol1 = lines_ff[7].data
lines_pol1 = np.fliplr(lines_pol1)
lines_pol2 = lines_ff[8].data
lines_pol2 = np.flipud(np.fliplr(lines_pol2))
for w in range(nwav):
    lines_pol2[:,:,w] = affine_transform(lines_pol2[:,:,w], affine12, mode='nearest')
lines_pol = lines_pol1+lines_pol2
zeros = 0*lines_pol
lines_pol_ = np.array([lines_pol, zeros, zeros, zeros])
lines_pol_ = np.moveaxis(lines_pol_, 0, 2)
#

# %%
# Invert template spectral profiles
sprefix = ['ii', 'qq', 'uu', 'vv']
tfile = open(tempdir_in+os.sep+'file_list.txt', 'w')
# sc = pf.open(pfile)[1].data
ny, nx, ns, nw = lines_pol_.shape
for s in np.arange(ns):
    for w in np.arange(nw):
        file_name = sprefix[s] + '_' + f'{w:02}' + '.fits'  
        hdu = pf.PrimaryHDU(lines_pol_[:,:,s,w])
        hdu.writeto(tempdir_in+os.sep+file_name, overwrite=True)
        tfile.write(file_name + '\n')
tfile.close()

# create the map file
params_file = 'pol_flat_params.dat'
stokes_file = 'pol_flat_stokes.dat'
#
tfile = open(invdir+os.sep+'hellride1.map', 'w')
tfile.write(f'xmin {1} \t \n')
tfile.write(f'xmax {nx} \t \n')
tfile.write(f'ymin {1} \t \n')
tfile.write(f'ymax {ny} \t \n')
tfile.write(f'dirin {tempdir_in+os.sep} \n')
tfile.write(f'dirout {tempdir_out+os.sep} \n')
tfile.write(f'invfileout {params_file} \n')
tfile.write(f'stokfileout {stokes_file} \n')
tfile.write(f'fitslist file_list.txt \n')
tfile.close()

# create the line file
noise = 0.001
wave_steps = np.arange(-nwav//2+1, nwav//2+1)*wave_step*1e3
#
tfile = open(invdir+os.sep+'hellride1.line', 'w')
tfile.write(f'cen_wav {wavelength:.3f} \n')
tfile.write(f'geff {geff:.3f} \n')
tfile.write(f'noise {noise:.5f} \n')
tfile.write(f'nwav {nwav} \n')
tfile.write(f'cont_pix {nwav} \n')
tfile.write(f'wavpos ')
for w in wave_steps: tfile.write(f'{w} ')
tfile.write('\n')
tfile.write('telluric ini 10 \n')
tfile.write('telluric end 11 \n')
tfile.write('instrument hellride \n')
tfile.close()

# create the filter file
et1_r, et2_r = 0.93, 0.93
et1_s = int(int(settings['Etalon_1']['ZSpacing'])*1e4)
et2_s = int(int(settings['Etalon_2']['ZSpacing'])*1e4)
tfile = open(invdir+os.sep+'hellride1.filter', 'w')
tfile.write(f'reflectivity et1 {et1_r:0.3} \n')
tfile.write(f'reflectivity et2 {et2_r:0.3} \n')
tfile.write(f'distance plates et1 {et1_s} \n')
tfile.write(f'distance plates et2 {et2_s} \n')
tfile.close()
# sp.run(f'mpirun -machinefile {invdir}/hostfile -n 16 {invdir}/vfisv_fpi.x -map={invdir}/hellride.map -line={invdir}/hellride.line -filter={invdir}/hellride.filter', shell=True)
sp.run(f'mpirun -n 16 {invdir}/vfisv_fpi.x -map={invdir}/hellride1.map -line={invdir}/hellride1.line -filter={invdir}/hellride1.filter', shell=True)



# %%
sc = pf.open(pol_files[-1])[1].data
plt.imshow(sc[:,:,3,4]/sc[:,:,0,4])

# %%
# Invert the observed stoke profiles
for pfile in pol_files:
    sprefix = ['ii', 'qq', 'uu', 'vv']
    tfile = open(tempdir_in+os.sep+'file_list.txt', 'w')
    sc = pf.open(pfile)[1].data
    ny, nx, ns, nw = sc.shape
    for s in np.arange(ns):
        for w in np.arange(nw):
            file_name = sprefix[s] + '_' + f'{w:02}' + '.fits'  
            hdu = pf.PrimaryHDU(sc[:,:,s,w])
            hdu.writeto(tempdir_in+os.sep+file_name, overwrite=True)
            tfile.write(file_name + '\n')
    tfile.close()

    # create the map file
    pfile_name = os.path.split(pfile)[-1]
    params_file = pfile_name.replace('.fits', '_params.dat')
    stokes_file = pfile_name.replace('.fits', '_stokes.dat')
    #
    tfile = open(invdir+os.sep+'hellride1.map', 'w')
    tfile.write(f'xmin {1} \t \n')
    tfile.write(f'xmax {nx} \t \n')
    tfile.write(f'ymin {1} \t \n')
    tfile.write(f'ymax {ny} \t \n')
    tfile.write(f'dirin {tempdir_in+os.sep} \n')
    tfile.write(f'dirout {tempdir_out+os.sep} \n')
    tfile.write(f'invfileout {params_file} \n')
    tfile.write(f'stokfileout {stokes_file} \n')
    tfile.write(f'fitslist file_list.txt \n')
    tfile.close()

    # create the line file
    noise = 1.0/np.sqrt(sc.mean())
    wave_steps = np.arange(-nwav//2+1, nwav//2+1)*wave_step*1e3
    #
    tfile = open(invdir+os.sep+'hellride1.line', 'w')
    tfile.write(f'cen_wav {wavelength:.3f} \n')
    tfile.write(f'geff {geff:.3f} \n')
    tfile.write(f'noise {noise:.5f} \n')
    tfile.write(f'nwav {nwav} \n')
    tfile.write(f'cont_pix {nwav} \n')
    tfile.write(f'wavpos ')
    for w in wave_steps: tfile.write(f'{w} ')
    tfile.write('\n')
    tfile.write('telluric ini 10 \n')
    tfile.write('telluric end 11 \n')
    tfile.write('instrument hellride \n')
    tfile.close()

    # create the filter file
    et1_r, et2_r = 0.93, 0.93
    et1_s = int(int(settings['Etalon_1']['ZSpacing'])*1e4)
    et2_s = int(int(settings['Etalon_2']['ZSpacing'])*1e4)
    tfile = open(invdir+os.sep+'hellride1.filter', 'w')
    tfile.write(f'reflectivity et1 {et1_r:0.3} \n')
    tfile.write(f'reflectivity et2 {et2_r:0.3} \n')
    tfile.write(f'distance plates et1 {et1_s} \n')
    tfile.write(f'distance plates et2 {et2_s} \n')
    tfile.close()
    # sp.run(f'mpirun -machinefile {invdir}/hostfile -n 16 {invdir}/vfisv_fpi.x -map={invdir}/hellride1.map -line={invdir}/hellride1.line -filter={invdir}/hellride1.filter', shell=True)
    sp.run(f'mpirun -n 16 {invdir}/vfisv_fpi.x -map={invdir}/hellride1.map -line={invdir}/hellride1.line -filter={invdir}/hellride1.filter', shell=True)


