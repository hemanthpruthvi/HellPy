# -*- coding: utf-8 -*-
"""
"""
import cv2
from scipy.ndimage import zoom, shift, rotate
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D
from lmfit import minimize, Parameters
from lmfit.models import GaussianModel
from lmfit.models import QuadraticModel
from process_files import *
from process_darks import *
import tqdm

iline = 0
# calib data
config = configobj.ConfigObj('config.ini')
dkdir = config['darks']['directory']
pcdir = config['pcalibration']['directory']
settings = [f for f in os.listdir(pcdir) if 'settings' in f]
settings = pcdir + os.sep + settings[0]
settings = configobj.ConfigObj(settings)
# select line for calib
pclines = [d for d in os.listdir(pcdir) if os.path.isdir(pcdir+os.sep+d)]
pcnlines = len(pclines)
pcilines = range(pcnlines)
pciline = pcilines[iline]
pcline = pclines[pciline]
#
dklines = [d for d in os.listdir(dkdir) if os.path.isdir(dkdir+os.sep+d)]
dknlines = len(dklines)
dkilines = [i for i in range(dknlines) if dklines[i] in pclines]
dkiline = dkilines[iline]
dkline = dklines[dkiline]
print('Calibration and dark data from: ', pcline, ', ', dkline)
# dark frames
dk0 = data_cube(dkdir, dkiline, 0, 0 )
dk0.data = np.array(np.average(dk0.data, axis=2), dtype=np.int16)
dk1 = data_cube(dkdir, dkiline, 1, 0)
dk1.data = np.array(np.average(dk1.data, axis=2), dtype=np.int16)
dk2 = data_cube(dkdir, dkiline, 2, 0)
dk2.data = np.array(np.average(dk2.data, axis=2), dtype=np.int16)

#
npol = int(settings['VTTControls']['ICUPolarizer\\NPositions'])
nret = int(settings['VTTControls']['ICURetarder\\NPositions'])
pang0 = float(settings['VTTControls']['ICUPolarizer\\Start'])
pang1 = float(settings['VTTControls']['ICUPolarizer\\Stop'])
poffset = float(settings['VTTControls']['ICUPolarizer\\Offset'])
pfactor = float(settings['VTTControls']['ICUPolarizer\\Factor'])
rang0 = float(settings['VTTControls']['ICURetarder\\Start'])
rang1 = float(settings['VTTControls']['ICURetarder\\Stop'])
roffset = float(settings['VTTControls']['ICURetarder\\Offset'])
rfactor = float(settings['VTTControls']['ICURetarder\\Factor'])
ncyc = npol*nret
ncycobs = len(os.listdir(pcdir+os.sep+pcline))//3
print('Intended and actual recordings of polcal data: ', ncyc, ', ', ncycobs)

# Analyze log file
logfile = [f for f in os.listdir(pcdir) if 'log' in f]
logfile = pcdir + os.sep + logfile[0]
with open(logfile) as f:
    log = f.readlines()
poltemp = [int(l.split()[-1][4:-1]) for l in log if 'X10A' in l][0:ncycobs]
rettemp = [int(l.split()[-1][4:-1]) for l in log if 'X11A' in l][0:ncycobs]
pang = (np.array(poltemp)-poffset)/pfactor
rang = (np.array(rettemp)-roffset)/rfactor
rnpos = sum(pang==pang0)
pnpos = sum(rang==rang1)

# Get intensity
roi = 300
halfw = 640
ind = halfw-roi//2
# BBI data
int0 = []
for i in tqdm.tqdm(range(rnpos)):
    dc = data_cube(pcdir, pciline, 0, i)
    int0.append(np.mean(dc.data[ind:-ind,ind:-ind,:], axis=(0,1)))
int0 = np.array(int0)
# Pol1 data
int1_mod = []
for i in tqdm.tqdm(range(rnpos)):
    dc = data_cube(pcdir, pciline, 1, i)
    int1_mod.append(np.mean(dc.data[ind:-ind,ind:-ind,:], axis=(0,1)))
int1_mod = np.array(int1_mod)
# Pol2 data
int2_mod = []
for i in tqdm.tqdm(range(rnpos)):
    dc = data_cube(pcdir, pciline, 2, i)
    int2_mod.append(np.mean(dc .data[ind:-ind,ind:-ind,:], axis=(0,1)))
int2_mod = np.array(int2_mod)

# Other numbers
def get_line_num(settings, linestr, iline):
    try:
        line = settings['Line_'+str(iline)]['Label']
    except:
        line = ''
        pass
    if (line == linestr):
        return iline
    else:
        iline += 1
        iline = get_line_num(settings, linestr, iline)
    return iline
linestr = 'Line_' + str(get_line_num(settings, pcline, iline))
nacc = int(settings[linestr]['Polarimeter\\NAccumulations'])
nwav = int(settings[linestr]['NWavePoints'])
filtstr = settings[linestr]['Filter']
modstr = settings[linestr]['Polarimeter\\Modulation']
nmod = int(settings[filtstr]['Polarimeter\\'+modstr+'\\NModulations'])
nfpc = nmod*nacc*nwav
nfpw = nmod*nacc
# Time stamps
tsfile = [f for f in os.listdir(pcdir) if 'timestamp' in f]
tsfile = pcdir + os.sep + tsfile[0]
ts = np.loadtxt(tsfile, delimiter=',')
im0ind, im1ind, im2ind = ts[:,3], ts[:,6], ts[:,9]
im0ind = im0ind[0:nfpc*rnpos].reshape(int0.shape)
im1ind = im1ind[0:nfpc*rnpos].reshape(int1_mod.shape)
im2ind = im2ind[0:nfpc*rnpos].reshape(int2_mod.shape)
#
im0ind = im0ind - im0ind[:,0:1]
im1ind = im1ind - im1ind[:,0:1]
im2ind = im2ind - im2ind[:,0:1]
#
im0ind = im0ind.reshape([rnpos, nwav, nacc, nmod])%4
im1ind = im1ind.reshape([rnpos, nwav, nacc, nmod])%4
im2ind = im2ind.reshape([rnpos, nwav, nacc, nmod])%4
#
def coadd_mod(intens, imod, nmod, nacc, nwav):
    intens_ = intens.reshape([nwav, nacc, nmod])
    imod_ = imod.reshape([nwav, nacc, nmod])%4
    modint = np.zeros([nmod,nwav])
    for w in range(nwav):
        accs = {}
        for m in range(nmod): accs[m] = []
        for a in range(nacc):
            temp = imod_[w,a,:]
            if (set(temp)==set(np.arange(nmod))):
                for i, m in enumerate(temp): accs[m].append(intens_[w,a,i])
        for m in range(nmod): modint[m,w] = np.median(np.array(accs[m]))
    return modint[:,0]      
            
# Data for one cycle
beam1 = np.zeros([nmod, rnpos])
for i in range(rnpos):
    beam1[:,i] = coadd_mod(int1_mod[i,:], im1ind[i,:], nmod, nacc, nwav)
beam2 = np.zeros([nmod, rnpos])
for i in range(rnpos):
    beam2[:,i] = coadd_mod(int2_mod[i,:], im2ind[i,:], nmod, nacc, nwav)

#
def get_instokes(npos, offset=0.0, wpret=90, rotang=0.0, ang_start=0.0, ang_stop=180.0):
    thetas = np.radians(np.linspace(ang_start,ang_stop,npos)+offset)
    in_i, in_q, in_u, in_v = 1, 1, 0, 0
    delta = np.radians(wpret)
    #
    i = np.ones(thetas.shape)
    q = in_q * ( np.cos(2*thetas)**2 + np.cos(delta)*np.sin(2*thetas)**2 ) + \
        in_u * ( np.sin(2*thetas)*np.cos(2*thetas) * (1-np.cos(delta)) ) - \
        in_v * np.sin(2*thetas)*np.sin(delta)
    u = in_q * ( np.sin(2*thetas)*np.cos(2*thetas) * (1-np.cos(delta)) ) + \
        in_u * ( np.sin(2*thetas)**2 + np.cos(delta)*np.cos(2*thetas)**2 ) + \
        in_v * np.cos(2*thetas)*np.sin(delta)
    v = in_q * np.sin(2*thetas)*np.sin(delta) - \
        in_u * np.cos(2*thetas)*np.sin(delta) + \
        in_v * np.cos(delta)
    in_stokes = np.array([i,q,u,v])
    return in_stokes
#
def get_modmatrix(in_stokes, mod_int):
    nmod = mod_int.shape[0]
    in_ = np.matrix(in_stokes)
    in_inv = np.transpose(in_)*np.linalg.inv(in_*np.transpose(in_))
    mod_mat = np.array(np.matrix(mod_int)*in_inv)
    mod_mat = mod_mat/mod_mat[:,0:1]
    return mod_mat
# Beam-1
s_in = get_instokes(13, wpret=61.5)
modmat_beam1 = get_modmatrix(s_in, beam1)
demodmat_beam1 = np.linalg.inv(modmat_beam1)
meas_in1 = np.array(np.matrix(demodmat_beam1)*np.matrix(beam1))
meas_in1[1::,:] /= meas_in1[0:1,:] 
meas_in1[0,:] /= meas_in1[0:1,:].mean()
plt.plot(meas_in1.transpose())
plt.plot(s_in.transpose())
# Beam-2
modmat_beam2 = get_modmatrix(s_in, beam2)
demodmat_beam2 = np.linalg.inv(modmat_beam2) 

# def find_residual():
    
meas_in2 = np.array(np.matrix(demodmat_beam2)*np.matrix(beam2))
 



















