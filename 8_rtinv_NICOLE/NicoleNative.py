from __future__ import print_function
import struct
import os
from numpy import *
from matplotlib.pyplot import *

# Read ASCII model file
def read_ascii_model(MODFILE):
    F = open(MODFILE)
    LIN8 = loadtxt(MODFILE, skiprows=2)
    HEAD = F.readline()
    PT2 = F.readline()
    PAR10 = [r'Macro turbulent velocity', r'Stray light fraction', 
             r'$log(\tau_{5000})$', r'T in $K$', r'Electron Presuure in $dyne/cm^2$', r'Microtubulence in $cm/s$', 
             r'$B_z$ in $Gauss$', r'$v_z$ in $cm/s$', r'$B_x$ in $Gauss$', r'$B_y$ in $Gauss$']
    return PAR10, HEAD, PT2, LIN8

# Create ASCII model file
def create_ascii_model(MODFILE, HEAD, PT2, LIN8):
    F = open(MODFILE, 'w')
    F.write(''.join([HEAD, PT2]))
    savetxt(F, LIN8, '%0.6f')
    F.close()

# Read NICOLE native format model file
def read_native_model(MODFILE):
    SIZE = os.path.getsize(MODFILE)
    F = open(MODFILE, 'rb')
    HEAD = F.read(16)
    NX = struct.unpack('@I', F.read(4))[0]
    NY = struct.unpack('@I', F.read(4))[0]
    NZ = struct.unpack('@Q', F.read(8))[0]
    SIG = 22*NZ+11+92
    F.seek(0)
    F.seek(SIG*8)
    S = []
    for i in range(int(SIZE/8-SIG)):
        S.append(struct.unpack('@d', F.read(8))[0])
    MOD = reshape(array(S, dtype=float64), (SIG, NY, NX), 'F')
    M = MOD[:,0,0]
    LIN22 = reshape(M[0:22*NZ], (NZ, 22), 'F')
    PT11 = M[22*NZ:22*NZ+11]
    ABU = M[22*NZ+11:22*NZ+11+92]
    PAR34 = [r'$z$', r'$log(\tau)$', r'T', r'$P_{gas}$', r'$\rho$', r'$P_{el}$', 
             r'$v_z$', r'$v_{mic}$', r'$B_z$', r'$B_x$', r'$B_y$', 
             r'$B_{z (local)}$', r'$B_{y (local)}$', r'$B_{x (local)}$', 
             r'$v_{z (local)}$', r'$v_{y (local)}$', r'$v_{x (local)}$',
             r'$nH$', r'$nH^-$', r'$nH^+$', r'$nH_2$', r'$nH_2^+$']
    try:
        MODERRFILE = MODFILE + '.err'
        F = open(MODERRFILE, 'rb')
        S = []
        for i in range(int(SIZE/8-SIG)):
            S.append(struct.unpack('@d', F.read(8))[0])
        MODERR = reshape(array(S, dtype=float64), (SIG, NY, NX), 'F')
        MERR = MODERR[:,0,0]
        LIN22E = reshape(MERR[0:22*NZ], (NZ, 22), 'F')
        PT11E = MERR[22*NZ:22*NZ+11]
        ABUE = MERR[22*NZ+11:22*NZ+11+92]
    except: 
        LIN22E = 0*copy(LIN22)
    return PAR34, HEAD, [LIN22, LIN22E], PT11, ABU

# Read NICOLE native profile file
def read_native_profile(PROFILES):
    STOKES = [0]*len(PROFILES)
    for p, PROFILE in enumerate(PROFILES):
        SIZE = os.path.getsize(PROFILE)
        F = open(PROFILE, 'rb')
        SIG = F.read(16)
        print(PROFILE, '\n', SIG)
        NX = struct.unpack('@I', F.read(4))[0]
        NY = struct.unpack('@I', F.read(4))[0]
        NW = struct.unpack('@Q', F.read(8))[0]
        print(NX, NY, NW)
        F.seek(0)
        F.seek(4*NW*8)
        S = []
        for i in range(int(SIZE/8-4*NW)):
            S.append(struct.unpack('@d', F.read(8))[0])
        STOKES[p] = reshape(array(S, dtype=float64), (4, NW, NY, NX), 'F')
    return STOKES

# Set NICOLE input parameters for NICOLE.input
def set_nicole_input_params(**kwargs):
    TEXT = ''
    for key, value in kwargs.items():
        STR = key + '=' + str(value) + '\n'
        TEXT += STR.replace('__', ' ')
    return TEXT
# Set region and line information for NICOLE.input
def set_nicole_regions_lines(TEXT, HEAD, **kwargs):
    TEXT += '[' + HEAD + ']\n'
    for key, value in kwargs.items():
        STR = '\t' + key + '=' + str(value) + '\n'
        TEXT += STR.replace('__', ' ')
    return TEXT

# Plot Stokes profiles
def plot_profile(FIG, STOKES, **kwargs):
    TIT = ['$I/I_0$', '$Q/I$', '$U/I$', '$V/I$']
    for S in STOKES:
        for i in range(4):
            AX = FIG.add_subplot(221+i, title=TIT[i])
            if(i != 0):
                TEMP = (S[i]/S[0]).ravel()
                AX.axhline(0, color='k', ls='--', lw=0.5)
            else:
                TEMP = S[0].ravel()
                AX.axhline(1, color='k', ls='--', lw=0.5)
            AX.plot(TEMP, **kwargs)
            
# Plot important model parameters
def plot_model_bvtp(FIG, MODELS, HEIGHT, **kwargs):
    TIT = ['$B_{los}$ (G)', '$B_x$ (G)', '$B_y$ (G)', '$v_{los}$ (cm/s)', '$T$ (K)', '$v_{mic}$ (cm/s)']
    AX = [FIG.add_subplot(231+i, title=TIT[i]) for i in range(len(TIT))]
    # AXD = [FIG.add_subplot(231+i, title=TIT[i],  frame_on=False) for i in range(len(TIT))]
    for M in MODELS:
        if (8 in M.shape):
            for i,j in enumerate([4,6,7,5,1,3]):
                AX[i].plot(M[:,0], M[:,j],**kwargs)
        else:
            if (HEIGHT): 
                for i,j in enumerate([8,9,10,6,2,7]):
                    AX[i].plot(M[:,0], M[:,j], **kwargs)
                    # AX[i].set_xlabel(r'Height (km)')
                    # AXD[i].plot(M[:,0], M[:,j], **kwargs)
                    # AXD[i].xaxis.tick_top()
            else:
                for i,j in enumerate([8,9,10,6,2,7]):
                    AX[i].plot(M[:,1], M[:,j], **kwargs)
                    # AX[i].set_xlabel(r'$log \tau$')