from __future__ import print_function
import os
from os import sep as SEP
import datetime as dt
import time
import random as rndm
try:
    from tkinter import Tk, filedialog
except:
    from Tkinter import Tk
    import tkFileDialog as filedialog
from numpy import *
from matplotlib.pyplot import *
from matplotlib.gridspec import *
from matplotlib.ticker import *
from scipy.signal import medfilt2d, argrelextrema, correlate2d
from scipy.ndimage import zoom ,shift, gaussian_filter1d, gaussian_filter
from astropy.io import fits as pf
from astropy import time as astrotime
from astropy import convolution
from astropy.modeling import models, fitting, polynomial

def setrcParams():
    rcParams['font.sans-serif'] = "Liberation Sans Narrow"
    rcParams['font.family'] = "sans-serif"
    rcParams['font.size'] = 12
    rcParams['font.weight'] = 1
    rcParams['image.origin'] = 'lower'
    
def getFileFilesDir(INDEX, **kwargs):
    ROOT = Tk()
    ROOT.attributes('-topmost', True)
    ROOT.withdraw()
    if(INDEX == 0):
        TEMP = filedialog.askopenfilename(**kwargs) # Select the file
    elif(INDEX == 1):
        TEMP = filedialog.askopenfilenames(**kwargs) # Select the file
    elif(INDEX==2):
        TEMP = filedialog.askdirectory(**kwargs) # Select the file
    return TEMP


def previewImage(I, time=0):
    if (I.ndim == 2):    
        figure()
        imshow(I, cmap='gray', interpolation='none')
    elif (I.ndim == 3):
        figure()
        imshow(I[0], cmap='gray', interpolation='none')
    else:
        print('Invalid image dimensions!')
        return
    if time is not 0:
        pause(time)
        close()
        
def previewPlot(*args, **kwargs):   
    figure()
    plot(*args)
    try:
        pause(kwargs['time'])
        close()
    except: pass
    
def round_off(X, base=5):
    return int32(base * floor(float32(X)/base)), int32(base * ceil(float32(X)/base))

def xy_disp(AX, I, PLATESCALE):
    BASE = 5
    #------------ x ticks
    XLOC = arange(I.shape[1]) - I.shape[1]/2
    XLAB = XLOC*PLATESCALE
    TEMP, XLIM = round_off(XLAB.max(), BASE)
    XLAB = arange(-XLIM, XLIM+BASE, 2*BASE)
    XLOC = XLAB/PLATESCALE + I.shape[1]/2
    #------------ y ticks
    YLOC = arange(I.shape[0]) - I.shape[0]/2
    YLAB = YLOC*PLATESCALE
    TEMP, YLIM = round_off(YLAB.max(), BASE)
    YLAB = arange(-YLIM, YLIM+BASE, 2*BASE)
    YLOC = YLAB/PLATESCALE + I.shape[0]/2
    #--------------------- display
    IM = AX.imshow(I, cmap='gray', interpolation='none')
    #--------------------- ticks, limits
    AX.set_xticks(XLOC)
    AX.set_xticklabels(map(str, XLAB))
    AX.set_yticks(YLOC)
    AX.set_yticklabels(map(str, YLAB))
    AX.autoscale(enable=True, axis='x', tight=True)
    AX.autoscale(enable=True, axis='y', tight=True)
    #--------------------- labels
    AX.set_xlabel('arcsec')
    AX.set_ylabel('arcsec')
    #--------------------- colorbar
    CLB = colorbar(IM)
    TICKLOCS = MaxNLocator(nbins=5)
    CLB.locator = TICKLOCS
    CLB.update_ticks()
    
def ly_disp(AX, I, DISPERSION, PLATESCALE, PIX_CAII, **kwargs):
    if(I.shape[0] > I.shape[1]): I = swapaxes(I, axis1=0, axis2=1)
    #------------- x ticks
    BASE = 2
    L = (arange(I.shape[1])-PIX_CAII)*DISPERSION
    TEMP, LMIN = round_off(rint(L.min()), BASE)
    TEMP, LMAX = round_off(rint(L.max()), BASE)
    LLAB = arange(LMIN, LMAX, BASE)
    LLOC = float32(LLAB)/DISPERSION + PIX_CAII
    #------------ y ticks
    BASE = 10
    YLOC = arange(I.shape[0]) - I.shape[0]/2
    YLAB = YLOC*PLATESCALE
    TEMP, YLIM = round_off(YLAB.max(), BASE)
    YLAB = arange(-YLIM, YLIM+BASE, 2*BASE)
    YLOC = YLAB/PLATESCALE + I.shape[0]/2
    #----------- display
    AX.imshow(I, cmap='gray', **kwargs)
    #----------- limits
    AX.set_xticks(LLOC)
    AX.set_xticklabels(map(str, int32(LLAB)))
    AX.set_yticks(YLOC)
    AX.set_yticklabels(map(str, YLAB))
    AX.autoscale(enable=True, axis='x', tight=True)
    AX.autoscale(enable=True, axis='y', tight=True)
    #----------------- labels
    AX.set_xlabel('Wavelength + 8542 $\AA$')
    
def l_disp(AX, L, STOKES_IND, DISPERSION, PIX_CAII, **kwargs):
    YLAB = [r'$I/I_{max}$', r'$Q/I$', r'$U/I$', r'$V/I$']
    W = arange(L.shape[0])
    W = (W-PIX_CAII)*DISPERSION
    AX.plot(W, L, **kwargs)
    if (L.min()<0): YLIM_1 = 1.1*L.min()
    else: YLIM_1 = 0.9*L.min()
    if (L.max()<0): YLIM_2 = 0.9*L.max()
    else: YLIM_2 = 1.1*L.max()
    AX.set_ylim([YLIM_1, YLIM_2])
    AX.autoscale(enable=True, axis='x', tight=True)
    AX.set_xlabel('Wavelength + 8542 $\AA$')
    AX.set_ylabel(YLAB[STOKES_IND])
    AX.locator_params(axis='y', nbins=5)
    
def draw_lines(AX, HOR, VER, **kwargs):
    for H in HOR:
        AX.axhline(H, **kwargs)
    for V in VER:
        AX.axvline(V, **kwargs)
        
def get_xy(IMAGE, XBIN, YBIN):
    figure()
    imshow(IMAGE, cmap='gray', interpolation='none')
    X, Y = ginput(1)[0]
    close()
    XBEG, YBEG = X-XBIN/2, Y-YBIN/2
    XEND, YEND = XBEG+XBIN, YBEG+YBIN
    return int(XBEG), int(XEND), int(YBEG), int(YEND)

def rebin(ARRAY, NEWSHAPE):
    SHAPE = NEWSHAPE[0], ARRAY.shape[0]//NEWSHAPE[0], NEWSHAPE[1], ARRAY.shape[1]//NEWSHAPE[1]
    return ARRAY.reshape(SHAPE).mean(-1).mean(1)
    