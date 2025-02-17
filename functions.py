import os, sys, time, tqdm, cv2, matplotlib, termcolor, configobj
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits as pf
from lmfit.models import *
from lmfit import Minimizer, Parameters, minimize, Model
from lmfit.lineshapes import gaussian, parabolic
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.ndimage import zoom, shift, rotate, gaussian_filter, median_filter, sobel, affine_transform
from scipy.signal import fftconvolve
from scipy.optimize import leastsq, curve_fit
from scipy.special import wofz
import datetime as dt
import subprocess as sp
import polarTransform as ptr
import matplotlib as mpl


def fit_2d_plane(z, w=1):
    """
    Fit the 2d data with the equation of z = ax + by + c
    Input:  2d array
    Output: fitted 2d array
    """
    ny, nx = z.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    s_1 = np.sum(np.ones([ny,nx])*w)
    s_x = np.sum(x*w)
    s_y = np.sum(y*w)
    s_z = np.sum(z*w)
    s_xx = np.sum(x**2*w)
    s_yy = np.sum(y**2*w)
    s_xy = np.sum(x*y*w)
    s_zx = np.sum(z*x*w)
    s_zy = np.sum(z*y*w)
    M_z = np.matrix([[s_z],[s_zx],[s_zy]])
    M_xy = np.matrix([[s_x, s_y, s_1],[s_xx, s_xy, s_x],[s_xy, s_yy, s_y]])
    M_abc = np.linalg.pinv(M_xy)*M_z
    a, b, c = np.array(M_abc).flatten()
    zfit = a*x + b*y + c
    return zfit

def generate_config():
    """
    Create config file by selecting directories for various data
    """
    config = configobj.ConfigObj('config.ini')
    top_dir = config['topdir']
    #
    termcolor.cprint('\nSelect the directory for science data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['science']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for darks data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['darks']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for flats data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['flats']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for pcalibration data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['pcalibration']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for targetplate data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['targetplate']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for pinhole data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['pinhole']['directory'] = direc
    config.write()
    
"""
Select directory for needed data, given top level directory
"""
def select_dir(top_dir):
    if not os.path.isdir(top_dir):
        print(top_dir, ' is not a valid directory!')
        return
    dir_tree = top_dir.split(os.sep)
    path = os.sep.join(dir_tree)
    list_dir(path)
    sub_dir = input('Select the next level sub-directory in ' 
                    + path +  '\nType # and ENTER to go back a level \nPress ENTER to finalize selection : ')
    if (sub_dir == ''):
        final_dir = os.sep.join(dir_tree)
        if os.path.isdir(final_dir):
            print('Selected directory is: ', final_dir)
        else:
            print(final_dir, ' is not a valid directory!')
        return final_dir
    
    elif (sub_dir == '#'):
        del dir_tree[-1]
        final_dir = select_dir(os.sep.join(dir_tree))
    else:
        dir_tree.append(sub_dir)
        if not os.path.isdir(os.sep.join(dir_tree)):
            print('Invalid selection!')
            del dir_tree[-1]
        final_dir = select_dir(os.sep.join(dir_tree)) 
    return final_dir

"""
List out the details of the directory
"""
def list_dir(name):
    if not os.path.isdir(name):
        print(name, ' is not a valid directory!')
        return
    dirs = sorted(os.listdir(name))
    print('List of sub-directories and files: ')
    for tempdir in dirs: 
        temp = name + os.sep + tempdir
        if(os.path.isdir(temp)): 
            file_count, file_size = get_dir_details(temp)
            print(termcolor.colored('%s \t %s \t %i'%(tempdir, file_size, file_count), 'cyan'))
    for tempdir in dirs:
        temp = name + os.sep + tempdir
        if(os.path.isfile(temp)): 
            file_count, file_size = get_file_details(temp)
            print(termcolor.colored('%s \t %s'%(tempdir, file_size), 'magenta'))

"""
Get size and number of files for the given directory
"""
def get_dir_details(name):
    file_count = 0
    file_size = 0
    KB = 1024.0
    MB = 1024*KB
    GB = 1024*MB
    TB = 1024*GB
    for temppath, tempdirs, tempfiles in os.walk(name):
        file_count += len(tempfiles)
        file_size += sum(os.path.getsize(os.path.join(temppath, name)) for name in tempfiles)
    if (file_size>=TB): 
        all_size = file_size/TB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' TB'
    elif (file_size<TB and file_size>=GB):
        all_size = file_size/GB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' GB'
    elif (file_size<GB and file_size>=MB):
        all_size = file_size/MB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' MB'
    elif (file_size<MB and file_size>=KB):
        all_size = file_size/KB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' KB'
    else:
        all_size = str(file_size) + ' B'
    return file_count, all_size

"""
Get size for the file
"""
def get_file_details(name):
    file_count = 1
    file_size = 0
    KB = 1024.0
    MB = 1024*KB
    GB = 1024*MB
    TB = 1024*GB
    file_size = os.path.getsize(name)
    if (file_size>=TB): 
        all_size = file_size/TB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' TB'
    elif (file_size<TB and file_size>=GB):
        all_size = file_size/GB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' GB'
    elif (file_size<GB and file_size>=MB):
        all_size = file_size/MB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' MB'
    elif (file_size<MB and file_size>=KB):
        all_size = file_size/KB
        all_size = int(10*all_size)/10.0
        all_size = str(all_size) + ' KB'
    else:
        all_size = str(file_size) + ' B'
    return file_count, all_size




def crop_data(data, hor, ver, width, height, mode='start'):
    """
    Crop the data to exclude unnecessary parts.
    Input:  3d data cube, 
            crop start x - coordinate, 
            crop start y - coordinate,
            cropped data x - width, and
            cropped data y - height in pixels.
    Output: cropped 3d data cube.
    """
    if (mode=='start'):
        new_data = data[ver:ver+height,hor:hor+width,:]
    elif (mode=='center'):
        new_data = data[ver-height//2:ver+height//2+height,hor-width//2:hor-width//2+width,:]
    else:
        new_data = data
    return new_data
    


def list_files(startpath):
    """
    | 
    |  
    |
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def compute_plane_residual(params, xdata, ydata):
    """
    |  Input : 
    |  Output : residual after fitting a plane
    """
    x, y = xdata[:,0], xdata[:,1]
    a, b, c = params['a'], params['b'], params['c']
    return (a*x + b*y + c - ydata)
    
def fit_plane(z):
    """
    """
    ny, nx = z.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    xdata = np.zeros([nx*ny, 2])
    xdata[:,0], xdata[:,1] = x.flatten(), y.flatten()
    ydata = z.flatten()
    #
    guess_params = Parameters()
    guess_params.add(name='a', value=0.0)
    guess_params.add(name='b', value=0.0)
    guess_params.add(name='c', value=0.0)
    #
    fitter = Minimizer(compute_plane_residual, 
                        guess_params, 
                        fcn_args=(xdata, ydata))
    fit_params = fitter.leastsq().params
    #
    a = fit_params['a'].value
    b = fit_params['b'].value
    c = fit_params['c'].value
    #
    return a*x + b*y + c

def load_settings(file_name):
    """
    Input:  name of the settings file including full path
    Output: settings in the form of a dictionary
    """
    settings = {}
    temp = np.loadtxt(file_name, dtype=str)
    for t in temp:
        key, value = t.split('=')
        settings[key] = value
    return settings

def process_dark_cube(dk, axis=2):
    """
    |   Process dark frame cube: averageing and median filtreing to remove hot/cold pixels
    |   Input : 3d data cube array (y,x,z)
    |   Output : 2d data array (y,x,1)
    """
    dkm = median_filter(np.average(dk, axis=axis), size=3)[:,:,np.newaxis]
    return dkm

def find_line_valley(line, npts=7):
    """
    |   Compute line center by sub-pixel interpolation of Poly2 minima
    |   Input : 1d line profile, number of points around minima to consider for Poly2 interpolation 
    |   Output : valley position (double)
    """
    line_min = np.argmin(line)
    x0 = line_min-npts//2
    x1 = x0 + npts
    x = np.arange(x0,x1)
    y = line[x0:x1]
    yy = np.matrix(y.reshape([len(y),1]))
    xx = np.matrix(np.array([x**2, x, x*0+1]).transpose())
    aa = np.linalg.inv((xx.transpose()*xx))*xx.transpose()*yy
    a = np.array(aa).flatten()
    xmin = -0.5*a[1]/a[0]
    return xmin
    
def poly2_2d(x_, *args):
    """
    Compute 2d quadratic(Poly2) function z = ax2 + by2 + cx + dy + e
    Input:  2d array (2,x)
            2d Poly2 coefficients (5,)
    Output: 1d array (1,x)
    """
    a, b, c, d, e = args
    y, x = x_[0], x_[1]
    z = a*x**2 + b*y**2 + c*x + d*y + e
    return z

# def parabolic_2d(x_, *args):
#     """
#     Compute 2d parabolic function z = a[(x-X0)2 + (y-Y0)2] + Z0
#     Input:  2d array (2,x)
#             2d Parabola coefficients (3,)
#     Output: 1d array (1,x)
#     """
#     a, X0, Y0, Z0 = args
#     y, x = x_[0], x_[1]
#     z = a*(x-X0)**2 + a*(y-Y0)**2 + Z0
#     return z

def parabolic_2d(x, a, x0, y0, z0):
    """
    Compute 2d parabolic function z = a[(x-X0)2 + (y-Y0)2] + Z0
    Input:  2d array (2,x)
            2d Parabola coefficients (3,)
    Output: 1d array (1,x)
    """
    yy, xx = x
    z = a*(xx-x0)**2 + a*(yy-y0)**2 + z0
    return z

def get_line_num(settings, linestr, iline=0):
    """
    Get index of the line in the settings file, using line ID string
    Input:  settings (configobj)
            line string
            (opt.) index guess
    Output: index of the line
    """
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

def show_img_series(data, fps=10, cmap='gray'):
    """
    Play the images in 3d data as a video
    Input:  3d data array
            (opt.) frame rate
    Output: video display
    """
    plt.figure()
    disp = plt.imshow(data[:,:,0], cmap=cmap)
    for i in range(data.shape[2]):
        disp.set_data(data[:,:,i])
        plt.pause(1.0/fps)
        plt.draw()

def blink_frames(frames, pause=0.5, repeat=1):
    """
    Blink the images in list
    Input:  list of images
            (opt.) blink pause time
            (opt.) number of repetitions
    Output: display
    """
    plt.figure()
    for i in range(repeat):
        disp = plt.imshow(frames[0], cmap='gray')
        plt.pause(pause)
        for frame in frames[1::]:
            disp.set_data(frame)
            plt.pause(pause)
            plt.draw()
    return

def compute_mean_profile(data, median=False):
    """
    Calculate template line profile from 3d array of lines
    Input:  3d data array
            (opt.) median/mean
    Output: template line profile
    """
    temp = data/np.mean(data, axis=2)[:,:,np.newaxis]
    if (median):
        line = np.mean(temp, axis=(0,1))
    else:
        line = np.median(temp, axis=(0,1))
    return line

def compute_logps(x, axis=0):
    """
    |   Compute log power spectra of a given array
    |   Input : array, axis along which power spectra is to be computed
    |   Output : array (same size as input)
    """
    xf = np.fft.fftshift(np.fft.fft(x, axis=axis))
    xf = xf*np.conjugate(xf)
    xfl = np.log10(xf)
    return xfl

def time_now():
    """
    |   Get current time in HH:MM:SS format 
    """
    return dt.datetime.now().strftime("%H:%M:%S")

def zoom_clipped(arr, mag):
    """
    |   Scipy zoom routine for the images, with shape preserved
    |   Input:  2d array
    |           magnification
    |   Output: zoomed array
    """
    Y, X = arr.shape
    arr_ = zoom(arr, mag, mode='nearest')
    Y_, X_ = arr_.shape
    if (mag>1): y, x = Y_-Y, X_-X
    elif (mag<1): y, x = Y-Y_, X-X_
    top, bot = int(y//2), y-int(y//2)
    lef, rig = int(x//2), x-int(x//2)
    if (mag<1 and mag>0):
        arr_z = np.pad(arr_, ((top,bot),(lef,rig)), mode='mean')
    elif (mag>1 and mag>0): 
        arr_z = arr_[top:-bot,lef:-rig]
    else:
        arr_z = arr
    return arr_z

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



def read_time_stamps_obs(dir_name, iline):
    """
    |   Extract absolute frame numbers corresponding to a line from the timestamps file
    |   Applicable only for science data
    |   Input:  data directory name to search the timestamps file
    |           spectral line id number
    |   Output: frame numbers for bbi, pol1 and pol2 channels
    """
    tsfile = [f for f in os.listdir(dir_name) if 'timestamp' in f]
    tsfile = dir_name + os.sep + tsfile[0]
    ts = np.loadtxt(tsfile, delimiter=',')
    ts_line = ts[np.where(ts[:,1]==iline)]
    im0ind, im1ind, im2ind = ts_line[:,3], ts_line[:,6], ts_line[:,9]
    return np.int64(im0ind), np.int64(im1ind), np.int64(im2ind)

def read_time_stamps_cal(dir_name, iline):
    """
    |   Extract absolute frame numbers corresponding to a line from the timestamps file
    |   Applicable only for flats, polcal data
    |   Input:  data directory name to search the timestamps file
    |           spectral line id number
    |   Output: frame numbers for bbi, pol1 and pol2 channels
    """
    tsfile = [f for f in os.listdir(dir_name) if 'timestamp' in f]
    tsfile = dir_name + os.sep + tsfile[0]
    ts = np.loadtxt(tsfile, delimiter=',')
    ts_line = ts[np.where(ts[:,0]==iline)]
    im0ind, im1ind, im2ind = ts_line[:,3], ts_line[:,6], ts_line[:,9]
    return im0ind, im1ind, im2ind

def surf_plot(img, ax=None):
    """
    Plot the 2d image as a surface
    Input:  2d array
            (opt.) plot axis
    Output: plot axis
    """  
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    Y, X = img.shape
    YY, XX = np.meshgrid(np.arange(Y), np.arange(X))
    ax.plot_surface(YY, XX, img)
    return ax

def get_fts_spectra(file_name, wavelength=630.2e-9, wave_range=0.3e-9):
    """
    Read fts spectrum from file to get intensity vs. wavelength data with 1 mA sampling
    Input:  file name
            central wavelength of the range
            wavelength range
    Output: wavelength values
            continuum normalized intensity values
    """
    wave_step=1e-13
    spectra = np.loadtxt(file_name, delimiter=',')
    waves_full = 1.0e-10*spectra[:,0]
    intens_full = 1.0e-2*spectra[:,1]
    wave_beg, wave_end = wavelength-wave_range/2.0, wavelength+wave_range/2.0
    waves = np.arange(wave_beg, wave_end, wave_step)
    iwave_line = np.argwhere((waves_full>waves[0]) & (waves_full<waves[-1]))
    waves_line = waves_full[iwave_line]
    intens_line = intens_full[iwave_line]
    intens = zoom(intens_line.flatten(), len(waves)/len(waves_line))
    nwaves = len(waves)
    return waves, intens

def get_imcenter(img, nxx, nyy):
    ny, nx = img.shape
    x1, y1 = (nx-nxx)//2, (ny-nyy)//2
    x2, y2 = x1+nxx, y1+nyy
    return img[y1:y2,x1:x2]

def compute_shift(img1, img2, roi=0.5):
    ny, nx = np.int16(roi*np.array(img1.shape))
    im1_cen = get_imcenter(img1, nx, ny)
    im1_int = fit_2d_plane(im1_cen, w=im1_cen)
    im2_cen = get_imcenter(img2, nx, ny)
    im2_int = fit_2d_plane(im2_cen, w=im2_cen)
    im1, im2 = im1_cen-im1_int, im2_cen-im2_int
    im1, im2 = im1/im1.mean(), im2/im2.mean()
    # Window
    hamm_0 = np.reshape(np.hamming(im1.shape[0]), (im1.shape[0], 1))
    # hamm_0[32:-32] = hamm_0[31]
    hamm_0 /= hamm_0.max()
    hamm_1 = np.reshape(np.hamming(im1.shape[1]), (1, im1.shape[1]))
    # hamm_1[32:-32] = hamm_1[31]
    hamm_1 /= hamm_1.max()
    w = hamm_0*hamm_1
    # coeff = 0.54
    # x = np.exp(-(4*(np.arange(nx)-nx//2)/nx)**4)
    # y = np.exp(-(4*(np.arange(ny)-ny//2)/ny)**4)
    # x = (x-x.min())/(x.max()-x.min())
    # y = (y-y.min())/(y.max()-y.min())
    # w = np.reshape(x,[1,nx])*np.reshape(y,[ny,1])
    # Images
    im1, im2 = w*im1, w*im2
    fim1, fim2 = np.fft.fft2(im1), np.fft.fft2(im2)
    corr = np.fft.ifftshift(np.abs(np.fft.ifft2(fim1*np.conj(fim2))))
    half = np.array([ny//2, nx//2])
    sh = np.argwhere(corr==corr.max()).flatten()[0:2]-half
    return sh

def compute_align_params(img0, img1, npts=4):
    """
    |   Compute affine transform between two images by interactively selecting key points of correspondence in both the images
    |   Input:  reference image
    |           second image
    |           (opt.) number of key points (default = 4)
    |   Output: affine matrix (I_sec = A*I_ref)
    |           alignment parameters array (x scale, y scale, rotation, x shift, y shift)
    """
    pts0, pts1 = [], []
    for i in range(npts):
        plt.figure()
        plt.imshow(img0, cmap='gray')
        plt.tight_layout()
        plt.title('Select key point %i in the first image'%(i+1))
        pts0_ = plt.ginput(n=1, timeout=0, mouse_add=None, mouse_pop=None, mouse_stop=None)
        plt.close()
        #
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.tight_layout()
        plt.title('Select key point %i in the second image'%(i+1))
        pts1_ = plt.ginput(n=1, timeout=0, mouse_add=None, mouse_pop=None, mouse_stop=None)
        plt.close()
        pts0.append(pts0_)
        pts1.append(pts1_)
    # print(pts0, pts1)
    #
    affine_mat, temp = cv2.estimateAffine2D(np.array(pts0), np.array(pts1))
    magx, magy = np.linalg.norm(affine_mat[:,0]), np.linalg.norm(affine_mat[:,1])
    rotang = np.degrees(np.arctan2(-affine_mat[0,1]/magy, affine_mat[0,0]/magx))
    shiftx, shifty = affine_mat[0,2], affine_mat[1,2]
    #
    return affine_mat, np.array([magx, magy, rotang, shiftx, shifty]), pts0, pts1

def compute_image_shift(img1, img2):
    """
    |   Compute the shift between the two images using fft-based correlation. To align the images use I1 = scipy.ndimage.shift(I2)
    |   Input:  first image (I1)
    |           second image (I2)
    |   Output: 1x2 shift array   
    """
    ny, nx = img1.shape
    roi = 1
    im1, im2 = (img1-img1.mean())/img1.std(), (img2-img2.mean())/img2.std()
    # Window
    hamm_0 = np.reshape(np.hamming(im1.shape[0]), (im1.shape[0], 1))
    hamm_0 /= hamm_0.max()
    hamm_1 = np.reshape(np.hamming(im1.shape[1]), (1, im1.shape[1]))
    hamm_1 /= hamm_1.max()
    w = hamm_0*hamm_1
    im1, im2 = w*im1, w*im2

    fim1, fim2 = np.fft.fft2(im1), np.fft.fft2(im2)
    corr = np.fft.ifftshift(np.abs(np.fft.ifft2(fim1*np.conj(fim2))))
    half = np.array([ny//2, nx//2])
    sh = np.argwhere(corr==corr.max()).flatten()[0:2]-half
    return sh

def shift_integer_pixel(arr, sh):
    """
    |   Efficiently compute scipy.ndimage.shift for 2D images when the shift is integer pixels, simply by using numpy indexing.
    |   Input:  image
    |           1x2 shift array
    |   Output: shifted image
    """
    sh = np.int64(sh)
    if (sh[0]>=0):
        arr = arr[sh[0]::,:,:,:]
        arr = np.pad(arr, ((sh[0],0),(0,0),(0,0),(0,0)), mode='mean')
    elif (sh[0]<0):
        arr = arr[0:sh[0],:,:,:]
        arr = np.pad(arr, ((0,-sh[0]),(0,0),(0,0),(0,0)), mode='mean')
    #
    if (sh[1]>=0):
        arr = arr[:,sh[1]::,:,:]
        arr = np.pad(arr, ((0,0),(sh[1],0),(0,0),(0,0)), mode='mean')
    elif (sh[1]<0):
        arr = arr[:,0:sh[1],:,:]
        arr = np.pad(arr, ((0,0),(0,-sh[1]),(0,0),(0,0)), mode='mean')
    return arr

def fit_quadratic(prof):
    """
    |   Fit a parabolic profile to data using lmfit
    |   Intended for fitting time varying instrumental crosstalks
    |   Input:  1d crosstalk profile
    |   Output: 1d best fit for the profile
    """
    x = np.arange(len(prof))
    qmodel = QuadraticModel()
    params = qmodel.guess(prof, x=x)
    result = qmodel.fit(prof, params, x=x)
    prof_fit = result.best_fit
    return prof_fit

def fit_line_slope(y, x):
    """
    |   Fit a line profile to ratio of the input data using lmfit
    |   Intended for fitting time varying instrumental crosstalks Q,U->V
    |   Input:  1d array y
                1d array x
    |   Output: best fit slope
    """
    qmodel = LinearModel()
    params = qmodel.guess(y, x=x)
    result = qmodel.fit(y, params, x=x, weights=np.abs(y))
    ratio = result.best_values['slope']
    return ratio
