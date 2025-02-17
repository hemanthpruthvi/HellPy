# -*- coding: utf-8 -*-
"""
"""
import cv2
from scipy.ndimage import zoom, shift, rotate
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D
from lmfit.models import GaussianModel
from lmfit.models import QuadraticModel
from process_files import *
from process_darks import *

class target_process:
    def __init__(self, name='raw.ini'):
        if isinstance(name, str):
            self.ds = data_set(name)
        else:
            self.ds = name
        self.meta = {}
        self.meta['dirtree'] = cp.deepcopy(self.ds.meta['dirtree'])
        self.meta['dirtree'][-2] += '_' + self.meta['dirtree'][-3]
        self.meta['dirtree'][-3] = 'Interim'
        self.meta['dir'] = os.sep.join(self.meta['dirtree'])
        self.meta['mag'] = 0.934
        self.meta['roi'] = 1200
        self.meta['minmag'] = 1.05
        self.meta['maxmag'] = 1.1
        try: 
            os.makedirs(self.meta['dir'])
        except: 
            pass
        self.read_interim_data()
        self.data = {}
        for i in range(self.ds.meta['nlines']):
            self.read_target_data(iline=i)
            self.analyze_target()
    
    def read_interim_data(self):
        tempds = data_set('dark.ini')
        for i in range(self.ds.meta['nlines']):
            self.meta[i] = {}
            temp = tempds.meta['dir'] + os.sep + self.ds.meta[i]['line']
            self.meta[i]['darkfiles'] = [temp+os.sep+file for file in os.listdir(temp)]
            tempn = int(len(self.meta[i]['darkfiles'])/3)
            self.meta[i]['darkfiles'] = np.reshape(self.meta[i]['darkfiles'], 
                                                   (3, tempn))           
    def read_dark_data(self, iline=0, icam=0, idata=0):
        self.dfile_name = self.meta[iline]['darkfiles'][icam][idata]
        hdu = fits.open(self.dfile_name)[0]
        self.ddata = hdu.data
        self.dheader = hdu.header
        
    def process_single_set(self, data):
        dd = np.average(data, axis=2) - self.ddata
        dd = np.array(dd, dtype=data.dtype)
        return dd
        
    def read_target_data(self, iline=0):
        for d in tqdm.tqdm(range(3)):
            self.ds.read_data(iline=iline, icam=d, idata=0)
            self.read_dark_data(iline=iline, icam=d, idata=0)
            self.data[d] = self.process_single_set(self.ds.data)
        self.data[1] = np.fliplr(self.data[1])
        self.data[2] = np.fliplr(np.flipud(self.data[2]))
            # tempdir = self.meta['dir'] + os.sep + self.ds.filename.split(os.sep)[-2]
            # try: os.makedirs(tempdir)
            # except: pass
            # tempname = tempdir + os.sep + self.ds.filename.split(os.sep)[-1]
            # tempname = tempname.replace('.dat', '.fits')
            # tempname = tempname.replace('.DAT', '.fits')
            # if (os.path.exists(tempname)): 
            #     pass
            # else:
            #     dd = self.process_single_set(self.ds.data)
            #     hdu = fits.PrimaryHDU(data=dd)
            #     # print('Writing file: ', tempname)
            #     hdu.writeto(tempname, overwrite=True)  
        # fig, axs = plt.subplots(1,3,figsize=(18,8))
        # for d in tqdm.tqdm(range(3)):
        #     axs[d].imshow(self.data[d])
        # plt.tight_layout()
        
    def analyze_target(self):
        bb = 1.0*self.data[0]
        p1 = 1.0*self.data[1]
        p2 = 1.0*self.data[2]
        # Rotation
        ang_bb = self.compute_angle(bb)
        ang_p1 = self.compute_angle(p1)
        ang_p2 = self.compute_angle(p2)
        an1 = ang_p1 - ang_bb
        an2 = ang_p2 - ang_bb
        p1 = rotate(p1, an1)
        p2 = rotate(p2, an2)
        print('Reference angles are: ', ang_bb, ang_p1, ang_p2)
        # Scaling
        mag1 = self.compute_magnification(bb, p1)
        mag2 = self.compute_magnification(bb, p2)
        self.meta['mag'] = 0.5*(mag1+mag2)
        print('Magnification: ', mag1, mag2)
        # bb = zoom(self.data[0], self.meta['mag'])
        p1 = zoom(p1, self.meta['mag'])
        p2 = zoom(p2, self.meta['mag'])
        hpad = p1.shape[1]-bb.shape[1]
        vpad = p1.shape[0]-bb.shape[0]
        lpad, rpad = int(hpad/2), hpad-int(hpad/2)
        tpad, bpad = int(vpad/2), vpad-int(vpad/2)
        bb = np.pad(bb, ((tpad,bpad),(lpad,rpad)))
        # Translation
        sh1 = self.compute_shift(bb, p1)
        sh2 = self.compute_shift(bb, p2)
        print('Shifts: ', sh1, sh2)
        p1 = shift(p1, sh1)
        p2 = shift(p2, sh2)
        # Crop 
        size = self.meta['roi']
        hcrop = (p1.shape[1]-size)//2
        vcrop = (p1.shape[0]-size)//2
        self.bb = bb[vcrop:vcrop+size, hcrop:hcrop+size] 
        self.p1 = p1[vcrop:vcrop+size, hcrop:hcrop+size]
        self.p2 = p2[vcrop:vcrop+size, hcrop:hcrop+size]
        self.view_targets()
           
    def view_targets(self):
        fig, axs = plt.subplots(1,3,figsize=(18,8))
        axs[0].imshow(self.bb)
        axs[1].imshow(self.p1)
        axs[2].imshow(self.p2)
        plt.tight_layout()
        
    def compute_shift(self, img1, img2):
        ny, nx = img1.shape
        nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
        im1 = img1[ny//4:3*ny//4,nx//4:3*nx//4]
        im2 = img2[ny//4:3*ny//4,nx//4:3*nx//4]
        im1, im2 = im1/im1.mean(), im2/im2.mean()
        # Window
        coeff = 0.54
        x = coeff - (1-coeff)*np.cos(2*np.pi*np.arange(nxx)/(nxx-1));
        y = coeff - (1-coeff)*np.cos(2*np.pi*np.arange(nyy)/(nyy-1));
        w = np.reshape(x,[1,nxx])*np.reshape(y,[nyy,1])
        # Images
        # im1, im2 = w*im1, w*im2
        fim1, fim2 = np.fft.fft2(im1), np.fft.fft2(im2)
        corr = np.fft.ifftshift(np.abs(np.fft.ifft2(fim1*np.conj(fim2))))
        half = np.array([nyy//2, nxx//2])
        sh = np.argwhere(corr==corr.max()).flatten()[0:2]-half
        return sh
    
    def compute_angle(self, img):
        ny, nx = img.shape
        nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
        td = 1.0*img[ny//4:ny//4+nyy, nx//4:nx//4+nxx]
        td = td/td.max()
        # thresh = 0.8
        thresh = self.compute_thresh(img)
        # print('Threshold: ', thresh)
        td = np.array(td<thresh, dtype=np.uint8)
        plt.figure()
        plt.imshow(td)
        lines = cv2.HoughLinesP(td, 1, np.pi/180, 50, 10, nxx//2, 4)
        ls = 0.0*td
        for line in lines:
            for x1, y1, x2, y2 in line:
                plt.plot([x1,x2],[y1,y2],color='red')
        angs = (lines[:,0,1]-lines[:,0,3])/(lines[:,0,0]-lines[:,0,2])
        angs = np.degrees(np.arctan(angs))
        angs[np.argwhere(angs<0)] += 90 
        ang = np.median(angs)
        # print('Reference angle: ', ang)
        return ang
    """
    Compute appropriate image threshold by analyzing histogram
    """
    def compute_thresh(self, img):
        # Normalize image
        ny, nx = img.shape
        nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
        td = 1.0*img[ny//4:ny//4+nyy, nx//4:nx//4+nxx]
        td = td/td.max()
        # Compute histogram
        hd = np.histogram(td, bins=nxx)
        x, y = hd[1][1::], hd[0]
        x, y = x[nxx//2::], y[nxx//2::]
        # Fit histogram
        gm1 = GaussianModel(prefix='g1_')
        gm2 = GaussianModel(prefix='g2_')
        model = gm1 + gm2
        params = model.make_params(g1_center=0.7, g2_center=0.9)
        result = model.fit(y, params, x=x)
        # Compute threshold
        c1 = result.params['g1_center'].value
        c2 = result.params['g2_center'].value
        s1 = result.params['g1_sigma'].value
        s2 = result.params['g2_sigma'].value
        thresh = (c1*s2+c2*s1)/(s1+s2)
        # plt.figure()
        # plt.plot(x, result.best_fit, x, y)
        return thresh
    """
    Compute magnification between broadband images and polarimeter images
    """
    def compute_magnification(self, img1, img2):
        # Select range of magnification
        zooms = np.linspace(self.meta['minmag'],self.meta['maxmag'], 101)
        corrs = 0.0*zooms
        half = img1.shape[0]//4
        ny1, nx1 = img1.shape
        im1 = img1[ny1//2-half:ny1//2+half, nx1//2-half:nx1//2+half]
        im1 = (im1-im1.mean())/im1.std()
        # Find valid correlation using convolution
        for i, fact in enumerate(tqdm.tqdm(zooms)):
            im2 = zoom(img2, fact)
            ny2, nx2 = img2.shape
            im2 = (im2-im2.mean())/im2.std()
            corr = fftconvolve(im1, im2, mode='valid')
            corrs[i] = corr.max()
        # Find maxima of corelation
        mi = np.argmax(corrs).flatten()[0]
        ni = 30
        x = zooms[mi-ni:mi+ni] 
        y = corrs[mi-ni:mi+ni]
        qmod = QuadraticModel(prefix='qm_')
        params = qmod.make_params()
        result = qmod.fit(y, params, x=x)
        # plt.figure()
        # plt.plot(zooms, corrs)
        # plt.plot(x, result.best_fit)
        a = result.params['qm_a'].value
        b = result.params['qm_b'].value
        c = result.params['qm_c'].value
        mag = -b/a/2
        return mag