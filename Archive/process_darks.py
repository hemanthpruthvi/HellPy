# -*- coding: utf-8 -*-
"""

"""

import tqdm
from astropy.io import fits
from scipy.ndimage import median_filter 
from process_files import *

class dark_process:
    def __init__(self, name='meta.ini'):
        if isinstance(name, str):
            self.ds = data_set(name)
        else:
            self.ds = name
        self.meta = {}
        self.meta['dirtree'] = cp.deepcopy(self.ds.meta['dirtree'])
        self.meta['dirtree'][-2] += '_' + self.meta['dirtree'][-3]
        self.meta['dirtree'][-3] = 'Interim'
        self.meta['dir'] = os.sep.join(self.meta['dirtree'])
        try: 
            os.makedirs(self.meta['dir'])
        except: 
            pass
        self.process_data_set()
    
    def process_data_set(self):
        ind = [(l,d,c) for l in range(self.ds.meta['nlines']) 
               for d in range(self.ds.meta[l]['nframes'][1])
               for c in range(self.ds.meta[l]['nframes'][0])]
        N = len(ind)
        for n, i in enumerate(tqdm.tqdm(ind)):
            l, d, c = i
            self.ds.read_data(iline=l, icam=d, idata=c)
            tempdir = self.meta['dir'] + os.sep + self.ds.filename.split(os.sep)[-2]
            try: os.makedirs(tempdir)
            except: pass
            tempname = tempdir + os.sep + self.ds.filename.split(os.sep)[-1]
            tempname = tempname.replace('.dat', '.fits')
            tempname = tempname.replace('.DAT', '.fits')
            if (os.path.exists(tempname)): 
                pass
            else:
                dd = self.process_single_set(self.ds.data)
                hdu = fits.PrimaryHDU(data=dd)
                # print('Writing file: ', tempname)
                hdu.writeto(tempname, overwrite=True)

    def process_single_set(self, data):
        dd = np.average(data, axis=2)
        dd = median_filter(dd, size=3)
        dd = np.array(dd, dtype=data.dtype)
        return dd
        
    

