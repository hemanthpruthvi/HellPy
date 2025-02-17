from functions import *
from func_flats import *


def coadd_modulated_imgs(data, imod, nmod, nacc, nwav):
    """
    Prepare sorted modulated images/frames
    Input:  3d data cube
            index extracted from time stramps
            number of modulation states
            number of accumulations
            number of wavelength points
    Output: sorted and co-added images/frames
    """
    Y, X, Z = data.shape
    data_ = data.reshape([Y, X, nmod, nacc, nwav], order='F')
    imod_ = np.swapaxes(imod, 0, 2)
    data_mod = np.zeros([Y, X, nmod, nwav])
    for w in range(nwav):
        accs = {}
        for m in range(nmod):
            accs[m] = []
        for a in range(nacc):
            temp = imod_[:,a,w]
            if (set(temp)==set(np.arange(nmod))):
                for i, m in enumerate(temp):
                    accs[m].append(data_[:,:,i,a,w])
        for m in range(nmod):
            data_mod[:,:,m,w] = np.mean(np.array(accs[m]), axis=0)
    return data_mod


def compute_line_center(data, lcont, rcont):
    """
    Input:  3d data cube
            number of pixels to be cropped
    Output: poly2 fits for 
            line center
    """
    Z, Y, X = data.shape
    y = np.float64(data[lcont:Z-rcont])
    x = np.arange(lcont,Z-rcont)[:,np.newaxis,np.newaxis]*np.ones([1,Y,X])
    #
    x4 = np.sum(x**4, axis=0)
    x3 = np.sum(x**3, axis=0)
    x2 = np.sum(x**2, axis=0)
    x1 = np.sum(x, axis=0)
    x0 = (Z-lcont-rcont)*np.ones([Y, X])
    y2 = np.sum(y*x**2, axis=0)
    y1 = np.sum(y*x, axis=0)
    y0 = np.sum(y, axis=0)
    #
    a, b, c = x4, x3, x2
    d, e, f = x3, x2, x1
    g, h, k = x2, x1, x0
    #
    A, B, C = e*k-f*h, f*g-d*k, d*h-e*g
    D, E, F = c*h-b*k, a*k-c*g, b*g-a*h
    G, H, K = b*f-c*e, c*d-a*f, a*e-b*d
    det = a*(e*k-f*h) + b*(f*g-d*k) + c*(d*h-e*g)
    A, B, C = A/det, B/det, C/det
    D, E, F = D/det, E/det, F/det
    G, H, K = G/det, H/det, K/det
    #
    deg2 = A*y2 + B*y1 + C*y0
    deg1 = D*y2 + E*y1 + F*y0
    deg0 = G*y2 + H*y1 + K*y0
    #
    cen = -deg1/deg2/2.0
    fit = deg2.reshape([1,Y,X])*x**2 + deg1.reshape([1,Y,X])*x + deg0.reshape([1,Y,X])
    return fit, cen


def shift_lines_4d(data, shifts):
    """
    Shift all the spectral lines in the 4d array by given shifts.
    Array is assumed to be Y*X*S*L where Y, X are spatial, S is Stokes and L is wavelength dimension
    Input:  4d array
            2d shifts array
    Output: shifted 4d array same size as input
    """    
    data_ = data*0.0
    for i in range(data.shape[2]):
        data_[:,:,i,:] = shift_3darray(data[:,:,i,:], shifts, axis=2)
    return data

def compute_offset_3d(data, plot=False):
    """
    Compute off-set values in Stokes peofiles of spectro-polarimetric data.
    Fourier-0 component is used for this.
    Input:  3d array with wavelength in last axis
            (opt.) plot the data
    Output: offset value; assuming it is the same throughout the frame
    """
    df = np.fft.fft(data, axis=2)
    df[:,:,1::] = 0
    data_ = np.fft.ifft(df, axis=2)
    offsets = np.real(np.average(data_, axis=2))
    if (plot):
        val, dist = np.histogram(offsets, 100)
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.plot(dist[1::], val)
        ax.set_xlabel('Crosstalk value')
        ax.set_ylabel('Number of samples')
        fig.tight_layout()
    return offsets

def correct_i2quv(data, plot=False):
    """
    Compute I->Q, I->U and I->V cross-talks
    Assumption is that they offset the profiles from zero
    Input:  4d data array
    Output: I->Q crosstalk
            I->U crosstalk
            I->V crosstalk
    """
    # 
    i2q = compute_offset_3d(data[:,:,1], plot=plot)
    i2q_ct = np.around(i2q.mean(), decimals=4)
    i2q_std = np.around(i2q.std(), decimals=4)
    print('I-->Q crosstalk distribution (mean, stdev):', i2q_ct, i2q_std)
    #
    i2u = compute_offset_3d(data[:,:,2], plot=plot)
    i2u_ct = np.around(i2u.mean(), decimals=4)
    i2u_std = np.around(i2u.std(), decimals=4)
    print('I-->U crosstalk distribution (mean, stdev):', i2u_ct, i2u_std)
    #
    i2v = compute_offset_3d(data[:,:,3], plot=plot)
    i2v_ct = np.around(i2v.mean(), decimals=4)
    i2v_std = np.around(i2v.std(), decimals=4)
    print('I-->V crosstalk distribution (mean, stdev):', i2v_ct, i2v_std)
    #
    data_corr = 1.0*data
    data_corr[:,:,1] -= i2q_ct
    data_corr[:,:,2] -= i2u_ct
    data_corr[:,:,3] -= i2v_ct
    return data_corr


def plot_stokes_figures(data_4d, npts, lcore=5, lwing=4):
    """
    Plot stokes images and profiles at selected points of the image
    Input:  4d spectro-polarimetric imaging data
            number of points to be plotted
            (opt.) line core position in pixels
            (opt.) line wing width in pixels
    Output: figure showing stokes images and profiles
    """
    plt.figure()
    plt.imshow(data_4d[:,:,0,0])
    pts = plt.ginput(npts)
    plt.close()
    #
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    circ_size=20
    #
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18,11))
    gs = axs[0,0].get_gridspec()
    for ax in axs[:,0]: ax.remove()
    for ax in axs[:,1]: ax.remove()
    axii = fig.add_subplot(gs[0:2,0])
    axqq = fig.add_subplot(gs[0:2,1])
    axuu = fig.add_subplot(gs[2::,0])
    axvv = fig.add_subplot(gs[2::,1])
    axi, axq, axu, axv = axs[:,-1]
    #
    axci = axii.imshow(data_4d[:,:,0,lcore], cmap='gray')
    axcq = axqq.imshow(np.sum(data_4d[:,:,1,lcore-lwing:lcore],axis=2), cmap='gray')
    axcu = axuu.imshow(np.sum(data_4d[:,:,2,lcore-lwing:lcore],axis=2), cmap='gray')
    axcv = axvv.imshow(np.sum(data_4d[:,:,3,lcore-lwing:lcore],axis=2), cmap='gray')
    #
    fig.colorbar(axci, ax=axii, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcq, ax=axqq, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcu, ax=axuu, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcv, ax=axvv, fraction=0.0475, pad=0.0125)

    #
    axi.set_ylabel('I (arbitrary units)')
    axq.set_ylabel('Q/I')
    axu.set_ylabel('U/I')
    axv.set_ylabel('V/I')
    axi.set_xlabel('Wavelength steps')
    axq.set_xlabel('Wavelength steps')
    axu.set_xlabel('Wavelength steps')
    axv.set_xlabel('Wavelength steps')
    #
    axii.set_title('Stokes-I map, line core')
    axqq.set_title('Stokes-Q map, added over blue wing')
    axuu.set_title('Stokes-U map, added over blue wing')
    axvv.set_title('Stokes-V map, added over blue wing')
    #
    # print(pts)
    for i, xy in enumerate(pts):
        x, y = xy
        axii.add_patch(plt.Circle((x,y),circ_size,fill=False,color=colors[i]))
        axqq.add_patch(plt.Circle((x,y),circ_size,fill=False,color=colors[i]))
        axuu.add_patch(plt.Circle((x,y),circ_size,fill=False,color=colors[i]))
        axvv.add_patch(plt.Circle((x,y),circ_size,fill=False,color=colors[i]))
        axi.plot(data_4d[int(y),int(x),0])
        axq.plot(data_4d[int(y),int(x),1])
        axu.plot(data_4d[int(y),int(x),2])
        axv.plot(data_4d[int(y),int(x),3])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.35)

def plot_stokes_figures_alt(data_4d, npts, pscale_im, pscale_wav, lcore=5, lwing=4):
    """
    Plot stokes images and profiles at selected points of the image
    Input:  4d spectro-polarimetric imaging data
            number of points to be plotted
            (opt.) line core position in pixels
            (opt.) line wing width in pixels
    Output: figure showing stokes images and profiles
    """
    plt.figure()
    plt.imshow(data_4d[:,:,0,0])
    pts = plt.ginput(npts)
    plt.close()
    #
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    circ_size=20*pscale_im
    ny, nx, nm, nw = data_4d.shape
    #
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(8,11))
    gs = axs[0,0].get_gridspec()
    for ax in axs[0:4,:]:
        for ax_ in ax:
            ax_.remove()
    axii = fig.add_subplot(gs[0:2,0])
    axqq = fig.add_subplot(gs[0:2,1])
    axuu = fig.add_subplot(gs[2:4,0])
    axvv = fig.add_subplot(gs[2:4,1])
    axi, axq, axu, axv = axs[4,0], axs[4,1], axs[5,0], axs[5,1]
    im_extent = [0,nx*pscale_im,0,ny*pscale_im]
    # print(im_extent)
    waves = (np.arange(nw)-5)*pscale_wav
    #
    axci = axii.imshow(data_4d[:,:,0,lcore], extent=im_extent, cmap='gray')
    axcq = axqq.imshow(np.mean(data_4d[:,:,1,lcore-lwing:lcore],axis=2), extent=im_extent, cmap='gray')
    axcu = axuu.imshow(np.mean(data_4d[:,:,2,lcore-lwing:lcore],axis=2), extent=im_extent, cmap='gray')
    axcv = axvv.imshow(np.mean (data_4d[:,:,3,lcore-lwing:lcore],axis=2), extent=im_extent, cmap='gray')
    #
    fig.colorbar(axci, ax=axii, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcq, ax=axqq, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcu, ax=axuu, fraction=0.0475, pad=0.0125)
    fig.colorbar(axcv, ax=axvv, fraction=0.0475, pad=0.0125)

    #
    axi.set_ylabel('I (arbitrary units)')
    axq.set_ylabel('Q/I')
    axu.set_ylabel('U/I')
    axv.set_ylabel('V/I')
    for ax in [axi, axq, axu, axv]:
        ax.set_xlabel('Wavelength from line core (in $\AA$)')
    #
    axii.set_title('Stokes-I map (core)')
    axqq.set_title('Stokes-Q map (wing)')
    axuu.set_title('Stokes-U map (wing)')
    axvv.set_title('Stokes-V map (wing)')
    #
    for ax in [axii, axqq, axuu, axvv]:
            ax.set_xlabel('arcsec')
            ax.set_ylabel('arcsec')
    #
    for i, xy in enumerate(pts):
        x, y = xy
        xa, ya = x*pscale_im, (ny-y)*pscale_im
        axii.add_patch(plt.Circle((xa,ya),circ_size,fill=False,color=colors[i]))
        axqq.add_patch(plt.Circle((xa,ya),circ_size,fill=False,color=colors[i]))
        axuu.add_patch(plt.Circle((xa,ya),circ_size,fill=False,color=colors[i]))
        axvv.add_patch(plt.Circle((xa,ya),circ_size,fill=False,color=colors[i]))
        axi.plot(waves, data_4d[int(y),int(x),0])
        axq.plot(waves, data_4d[int(y),int(x),1])
        axu.plot(waves, data_4d[int(y),int(x),2])
        axv.plot(waves, data_4d[int(y),int(x),3])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35, hspace=0.4)


