from functions import *

def compute_thresh(img, plot=False):
    """
    Compute appropriate image threshold by analyzing histogram of the image, assuming there are only 2 main brightness levels
    Input:  2d image array
    Output: intensity threshold number
    """
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
    if (plot):
        fig, ax = plt.subplots(111)
        ax.plot(x, result.best_fit, x, y)
    return thresh

def compute_angle(img):
    """ 
    Compute the angle of the lines in the image of a grid
    Input:  2d image array
    Output: angle [0,90)
    """
    ny, nx = img.shape
    nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
    td = 1.0*img[ny//4:ny//4+nyy, nx//4:nx//4+nxx]
    td = td/td.max()
    # compute threshold to generate binary image
    thresh = compute_thresh(img)
    td = np.array(td<thresh, dtype=np.uint8)
    plt.figure()
    plt.imshow(td)
    # detect lines using Hough transform
    lines = cv2.HoughLinesP(td, 1, np.pi/180, nxx//8, 50, nxx//2, 4)
    ls = 0.0*td
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot([x1,x2],[y1,y2],color='red')
    # compute slope of the lines
    angs = (lines[:,0,1]-lines[:,0,3])/(lines[:,0,0]-lines[:,0,2])
    angs = np.degrees(np.arctan(angs))
    angs[np.argwhere(angs<0)] += 90 
    ang = np.median(angs)
    return ang

def compute_magnification(img1, img2, minmag=0.98, maxmag=1.02):
    """ 
    Compute the magnification between two images, by brute force using maximizing correlation
    Input:  two 2d image arrays
    Output: magnification
    """
    # Select range of magnification
    zooms = np.linspace(minmag, maxmag, 101)
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
    # Find maxima of correlations
    mag = zooms[np.argmax(corrs).flatten()[0]]
    return mag


# def compute_shift(img1, img2):
#     """
#     Compute x and y shift between the given two images, using fft based correlation method
#     Input:  image-1
#             image-2
#     Output: (y,x) shift array
#     """
#     ny, nx = img1.shape
#     nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
#     im1 = img1[ny//4:3*ny//4,nx//4:3*nx//4]
#     im2 = img2[ny//4:3*ny//4,nx//4:3*nx//4]
#     im1, im2 = (im1-im1.mean())/im1.std(), (im2-im2.mean())/im2.std()
#     # Window
#     coeff = 0.54
#     x = coeff - (1-coeff)*np.cos(2*np.pi*np.arange(nxx)/(nxx-1))
#     y = coeff - (1-coeff)*np.cos(2*np.pi*np.arange(nyy)/(nyy-1))
#     w = np.reshape(x,[1,nxx])*np.reshape(y,[nyy,1])
#     # Images
#     im1, im2 = w*im1, w*im2
#     fim1, fim2 = np.fft.fft2(im1), np.fft.fft2(im2)
#     corr = np.fft.ifftshift(np.abs(np.fft.ifft2(fim1*np.conj(fim2))))
#     half = np.array([nyy//2, nxx//2])
#     sh = np.argwhere(corr==corr.max()).flatten()[0:2]-half
#     return sh


def compute_angle1(img):
    """ 
    |   Compute the angle of the lines in the image of a grid
    |   Input : 2d image array
    |   Output : angle [0,90)
    """
    td = np.uint8(255*(img-img.min())/(img.max()-img.min()))
    # ny, nx = img.shape
    # nxx, nyy = 3*nx//4-nx//4, 3*ny//4-ny//4
    # td = 1.0*td[ny//4:ny//4+nyy, nx//4:nx//4+nxx]
    # td_plane = fit_2d_plane(td, w=td)
    # td -= td_plane
    # compute threshold to generate binary image
    # plt.figure(); plt.imshow(td)
    td = sobel(td)
    # plt.figure(); plt.imshow(td)
    thresh = 10
    td = np.array(td>thresh, dtype=np.uint8)
    plt.figure(); plt.imshow(td)
    # detect lines using Hough transform
    lines = cv2.HoughLinesP(td, rho=1, theta=np.pi/180, threshold=nxx//4, lines=50, minLineLength=nxx//1.5, maxLineGap=8)
    ls = 0.0*td
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot([x1,x2],[y1,y2],color='red')
    # compute slope of the lines
    angs = (lines[:,0,1]-lines[:,0,3])/(lines[:,0,0]-lines[:,0,2])
    angs = np.degrees(np.arctan(angs))
    angs[np.argwhere(angs<0)] += 90 
    ang = np.median(angs)
    # remove extremes
    try:
        extremes = np.argwhere(np.abs(angs-ang)>3).flatten()
        angs = list(angs)
        del angs[extremes]
        ang = np.median(np.array(angs))
    except: pass
    return ang

def fit_2d_plane(z, w=1):
    """
    |   Fit the 2d data with the equation of z = ax + by + c
    |   Input:  2d array
    |   Output: fitted 2d array
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