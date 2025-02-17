from functions import *

def fit_gauss_poly2(spectral_line, line_width=3.0, gain_factor=1.0, plot=False):
    """
    Compute best fit for the line+continuum in the form of f(x)*[1-g(x)] where
    x is the pixel number starting at 0
    f(x) is quadratic (3 params)
    g(x) is Gaussian (3 params)
    Input:  1d-array spectral line
            (opt.) estimated width of the line
            (opt.) gain factor to estimate weights 
            (opt.) plot the results
    Output: list of inital guess parameters
            list of best fit parameters
    """
    xdata = np.arange(len(spectral_line))
    ydata = np.float64(spectral_line)
    ydata_error = np.sqrt(spectral_line.mean()/gain_factor)
    #
    deg0_guess = ydata[0]
    deg1_guess = (ydata[0]-ydata[-1])/(xdata[0]-xdata[-1])
    deg2_guess = 0
    amp_guess = 1-ydata.min()/deg0_guess
    cen_guess = xdata[np.argwhere(ydata==ydata.min())].flatten()[0]
    sig_guess = line_width/2.355
    #
    guess_params = Parameters()
    guess_params.add(name='amp', value=amp_guess, max=1)
    guess_params.add(name='cen', value=cen_guess, min=xdata.min(), max=xdata.max())
    guess_params.add(name='sig', value=sig_guess, min=0, max=xdata.max())
    guess_params.add(name='deg2', value=deg2_guess)
    guess_params.add(name='deg1', value=deg1_guess)
    guess_params.add(name='deg0', value=deg0_guess)
    #
    model = Model(gauss_poly2)
    result = model.fit(ydata, guess_params, xdata=xdata)
    if(plot):
        fig, ax = plt.subplots(1,1)
        ax.plot(ydata, 'k--')
        ax.plot(result.init_fit, 'k:')
        ax.plot(result.best_fit, 'k-')
        ax.legend(['Data', 'Initial', 'Fitted'])
    return result
    # fitter = Minimizer(residual_gauss_poly2, 
    #                     guess_params, 
    #                     fcn_args=(xdata, ydata), 
    #                     fcn_kws={'sigma': ydata_error}, 
    #                     scale_covar=True)
    # fit_params = fitter.leastsq().params
    # #
    # deg2_fit = fit_params['deg2'].value
    # deg1_fit = fit_params['deg1'].value
    # deg0_fit = fit_params['deg0'].value
    # amp_fit = fit_params['amp'].value
    # cen_fit = fit_params['cen'].value
    # sig_fit = fit_params['sig'].value
    # #
    # guess_params = [deg2_guess, deg1_guess, deg0_guess, amp_guess, cen_guess, sig_guess]
    # fit_params = [deg2_fit, deg1_fit, deg0_fit, amp_fit, cen_fit, sig_fit]
    # return guess_params, fit_params

def fit_gauss_poly2_parallel(spectral_line, indy, indx):
    """
    Compute best fit for the line+continuum in the form of f(x)*[1-g(x)] where
    x is the pixel number starting at 0
    f(x) is quadratic (3 params)
    g(x) is Gaussian (3 params)
    Input:  1d-array spectral line
            y index of array
            x index of array 
    Output: list of inital guess parameters
            list of best fit parameters
    """
    line_width=3.0
    gain_factor=1.0
    xdata = np.arange(len(spectral_line))
    ydata = np.float64(spectral_line)
    ydata_error = np.sqrt(spectral_line.mean()/gain_factor)
    #
    deg0_guess = ydata[0]
    deg1_guess = (ydata[0]-ydata[-1])/(xdata[0]-xdata[-1])
    deg2_guess = 0
    amp_guess = 1-ydata.min()/deg0_guess
    cen_guess = xdata[np.argwhere(ydata==ydata.min())].flatten()[0]
    sig_guess = line_width/2.355
    #
    guess_params = Parameters()
    guess_params.add(name='amp', value=amp_guess, max=1)
    guess_params.add(name='cen', value=cen_guess, min=xdata.min(), max=xdata.max())
    guess_params.add(name='sig', value=sig_guess, min=0, max=xdata.max())
    guess_params.add(name='deg2', value=deg2_guess)
    guess_params.add(name='deg1', value=deg1_guess)
    guess_params.add(name='deg0', value=deg0_guess)
    #
    model = Model(gauss_poly2)
    result = model.fit(ydata, guess_params, xdata=xdata)
    # print(indy, indx)
    return [indy, indx, result]

def residual_gauss_poly2(params, xdata, ydata, sigma=1.0):
    """
    Compute residula between y-data and fit data, for the Minimizer of the lmfit
    Input:  parameters for the fitting model
            1d x-data array
            1d y-data array
            (opt.) error corresponding to y data to compute weights
    Output: residual corresponding to the fit
    """
    yGauss = params['amp']*np.exp(-0.5*(xdata-params['cen'])**2/params['sig']**2)
    yPoly2 = params['deg2']*xdata**2 + params['deg1']*xdata + params['deg0']
    model = yPoly2*(1 - yGauss)
    return (model - ydata) / sigma

def model_gauss_poly2(params, xdata):
    """
    Compute the model spectral line(Gaussian) + continuum(quadratic)
    Input:  Parameters() object from lmfit
            1d x-data array
    Output: computed y-data corresponding to the parameters
    """
    # amp, cen, sig = params['amp'], params['cen'], params['sig']
    # deg2, deg1, deg0 = params['deg2'], params['deg1'], params['deg0']
    # yGauss = amp*np.exp(-0.5*(xdata-cen)**2/sig**2)
    # yPoly2 = deg2*xdata**2 + deg1*xdata + deg0
    yGauss = params['amp']*np.exp(-0.5*(xdata-params['cen'])**2/params['sig']**2)
    yPoly2 = params['deg2']*xdata**2 + params['deg1']*xdata + params['deg0']
    return yPoly2*(1 - yGauss)

def gauss_poly2(xdata, amp, cen, sig, deg0, deg1, deg2):
    """
    Compute the spectral line(Gaussian) + continuum(Lorentzian) 
    Input:  list of parametres
            x-data array
    Output: spectral line    
    """
    # a, b, c, amp, cen, sig = params
    return (deg2*xdata**2 + deg1*xdata + deg0)*(1 - amp*np.exp(-0.5*(xdata-cen)**2/sig**2))

def fit_gauss_lorentz(spectral_line, line_width=5.0, gain_factor=1.0, ncav=2, plot=False):
    """
    Computes the a fit for the spectrum in the form of f(x)*[1-g(x)] where
    x is the pixel number starting at 0
    f(x) is Lorentzian
    g(x) is Gaussian
    Input:  1d-array spectral line, 
            approximate FWHM of the line, 
            detector gain, and 
            plot (YES/NO)
    Output: list of inital guess parameters,
            list of final fit parameters
    """
    xdata = np.arange(len(spectral_line))
    ydata = np.float64(spectral_line)
    ydata_error = np.sqrt(spectral_line.mean()/gain_factor)
    #
    amp_guess = ydata.min()/ydata[0]
    cen_guess = xdata[np.argwhere(ydata==ydata.min())].flatten()[0]
    sig_guess = line_width/2.355
    wcen_guess = 1.0*cen_guess
    fwhm_guess = 50*line_width
    tran_guess = ydata.max()
    #
    guess_params = Parameters()
    guess_params.add(name='amp', value=amp_guess, max=1)
    guess_params.add(name='cen', value=cen_guess, min=xdata.min(), max=xdata.max())
    guess_params.add(name='sig', value=sig_guess, min=0, max=xdata.max())
    guess_params.add(name='tran', value=tran_guess, max=2*ydata.max())
    guess_params.add(name='wcen', value=wcen_guess, min=xdata.min(), max=xdata.max())
    guess_params.add(name='fwhm', value=fwhm_guess, min=0, max=1000*line_width)
    #
    fitter = Minimizer(residual_gauss_lorentz, 
                        guess_params, 
                        fcn_args=(xdata, ydata), 
                        fcn_kws={'sigma': ydata_error, 'ncav': ncav}, 
                        scale_covar=True)
    fit_params = fitter.leastsq().params
    #
    tran_fit = fit_params['tran'].value
    wcen_fit = fit_params['wcen'].value
    fwhm_fit = fit_params['fwhm'].value
    amp_fit = fit_params['amp'].value
    cen_fit = fit_params['cen'].value
    sig_fit = fit_params['sig'].value
    #
    if(plot):
        guess_fit = model_gauss_lorentz(guess_params, xdata)
        final_fit = model_gauss_lorentz(fit_params, xdata)
        fig, ax = plt.subplots(1,1)
        ax.plot(ydata, 'k--')
        ax.plot(guess_fit, 'k:')
        ax.plot(final_fit, 'k-')
        ax.legend(['Data', 'Guess', 'Fitted'])
    guess_params = [tran_guess, wcen_guess, fwhm_guess, amp_guess, cen_guess, sig_guess]
    fit_params = [tran_fit, wcen_fit, fwhm_fit, amp_fit, cen_fit, sig_fit]
    return guess_params, fit_params

def fit_gauss_lorentz_parallel(spectral_line, indy, indx):
    """
    Computes the a fit for the spectrum in the form of f(x)*[1-g(x)] where
    x is the pixel number starting at 0
    f(x) is Lorentzian
    g(x) is Gaussian
    Input:  1d-array spectral line, 
            approximate FWHM of the line, 
            detector gain, and 
            plot (YES/NO)
    Output: list of inital guess parameters,
            list of final fit parameters
    """
    line_width=3.0
    gain_factor=1.0
    ncav=2
    xdata = np.arange(len(spectral_line))
    ydata = np.float64(spectral_line)
    ydata_error = np.sqrt(spectral_line.mean()/gain_factor)
    #
    amp_guess = 1-ydata.min()/ydata[0]
    cen_guess = xdata[np.argwhere(ydata==ydata.min())].flatten()[0]
    sig_guess = line_width/2.355
    wcen_guess = 1.0*cen_guess
    fwhm_guess = 50*line_width
    tran_guess = ydata.max()
    #
    guess_params = Parameters()
    guess_params.add(name='amp', value=amp_guess, max=1)
    guess_params.add(name='cen', value=cen_guess, min=xdata.min(), max=xdata.max())
    guess_params.add(name='sig', value=sig_guess, min=0, max=xdata.max())
    guess_params.add(name='tran', value=tran_guess, max=2*ydata.max())
    guess_params.add(name='wcen', value=wcen_guess, min=-3*xdata.max(), max=4*xdata.max())
    guess_params.add(name='fwhm', value=fwhm_guess, min=0, max=1000*line_width)
    guess_params.add(name='ncav', value=2, vary=False)
    #
    model = Model(gauss_lorentz)
    result = model.fit(ydata, guess_params, xdata=xdata)
    return [indy, indx, result]

def residual_gauss_lorentz(params, xdata, ydata, sigma=1.0, ncav=2):
    """
    Compute residula between y-data and fit data, for the Minimizer of the lmfit
    Input : parameters for the fitting model,
            x - data,
            y - data, and 
            error corresponding to y data.
    Output: residual corresponding to this fit. 
    """
    yGauss = params['amp']*np.exp(-0.5*(xdata-params['cen'])**2/params['sig']**2)
    yPrefilt = params['tran']/(1+(2*(xdata-params['wcen'])/params['fwhm'])**(2*ncav))
    model = yPrefilt*(1 - yGauss)
    return (model - ydata) / sigma

def model_gauss_lorentz(params, xdata, ncav=2):
    """
    Compute the model spectral line(Gaussian) + continuum(Lorentzian)
    Input:  Parameters() object from lmfit
            1d x-data array
            (opt.) number of cavities of filter (for Lorentzian)
    Output: y-data corresponding to the parameters
    """
    yGauss = params['amp']*np.exp(-0.5*(xdata-params['cen'])**2/params['sig']**2)
    yLorentz = params['tran']/(1+(2*(xdata-params['wcen'])/params['fwhm'])**(2*ncav))
    return yLorentz*(1 - yGauss)

def gauss_lorentz(xdata, tran, wcen, fwhm, amp, cen, sig, ncav=2):
    """
    Compute the spectral line(Gaussian) + continuum(Lorentzian)
    Input:  list of 6 parameters
            1d x-data array
            (opt.) number of cavities of filter (for Lorentzian)
    Output: y-data corresponding to the parameters
    """
    # tran, wcen, fwhm, amp, cen, sig = params
    yGauss = (1 - amp*np.exp(-0.5*(xdata-cen)**2/sig**2))
    yLorentz =  tran/(1+(2*(xdata-wcen)/fwhm)**(2*ncav))
    return yGauss*yLorentz

def get_mean_flat(file_names):
    """
    Input : list of names of flat files to be averaged.
    Returns : 3d array containing average flat.
    All selected 3d arrays of flats are averaged.
    """
    nfiles = len(file_names) 
    print('Reading flat files... ')
    for i in tqdm.tqdm(range(nfiles)):
        f = file_names[i]
        # print(f)
        if (i==0):
            data = np.float64(pf.open(f)[0].data)
        else :
            data += pf.open(f)[0].data
    return data/nfiles

def compute_fit_for_row(row, row_index):
    """
    This function is created to be used in parallelized spectra fitting
    Input:  2d array of spectra corresponding to a row of the image, and
            index of the row
    Output: 2d array of fit parameters for the row, and
            index of the row
    """
    Z, X = row.shape
    fit_params = np.zeros([6, X])
    guess_params = np.zeros([6, X])
    print('Fitting spectral lines... ')
    for x in range(X):
        guess_params[:,x], fit_params[:,x] = fit_gauss_poly2(row[:,x])
    return [fit_params, row_index]

def compute_fit_parallel(data):
    """
    Compute the spectra fittings using multiple cores
    Input:  3d data cube containing spectra to be fitted
    Output: 3d data cube containing fitted parameters for the data cube
    """
    Z, Y, X = data.shape
    fit_params = np.zeros([6, Y, X])
    #
    pool = mp.Pool(mp.cpu_count())
    fit_results = [pool.apply(compute_fit_for_row, args=(data[:,row_index,:], row_index)) 
                        for row_index in tqdm.tqdm(range(Y))]
    pool.close()
    #
    for item in fit_results:
        fit_params[:,item[1],:] = item[0]
    return fit_params

def compute_shifted_3darray(data, shifts):
    """
    Shift all spectra by given amount
    Input:  3d data cube containing spectra
            2d array contaning shifts to be applied to each spectrum
    Output: 3d data cube containing shiftted spectra
    """
    correct_data = 0.0*data
    Z, Y, X = data.shape
    for y in tqdm.tqdm(range(Y)):
        for x in range(X):
            correct_data[:,y,x] = shift(data[:,y,x], shifts[y,x], mode='nearest')
    return correct_data


def shift_3darray(data, shifts, axis=2):
    """
    Shift all the spectral lines in a 3d array by given amount
    Input:  3d data cube containing spectra, and
            2d array contaning shifts to be applied to each spectrum.
    Output: 3d data cube containing shifted spectra.
    """
    correct_data = 0.0*data
    if (axis==2):
        Y, X, Z = data.shape
        for y in tqdm.tqdm(range(Y)):
            for x in range(X):
                correct_data[y,x,:] = shift(data[y,x,:], shifts[y,x], mode='nearest')
    elif (axis==0):
        Z, Y, X = data.shape
        for y in tqdm.tqdm(range(Y)):
            for x in range(X):
                correct_data[:,y,x] = shift(data[:,y,x], shifts[y,x], mode='nearest')
    return correct_data

def fit_lines_3d(data, pf='quadratic'):
    """
    Fit 3d cube of spectral lines (spatial axis=0,1; wavelength axis=2)
    Fit function is Poly2*(1-Gaussian) with 6 parameters (3 for Poly2, 3 for gaussian)
    Input:  3d data cube array (y,x,w)
            assumed prefilter profile ('quadratic' or 'lorentzian')
    Output: initial guess parameters(y,x,6)
            best fit parameters(y,x,6)
    """
    Y, X, Z = data.shape
    fit_params = np.zeros([Y, X, 6])
    guess_params = np.zeros([Y, X, 6])
    for x in tqdm.tqdm(range(X)):
        for y in range(Y):
            if (pf=='quadratic'):
                result = fit_gauss_poly2(data[y,x,:])
            elif (pf == 'lorentzian'):
                result = fit_gauss_lorentz(data[y,x,:])
            else:
                print('Incorrect mode for fitting!')
                return 
            guess_params[y,x,:] = np.array(list(result.init_values.values()))
            fit_params[y,x,:] = np.array(list(result.best_values.values()))
    return guess_params, fit_params

def fit_lines_3d_parallel(data, pf='poly2', nparallel=16):
    """
    Fit 3d cube of spectral lines (spatial axis=0,1; wavelength axis=2)
    Fit function is Poly2*(1-Gaussian) with 6 parameters (3 for Poly2, 3 for gaussian)
    Input:  3d data cube array (y,x,w)
            assumed prefilter profile ('poly2' or 'lorentzian')
    Output: initial guess parameters(y,x,6)
            best fit parameters(y,x,6)
    """
    Y, X, Z = data.shape
    fit_params = np.zeros([Y, X, 6])
    pool = mp.Pool(nparallel)
    mp_args = []
    for y in range(Y):
        for x in range(X):
            ydata = data[y,x]
            mp_args.append([ydata, y, x])
    #
    if (pf=='poly2'):
        results = pool.starmap(fit_gauss_poly2_parallel, mp_args)
    elif (pf == 'lorentzian'):
        results = pool.starmap(fit_gauss_lorentz_parallel, mp_args)
    else:
        print('Incorrect mode for fitting!')
    for r in results:
        y, x, params = r
        fit_params[y,x,:] = np.fromiter(params.best_values.values(), dtype=float)[0:6]
    return fit_params

def fit_lines_3d_parallel_poly2(data, nparallel=8):
    """
    Fit 3d cube of spectral lines (spatial axis=0,1; wavelength axis=2)
    Fit function is Poly2*(1-Gaussian) with 6 parameters (3 for Poly2, 3 for gaussian)
    Input:  3d data cube array (y,x,w)
            assumed prefilter profile ('poly2' or 'lorentzian')
    Output: initial guess parameters(y,x,6)
            best fit parameters(y,x,6)
    """
    Y, X, Z = data.shape
    fit_params = np.zeros([Y, X, 6])
    pool = mp.Pool(nparallel)
    mp_args = []
    for y in range(Y):
        for x in range(X):
            ydata = data[y,x]
            mp_args.append([ydata, y, x])
    results = pool.starmap(fit_gauss_poly2_parallel, mp_args)
    for r in results:
        y, x, params = r
        fit_params[y,x,:] = np.fromiter(params.best_values.values(), dtype=float)[0:6]
    pool.terminate()
    return fit_params

def compute_line_params_3d(fit_params, w):
    """
    Compute line properties: continuum and line shifts
    Input:  fit parameters (y,x,6)
            wavepoints index array (e.g., [0,1,2...,w-1]) 
    Output: line continuum (y,x,w)
            reltive line center shifts (y,x,1)
    """
    w = w.reshape([1,1,len(w)])
    deg2 = fit_params[:,:,0:1]
    deg1 = fit_params[:,:,1:2]
    deg0 = fit_params[:,:,2:3]
    continuum = deg2*w**2 + deg1*w + deg0
    # Compute line shifts from gaussian line profile
    line_shifts = gaussian_filter(fit_params[:,:,4], 2)
    line_shifts = line_shifts.min()-line_shifts
    return continuum, line_shifts

def compute_shifted_lines(line, line_shifts):
    """
    Compute 3d array comprising of 1d template spectra shifted by 2d array values
    Input:  template spectral line
            2d array of shifts
    Output: 3d array of shifted lines
    """
    line_ = line.ravel()
    Y, X = line_shifts.shape
    Z = len(line_)
    data = np.zeros([Y,X,Z])
    for x in tqdm.tqdm(range(X)):
        for y in range(Y):
            data[y,x,:] = shift(line_, line_shifts[y,x], mode='nearest')
    return data

def gaussian_absroption_line(x, amp, cen, sig):
    """
    Compute the absorption spectral line (Gaussian)
    Input:  list of 3 parameters (amplitude, center and std)
            1d x-data array
    Output: y-data corresponding to the parameters
    """
    return (1 - amp*np.exp(-0.5*(x-cen)**2/sig**2))

def poly2_continuum(x, deg2, deg1, deg0):
    """
    Compute the absorption spectral line (Gaussian)
    Input:  1d x-data array
            3 parameters a, b, c of second degree polynomial
    Output: y-data corresponding to the parameters
    """
    return (deg2*x**2 + deg1*x + deg0)

def coadd_accumulations(data, ind):
    """
    Prepare sorted modulated images/frames
    Input:  5d data cube
            3d index array extracted from time stamps
    Output: sorted and co-added images/frames
    """
    # Y, X, Z = data.shape
    # data_ = data.reshape([Y, X, nmod, nacc, nwav], order='F')
    Y, X, nmod, nacc, nwav = data.shape
    ind_ = np.swapaxes(ind, 0, 2)
    data_mod = np.zeros([Y, X, nmod, nwav])
    for w in range(nwav):
        accs = {}
        for m in range(nmod):
            accs[m] = []
        for a in range(nacc):
            temp = ind_[:,a,w]
            if (set(temp)==set(np.arange(nmod))):
                for i, m in enumerate(temp):
                    accs[m].append(data[:,:,i,a,w])
        for m in range(nmod):
            data_mod[:,:,m,w] = np.mean(np.array(accs[m]), axis=0)
    return data_mod

def coadd_del_accumulations(data, imod):
    """
    Coadd modulated images/frames after deleting the first frame after wavelength tuning
    (to account for the FPI stabilization)
    Input:  5d data cube
            3d index array extracted from time stamps
    Output: sorted and co-added images/frames
    """
    Y, X, nmod, nacc, nwav = data.shape
    data_ = data.reshape([Y, X, nmod*nacc, nwav], order='F')
    imod_ = imod.reshape([nwav, nacc*nmod])
    imod_ = np.swapaxes(imod_, 0, 1)
    imod_ = np.delete(imod_, [0], axis=0)
    data_ = np.delete(data_, [0], axis=2)
    data_mod = np.zeros([Y, X, nmod, nwav])
    for w in range(nwav):
        accs = {}
        for m in range(nmod):
            accs[m] = []
        #
        temp = imod_[:,w]
        for i, m in enumerate(temp):
            if (set(temp)==set(np.arange(nmod))):
                accs[m].append(data_[:,:,i,w])
        for m in range(nmod):
            data_mod[:,:,m,w] = np.mean(np.array(accs[m]), axis=0)
    return data_mod

def fit_et_blue_shifts(linesh):
    """
    Fit a 2d parabola to system blue shifts of the etalon transmission
    z = a((x-x0)²+(y-y²))+z0
    Input:  2d array of line shifts
    Output: 2d array of best fit parabola
            best fit parameters corresponding to the parabola
    """ 
    model = Model(parabolic_2d)
    Y, X = linesh.shape
    YY, XX = np.meshgrid(np.arange(Y), np.arange(X))
    params = model.make_params()
    params['a'].set(value=-2e-6, min=-1e-3, max=0)
    params['x0'].set(value=X//2, min=0, max=X) 
    params['y0'].set(value=Y//2, min=0, max=Y) 
    params['z0'].set(value=0)
    res = model.fit(linesh, params, x=[YY,XX])
    vals = np.array(list(res.best_values.values()))
    return res.best_fit, vals

def read_flat_fit(ffit_name):
    """
    Read the flat fit file and return individual datasets from fits
    Input:  Name of the fits file
    Output: Average flat
            Continnum fit for the average flat
            Line fit for the average flat
            Line shifts map
            Line broadening map
    """ 
    fits = pf.open(ffit_name)
    ff = fits[0].data
    ff_cont = fits[1].data
    ff_line = fits[2].data
    ff_linesh = fits[3].data
    ff_broad = fits[4].data
    return ff, ff_cont, ff_line, ff_linesh, ff_broad

def gaussians_row(x, amp=1, cen=0, sig=1):
    """
    Compute 2d array where each row is a Gaussian with given parameters
    Input:  x 1d array, same for all the rows
            (opt.) amplitude(s)
            (opt.) center(s)
            (opt.) sigma(s)
    Output: 2d array, each row is a 1d Gaussian 
    """ 
    return amp*np.exp(-0.5*(x[:,np.newaxis]-cen)**2/sig**2)

def voigt_row(x, amp=1, cen=0, sig=1, gam=1):
    """
    Compute 2d array where each row is a Voigt profile with given parameters
    Input:  x 1d array, same for all the rows
            (opt.) amplitude(s)
            (opt.) center(s)
            (opt.) sigma(s)
            (opt.) gamma(s)
    Output: 2d array, each row is a 1d Voigt profile 
    """ 
    z = np.sqrt(0.5)*(x[:,np.newaxis]-cen+1j*gam)/sig
    return amp*np.real(wofz(z))

def real_spectral_line(x, spectrum, broad, linesh, wavescan_range, wavelength):
    """
    Compute realistic spectral line by applying given broadening and sampling, to be used to fit the observed spectrum
    Input:  1d array of integers
            spectrum that is read using the function get_fts_spectra
            Instrumental broadening value
            Spectral shift
            Wavelength range
            Reference wavelength
    Output: Simulated observed spectrum
    """
    waves, intens = spectrum
    nwav = len(x)
    wavestep = wavescan_range/(nwav-1)
    waves_obs = (x-nwav//2)*wavestep+wavelength+linesh*wavestep
    waves_ = waves*1e10
    tune_profs = gaussians_row(waves_, cen=waves_obs, sig=broad*wavestep)
    # tune_profs = voigt_row(waves_, cen=waves_obs, sig=broad*wavestep, gam=broad2*wavestep)
    tune_profs /= np.sum(tune_profs, axis=0)
    intens_broad = intens[:,np.newaxis]*tune_profs
    intens_broad = np.sum(intens_broad, axis=0)
    return intens_broad