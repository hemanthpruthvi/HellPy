from functions import *

def compute_input_stokes2(npos, offset=0.0, wpret=90, rotang=0.0, range=180.0):
    """
    |   Compute array input Stokes parameters generated from a combination of rotating polarizer and retarder
    |   Input : number of stokes states, angle offset for the system, retardance, angle offset between pol and ret, range of rotation
    |   Output : array of angles(1xN), array of Stokes parameters (4xN)
    """
    ang_start, ang_stop = 0.0, range
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
    #
    rotang = np.radians(rotang)
    qr = q*np.cos(2*rotang) - u*np.sin(2*rotang)
    ur = q*np.sin(2*rotang) + u*np.cos(2*rotang)
    in_stokes = np.array([i,qr,ur,v])
    return np.degrees(thetas), in_stokes

def compute_pcu_generated_stokes(pnpos, rnpos, pa_pcusys=0.0, pa_reloff=0.0, wpret=90, span=180.0):
    """
    |   Compute stokes parameters of the beam passed through the polarimetric calibration unit (pcu)
    |   Input:  number of polarizer positions
    |           number of retarder positions
    |           (opt.) position angle for the polariemtric calibration unit
    |           (opt.) position angle offset between polarizer and retarder
    |           (opt.) retardance of the waveplate
    |           (opt.) rotation span for polarizer and retarder
    |   Output: Nx4 array of stokes parameters
    """
    pang_offset = 0.0 + pa_pcusys
    rang_offset = pa_pcusys + pa_reloff
    pang_start, pang_stop = 0.0, span
    pangs = np.linspace(pang_start, pang_stop, pnpos) - pang_offset
    rang_start, rang_stop = 0.0, span
    rangs = np.linspace(rang_start, rang_stop, rnpos) - rang_offset
    stokes_in = np.matrix([[1],[0],[0],[0]])
    stokes_out = []
    for p in pangs:
        for r in rangs:
            mm = mueller_matrix_retarder(wpret, r)*mueller_matrix_polarizer(p)
            stokes_out.append(mm[:,0])
    angs = np.meshgrid(pangs,rangs)
    angs[0] = angs[0].T.flatten()
    angs[1] = angs[1].T.flatten()
    stokes_out = np.array(stokes_out).reshape([pnpos*rnpos, 4])
    stokes_out = stokes_out/stokes_out[:,0,np.newaxis]
    return stokes_out.T

def compute_modulation_matrix(in_stokes, mod_int):
    """
    |   Compute modulation matrix
    |   Input : array of input Stokes parameters (Nx4), array of modulation intensities (MxN)
    |   Output : modulation matrix (Mx4)
    """
    nmod = mod_int.shape[0]
    in_ = np.matrix(in_stokes)
    in_inv = np.transpose(in_)*np.linalg.inv(in_*np.transpose(in_))
    mod_mat = np.array(np.matrix(mod_int)*in_inv)
    mod_mat = mod_mat/mod_mat[:,0:1]
    return mod_mat

def compute_modulation_efficiency(demodmat):
    """
    |   Compute the so-called modulation efficiency
    |   Input : demodulation matrix (pseudo-inverse of modulation matrix) (4xM)
    |   Output : modulation efficiency vector (1x4)
    """
    modeff = np.sqrt(demodmat.shape[1]*np.sum((np.array(demodmat))**2, axis=1))
    modeff = 1.0/modeff
    return modeff

def get_modulation_intens(in_stokes, mod_mat):
    """
    |   Compute the modulated intensity of the polarimeter
    |   Input:  stokes parameters of the input beam
    |           modulation matrix
    |   Output: modulated intensity
    """
    in_stokes_ = np.matrix(in_stokes)
    mod_mat_ = np.matrix(mod_mat)
    mod_intens_ = mod_mat_*in_stokes_
    mod_intens = np.array(mod_intens_)
    return mod_intens

def ideal_mod_matrix(axis, mode='balanced'):
    """
    |   Generate modulation matrices for various schemes of dual-beam polarimetry
    |   Input:  analyzer axis 1 or -1
    |           (opt.) mode "balanced" or "definition"
    |   Output: nx4 modulation matrix
    """
    mod_matrix = []
    if (mode=='balanced'):
        w = 1.0/np.sqrt(3)
        if (axis==1):
            mod_matrix = np.array([[1, -w, -w, -w],[1, -w, w, w],[1, w, -w, w],[1, w, w, -w]])
        elif (axis==-1):
            mod_matrix = np.array([[1, w, w, w],[1, w, -w, -w],[1, -w, w, -w],[1, -w, -w, w]])
        else:
            print('Invalid input!')
    elif (mode=='definition'):
        if (axis==1 or axis==-1):
            mod_matrix = np.array([[1,1,0,0],[1,-1,0,0],[1,0,1,0],[1,0,-1,0],[1,0,0,1],[1,0,0,-1]])
        else:
            print('Invalid input!')
    else:
        print('Invalid mode!')
    return mod_matrix


def compute_zerofreq_3d(data):
    """
    |   Compute zero frequencies of 3d data consisting of 1d arrays
    |   Input : 3d array (series axis is assumed to be 2)
    |   Output : 2d array
    """
    df = np.fft.fft(data, axis=2)
    amp_zero = 0.5*df[:,:,0]/data.shape[2]
    return np.abs(amp_zero)

def compute_zerofreq_1d(data):
    """
    |   Compute zero frequency of 1d series
    |   Input : 1d array
    |   Output : scalar
    """
    df = np.fft.fft(data)
    amp_zero = 0.5*df[0]/len(data)
    return np.abs(amp_zero)


def compute_modmat_residual(params, int_mod, pnpos, rnpos, weights, del_dat=[]):
    """
    |   Compute residual Stokes afer reconstructing input Stokes with computed modulation matrix
    |   Input : parameters to compute input Stokes (lmfit datatype), modulated intensities (MxN), weights, list of bad data (indices)
    |   Output : residual Stokes (1x4N)
    """
    # npos = int_mod.shape[1]
    pa_pcusys = params['pa_pcusys'].value
    pa_reloff = params['pa_reloff'].value
    wpret = params['wpret'].value
    span = params['span'].value
    s_in = compute_pcu_generated_stokes(pnpos, rnpos, pa_pcusys=pa_pcusys, pa_reloff=pa_reloff, wpret=wpret, span=span)
    s_in_ = np.delete(s_in, del_dat, 1)
    int_mod_ = np.delete(int_mod, del_dat, 1)
    weights_ = np.delete(weights, del_dat, 1)
    modmat = compute_modulation_matrix(s_in_, int_mod_)
    demodmat = np.linalg.inv(modmat)
    int_meas = np.array(np.matrix(demodmat)*np.matrix(int_mod_))
    int_meas[1::,:] /= int_meas[0:1,:]
    int_meas[0,:] /= int_meas[0:1,:].mean()
    int_meas += -compute_zerofreq_3d(int_meas[:,np.newaxis,:]) + -compute_zerofreq_3d(s_in_[:,np.newaxis,:])
    resid = (int_meas-s_in_)*weights_
    return resid.ravel()

# def compute_modmat_residual(params, int_mod, weights, del_dat=[]):
#     """
#     |   Compute residual Stokes afer reconstructing input Stokes with computed modulation matrix
#     |   Input : parameters to compute input Stokes (lmfit datatype), modulated intensities (MxN), weights, list of bad data (indices)
#     |   Output : residual Stokes (1x4N)
#     """
#     npos = int_mod.shape[1]
#     offset = params['offset']
#     wpret = params['wpret']
#     rotang = params['rotang']
#     range = params['range']
#     # offset, wpret, rotang, range = params
#     thetas, s_in = compute_input_stokes(npos, offset=offset, wpret=wpret, rotang=rotang, range=range)
#     s_in_ = np.delete(s_in, del_dat, 1)
#     int_mod_ = np.delete(int_mod, del_dat, 1)
#     weights_ = np.delete(weights, del_dat, 1)
#     modmat = compute_modulation_matrix(s_in_, int_mod_)
#     demodmat = np.linalg.inv(modmat)
#     int_meas = np.array(np.matrix(demodmat)*np.matrix(int_mod_))
#     int_meas[1::,:] /= int_meas[0:1,:]
#     int_meas[0,:] /= int_meas[0:1,:].mean()
#     int_meas += -compute_zerofreq_3d(int_meas[:,np.newaxis,:]) + -compute_zerofreq_3d(s_in_[:,np.newaxis,:])
#     resid = (int_meas-s_in_)*weights_
#     return resid.ravel()

def compute_residual_twobeam(params, int_mod,  pnpos, rnpos, weights, del_dat=[]):
    """
    |   Compute residual Stokes afer reconstructing input Stokes with computed modulation matrix
    |   Input : parameters to compute input Stokes (lmfit datatype), modulated intensities from two beams(2, MxN), weights, list of bad data (indices)
    |   Output : residual Stokes (1x8N)
    """
    beam1, beam2 = int_mod
    resid1 = compute_modmat_residual(params, beam1,  pnpos, rnpos, weights, del_dat=del_dat)
    resid2 = compute_modmat_residual(params, beam2,  pnpos, rnpos, weights, del_dat=del_dat)
    resid = np.array([resid1, resid2]).ravel()
    return resid.ravel()

def coadd_modulated_intens(intens, imod, nmod, nacc, nwav):
    """
    |   Prepare sorted modulated intensities
    |   Input:  intensities array
    |           index extracted from time stramps
    |           number of modulation states
    |           number of accumulations
    |           number of wavelength points
    |   Output: sorted and co-added intensity array
    """
    intens_ = intens.reshape([nwav, nacc, nmod])
    imod_ = imod.reshape([nwav, nacc, nmod])%nmod
    modint = np.zeros([nmod,nwav])
    for w in range(nwav):
        accs = {}
        for m in range(nmod): accs[m] = []
        for a in range(nacc):
            temp = imod_[w,a,:]
            if (set(temp)==set(np.arange(nmod))):
                for i, m in enumerate(temp): accs[m].append(intens_[w,a,i])
        for m in range(nmod): modint[m,w] = np.median(np.array(accs[m]))
    return modint 


def mueller_matrix_rotation(theta):
    """
    |   Mueller matrix corresponding to the rotation
    |   Input:  angle of rotation in degrees
    |   Output: 4x4 Mueller matrix  
    """
    th = 2*np.radians(theta)
    mm = np.matrix([[1, 0, 0, 0],
                    [0, np.cos(th), np.sin(th), 0],
                    [0, -np.sin(th), np.cos(th), 0],
                    [0, 0, 0, 1]])
    return mm

def mueller_matrix_retarder(delta, theta=0.0):
    """
    |   Mueller matrix for the retarder
    |   Input:  angle of retardance in degrees
    |          (opt.) position angle of the retarder in degrees
    |   Output: 4x4 Mueller matrix  
    """
    de = np.radians(delta)
    mm = np.matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.cos(de), np.sin(de)],
                    [0, 0, -np.sin(de), np.cos(de)]])
    mm = mueller_matrix_rotation(-theta)*mm*mueller_matrix_rotation(theta)
    return mm

def mueller_matrix_polarizer(theta=0.0):
    """
    |   Mueller matrix corresponding to the polarizer
    |   Input:  (opt.) position angle of the polarizer in degrees
    |   Output: 4x4 Mueller matrix  
    """
    mm = 0.5*np.matrix([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    mm = mueller_matrix_rotation(-theta)*mm*mueller_matrix_rotation(theta)
    return mm

def compute_input_stokes(npol, nret, polang, retang, wpret):
    """
    |   Compute stokes parameters of the beam passed through the polarimetric calibration unit (pcu)
    |   Input:  number of polarizer positions
    |           number of retarder positions
    |           (opt.) position angle for the polariemtric calibration unit
    |           (opt.) position angle offset between polarizer and retarder
    |           (opt.) retardance of the waveplate
    |           (opt.) rotation span for polarizer and retarder
    |   Output: Nx4 array of stokes parameters
    """
    pangs = np.linspace(0,180,npol) + polang
    rangs = np.linspace(0,180,nret) + retang
    pa, ra = np.meshgrid(pangs, rangs)
    pangs, rangs = pa.T.flatten(), ra.T.flatten()
    stokes_unp = np.matrix([1,0,0,0]).T
    stokes_ = []
    for p, r in zip(pangs, rangs):
        stokes_temp = mueller_matrix_retarder(wpret, r)*mueller_matrix_polarizer(p)*stokes_unp
        stokes_.append(stokes_temp)
    stokes_ = np.array(stokes_)[:,:,0].T
    stokes_ = np.matrix(stokes_/stokes_[0:1,:])
    return stokes_

def fit_beam_mods(xdata, polang, retang, wpret, intens1, intens2):
    beam_ = np.reshape(xdata, [8,len(xdata)//8])
    beam1 = beam_[0:4,:]
    beam2 = beam_[4::,:]
    #
    npol, nret = 1, 9
    s_in = compute_input_stokes(npol, nret, polang, retang, wpret)
    #
    mod_mat1 = np.matrix(beam1)*np.linalg.pinv(s_in)
    mod_mat1 = np.matrix(np.array(mod_mat1)/np.array(mod_mat1[:,0:1]))
    mod_int1 = intens1*mod_mat1*np.matrix(s_in)
    #
    mod_mat2 = np.matrix(beam2)*np.linalg.pinv(s_in)
    mod_mat2 = np.matrix(np.array(mod_mat2)/np.array(mod_mat2[:,0:1]))
    mod_int2 = intens2*mod_mat2*np.matrix(s_in)
    #
    mod_int = np.concatenate([mod_int1, mod_int2], axis=0).flatten()
    return mod_int