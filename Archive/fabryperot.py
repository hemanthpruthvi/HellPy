import numpy as np
import random as rd
from matplotlib import lines, pyplot as plt
from scipy.ndimage import zoom as zoom
from scipy.interpolate import UnivariateSpline
from IPython.display import Math
from tqdm import tqdm as progressbar
plt.rcParams.update({'font.size': 16})

class fp_etalon:
    """
    Fabry-Pérot etalon object: cavity enclosed by two parallel surfaces, inner faces with reflective coating
    """
    def __init__(self, thickness, material, coating):
        """
        Initialize a Fabry-Pérot etalon object: all spatial units are in meters
        Input:  thickness of the cavity
                name of the material file (refractive index vs wavelength data)
                name of coating file (reflectivity vs wavelength data)
        """
        self.thickness = thickness
        self.coating = np.loadtxt(coating, delimiter=',')
        self.material = np.loadtxt(material, delimiter=',')
        self.coating[:,0] *= 1e-9
        self.material[:,0] *= 1e-9
        self.coating[:,1] *= 1e-2
        # self.R = 0.95
        # self.A = 0.0
        # self.wavelength = 632.8e-9
        # self.compute_defects()
        # self.compute_params()
        self.err_curv = 4.2e-9      # curvature error, peak to valley
        self.err_rough = 1e-9       # micro roughness, rms error
        self.err_paral = 0.5e-9     # flatness error, peak to valley

    def compute_params(self, wavelength):
        """
        Compute various etalon parameters: finesse, free spectral range etc. 
        Compute finesse due to defects: curvature error, micro roughness and flatness error of the cavity
        Input:  wavelength
        """
        #
        self.df_curv = wavelength/(2*self.err_curv)
        self.df_rough = wavelength/(6.9*self.err_rough)
        self.df_paral = wavelength/(1.7*self.err_paral)
        self.defect_finesse = np.array([self.df_curv, self.df_rough, self.df_paral])
        #
        self.R = self.get_reflectivity(wavelength)
        self.RI = self.get_refractive_index(wavelength)
        self.order = int(2*self.RI*self.thickness/wavelength)
        self.reflect_finesse = 4*self.R/(1-self.R)**2
        self.finesse = (1/self.reflect_finesse**2 + np.sum(1/self.defect_finesse**2))**(-0.5)
        self.FSR = 0.5*wavelength**2/(self.RI*self.thickness)
        self.FWHM = self.FSR/self.finesse

    def get_reflectivity(self, wavelength):
        """
        Get reflectivity of the coating at given wavelength, from reflectivity data
        This data is read from a csv file, when the object is created
        Input:  wavelength
        Output: fractional reflectivity
        """
        temp_ind = np.argmin(np.abs(self.coating[:,0]-wavelength))
        R = self.coating[temp_ind,1]
        return R

    def get_refractive_index(self, wavelength):
        """
        Get refractive index of etalon at given wavelength, from material data
        This data is read from a csv file, when the object is created
        Input:  wavelength
        Output: refrative index
        """
        temp_ind = np.argmin(np.abs(self.material[:,0]-wavelength))
        RI = self.material[temp_ind,1]
        return RI

    def compute_trans_ideal(self, thickness, waves, inc=0):
        """
        Compute ideal transmission profile (transmission vs wavelength) of the etalon in the given wavelength range
        Input:  thickness of the etalon
                wavelengths at which transmission is to be computed (1d array)
                incident angle
        Output: transmission profile (1d array)
        """
        self.compute_params(waves[len(waves)//2])
        trans_profile_ideal = 1.0/(1.0+self.reflect_finesse*(np.sin(2*np.pi*self.RI*thickness*np.cos(inc)/waves))**2)
        return trans_profile_ideal

class fp_air_spaced_piezo(fp_etalon):
    """
    Air-spaced piezo-tunable FP etalon
    """
    def __init__(self, thickness, coating, tune_step=1e-9, tune_DAC=12):
        """
        Initialize the object with FP parameters and tuning parameters, except that material is loaded from "air.csv" file
        Input:  spacing between the plates
                name of coating file (reflectivity vs wavelength data)
                (opt.) tune step wavelength per count
                (opt.) controller DAC bit depth, decides electronic tuning range
        """
        super().__init__(thickness, 'air.csv', coating)
        self.tune_step = tune_step
        self.tune_DAC = tune_DAC
        self.tune_range = (2**self.tune_DAC)*self.tune_step
        self.tuned_counts = 0
        self.tuned_spacing = np.copy(self.thickness)

    def tune_to_wavelength(self, wavelength, waves):
        """
        Change the spacing in such a way that etalon transmission peak matches with the given wavelength
        Input:  wavelength at which maximum transmission is expected
                wavelength array (1d)
        Output: transmission profile (1d array)
        """
        self.waves = waves
        self.tuned_wavelength = wavelength
        self.compute_params(wavelength)
        self.tuned_spacing = self.order*self.tuned_wavelength/(2*self.RI)
        self.trans_profile_ideal = self.compute_trans_ideal(self.tuned_spacing, waves)
        self.apply_defect_broadening()
        return self.trans_profile

    def tune_wavelength_by(self, wave_step, waves):
        """
        Discreetly tune the counts and spacing in such a way to tune the transmission by given amount of wavelength step
        Input:  wavelength step
                wavelength array (1d)
        Output: transmission profile (1d)
        """
        self.tuned_wavelength += wave_step
        self.compute_params(self.tuned_wavelength)
        self.tuned_spacing = self.order*self.tuned_wavelength/(2*self.RI)
        self.tuned_spacing = self.thickness + self.tune_step*np.rint((self.tuned_spacing-self.thickness)/self.tune_step)
        # self.tuned_wavelength = self.tuned_spacing*2*self.RI/self.order
        self.trans_profile_ideal = self.compute_trans_ideal(self.tuned_spacing, waves)
        self.apply_defect_broadening()
        return self.trans_profile

    def plot_trans_profile(self, ax, *args, **kwargs):
        """
        Plot the transmission profile vs. wavelength
        """
        ax.plot(self.waves, self.trans_profile, *args, **kwargs)

    def apply_defect_broadening(self):
        """
        Apply effects of surface defects to the transmission profile
        Convolve the transmission profile by various defect functions 
        that correspond to curvature, micro-roughness and parallelism errors
        """
        self.df_curv_dist = self.dist_err_curv()/np.sum(self.dist_err_curv())
        self.df_paral_dist = self.dist_err_paral()/np.sum(self.dist_err_paral())
        self.df_rough_dist = self.dist_err_rough()/np.sum(self.dist_err_rough())
        trans_broad1 = np.convolve(self.trans_profile_ideal, self.df_curv_dist, mode='same')
        trans_broad2 = np.convolve(trans_broad1, self.df_paral_dist, mode='same')
        self.trans_profile = np.convolve(trans_broad2, self.df_rough_dist, mode='same')
        
    def dist_err_curv(self):
        """
        Compute phase distribution function for curvature error
        The function is D(phi) = 1/(2*phi_max) for -phi_max < phi < phi_max
        """
        wavelength = self.waves[len(self.waves)//2]
        phase_max = 2*self.err_curv*2*np.pi/wavelength
        phase = (self.waves-wavelength)*4*np.pi*self.tuned_spacing/wavelength**2
        dist = 0.0*phase
        dist[np.argwhere(np.abs(phase)<phase_max)] = 1.0/(2*phase_max)
        return dist
    
    def dist_err_rough(self):
        """
        Compute phase distribution function for micro-roughness error
        The function is D(phi) = Gaussian(phi)
        """
        wavelength = self.waves[len(self.waves)//2]
        phase = (self.waves-wavelength)*4*np.pi*self.tuned_spacing/wavelength**2
        phase_std = np.sqrt(2)*self.err_rough*2*np.pi/wavelength
        phase = (self.waves-wavelength)*4*np.pi*self.tuned_spacing/wavelength**2
        dist = np.exp(-phase**2/(2*phase_std**2))/(np.sqrt(2*np.pi)*phase_std)
        return dist

    def dist_err_paral(self):
        """
        Compute phase distribution function for parallelism error
        The function is D(phi) = 
        """
        wavelength = self.waves[len(self.waves)//2]
        phase = (self.waves-wavelength)*4*np.pi*self.tuned_spacing/wavelength**2
        phase_max = 2*self.err_paral*2*np.pi/wavelength
        dist = (2.0/(np.pi*phase_max))*np.sqrt(1-phase**2/phase_max**2)
        dist[np.argwhere(np.abs(phase)>phase_max)] = 0.0
        return dist

class line_spectrum:
    """
    Solar spectrum of selected wavelength region
    """
    def __init__(self, spectrum):
        """
        Initialize the oject by reading spectrum from the file
        Input:  spectrum file name with intensity vs. wavelength data
        """
        self.spectrum = spectrum
        self.waves_full = 1.0e-10*spectrum[:,0]
        self.intens_full = 1.0e-2*spectrum[:,1]

    def set_spectral_line(self, wavelength, wave_range, wave_step, plot=False):
        """
        Select the spectral line from the full spectrum
        Input:  central wavelength of the range
                wavelength range
                wavelength step size (sample resolution)
                (opt.) plot the selection
        """
        self.wavelength = wavelength
        self.wave_range = wave_range
        self.wave_step = wave_step
        wave_beg, wave_end = wavelength-wave_range/2.0, wavelength+wave_range/2.0
        self.waves = np.arange(wave_beg, wave_end, wave_step)
        iwave_line = np.argwhere((self.waves_full>self.waves[0]) & (self.waves_full<self.waves[-1]))
        self.waves_line = self.waves_full[iwave_line]
        self.intens_line = self.intens_full[iwave_line]
        self.intens = zoom(self.intens_line.flatten(), len(self.waves)/len(self.waves_line))
        self.nwaves = len(self.waves)
        if (plot):
            fig, ax = plt.subplots(1,1)
            self.plot_spectral_line(ax)
            fig.tight_layout()

    def plot_spectral_line(self, ax, *args, **kwargs):
        """
        Plot the spectrum intensity vs. wavelength
        """
        ax.plot(self.waves, self.intens, *args, **kwargs)
        
class interference_filter:
    """
    Simple ideal multi-cavity interference filter
    """
    def __init__(self, wavelength, passband, ncavity=2, trans_peak=1.0, od=np.inf, RI_eff=1.5, fnum=np.inf, xtilt=0, ytilt=0):
        """
        Initialize object with its core properties
        Input:  central wavelength of maximum transmission
                full width at half maximum of transmission passband
                (opt.) number of cavities
                (opt.) fractional peak transmission
                (opt.) effective refractive index of the filter
                (opt.) f-ratio of the incoming beam
                (opt.) angle between the filter normal and the chief ray
                (opt.) level of discreetness for sampling
        """
        self.ncavity = ncavity
        self.trans_peak = trans_peak
        self.wavelength = wavelength
        self.passband = passband
        self.OD = od
        self.RI_eff = RI_eff
        self.xtilt = xtilt
        self.ytilt = ytilt
        self.fnum = fnum
    def compute_transmission(self):
        """
        Butterworth filter equation
        """
        return self.trans_peak/(1+(2*(self.waves-self.wavelength)/self.passband)**(2*self.ncavity))


    def get_trans_profile(self, waves, plot=False):
        """
        Compute transmission profile of the filter in a given wavelength range for given filter and beam properties
        Input:  wavelength array(1d)
                (opt.) plot the transmission profile
        Output: transmission profile
        """
        nsamp = 100
        self.waves = waves
        xtilt_ = np.radians(self.xtilt)                   
        ytilt_ = np.radians(self.ytilt)
        theta = np.arctan(0.5/self.fnum)                         # cone half angle
        self.cha = np.degrees(theta)                        # cone half angle
        cwl_original = 1.0*self.wavelength
        if (self.fnum == np.inf):                           # special case of collimated beam
            tilt = np.arccos(np.cos(xtilt_)*np.cos(ytilt_))
            cwl_shifted = self.wavelength*np.sqrt(1-np.sin(tilt)**2/self.RI_eff**2)
            self.wavelength = 1.0*cwl_shifted
            self.trans_profile = self.compute_transmission()
        else:
            ang_range = np.array([])
            theta_range = np.linspace(0, theta, nsamp)
            for th in theta_range:
                temp_n = int(nsamp*np.sin(th)/np.sin(theta))
                phi_range = np.linspace(0, 2*np.pi, temp_n)
                temp = -np.sin(ytilt_)*np.sin(th)*np.cos(phi_range) + np.cos(ytilt_)*(np.sin(xtilt_)*np.sin(th)*np.sin(phi_range)+np.cos(xtilt_)*np.cos(th))
                ang_range = np.append(ang_range, np.arccos(temp))
            cwl_shifted = self.wavelength*np.sqrt(1-np.sin(ang_range)**2/self.RI_eff**2)
            # weights = np.tan(ang_range)
            trans_arr = []
            for c in cwl_shifted:
                self.wavelength = 1.0*c
                trans_arr.append(self.compute_transmission())
            self.trans_profile = np.average(np.array(trans_arr), axis=0)
            self.trans_arr = np.array(trans_arr)
        self.trans_profile[np.argwhere(self.trans_profile < 10**(-self.OD))] = 10**(-self.OD)
        self.wavelength = 1.0*cwl_original
        return self.trans_profile
        #
        if (plot):
            fig, ax = plt.subplots(1,1)
            self.plot_trans_profile(ax)
            fig.tight_layout()
        return self.trans_profile

    def plot_trans_profile(self, ax, *args, **kwargs):
        """
        Plot the filter's transmission profile vs. wavelength 
        """
        ax.plot(self.waves, self.trans_profile)


class fp_filter_system:
    """
    An optical system consisting of multiple interference filters + multiple etalons in tandem
    """
    def __init__(self, spectrum, filters, etalons):
        """
        Initialize the object with components
        Input:  spectrum
                list of interference filters
                list of etalons
        """
        self.spectrum = spectrum
        self.etalons = etalons
        self.filters = filters

    def set_spectral_line(self, line_id, wavelength, wave_range, wave_step, plot=False):
        """
        Select spectral line for the analysis
        Input:  string id of the spectral line
                central wavelength (to be applied to the spectrum, filters and etalons)
                wavelength range
                wavelength step (sample resolution)
                (opt.) plot various profiles
        Output: transmission profile of the system in the given wavelength range
        """
        self.line_id = line_id
        self.wavelength = wavelength
        self.wave_range = wave_range
        self.wave_step = wave_step
        self.spectrum.set_spectral_line(wavelength, wave_range, wave_step)
        self.waves = self.spectrum.waves
        self.filters_trans = []
        for f in self.filters:
            self.filters_trans.append(f.get_trans_profile(self.spectrum.waves))
        self.etalon_trans = []
        for e in self.etalons:
            self.etalon_trans.append(e.tune_to_wavelength(wavelength, self.spectrum.waves))
        self.trans_profile = np.prod(self.filters_trans,axis=0)*np.prod(self.etalon_trans,axis=0)
        if (plot):
            fig, ax = plt.subplots(1,1)
            self.spectrum.plot_spectral_line(ax)
            for f in self.filters:
                f.plot_trans_profile(ax)
            for e in self.etalons:
                e.plot_trans_profile(ax)
            self.plot_trans_profile(ax, 'k')
        return self.trans_profile

    def plot_trans_profile(self, ax, *args, **kwargs):
        """
        Plot the transmission profile vs. wavelength of the system 
        """
        ax.plot(self.waves, self.trans_profile, *args, **kwargs)

    def tune_to_wavelength(self, wavelength):
        """
        Tune the spacings of all the etalons in such a way that their transmission peak matches with the given wavelength
        Input:  wavelength at which maximum transmission is expected
        Output: transmission profile of the system (1d array)
        """
        self.etalon_trans = []
        for e in self.etalons:
            self.etalon_trans.append(e.tune_to_wavelength(wavelength, self.spectrum.waves))
        self.trans_profile = np.prod(self.filters_trans,axis=0)*np.prod(self.etalon_trans,axis=0)
        return self.trans_profile

    def tune_wavelength_by(self, wave_step):
        """
        Discreetly tune the counts and spacings of all the etalons in such a way to tune the transmission by given amount of wavelength step
        Input:  wavelength step
        Output: transmission profile of the system (1d)
        """
        self.etalon_trans = []
        for e in self.etalons:
            self.etalon_trans.append(e.tune_wavelength_by(wave_step, self.spectrum.waves))
        self.trans_profile = np.prod(self.filters_trans,axis=0)*np.prod(self.etalon_trans,axis=0)
        return self.trans_profile
    
    def scan_spectral_line(self, wavelength, wave_range, nsamples, plot=False):
        """
        Scan the spectral line with computed system response to generate observed spectrum
        Noe that all the parameters have to be within the pre-selected spectral line's range
        Input:  central wavelength
                wavelength range
                number of samples for scanning
                (opt.) plot the scanned spectral line vs. wavelength
        Output: wavelength points
                computed line intensity
        """
        waves = np.linspace(wavelength-wave_range/2.0, wavelength+wave_range/2.0,nsamples)
        intens = []
        weights = []
        for w in progressbar(waves):
            intens.append(np.sum(self.tune_to_wavelength(w)*self.spectrum.intens))
            weights.append(np.sum(self.trans_profile))
        weights = np.array(weights)
        intens = np.array(intens)/np.mean(weights)
        if (plot):
            fig, ax = plt.subplots(1,1)
            ax.plot(waves, intens)
            self.spectrum.plot_spectral_line(ax)
        return waves, intens

    def scan_spectral_line_discreet(self, wavelength, wave_range, nsamples, plot=False):
        """
        Discreetly scan the spectral line in steps with computed system response to generate observed spectrum
        Noe that all the parameters have to be within the pre-selected spectral line's range
        Input:  central wavelength
                wavelength range
                number of samples for scanning
                (opt.) plot the scanned spectral line vs. wavelength
        Output: wavelength points
                computed line intensity
        """
        waves = np.linspace(wavelength-wave_range/2.0, wavelength+wave_range/2.0,nsamples)
        wave_step = wave_range/(nsamples-1)
        intens = self.tune_to_wavelength(waves[0]-wave_step)
        intens = []
        weights = []
        for i in progressbar(range(nsamples)):
            intens.append(np.sum(self.tune_wavelength_by(wave_step)*self.spectrum.intens))
            weights.append(np.sum(self.trans_profile))
        weights = np.array(weights)
        intens = np.array(intens)/np.mean(weights)
        if (plot):
            fig, ax = plt.subplots(1,1)
            ax.plot(waves, intens)
            self.spectrum.plot_spectral_line(ax)
        return waves, intens

def compute_parasitic_light(waves, intens, main_band=1e-10):
    """
    Compute fractional intensity transmitted out of the main transmission band of a bandpass system
    Input:  wavelength points
            transmission at given wavelength points
            (opt.) mainband limit
    Output: fraction of parasitic light
    """
    wavelength = waves[np.argmax(intens)]
    intens_main = np.sum(intens[np.argwhere(np.abs(waves-wavelength)<main_band/2.0)])
    paras_frac = (np.sum(intens)-intens_main)/intens_main
    return paras_frac


def compute_transmission_properties(waves, trans_profile):
    """
    Compute central wavelength, full width at half maximum, full width at one percent of maximum and 75% encircled energy wavelength passband
    Input:  wavelength points
            transmission profile
    Output: central wavelength
            full width at half maximum passband
            full width at one percent maximum passband 
            75% encircled energy passband
    """
    iwave_peak_trans = np.argmax(trans_profile)
    iwave_peak_trans = int(np.mean(iwave_peak_trans))
    waves_cwl = waves[iwave_peak_trans]
    #
    spline_interp = UnivariateSpline(waves, trans_profile-trans_profile.max()/2, s=0)
    root1, root2 = spline_interp.roots() # find the roots
    waves_fwhm = np.abs(root1-root2)
    #
    spline_interp = UnivariateSpline(waves, trans_profile-trans_profile.max()/100, s=0)
    root1, root2 = spline_interp.roots() # find the roots
    waves_fwopm = np.abs(root1-root2)
    #
    enc_energ = [np.sum(trans_profile[iwave_peak_trans-j:iwave_peak_trans+j]) for j in range(iwave_peak_trans//2)]
    enc_75 = np.argmin(abs(0.75-enc_energ/np.sum(trans_profile)))*(waves[1]-waves[0])*2
    return waves_cwl, waves_fwhm, waves_fwopm, enc_75

def compute_et_trans_shift(theta, RI):
    """
    Compute the shift in etalon transmission
    Input:  angle of incidence in degrees
            refractive index of the cavity
    Output: fractional shift in terms of wavelength
    """
    sh = np.sqrt(1-(np.sin(np.radians(theta))/RI)**2)-1
    return sh

def compute_parasitic_light_extended(et, syst, full_wave_range=200e-9):
    """
    Compute the parasitic light parameters for etalon and the system combination
    Input:  etalon object
            etalon filter system object
            (opt.) full wavelength range of incoming light
    Output: percentage parasitic light from the side lobes (first order)
            percentage parasitic light from the full wavelength range
            phono noise limited snr and parasitic light limited snr
    """
    imainband = np.argwhere(np.abs(et.waves-syst.wavelength) < et.FSR/2.0)
    nsteps_FSR = int(et.FSR/syst.wave_step)
    nFSR = full_wave_range/et.FSR
    mainband, sideband = np.sum(syst.trans_profile[imainband]), np.sum(syst.trans_profile[0:nsteps_FSR])
    paras_first = 100*compute_parasitic_light(syst.waves, syst.trans_profile, main_band=et.FSR)
    paras_total = 100*nFSR*sideband/mainband + paras_first
    paras_snr = mainband/np.sqrt(mainband+paras_total*mainband/100)
    max_snr = np.sqrt(mainband)
    return paras_first, paras_total, [max_snr, paras_snr]

# def compute_parasitic_snr(line, et, syst):
#     imainband = np.argwhere(np.abs(et.waves-syst.wavelength) < et.FSR/2.0)
#     nsteps_FSR = int(et.FSR/syst.wave_step)
#     full_wave_range = 200e-9
#     nFSR = full_wave_range/et.FSR
#     trans_profile = syst.trans_profile*line.intens
#     mainband, sideband = np.sum(trans_profile[imainband]), np.sum(trans_profile[0:nsteps_FSR])
#     paras_first = compute_parasitic_light(syst.waves, trans_profile, main_band=et.FSR)
#     paras_total = nFSR*sideband/mainband + paras_first
#     paras_snr = mainband/np.sqrt(mainband+paras_total)
#     return paras_snr
    
def get_random_filter_specs(cen, tol, nsamp):
    """
    Get random samples from uniform distribution
    Input:  mean value
            half range
            number of samples
    Output: array of samples
    """
    samps = []
    for i in range(nsamp):
        samps.append(rd.uniform(cen-tol, cen+tol))
    return np.array(samps)