# Data Processing Routines for HELioseismic Large Regions Interferometric DEvice (HELLRIDE)
HELLRIDE is a 2-D imaging spectropolarimeter at the back-end of Vaccum Tower Telescope (VTT), Teide Observatory (OT), Tenerife. It uses two Fabry-Pérot interferometers for 2-D spectroscopy and liquid-crystal variable retarders for polarimetry. The details of the instrument can be found in the instrument paper.  
[SoPh]

This package is a collection of subroutines that process the data recorded by HELLRIDE.  
1. Before the science data can be processed, first the calibration data needs to be processed. Master dark and average flat are generated from the dark and flat-fielding recordings respectively, for all the spectral regions or filters.  
2. Polarimetric calibration data is processed to compute the modulation matrix for all the filters.
3. Average flat data computed from step-1 is fitted with a parametric model consisting of atlas solar spectrum and instrumental effects. The output of this step is the fitted model.  
4. Output of step-3 is further processed to generate the master flat and wavelength calibration data for all the filters. Master flat corrects for the imaging as well as spectral continuum levels for all states of modulation.   
5. Target plate data is processed to deduce the alignment parameters (affine matrix) for the three imaging channels. Channel-1 of the polarimeter is taken as the reference. This process is interactive.    
6. Science data is corrected for dark and flat, and demodulated. In case of step-6a, the channels are aligned and combined. However, the default option is to prepare the data for the image reconstruction using MOMFBD (cite). At the moment, MOMFBD is run in a separate environment (as we don't have a wrapper that can be integrated onto this package) and the resulting data is read out by the package. The output of this step is Stokes parameters and context images.  
7. Stokes parameters are corrected for the instrumental polarization and prepared for the radiative transfer inversions. Alternatively, using step-7a only Stokes-I/intensity profiles are processed.  
8. Stokes parameters are inverted to produce physical parameters/atmospheric model of the Sun. The package uses VFISV (cite) for the inversions. Due to the way VFISV functions, each filter needs its own codefile in its own directory. Hence, code files corresponding to step-8 are structured differently.  
9. The data of the entire time series of observations is co-aligned and put-together for scientific usage as well as visualization purposes.   