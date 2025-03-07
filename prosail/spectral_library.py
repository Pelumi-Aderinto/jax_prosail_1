#!/usr/bin/env python
"""Spectral libraries for PROSPECT + SAIL
"""
import pkgutil
# from io import StringIO
from io import BytesIO
from collections import namedtuple

import numpy as np
import jax.numpy as jnp

Spectra = namedtuple('Spectra',  'prospectpro prospect5 prospectd soil light')
ProspectproSpectra = namedtuple('ProspectproSpectra', 
                                'nr kab kcar kbrown kw km kant kcp kcbc')
Prospect5Spectra = namedtuple('Prospect5Spectra', 
                                'nr kab kcar kbrown kw km')
ProspectDSpectra = namedtuple('ProspectDSpectra', 
                                'nr kab kcar kbrown kw km kant')
SoilSpectra = namedtuple("SoilSpectra", "rsoil1 rsoil2")
LightSpectra = namedtuple("LightSpectra", "es ed")


def get_spectra():
    """Reads the spectral information and stores is for future use."""
    
    #PROSPECT-PRO
    prospect_pro_spectraf = pkgutil.get_data('prosail', 'prospect_pro_spectra.txt')
    _, nr, kab, kcar, kant, kbrown, kw, km, kcp, kcbc =  jnp.array(np.loadtxt(BytesIO(prospect_pro_spectraf), 
                                                unpack=True))
    prospect_pro_spectra = ProspectproSpectra(nr, kab, kcar, kbrown, kw, km, kant, kcp, kcbc)

    # PROSPECT-D
    prospect_d_spectraf = pkgutil.get_data('prosail', 'prospect_d_spectra.txt')
    _, nr, kab, kcar, kant, kbrown, kw, km= jnp.array(np.loadtxt( 
        BytesIO(prospect_d_spectraf), unpack=True))
    prospect_d_spectra = ProspectDSpectra(nr, kab, kcar, kbrown, kw, km, kant)
    # PROSPECT 5
    prospect_5_spectraf = pkgutil.get_data('prosail', 'prospect5_spectra.txt')
    nr, kab, kcar, kbrown, kw, km =  jnp.array(np.loadtxt(BytesIO(prospect_5_spectraf), 
                                                unpack=True))
    prospect_5_spectra = Prospect5Spectra(nr, kab, kcar, kbrown, kw, km)
    # SOIL
    soil_spectraf = pkgutil.get_data('prosail', 'soil_reflectance.txt')
    rsoil1, rsoil2 =  jnp.array(np.loadtxt(BytesIO(soil_spectraf), 
                                                unpack=True))
    soil_spectra = SoilSpectra(rsoil1, rsoil2)    
    # LIGHT
    light_spectraf = pkgutil.get_data('prosail', 'light_spectra.txt')
    es, ed =  jnp.array(np.loadtxt(BytesIO(light_spectraf), 
                                                unpack=True))
    light_spectra = LightSpectra(es, ed)
    spectra = Spectra(prospect_pro_spectra, prospect_5_spectra, prospect_d_spectra, 
                      soil_spectra, light_spectra)
    return spectra

# get_spectra()