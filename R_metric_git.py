## Implementation of the R-metric algorithm as given in Röthlisberger et al. 2018
## Wavenumber filtering algoritm is from Zimmin et al

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
# import glob
import sys
from scipy.fftpack import fft, ifft
import math
import cmath as cm

t1 = datetime.now()

def calc_hov(da, lat1, lat2, dim='lat'):
    """Function to calculate Hovmoeller field
    ---------------------------------------------
    Parameters:
    da: xarray dataarray
    lat1, lat2: (int) latitude band to use for averaging
    dim: (str) dimension to average along
    ----------------------------------------------
    Returns:
    xarray dataarray
        """
    return da.sel(lat=slice(lat1, lat2)).mean(dim)

def wave_filter(da):
    """
    Function to apply wave-number filtering as per Zimin et al. 
    -----------------------------------------------------------
    Parameters:
    da: xarray datarray, Hovmoeller fields to apply wavenumber filter to
    -----------------------------------------------------------
    Returns:
    k_filter: numpy array containing complex value data
    """
    
    # filter values are decided as per Roethlisberger et al. 2018
    kmin = 4
    kmax = 15
    k = np.arange(kmin, kmax+1) # since last element is not included
    ### Calculating FFT ###

    V_fft = fft(da.values)/len(da.lon)

    time_steps = len(da.time)
    lon_grids = len(da.lon)

    ###--------------------------------------------------------------------------------------------###


    ### Wave number filtering ###
    ## loop to apply wavenumber filtering (Zimmin et al Roethlisberger et al. 
    ## To Do: Vectorize instead of looping

    # assert, filtering works for only shape 2
    assert len(da.shape) == 2
    k_filter = np.zeros(shape = da.shape, dtype=complex)

    print("Starting R-metric calculations at", datetime.now())
    for t in range(0,time_steps):
        # looping over all longitudes
        for lo in range(0,lon_grids):
            k_filter[t, lo] = \
            2 * (V_fft[t, kmin:kmax+1] * math.e ** (2*math.pi*1j* k * lo / lon_grids)).sum()
            # kmax+1 since last element is not included in the slice
            
    return k_filter

def main():
    input_path='/path/to/velocity_fields'
    out_path='/path/to/save'
    
    da_V = xr.open_dataarray(input_path)
    # calculate Hov. fields
    hov_V = calc_hov(da_V)
    
    # Time Filtering

    ## time-average for 14 day running mean, 14x4 = 56 +1 to keep it centered
    ## dropping the NAN values at the start and end, some time-steps are dropped
    ## 7 or 8 time steps at both end will be dropped
    hov_V = hov_V.rolling(time=57, center=True).mean().dropna('time')
    
    # make sure var_name is "V"
    new_var_name='V'
    hov_V.rename({'old_var_name': new_var_name})
    
    big_ds = hov_V.to_dataset()
    big_ds.V.attrs['Comment'] = '14-day time filtered meridional velocity'
    k_filter = wave_filter(big_ds.V)
    
    # absolute part of the complex number is the Envelop
    big_ds['R_metric'] = (('time', 'lon'), np.absolute(k_filter)) # to add variables use dictionary syntax
    # real part is the filtered wave
    big_ds['Filtered_wave'] = (('time', 'lon'), np.real(k_filter)) # real part
    
    big_ds.to_netcdf(f'{out_path}R_metric.nc', mode='w', 
    encoding={
        'V':{'_FillValue': -999.0},
        'R_metric':{'_FillValue': -999.0},
        'Filtered_wave':{'_FillValue': -999.0}
        }
    )
if __name__== "__main__":
    # main will not be called when imported as a module
    main()