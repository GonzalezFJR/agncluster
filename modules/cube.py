import os
from astropy.io import fits
import numpy as np
import astropy.units as u

class cube:

    def __init__(self, filename, wavelength_command=None,ext=1):
        self.filename = filename

        self.data = None # raw data
        self.cube = None # processed data
        self.header = None # header information
        self.header0 = None # header information
        self.extent = None # physical extent of the datacube in arcseconds
        self.wavelength = None # wavelength array in Angstroms
        self.lambdas = None # wavelength array in Angstroms after cutting

        self.set_wavelength_command(wavelength_command)
        self.load_data(ext=ext)

    ### Load data
    ##################################################################

    def set_wavelength_command(self, wavelength_command=None):
        ''' Set the wavelength command '''
        if wavelength_command is None:
            self.wavelength_command = "header['CRVAL3']+(header['CDELT3']*np.arange(abs(header['CRPIX3']-1),abs(header['CRPIX3']-1)+len(data),1))"
        else:
            self.wavelength_command = wavelength_command

    def set_extent_commmand(self, extent_command=None):
        ''' Get the extent of the datacube from the header information''' 
        if extent_command is not None:
            self.extent_command = "[header['CDELT1']*u.deg.to(u.arcsec)*header['NAXIS1']/2,-header['CDELT1']*u.deg.to(u.arcsec)*(header['NAXIS1']/2),-header['CDELT2']*u.deg.to(u.arcsec)*(header['NAXIS2']/2),header['CDELT2']*u.deg.to(u.arcsec)*(header['NAXIS2']/2)]"
        else:
            self.extent_command = extent_command

    def load_data(self,ext=1):
        ''' Load the data and wavelength array from a fits file '''
        data = fits.getdata(self.filename, ext=ext)
        header = fits.getheader(self.filename, ext=ext)
        if ext == 1:
            header0 = fits.getheader(self.filename, ext=0)
            self.header0 = header0
        #    extent = eval(self.extent_command("[header0['CDELT1']*u.deg.to(u.arcsec)*header0['NAXIS1']/2,-header0['CDELT1']*u.deg.to(u.arcsec)*(header0['NAXIS1']/2),-header0['CDELT2']*u.deg.to(u.arcsec)*(header0['NAXIS2']/2),header0['CDELT2']*u.deg.to(u.arcsec)*(header0['NAXIS2']/2)]"))
        #else: 
        #    extent = eval(self.extent_command)
        # Construct the wavelength array from the CD1_1, CRVAL1, and CRPIX1 keywords
        wavelength = eval(self.wavelength_command)
        self.data = data
        self.header = header
        self.wavelength = wavelength
        #self.extent = extent
        #self.remove_nans()
        self.mask = ~((np.isnan(data).any(axis=0)) | (data == 0).any(axis=0))
        print('[INFO]: Loaded datacube from', self.filename)
        print('[INFO]: Wavelength range:', np.min(self.wavelength), np.max(self.wavelength))
        print('[INFO]: Data shape:', self.data.shape)
        print('[INFO]: Mask shape:', self.mask.shape)
    
    def remove_nans(self):
        ''' Extrapolate with values in the wavelength axis '''
        zz, yy, xx = self.data.shape
        print('Inital shape of the cube:', zz, yy, xx)
        for i in range(yy):
            for j in range(xx):
                mask = np.isnan(self.data[:, i, j])
                # if all the values are nan, set them to 0
                if np.all(mask):
                    self.data[:, i, j] = 0
                # if more than 50% of the values are nan, set them to 0 as well
                elif np.sum(mask) > 0.2 * zz:
                    self.data[:, i, j] = 0
                else:
                    self.data[mask, i, j] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.data[~mask, i, j])

        print('Final shape of the cube:', self.data.shape)


    ### Process data
    ##################################################################
    def set_data_limits(self, xfrom=15, xto=300, yfrom=40, yto=320, lambda_from=0, lambda_to=8500, lambda_idx=True):
        ''' Set the data limits '''
        self.xfrom = xfrom
        self.xto = xto
        self.yfrom = yfrom
        self.yto = yto
        if lambda_idx:
            self.lambda_from_idx = max(lambda_from, 0)
            self.lambda_to_idx = min(lambda_to, len(self.wavelength)-1)
            #self.lambda_to_idx = min(max(lambda_to, len(self.wavelength)-1), len(self.wavelength)-1)
            self.lambda_from = self.wavelength[self.lambda_from_idx]
            self.lambda_to = self.wavelength[self.lambda_to_idx]
        else:
            self.lambda_from = lambda_from
            self.lambda_to = lambda_to
            self.lambda_from_idx = np.where(self.wavelength > lambda_from)[0][0]
            self.lambda_to_idx = np.where(self.wavelength < lambda_to)[0][-1]

    def set_continuum_limits(self, lambda_from=0, lambda_to=8500, lambda_idx=True):
        ''' Set the continuum limits '''
        if lambda_idx:
            self.continuum_from_idx = lambda_from
            self.continuum_to_idx = lambda_to
            self.continuum_from = self.wavelength[lambda_from]
            self.continuum_to = self.wavelength[lambda_to]
        else:
            self.continuum_from = lambda_from
            self.continuum_to = lambda_to
            self.continuum_from_idx = np.where(self.wavelength > lambda_from)[0][0]
            self.continuum_to_idx = np.where(self.wavelength < lambda_to)[0][-1]

    def cut_data(self):
        ''' Cut the datacube to a smaller region and a smaller wavelength range '''
        print('Cutting the datacube to the region:', self.xfrom, self.xto, self.yfrom, self.yto)
        valid_idx = np.where((self.wavelength < self.lambda_to) & (self.wavelength > self.lambda_from))[0]
        if len(valid_idx) == 0:
            raise ValueError('No wavelength channels found within the selected limits.')

        start_idx = valid_idx[0]
        stop_idx = valid_idx[-1] + 1
        datacube = self.data[start_idx:stop_idx, self.yfrom:self.yto, self.xfrom:self.xto]

        # Cut the wavelength array to match the datacube
        self.lambdas = self.wavelength[start_idx:stop_idx]
        self.cube = datacube
        self.cube_mask = self.mask[self.yfrom:self.yto, self.xfrom:self.xto]

    def continuum(self, x, y):
        ''' Get the continuum for a given pixel '''
        #return np.median(self.cube[self.continuum_from_idx:self.continuum_to_idx, x, y])
        return np.median(self.data[self.continuum_from_idx:self.continuum_to_idx, x, y])

    def normalize_cont(self):
        ''' Normalize the pixels in the datacube using continuum '''
        zz, yy, xx = self.cube.shape
        print('Inital shape of the cube:', zz, yy, xx)
        for i in range(yy):
            for j in range(xx):
                cont = self.continuum(i, j)
                self.cube[:, i, j] = self.cube[:, i, j] / cont
        print('Final shape of the cube:', self.cube.shape)
    
    def normalize_flux(self):
        ''' Normalize the flux by integrating the total flux per pixel '''
        zz, yy, xx = self.cube.shape
        print('Inital shape of the cube:', zz, yy, xx)
        total_flux = np.sum(self.cube, axis=0)  # Integrate total flux per pixel
        print('Shape of total flux:', total_flux.shape)
        total_flux[total_flux == 0] = 1  # Avoid division by zero
        self.cube = self.cube / total_flux  # Normalize the cube
        print('Final shape of the cube:', self.cube.shape)
        
    def reset_cube(self):
        ''' Reset the cube to the original data '''
        self.cube = self.data
        self.cut_data()

    def get_mask(self):
        ''' Mask for x,y pixels that are empty '''
        mask = np.all(self.data == 0, axis=0)
        # add pixels where more than 50% of the values are 0
        mask = mask | (np.sum(self.data == 0, axis=0) > 0.5 * self.data.shape[0])
        return mask

    def subtract_continuum(self):
        ''' Subtract the continuum from the datacube '''
        zz, yy, xx = self.cube.shape
        for i in range(yy):
            for j in range(xx):
                # the continuum would be the median of the flux in the first 10 pixels
                cont = np.median(self.cube[0:20, i, j])
                self.cube[:, i, j] = self.cube[:, i, j] - cont

    ### Get data
    ##################################################################

    def get_image(self, lambda_select, processed=True, log=True, mask=True, traspose=False):
        ''' Get an image at a given wavelength '''
        data = self.data[lambda_select] if not processed else self.cube[lambda_select]
        if mask and processed:
            data[self.get_mask()] = 0
        if traspose:
            data = data.T
        if log:
            data = np.log(data + 1e-4)
            # Normalize the data to [0, 1]
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def get_spectrum(self, y, x, processed=True):
        ''' Get the wavelength array for a given pixel '''
        return self.cube[:, y, x] if processed else self.data[:, y, x]

    def get_cube(self, normalize_flux=False, normalize_cont=False, subtract=False):
        ''' Get the datacube '''
        self.reset_cube()
        if subtract:
            self.subtract_continuum()
        if normalize_cont:
            self.normalize_cont()
        if normalize_flux:
            self.normalize_flux()
        return self.cube
