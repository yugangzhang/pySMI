#import sys
import re # Regular expressions
import numpy as np
import pylab as plt
import matplotlib as mpl
#from scipy.optimize import leastsq
#import scipy.special
import PIL # Python Image Library (for opening PNG, etc.) 
import sys, os


# Mask
################################################################################    
class Mask(object):
    '''Stores the matrix of pixels to be excluded from further analysis.'''
    
    def __init__(self, infile=None, format='auto'):
        '''Creates a new mask object, storing a matrix of the pixels to be 
        excluded from further analysis.'''
        
        self.data = None
        
        if infile is not None:
            self.load(infile, format=format)
        
        
    def load(self, infile, format='auto', invert=False):
        '''Loads a mask from a a file. If this object already has some masking
        defined, then the new mask is 'added' to it. Thus, one can load multiple
        masks to exlude various pixels.'''
        
        if format=='png' or infile[-4:]=='.png':
            self.load_png(infile, invert=invert)
            
        elif format=='hdf5' or infile[-3:]=='.h5' or infile[-4:]=='.hd5':
            self.load_hdf5(infile, invert=invert)
            
        else:
            print("Couldn't identify mask format for %s."%(infile))
            
            
    def load_blank(self, width, height):
        '''Creates a null mask; i.e. one that doesn't exlude any pixels.'''
        
        # TODO: Confirm that this is the correct order for x and y.
        self.data = np.ones((height, width))
        
            
    def load_png(self, infile, threshold=127, invert=False):
        '''Load a mask from a PNG image file. High values (white) are included, 
        low values (black) are exluded.'''
        
        # Image should be black (0) for excluded pixels, white (255) for included pixels
        img = PIL.Image.open(infile).convert("L") # black-and-white
        img2 = img.point(lambda p: p > threshold and 255)
        data = np.asarray(img2)/255
        data = data.astype(int)
        
        if invert:
            data = -1*(data-1)
        
        if self.data is None:
            self.data = data
        else:
            self.data *= data
        
        
    def load_hdf5(self, infile, invert=False):
        
        with h5py.File(infile, 'r') as f:
            data = np.asarray( f['mask'] )

        if invert:
            data = -1*(data-1)
        
        if self.data is None:
            self.data = data
        else:
            self.data *= data

        
    def invert(self):
        '''Inverts the mask. Can be used if the mask file was written using the
        opposite convention.'''
        self.data = -1*(self.data-1)


    # End class Mask(object)
    ########################################
    
    
    
    
    
    

# Calibration
################################################################################    
class Calibration(object):
    '''Stores aspects of the experimental setup; especially the calibration
    parameters for a particular detector. That is, the wavelength, detector
    distance, and pixel size that are needed to convert pixel (x,y) into
    reciprocal-space (q) value.
    
    This class may also store other information about the experimental setup
    (such as beam size and beam divergence).
    '''
    
    def __init__(self, wavelength_A=None, distance_m=None, pixel_size_um=None):
        
        self.wavelength_A = wavelength_A
        self.distance_m = distance_m
        self.pixel_size_um = pixel_size_um
        
        
        # Data structures will be generated as needed
        # (and preserved to speedup repeated calculations)
        self.clear_maps()
    
    
    # Experimental parameters
    ########################################
    
    def set_wavelength(self, wavelength_A):
        '''Set the experimental x-ray wavelength (in Angstroms).'''
        
        self.wavelength_A = wavelength_A
    
    
    def get_wavelength(self):
        '''Get the x-ray beam wavelength (in Angstroms) for this setup.'''
        
        return self.wavelength_A
    
        
    def set_energy(self, energy_keV):
        '''Set the experimental x-ray beam energy (in keV).'''
        
        energy_eV = energy_keV*1000.0
        energy_J = energy_eV/6.24150974e18
        
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        wavelength_m = (h*c)/energy_J
        self.wavelength_A = wavelength_m*1e+10
    
    
    def get_energy(self):
        '''Get the x-ray beam energy (in keV) for this setup.'''
        
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        wavelength_m = self.wavelength_A*1e-10 # m
        E = h*c/wavelength_m # Joules
        
        E *= 6.24150974e18 # electron volts
        
        E /= 1000.0 # keV
        
        return E
    
    
    def get_k(self):
        '''Get k = 2*pi/lambda for this setup, in units of inverse Angstroms.'''
        
        return 2.0*np.pi/self.wavelength_A
    
    
    def set_distance(self, distance_m):
        '''Sets the experimental detector distance (in meters).'''
        
        self.distance_m = distance_m
        
    
    def set_pixel_size(self, pixel_size_um=None, width_mm=None, num_pixels=None):
        '''Sets the pixel size (in microns) for the detector. Pixels are assumed
        to be square.'''
        
        if pixel_size_um is not None:
            self.pixel_size_um = pixel_size_um
            
        else:
            if num_pixels is None:
                num_pixels = self.width
            pixel_size_mm = width_mm*1./num_pixels
            self.pixel_size_um = pixel_size_mm*1000.0
        
        
    def set_beam_position(self, x0, y0):
        '''Sets the direct beam position in the detector images (in pixel 
        coordinates).'''
        
        self.x0 = x0
        self.y0 = y0
        
        
    def set_image_size(self, width, height=None):
        '''Sets the size of the detector image, in pixels.'''
        
        self.width = width
        if height is None:
            # Assume a square detector
            self.height = width
        else:
            self.height = height
    
    
    def get_q_per_pixel(self):
        '''Gets the delta-q associated with a single pixel. This is computed in
        the small-angle limit, so it should only be considered approximate.
        For instance, wide-angle detectors will have different delta-q across
        the detector face.'''
        
        if self.q_per_pixel is not None:
            return self.q_per_pixel
        
        c = (self.pixel_size_um/1e6)/self.distance_m
        twotheta = np.arctan(c) # radians
        
        self.q_per_pixel = 2.0*self.get_k()*np.sin(twotheta/2.0)
        
        return self.q_per_pixel
    
    
    # Maps
    ########################################
    
    def clear_maps(self):
        self.r_map_data = None
        self.q_per_pixel = None
        self.q_map_data = None
        self.angle_map_data = None
        
        self.qx_map_data = None
        self.qy_map_data = None
        self.qz_map_data = None
        self.qr_map_data = None

    
    def r_map(self):
        '''Returns a 2D map of the distance from the origin (in pixel units) for
        each pixel position in the detector image.'''
        
        if self.r_map_data is not None:
            return self.r_map_data

        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        self.r_map_data = R
        
        return self.r_map_data
        
    
    def q_map(self):
        '''Returns a 2D map of the q-value associated with each pixel position
        in the detector image.'''

        if self.q_map_data is not None:
            return self.q_map_data
        
        c = (self.pixel_size_um/1e6)/self.distance_m
        twotheta = np.arctan(self.r_map()*c) # radians
        
        self.q_map_data = 2.0*self.get_k()*np.sin(twotheta/2.0)
        
        return self.q_map_data
        
    
    def angle_map(self):
        '''Returns a map of the angle for each pixel (w.r.t. origin).
        0 degrees is vertical, +90 degrees is right, -90 degrees is left.'''

        if self.angle_map_data is not None:
            return self.angle_map_data
        
        x = (np.arange(self.width) - self.x0)
        y = (np.arange(self.height) - self.y0)
        X,Y = np.meshgrid(x,y)
        #M = np.degrees(np.arctan2(Y, X))
        # Note intentional inversion of the usual (x,y) convention.
        # This is so that 0 degrees is vertical.
        #M = np.degrees(np.arctan2(X, Y))

        # TODO: Lookup some internal parameter to determine direction
        # of normal. (This is what should befine the angle convention.)
        M = np.degrees(np.arctan2(X, -Y))

        
        self.angle_map_data = M
        
        return self.angle_map_data
    
    
    def qx_map(self):
        if self.qx_map_data is not None:
            return self.qx_map_data
        
        self._generate_qxyz_maps()
        
        return self.qx_map_data    

    def qy_map(self):
        if self.qy_map_data is not None:
            return self.qy_map_data
        
        self._generate_qxyz_maps()
        
        return self.qy_map_data    

    def qz_map(self):
        if self.qz_map_data is not None:
            return self.qz_map_data
        
        self._generate_qxyz_maps()
        
        return self.qz_map_data    
    
    def qr_map(self):
        if self.qr_map_data is not None:
            return self.qr_map_data

        self._generate_qxyz_maps()

        return self.qr_map_data



    def _generate_qxyz_maps(self):

        # Conversion factor for pixel coordinates
        # (where sample-detector distance is set to d = 1)
        c = (self.pixel_size_um/1e6)/self.distance_m
        
        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        #twotheta = np.arctan(self.r_map()*c) # radians
        theta_f = np.arctan2( X*c, 1 ) # radians
        #alpha_f_prime = np.arctan2( Y*c, 1 ) # radians
        alpha_f = np.arctan2( Y*c*np.cos(theta_f), 1 ) # radians
        
        
        self.qx_map_data = self.get_k()*np.sin(theta_f)*np.cos(alpha_f)
        self.qy_map_data = self.get_k()*( np.cos(theta_f)*np.cos(alpha_f) - 1 ) # TODO: Check sign
        self.qz_map_data = -1.0*self.get_k()*np.sin(alpha_f)
        
        self.qr_map_data = np.sign(self.qx_map_data)*np.sqrt(np.square(self.qx_map_data) + np.square(self.qy_map_data))

        
    
    
    # End class Calibration(object)
    ########################################
    
    
    
# CalibrationGonio
################################################################################    
class CalibrationGonio(Calibration):
    """
    The geometric claculations used here are described:
    http://gisaxs.com/index.php/Geometry:WAXS_3D
    
    """    
    
    # Experimental parameters
    ########################################
    
    def set_angles(self, det_phi_g=0., det_theta_g=0., offset_x = 0, offset_y =0, offset_z=0):
        '''YG add offset corrections at Sep 21, 2017
        For SMI, because only rotate along y-axis, (det_theta_g=0.), only care about 
        offset_x, offset_z '''
        #print('Set angles here')
        self.det_phi_g = det_phi_g
        self.det_theta_g = det_theta_g
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
    def get_ratioDw(self):
        
        width_mm = self.width*self.pixel_size_um/1000.
        return self.distance_m/(width_mm/1000.)


    # Maps
    ########################################

    def q_map(self):
        if self.q_map_data is None:
            self._generate_qxyz_maps()
        
        return self.q_map_data
    
    def angle_map(self):
        if self.angle_map_data is not None:
            self._generate_qxyz_maps()
        
        return self.angle_map_data
    

    def _generate_qxyz_maps_no_offest(self):
        """
        The geometric claculations used here are described:
        http://gisaxs.com/index.php/Geometry:WAXS_3D
        
        """    
        
        d = self.distance_m
        pix_size = self.pixel_size_um/1e6
        phi_g = np.radians(self.det_phi_g)
        theta_g = np.radians(self.det_theta_g)
        
        xs = (np.arange(self.width) - self.x0)*pix_size
        ys = (np.arange(self.height) - self.y0)*pix_size
        #ys = ys[::-1]

        X_c, Y_c = np.meshgrid(xs, ys)
        Dprime = np.sqrt( np.square(d) + np.square(X_c) + np.square(Y_c) )
        k_over_Dprime = self.get_k()/Dprime
        
        
        qx_c = k_over_Dprime*( X_c*np.cos(phi_g) - np.sin(phi_g)*(d*np.cos(theta_g) - Y_c*np.sin(theta_g)) )
        qy_c = k_over_Dprime*( X_c*np.sin(phi_g) + np.cos(phi_g)*(d*np.cos(theta_g) - Y_c*np.sin(theta_g)) - Dprime )
        qz_c = -1*k_over_Dprime*( d*np.sin(theta_g) + Y_c*np.cos(theta_g) )

        qr_c = np.sqrt(np.square(qx_c) + np.square(qy_c))        
        q_c = np.sqrt(np.square(qx_c) + np.square(qy_c) + np.square(qz_c))
        
        
        
        
        # Conversion factor for pixel coordinates
        # (where sample-detector distance is set to d = 1)
        c = (self.pixel_size_um/1e6)/self.distance_m
        
        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        #twotheta = np.arctan(self.r_map()*c) # radians
        theta_f = np.arctan2( X*c, 1 ) # radians
        #alpha_f_prime = np.arctan2( Y*c, 1 ) # radians
        alpha_f = np.arctan2( Y*c*np.cos(theta_f), 1 ) # radians
        
        
        self.qx_map_data = self.get_k()*np.sin(theta_f)*np.cos(alpha_f)
        self.qy_map_data = self.get_k()*( np.cos(theta_f)*np.cos(alpha_f) - 1 ) # TODO: Check sign
        self.qz_map_data = -1.0*self.get_k()*np.sin(alpha_f)
        
        self.qr_map_data = np.sign(self.qx_map_data)*np.sqrt(np.square(self.qx_map_data) + np.square(self.qy_map_data))

        
        self.qx_map_data = qx_c
        self.qy_map_data = qy_c
        self.qz_map_data = qz_c
        self.q_map_data = q_c    
    
    
    def _generate_qxyz_maps(self):
        """
        The geometric claculations used here are described:
        http://gisaxs.com/index.php/Geometry:WAXS_3D
        
        YG add offset corrections at Sep 21, 2017
        """    
        
        #print('Here to get qmap without offset.')
        
        d = self.distance_m #
        pix_size = self.pixel_size_um/1e6  #in meter
        phi_g = np.radians(self.det_phi_g)
        theta_g = np.radians(self.det_theta_g)        
        
        offset_x = self.offset_x  *pix_size  #in meter
        offset_y = self.offset_y  *pix_size
        offset_z = self.offset_z  *pix_size
        
        xs = (np.arange(self.width) - self.x0)*pix_size 
        ys = (np.arange(self.height) - self.y0)*pix_size 
        
        xsprime = xs -  offset_x
        dprime  = d -   offset_y
        ysprime = ys  - offset_z
        #ys = ys[::-1]

        X_c, Y_c = np.meshgrid(xsprime, ysprime)
        #Dprime = np.sqrt( np.square(d) + np.square(X_c) + np.square(Y_c) )
        #k_over_Dprime = self.get_k()/Dprime
        yprime = dprime*np.cos(theta_g) - Y_c*np.sin(theta_g)
        Dprime = np.sqrt(   np.square(dprime) + np.square(X_c) + np.square(Y_c) + 
                            offset_x**2 + offset_y**2 + offset_z**2 + 
                           2*offset_x*(   X_c*np.cos(phi_g) - np.sin(phi_g) * yprime ) +
                           2*offset_y*(   X_c*np.sin(phi_g) + np.cos(phi_g) * yprime ) +
                           2*offset_z*(    dprime*np.sin(theta_g) + Y_c*np.cos(theta_g) )
                        )                      
                        
        k_over_Dprime = self.get_k()/Dprime          
        
        qx_c = k_over_Dprime*( X_c*np.cos(phi_g) - np.sin(phi_g) * yprime +  offset_x) 
        qy_c = k_over_Dprime*( X_c*np.sin(phi_g) + np.cos(phi_g) * yprime +  offset_y - Dprime) 
        qz_c = -1*k_over_Dprime*( dprime*np.sin(theta_g) + Y_c*np.cos(theta_g) +  offset_z )

        qr_c = np.sqrt(np.square(qx_c) + np.square(qy_c)) 
        q_c = np.sqrt(np.square(qx_c) + np.square(qy_c) + np.square(qz_c))  
        
        
        
        if False:#True:
            # Conversion factor for pixel coordinates
            # (where sample-detector distance is set to d = 1)
            c = (self.pixel_size_um/1e6)/self.distance_m

            x = np.arange(self.width) - self.x0
            y = np.arange(self.height) - self.y0
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)

            #twotheta = np.arctan(self.r_map()*c) # radians
            theta_f = np.arctan2( X*c, 1 ) # radians
            #alpha_f_prime = np.arctan2( Y*c, 1 ) # radians
            alpha_f = np.arctan2( Y*c*np.cos(theta_f), 1 ) # radians


            self.qx_map_data1 = self.get_k()*np.sin(theta_f)*np.cos(alpha_f)
            self.qy_map_data1 = self.get_k()*( np.cos(theta_f)*np.cos(alpha_f) - 1 ) # TODO: Check sign
            self.qz_map_data1 = -1.0*self.get_k()*np.sin(alpha_f)

            self.qr_map_data1 = np.sign(self.qx_map_data1)*np.sqrt(np.square(self.qx_map_data1) + np.square(self.qy_map_data1))

        
        self.qx_map_data = qx_c
        self.qy_map_data = qy_c
        self.qz_map_data = qz_c
        self.q_map_data = q_c
        self.qr_map_data = qr_c
        
        
    