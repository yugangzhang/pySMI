#import sys
import re # Regular expressions
import numpy as np
import pylab as plt
import matplotlib as mpl
#from scipy.optimize import leastsq
#import scipy.special
import PIL # Python Image Library (for opening PNG, etc.) 
import sys, os

from skbeam.core.accumulators.binned_statistic import BinnedStatistic2D,BinnedStatistic1D




def convert_Qmap( img, qx_map, qy_map=None, bins=None, rangeq=None,  origin=None, mask=None, statistic='mean'):
    """Y.G. Nov 3@CHX 
    Convert a scattering image to a qmap by giving qx_map and qy_map
    Return converted qmap, x-coordinates and y-coordinates
    """
    if qy_map is not None:           
        if rangeq is None:
            qx_min,qx_max = qx_map.min(), qx_map.max()
            qy_min,qy_max = qy_map.min(), qy_map.max()
            rangeq = [ [qx_min,qx_max], [qy_min,qy_max] ] 
            #rangeq =  [qx_min,qx_max ,  qy_min,qy_max]
        if bins is None:
            bins = qx_map.shape

        if mask is not None:
            m = mask.ravel()
        else:
            m = None
            
        b2d = BinnedStatistic2D( qx_map.ravel(), qy_map.ravel(), 
                                    statistic=statistic, bins=bins, mask=m,
                                    range=rangeq)
        remesh_data, xbins, ybins = b2d( img.ravel() ), b2d.bin_centers[0], b2d.bin_centers[1]
        
    else:
        if rangeq is None:
            qx_min,qx_max = qx_map.min(), qx_map.max()
            rangeq =   [qx_min,qx_max]       
        if bins is None:
            bins = [ qx_map.size ]
        #print( rangeq, bins )
        if mask is not None:
            m = mask.ravel()
        else:
            m = None
        b1d = BinnedStatistic1D(  qx_map.ravel(), bins= bins, mask=m  )
        
        remesh_data  = b1d(  img.ravel()  )
        #print('Here')
        xbins= b1d.bin_centers     
        ybins=None
        
    return remesh_data, xbins, ybins






def qphiavg(image, q_map=None, phi_map=None, mask=None, bins= None,
            origin=None, range=None, statistic='mean'):
    ''' quick qphi average calculator.
        ignores bins for now
    '''
    # TODO : replace with method that takes qphi maps
    # TODO : also return q and phi of this...
    # print("In qphi average stream")
    shape = image.shape
    if bins is None:
        bins=shape
        #print(bins)
    if origin is None:
        origin = (shape[0] - 1) / 2., (shape[1] - 1) / 2.

    from skbeam.core.utils import radial_grid, angle_grid

    if q_map is None:
        q_map = radial_grid(origin, shape)
    if phi_map is None:
        phi_map = angle_grid(origin, shape)

    expected_shape = tuple(shape)
    if mask is not None:
        if mask.shape != expected_shape:
            raise ValueError('"mask" has incorrect shape. '
                             ' Expected: ' + str(expected_shape) +
                             ' Received: ' + str(mask.shape))
        mask = mask.reshape(-1)

    rphibinstat = BinnedStatistic2D(q_map.reshape(-1), phi_map.reshape(-1),
                                    statistic=statistic, bins=bins, mask=mask,
                                    range=range)

    sqphi = rphibinstat(image.ravel())
    qs = rphibinstat.bin_centers[0]
    phis = rphibinstat.bin_centers[1]
    return  sqphi,  qs,  phis 





def convert_Qmap_old( img, qx_map, qy_map=None, bins=None, rangeq=None):
    """Y.G. Nov 3@CHX 
    Convert a scattering image to a qmap by giving qx_map and qy_map
    Return converted qmap, x-coordinates and y-coordinates
    """
    if qy_map is not None:           
        if rangeq is None:
            qx_min,qx_max = qx_map.min(), qx_map.max()
            qy_min,qy_max = qy_map.min(), qy_map.max()
            rangeq = [ [qx_min,qx_max], [qy_min,qy_max] ]        
        if bins is None:
            bins = qx_map.shape

        remesh_data, xbins, ybins = np.histogram2d(qx_map.ravel(), qy_map.ravel(), 
                        bins=bins, range= rangeq, normed=False, weights= img.ravel() )
        
    else:
        if rangeq is None:
            qx_min,qx_max = qx_map.min(), qx_map.max()
            rangeq =   [qx_min,qx_max]       
        if bins is None:
            bins = qx_map.size 
        else:
            if isinstance( bins, list):
                bins = bins[0] 
        print( rangeq, bins )
        remesh_data, xbins  = np.histogram(qx_map.ravel(), 
                        bins=bins, range= rangeq, normed=False, weights= img.ravel() )        
        ybins=None
    return remesh_data, xbins, ybins




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
    ''' Devloped by Kevin. Yager 
    Copied from  https://github.com/CFN-softbio/SciAnalysis/blob/master/SciAnalysis/XSAnalysis/Data.py
    Till Nov 20, 2018--> Without modification
    
    Stores aspects of the experimental setup; especially the calibration
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
    Initially developed by Kevin. Yager
    Add offest and incident angles by Yugang Zhang
    
    The geometric claculations used here are described:
    http://gisaxs.com/index.php/Geometry:WAXS_3D    
    
    
    """    
    
    # Experimental parameters
    ########################################
    
    def set_angles(self, det_phi_g=0., det_theta_g=0., incident_angle=0,
                   sam_phi=0, sam_chi=0, sam_theta=0,
                   offset_x = 0, offset_y =0, offset_z=0):
        '''
        
        
        YG. Add sample rotation angles that convert qmap from lab frame to sample frame
        All the angles are given in degrees
        
        sam_phi, rotate along lab-frame x, CHX phi
        sam_chi, rotate along lab-frame z, CHX chi
        sam_theta, rotate along lab-frame y, CHX theta
        
        YG add offset corrections at Sep 21, 2017
        
        det_phi_g, rotate along y-axis, delta at CHX
        det_theta_g, away from z-plane, gamma at CHX
        
        For SMI, because only rotate along y-axis, (det_theta_g=0.), only care about 
        offset_x, offset_z '''
        #print('Set angles here')
        
        
        self.det_phi_g = det_phi_g
        self.det_theta_g = det_theta_g
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.incident_angle  = incident_angle
        self.sam_phi=sam_phi   
        self.sam_chi= sam_chi
        self.sam_theta=sam_theta
        
    def rotation_matix(self,  sam_phi, sam_theta, sam_chi,  degrees=True):
        '''        
        sam_phi, rotate along lab-frame x, CHX phi
        sam_chi, rotate along lab-frame z, CHX chi
        sam_theta, rotate along lab-frame y, CHX theta        
        '''
        
        if degrees:
            sam_phi, sam_chi, sam_theta = np.radians(sam_phi), np.radians(sam_chi), np.radians(sam_theta)            
        
        Rx = np.array( [ [1,        0,                0            ],
                         [0,  np.cos( sam_phi ), np.sin( sam_phi ) ],
                         [0, -np.sin( sam_phi ), np.cos( sam_phi ) ]
                       ]
                     )
        
        Rz = np.array( [ [ np.cos( sam_chi ), np.sin( sam_chi ),  0          ],
                         [-np.sin( sam_chi ), np.cos( sam_chi ),  0           ],
                         [0,        0,                1 ]
                       ]
                     ) 
         
        Ry = np.array( [ [np.cos(  sam_theta ), 0,    np.sin( sam_theta ) ],
                         [0,                   1,       0                ],
                         [-np.sin( sam_theta ), 0,    np.cos( sam_theta ) ]
                       ]
                     )
        Rxy = np.dot(Rx,Ry)
        return np.dot(Rxy,Rz)
    
    def _generate_qxyz_map_SF_from_Lab(self,qx,qy,qz,
                                       sam_phi, sam_theta, sam_chi,
                                       degrees=True):
        '''
        Convert qmap from Lab frame to sample frame 
        '''
        self.Rot = self.rotation_matix( sam_phi,  sam_theta,  sam_chi, degrees=degrees )        
        qsx, qsy, qsz = np.dot(self.Rot, [ np.ravel(qx), np.ravel(qy), np.ravel(qz)] )
        return qsx.reshape( qx.shape), qsy.reshape( qy.shape),qsz.reshape( qz.shape)
            
    def _generate_qxyz_maps_samFrame(self, degrees=True):
        """
        Get lab frame qmap
        """
        self._generate_qxyz_maps()
        self.qx_map_lab_data,self.qy_map_lab_data,self.qz_map_lab_data=  self._generate_qxyz_map_SF_from_Lab( 
                          self.qx_map_data,self.qy_map_data,self.qz_map_data, 
                          self.sam_phi, self.sam_theta, self.sam_chi,
                           degrees=degrees ) 
        self.qr_map_lab_data = np.sqrt(np.square(self.qx_map_lab_data) + np.square(self.qy_map_lab_data))
        
        self.q_map_lab_data =  np.sqrt(np.square(self.qx_map_lab_data) +
                                       np.square(self.qy_map_lab_data) +
                                       np.square(self.qz_map_lab_data)
                                     )       
        
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
        
        In this calculation, X-ray along y direction, up is z-direction, outward is x dir
        det_phi_g, rotate along z dir
        det_theta_g, rotate along x dir
        
        For SMI, only rotate along z-dir (only det_phi_g !=0)
        
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
        
        
        
        #######why use this following part
        # Conversion factor for pixel coordinates
        # (where sample-detector distance is set to d = 1)
        c = (self.pixel_size_um/1e6)/self.distance_m        
        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)        
        #twotheta = np.arctan(self.r_map()*c) # radians
        theta_f = np.arctan2( X*c, 1 ) # radians
        alpha_f_prime = np.arctan2( Y*c, 1 ) # radians
        alpha_f = np.arctan2( Y*c*np.cos(theta_f), 1 ) # radians            
        #print(c)
        self.alpha_f = alpha_f 
        self.theta_f  = theta_f
        self.qx_map_data2 = self.get_k()*np.sin(theta_f)*np.cos(alpha_f_prime)
        self.qy_map_data2 = self.get_k()*( np.cos(theta_f)*np.cos(alpha_f_prime) - 1 ) # TODO: Check sign
        self.qz_map_data2 = -1.0*self.get_k()*np.sin(alpha_f_prime) 
        
        self.qr_map_data2 = np.sign(self.qx_map_data2)*np.sqrt(np.square(self.qx_map_data2) + np.square(self.qy_map_data2))   
        
        ##################  
        
        self.qx_map_data = qx_c
        self.qy_map_data = qy_c
        self.qz_map_data = -qz_c
        self.q_map_data = q_c    
        self.qr_map_data = np.sign(self.qx_map_data)*np.sqrt(np.square(self.qx_map_data) + np.square(self.qy_map_data)) 
 
        # convert to sample coordinates
        alpha = np.radians(self.sam_phi)
        Qn = -qz_c * np.cos(alpha) + qy_c * np.sin(alpha)
        Qr = np.sqrt(q_c*q_c-Qn*Qn)*np.sign(qx_c)        
        self.qr_map_data = Qr
        self.qn_map_data = Qn 
        #print('here')
    
    def _generate_qxyz_maps(self):
        """
        The geometric claculations used here are described:
        http://gisaxs.com/index.php/Geometry:WAXS_3D
        
        YG add offset corrections at Sep 21, 2017
        
        incident beam is along y-direction
        
        """    
        
        #print('Here to get qmap with offset.')
        
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
        
        # convert to sample coordinates        
        if self.incident_angle == 0:
            self.qx_map_data = qx_c
            self.qy_map_data = qy_c
            self.qz_map_data = qz_c  
        else:
            ai = np.radians( self.incident_angle )
            self.qx_map_data = qx_c 
            self.qy_map_data =  qy_c * np.cos( ai ) + qz_c * np.sin( ai )
            self.qz_map_data =  -qy_c * np.sin( ai ) + qz_c * np.cos( ai )
        M = (self.qx_map_data ==0 )
        sign = np.sign( self.qx_map_data  )
        sign[M] = 1    
        self.qr_map_data =  ( np.sqrt(np.square(self.qx_map_data) + np.square(self.qy_map_data))  ) * sign     
        self.q_map_data =  np.sqrt(  np.square(self.qx_map_data) + np.square(self.qy_map_data) + np.square(self.qz_map_data)      )
            
        #alpha = np.radians(self.sam_phi)
        #Qn = -qz_c * np.cos(alpha) + qy_c * np.sin(alpha)
        #Qr = np.sqrt(q_c*q_c-Qn*Qn)*np.sign(qx_c)        
        #self.qr_map_data = Qr
        #self.qn_map_data = Qn 
        #self.qz_map_data = Qn 
        #print('here')
        
        
        
        
 # CalibrationRQconv
################################################################################    
class CalibrationRQconv(Calibration):
    """
    Developed by K.Yager
    Copied from https://github.com/CFN-softbio/SciAnalysis/blob/master/SciAnalysis/XSAnalysis/DataRQconv.py
    
    The geomatric claculations used here are described in Yang, J Synch Rad (2013) 20, 211–218
    http://dx.doi.org/10.1107/S0909049512048984
    
    """    
    
    
    def __init__(self, wavelength_A=None, distance_m=None, pixel_size_um=None, det_orient=0., det_tilt=0., det_phi=0., incident_angle=0., sample_normal=0.):
        
        self.det_orient = det_orient
        self.det_tilt = det_tilt
        self.det_phi = det_phi
        
        self.incident_angle = incident_angle
        self.sample_normal = sample_normal
        
        
        self.rot_matrix = None
        super().__init__(wavelength_A=wavelength_A, distance_m=distance_m, pixel_size_um=pixel_size_um)
            
    
    # Experimental parameters
    ########################################
    
    def set_angles(self, det_orient=0, det_tilt=0, det_phi=0, det_roty=0, incident_angle=0., 
                   sample_normal=0.):
        self.det_orient = det_orient
        self.det_tilt = det_tilt
        self.det_phi = det_phi
        self.det_roty = det_roty

        self.incident_angle = incident_angle
        self.sample_normal = sample_normal
    
    def get_ratioDw(self):
        
        width_mm = self.width*self.pixel_size_um/1000.
        return self.distance_m/(width_mm/1000.)


    # Maps
    ########################################
    
    def q_map(self):
        '''Returns a 2D map of the q-value associated with each pixel position
        in the detector image.'''

        if self.q_map_data is None:
            self._generate_maps()
        
        return self.q_map_data
    
    def angle_map(self):
        '''Returns a map of the angle for each pixel (w.r.t. origin).
        0 degrees is vertical, +90 degrees is right, -90 degrees is left.'''

        if self.angle_map_data is not None:
            self._generate_maps()
        
        return self.angle_map_data
    
    def qx_map(self):
        if self.qx_map_data is None:
            self._generate_maps()
        
        return self.qx_map_data    

    def qy_map(self):
        if self.qy_map_data is None:
            self._generate_maps()
        
        return self.qy_map_data    

    def qz_map(self):
        if self.qz_map_data is None:
            self._generate_maps()
        
        return self.qz_map_data    
    
    def qr_map(self):
        if self.qr_map_data is None:
            self._generate_maps()

        return self.qr_map_data
    
    
    def _generate_maps(self):
        """
        calculate all coordinates (pixel position as well as various derived values)
        all coordinates are stored in 2D arrays, as is the data itself in Data2D
        """
        
        #self.calc_rot_matrix()
        self.calc_rot_matrix_SMI()
        
        (w,h) = (self.width, self.height)
        self.X = np.repeat(np.arange(w), h).reshape((w, h)).T
        X = self.X.flatten()
        Y = np.repeat(np.arange(h), w)
        self.Y = Y.reshape((h, w))
        
        (Q, Phi, Qx, Qy, Qz, Qr, Qn, FPol, FSA) = self.calc_from_XY(X, Y, calc_cor_factors=True)
                
        self.q_map_data = Q.reshape((h, w))
        
        self.qx_map_data = Qx.reshape((h, w))
        self.qy_map_data = -Qy.reshape((h, w))  #YG add -
        self.qz_map_data = -Qz.reshape((h, w))  #YG add -
        
        self.qr_map_data = Qr.reshape((h, w))        
        self.qn_map_data = -Qn.reshape((h, w))  #YG add -
        
        self.angle_map_data = np.degrees( Phi.reshape((h, w)) )
        self.FPol_map_data = FPol.reshape((h, w))
        self.FSA_map_data = FSA.reshape((h, w))
        
        
        
        
    # q calculation
    ########################################
        
    def calc_from_XY(self, X, Y, calc_cor_factors=False):
        """
        calculate Q values from pixel positions X and Y
        X and Y are 1D arrays
        returns reciprocal/angular coordinates, optionally returns 
        always calculates Qr and Qn, therefore incident_angle needs to be set 
        Note that Phi is saved in radians; but the angles in ExpPara are in degrees
        """
        if self.rot_matrix is None:
            raise ValueError('the rotation matrix is not yet set.')

        # the position vectors for each pixel, origin at the postion of beam impact
        # R.shape should be (3, w*h), but R.T is more convinient for matrix calculation
        # RT.T[i] is a vector
        RT = np.vstack((X - self.x0, -(Y - self.y0), 0.*X))
        
        dr = self.get_ratioDw()*self.width
        # position vectors in lab coordinates, sample at the origin
        [X1, Y1, Z1] = np.dot(self.rot_matrix, RT)
        Z1 -= dr
        
        # angles
        r3sq = X1*X1+Y1*Y1+Z1*Z1
        r3 = np.sqrt(r3sq)
        r2 = np.sqrt(X1*X1+Y1*Y1)
        Theta = 0.5*np.arcsin(r2/r3)
        Phi = np.arctan2(Y1, X1) + np.radians(self.sample_normal)

        Q = 2.0*self.get_k()*np.sin(Theta)

        # lab coordinates
        Qz = Q*np.sin(Theta)
        Qy = Q*np.cos(Theta)*np.sin(Phi)
        Qx = Q*np.cos(Theta)*np.cos(Phi)

        # convert to sample coordinates
        alpha = np.radians(self.incident_angle)
        Qn = Qy*np.cos(alpha) + Qz*np.sin(alpha)
        Qr = np.sqrt(Q*Q-Qn*Qn)*np.sign(Qx)
        
        if calc_cor_factors==True:
            FPol = (Y1*Y1+Z1*Z1)/r3sq
            FSA = np.power(np.fabs(Z1)/r3, 3)
            return (Q, Phi, Qx, Qy, Qz, Qr, Qn, FPol, FSA)
        else:
            return (Q, Phi, Qx, Qy, Qz, Qr, Qn)
    
    
    # Rotation calculation
    ########################################
    
    
    def RotationMatrix(self, axis, angle):
        if axis=='x' or axis=='X':
            rot = np.asarray(
                [[1., 0., 0.],
                [0., np.cos(angle), -np.sin(angle)],
                [0., np.sin(angle),  np.cos(angle)]])
        elif axis=='y' or axis=='Y':
            rot = np.asarray(
                [[ np.cos(angle), 0., np.sin(angle)],
                [0., 1., 0.],
                [-np.sin(angle), 0., np.cos(angle)]])
        elif axis=='z' or axis=='Z':
            rot = np.asarray(
                [[np.cos(angle), -np.sin(angle), 0.],
                [np.sin(angle),  np.cos(angle), 0.],
                [0., 0., 1.]])
        else:
            raise ValueError('unkown axis %s' % axis)
        
        return rot 


    def calc_rot_matrix(self):
        
        # First rotate detector about x-ray beam
        tm1 = self.RotationMatrix('z', np.radians(-self.det_orient))
        
        # Tilt detector face (so that it is not orthogonal to beam)
        tm2 = self.RotationMatrix('y', np.radians(self.det_tilt))
        
        # Rotate detector about x-ray beam
        tm3 = self.RotationMatrix('z', np.radians(self.det_orient+self.det_phi))
        self.rot_matrix = np.dot(np.dot(tm3, tm2), tm1)
            
            
    def calc_rot_matrix_SMI(self):        
        tm1 = self.RotationMatrix('z', np.radians(self.det_roty)) 
        self.rot_matrix =   tm1            
            
    
    
    
    # End class CalibrationRQconv(Calibration)
    ########################################       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
