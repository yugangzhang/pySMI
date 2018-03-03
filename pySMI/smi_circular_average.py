import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
RUN_GUI = False
    
    
    
def radial_grid(center, shape, pixel_size=None):
    """Convert a cartesian grid (x,y) to the radius relative to some center
    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.
        Order is (rr, cc).
    pixel_size : sequence, optional
        The physical size of the pixels.
        len(pixel_size) should be the same as len(shape)
        defaults to (1,1)
    Returns
    -------
    r : array
        The distance of each pixel from `center`
        Shape of the return value is equal to the `shape` input parameter
    """

    if pixel_size is None:
        pixel_size = (1, 1)

    X, Y = np.meshgrid(pixel_size[1] * (np.arange(shape[1]) - center[1]),
                       pixel_size[0] * (np.arange(shape[0]) - center[0]))
    return np.sqrt(X * X + Y * Y)


def angle_grid(center, shape, pixel_size=None):
    """
    Make a grid of angular positions.
    Read note for our conventions here -- there be dragons!
    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).
    Returns
    -------
    agrid : array
        angular position (in radians) of each array element in range [-pi, pi]
    Note
    ----
    :math:`\\theta`, the counter-clockwise angle from the positive x axis,
    assuming the positive y-axis points upward.
    :math:`\\theta \\el [-\pi, \pi]`.  In array indexing and the conventional
    axes for images (origin in upper left), positive y is downward.
    """

    if pixel_size is None:
        pixel_size = (1, 1)

    # row is y, column is x. "so say we all. amen."
    x, y = np.meshgrid(pixel_size[1] * (np.arange(shape[1]) -
                                        center[1]),
                       pixel_size[0] * (np.arange(shape[0]) -
                                        center[0]))
    return np.arctan2(y, x)


def radius_to_twotheta(dist_sample, radius):
    """
    Converts radius from the calibrated center to scattering angle
    (2:math:`2\\theta`) with known detector to sample distance.
    Parameters
    ----------
    dist_sample : float
        distance from the sample to the detector (mm)
    radius : array
        The L2 norm of the distance of each pixel from the calibrated center.
    Returns
    -------
    two_theta : array
        An array of :math:`2\\theta` values
    """
    return np.arctan(radius / dist_sample)

def bin_edges_to_centers(input_edges):
    """
    Helper function for turning a array of bin edges into
    an array of bin centers
    Parameters
    ----------
    input_edges : array-like
        N + 1 values which are the left edges of N bins
        and the right edge of the last bin
    Returns
    -------
    centers : ndarray
        A length N array giving the centers of the bins
    """
    input_edges = np.asarray(input_edges)
    return (input_edges[:-1] + input_edges[1:]) * 0.5



def bin_1D(x, y, nx=None, min_x=None, max_x=None):
    """
    Bin the values in y based on their x-coordinates
    Parameters
    ----------
    x : array
        position
    y : array
        intensity
    nx : integer, optional
        number of bins to use defaults to default bin value
    min_x : float, optional
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    edges : array
        edges of bins, length nx + 1
    val : array
        sum of values in each bin, length nx
    count : array
        The number of counts in each bin, length nx
    """

    # handle default values
    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)
    if nx is None:
        nx = int(max_x - min_x) 

    #print ( min_x, max_x, nx)    
    
    
    # use a weighted histogram to get the bin sum
    bins = np.linspace(start=min_x, stop=max_x, num=nx+1, endpoint=True)
    #print (x)
    #print (bins)
    val, _ = np.histogram(a=x, bins=bins, weights=y)
    # use an un-weighted histogram to get the counts
    count, _ = np.histogram(a=x, bins=bins)
    # return the three arrays
    return bins, val, count



def circular_average(image, calibrated_center, threshold=0, nx=None,
                     pixel_size=(1, 1),  min_x=None, max_x=None, mask=None):
    """Circular average of the the image data
    The circular average is also known as the radial integration
    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    calibrated_center : tuple
        The center of the image in pixel units
        argument order should be (row, col)
    threshold : int, optional
        Ignore counts above `threshold`
        default is zero
    nx : int, optional
        number of bins in x
        defaults is 100 bins
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    bin_centers : array
        The center of each bin in R. shape is (nx, )
    ring_averages : array
        Radial average of the image. shape is (nx, ).
    """
    radial_val = radial_grid(calibrated_center, image.shape, pixel_size)     
    
    if mask is not None:  
        #maks = np.ones_like(  image )
        mask = np.array( mask, dtype = bool)
        binr = radial_val[mask]
        image_mask =     np.array( image )[mask]
        
    else:        
        binr = np.ravel( radial_val ) 
        image_mask = np.ravel(image) 
        
    #if nx is None: #make a one-pixel width q
    #   nx = int( max_r - min_r)
    #if min_x is None:
    #    min_x= int( np.min( binr))
    #    min_x_= int( np.min( binr)/(np.sqrt(pixel_size[1]*pixel_size[0] )))
    #if max_x is None:
    #    max_x = int( np.max(binr )) 
    #    max_x_ = int( np.max(binr)/(np.sqrt(pixel_size[1]*pixel_size[0] ))  )
    #if nx is None:
    #    nx = max_x_ - min_x_
    
    #binr_ = np.int_( binr /(np.sqrt(pixel_size[1]*pixel_size[0] )) )
    binr_ =   binr /(np.sqrt(pixel_size[1]*pixel_size[0] ))
    #print ( min_x, max_x, min_x_, max_x_, nx)
    bin_edges, sums, counts = bin_1D(      binr_,
                                           image_mask,
                                           nx=nx,
                                           min_x=min_x,
                                           max_x=max_x)
    
    #print  (len( bin_edges), len( counts) )
    th_mask = counts > threshold
    
    #print  (len(th_mask) )
    ring_averages = sums[th_mask] / counts[th_mask]

    bin_centers = bin_edges_to_centers(bin_edges)[th_mask]
    
    #print (len(  bin_centers ) )

    return bin_centers, ring_averages 

def twotheta_to_q(two_theta, wavelength):
    r"""
    Helper function to convert two-theta to q
    By definition the relationship is
    .. math::
        \sin\left(\frac{2\theta}{2}\right) = \frac{\lambda q}{4 \pi}
    thus
    .. math::
        q = \frac{4 \pi \sin\left(\frac{2\theta}{2}\right)}{\lambda}
    Parameters
    ----------
    two_theta : array
        An array of :math:`2\theta` values
    wavelength : float
        Wavelength of the incoming x-rays
    Returns
    -------
    q : array
        An array of :math:`q` values in the inverse of the units
        of ``wavelength``
    """
    two_theta = np.asarray(two_theta)
    wavelength = float(wavelength)
    pre_factor = ((4 * np.pi) / wavelength)
    return pre_factor * np.sin(two_theta / 2) 

def get_circular_average( avg_img, mask, pargs, show_pixel=True,  min_x=None, max_x=None,
                          nx=None, plot_ = False ,   save=False, *argv,**kwargs):   
    """get a circular average of an image        
    Parameters
    ----------
    
    avg_img: 2D-array, the image
    mask: 2D-array  
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength    
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
    nx : int, optional
        number of bins in x
        defaults is 1500 bins
        
    plot_: a boolen type, if True, plot the one-D curve
    plot_qinpixel:a boolen type, if True, the x-axis of the one-D curve is q in pixel; else in real Q
    
    Returns
    -------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)
     
     
    """   
    
    center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    uid = pargs['uid']    
    qp, iq = circular_average(avg_img, 
        center, threshold=0, nx=nx, pixel_size=(dpix, dpix), mask=mask, min_x=min_x, max_x=max_x) 
    qp_ = qp * dpix
    #  convert bin_centers from r [um] to two_theta and then to q [1/px] (reciprocal space)
    two_theta = radius_to_twotheta(Ldet, qp_)
    q = twotheta_to_q(two_theta, lambda_)    
    if plot_:
        if  show_pixel: 
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            #ax2 = ax1.twiny()        
            ax1.semilogy(qp, iq, '-o')
            #ax1.semilogy(q,  iq , '-o')
            
            ax1.set_xlabel('q (pixel)')             
            #ax1.set_xlabel('q ('r'$\AA^{-1}$)')
            #ax2.cla()
            ax1.set_ylabel('I(q)')
            title = ax1.set_title('uid= %s--Circular Average'%uid)      
            
        else:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            ax1.semilogy(q,  iq , '-o') 
            ax1.set_xlabel('q ('r'$\AA^{-1}$)')        
            ax1.set_ylabel('I(q)')
            title = ax1.set_title('uid= %s--Circular Average'%uid)     
            ax2=None                     
        if 'xlim' in kwargs.keys():
            ax1.set_xlim(    kwargs['xlim']  )    
            x1,x2 =  kwargs['xlim']
            w = np.where( (q >=x1 )&( q<=x2) )[0] 
        if 'ylim' in kwargs.keys():
            ax1.set_ylim(    kwargs['ylim']  )       
          
        title.set_y(1.1)
        fig.subplots_adjust(top=0.85)
        path = pargs['path']
        fp = path + '%s_q_Iq'%uid  + '.png'  
        fig.savefig( fp, dpi=fig.dpi)
    if save:
        path = pargs['path']
        save_lists(  [q, iq], label=['q_A-1', 'Iq'],  filename='%s_q_Iq.csv'%uid, path= path  )        
    return  qp, iq, q

 
 
def plot_circular_average( qp, iq, q,  pargs, show_pixel= False, loglog=False, 
                          save=True,return_fig=False, *argv,**kwargs):
    
    if RUN_GUI:
        fig = Figure()
        ax1 = fig.add_subplot(111)
    else:
        fig, ax1 = plt.subplots()

    uid = pargs['uid']

    if  show_pixel:  
        if loglog:
            ax1.loglog(qp, iq, '-o')
        else:    
            ax1.semilogy(qp, iq, '-o')
        ax1.set_xlabel('q (pixel)')  
        ax1.set_ylabel('I(q)')
        title = ax1.set_title('%s_Circular Average'%uid)  
    else:
        if loglog:
            ax1.loglog(qp, iq, '-o')
        else:            
            ax1.semilogy(q,  iq , '-o') 
        ax1.set_xlabel('q ('r'$\AA^{-1}$)')        
        ax1.set_ylabel('I(q)')
        title = ax1.set_title('%s_Circular Average'%uid)     
        ax2=None 
    if 'xlim' in kwargs.keys():
        xlim =  kwargs['xlim']
    else:
        xlim=[q.min(), q.max()]
    if 'ylim' in kwargs.keys():
        ylim =  kwargs['ylim']
    else:
        ylim=[iq.min(), iq.max()]        
        
    ax1.set_xlim(   xlim  )  
    ax1.set_ylim(   ylim  ) 

    title.set_y(1.1)
    fig.subplots_adjust(top=0.85)
    if save:
        path = pargs['path']
        fp = path + '%s_q_Iq'%uid  + '.png'  
        fig.savefig( fp, dpi=fig.dpi)
    if return_fig:
        return fig        
