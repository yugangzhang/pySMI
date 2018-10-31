from pySMI.smi_libs import *
#from tqdm import *
from pySMI.smi_libs import  ( colors,  markers )
from scipy.special import erf

from skimage.filters import  prewitt
from skimage.draw import line_aa, line, polygon, ellipse, circle

from modest_image import ModestImage, imshow
import matplotlib.cm as mcm
import copy, scipy 
import PIL    



def angulars(edges, center, shape):
    """
    Draw annual (angluar-shaped) shaped regions of interest.
    Each ring will be labeled with an integer. Regions outside any ring will
    be filled with zeros.
    Parameters
    ----------
    edges: list
        giving the inner and outer angle in unit of radians
        e.g., [(1, 2), (11, 12), (21, 22)]
    center: tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges.
    """
    edges = np.atleast_2d(np.asarray(edges)).ravel() 
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each angular")
    if not np.all( np.diff(edges) > 0):         
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each angular from "
                         "r=0 outward")

    angle_val = utils.angle_grid( center,  shape) .ravel()        
     
    return _make_roi(angle_val, edges, shape)


def _make_roi(coords, edges, shape):
    """ Helper function to create ring rois and bar rois
    Parameters
    ----------
    coords : array
        shape is image shape
    edges : list
        List of tuples of inner (left or top) and outer (right or bottom)
        edges of each roi.
        e.g., edges=[(1, 2), (11, 12), (21, 22)]
    shape : tuple
        Shape of the image in which to create the ROIs
        e.g., shape=(512, 512)
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are
        specified in `edges`.
        Has shape=`image shape`
    """
    label_array = np.digitize(coords, edges, right=False)
    # Even elements of label_array are in the space between rings.
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    return label_array.reshape(shape)




def get_angular_mask( mask,  inner_angle= 0, outer_angle = 360, width = None, edges = None,
                     num_angles = 12, center = None, dpix=[1,1], flow_geometry=False    ):
     
    ''' 
    mask: 2D-array 
    inner_angle # the starting angle in unit of degree
    outer_angle #  the ending angle in unit of degree
    width       # width of each angle, in degree, default is None, there is no gap between the neighbour angle ROI
    edges: default, None. otherwise, give a customized angle edges
    num_angles    # number of angles
     
    center: the beam center in pixel    
    dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
    flow_geometry: if True, the angle should be between 0 and 180. the map will be a center inverse symmetry
    
    Returns
    -------
    ang_mask: a ring mask, np.array
    ang_center: ang in unit of degree
    ang_val: ang edges in degree
    
    '''
    
    #center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    
    #spacing =  (outer_radius - inner_radius)/(num_rings-1) - 2    # spacing between rings
    #inner_angle,outer_angle = np.radians(inner_angle),  np.radians(outer_angle)
    
    #if edges is  None:
    #    ang_center =   np.linspace( inner_angle,outer_angle, num_angles )  
    #    edges = np.zeros( [ len(ang_center), 2] )
    #    if width is None:
    #        width = ( -inner_angle + outer_angle)/ float( num_angles -1 + 1e-10 )
    #    else:
    #        width =  np.radians( width )
    #    edges[:,0],edges[:,1] = ang_center - width/2, ang_center + width/2 
        

    if flow_geometry:
        if edges is  None:
            if inner_angle<0:
                print('In this flow_geometry, the inner_angle should be larger than 0')
            if outer_angle >180:
                print('In this flow_geometry, the out_angle should be smaller than 180')
            

    if edges is None:
        if num_angles!=1:
            spacing =  (outer_angle - inner_angle - num_angles* width )/(num_angles-1)      # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_angle, width, spacing, num_angles) 

    #print (edges)
    angs = angulars( np.radians( edges ), center, mask.shape)    
    ang_center = np.average(edges, axis=1)        
    ang_mask = angs*mask
    ang_mask = np.array(ang_mask, dtype=int)
        
    if flow_geometry:        
        outer_angle -= 180
        inner_angle -= 180 
        edges2 = roi.ring_edges(inner_angle, width, spacing, num_angles)
        #print (edges)
        angs2 = angulars( np.radians( edges2 ), center, mask.shape)
        ang_mask2 = angs2*mask
        ang_mask2 = np.array(ang_mask2, dtype=int)        
        ang_mask +=  ang_mask2    
    else:
        for i, (al, ah) in enumerate( edges ):
            if al<=-180. and ah >-180:
                #print(i+1, al,ah)
                edge3 = np.array([  [ al + 360, 180 ]  ])       
                ang3 = angulars( np.radians( edge3 ), center, mask.shape) * mask
                w = np.ravel(  ang3 )==1
                #print(w)
                np.ravel( ang_mask )[w] = i+1
                
    
    labels, indices = roi.extract_label_indices(ang_mask)
    nopr = np.bincount( np.array(labels, dtype=int) )[1:]
   
    if len( np.where( nopr ==0 )[0] !=0):
        #print (nopr)
        print ("Some angs contain zero pixels. Please redefine the edges.")      
    return ang_mask, ang_center, edges







def get_goodlist( lists, good_pattern_list):
    '''YG Dev Octo 2018@SMI find a subset of a list by  good pattern
    Input:
        lists: list, the list for searching
        good_pattern_list: list, the list contains good patterns
    Output:
        return the good list contains pattern defined in good_pattern_list
    
    '''
    gl = []
    for l in lists:
        for g in  good_pattern_list:
            if g in l:
                gl.append( l ) 
    return gl



def recover_img_from_iq(  qp, iq, center, mask):
    '''YG. develop at CHX, 2017 July 18,
    Recover image a circular average
    '''        
    norm = get_pixelist_interp_iq( qp,  iq, np.ones_like(mask), center)
    img_ = norm.reshape( mask.shape)*mask
    return img_

def get_pixelist_interp_iq( qp, iq, ring_mask, center):
    
    qind, pixelist = roi.extract_label_indices(  ring_mask  )
    #pixely = pixelist%FD.md['nrows'] -center[1]  
    #pixelx = pixelist//FD.md['nrows'] - center[0]
    
    pixely = pixelist%ring_mask.shape[1] -center[1]  
    pixelx = pixelist//ring_mask.shape[1]  - center[0]
    
    r= np.hypot(pixelx, pixely)              #leave as float.
    #r= np.int_( np.hypot(pixelx, pixely)  +0.5  ) + 0.5  
    return np.interp( r, qp, iq ) 



def sum_array_withNan( array,  axis=0, mask=None):
    '''YG. Jan 23, 2018
       Average array invovling np.nan along axis       
        
       Input:
           array: ND array, actually should be oneD or twoD at this stage..TODOLIST for ND
           axis: the average axis
           mask: bool, same shape as array, if None, will mask all the nan values 
       Output:
           avg: averaged array along axis
    '''
    shape = array.shape
    if mask is None:
        mask = np.isnan(array)
        #mask = np.ma.masked_invalid(array).mask 
    array_ = np.ma.masked_array(array, mask=mask) 
    try:
        sums = np.array( np.ma.sum( array_[:,:], axis= axis ) )
    except:
        sums = np.array( np.ma.sum( array_[:], axis= axis ) )
        
    #cts = np.sum(~mask,axis=axis)
    #print(cts)
    return sums 


def average_array_withNan( array,  axis=0, mask=None):
    '''YG. Jan 23, 2018
       Average array invovling np.nan along axis       
        
       Input:
           array: ND array, actually should be oneD or twoD at this stage..TODOLIST for ND
           axis: the average axis
           mask: bool, same shape as array, if None, will mask all the nan values 
       Output:
           avg: averaged array along axis
    '''
    shape = array.shape
    if mask is None:
        mask = np.isnan(array)
        #mask = np.ma.masked_invalid(array).mask 
    array_ = np.ma.masked_array(array, mask=mask) 
    try:
        sums = np.array( np.ma.sum( array_[:,:], axis= axis ) )
    except:
        sums = np.array( np.ma.sum( array_[:], axis= axis ) )
        
    cts = np.sum(~mask,axis=axis)
    #print(cts)
    return sums/cts

def deviation_array_withNan( array,  axis=0, mask=None):
    '''YG. Jan 23, 2018
       Get the deviation of array invovling np.nan along axis       
        
       Input:
           array: ND array
           axis: the average axis
           mask: bool, same shape as array, if None, will mask all the nan values 
       Output:
           dev: the deviation of array along axis
    '''
    avg2 = average_array_withNan( array**2, axis = axis, mask = mask )
    avg = average_array_withNan( array, axis = axis, mask = mask )
    return  np.sqrt( avg2 - avg**2 )



def find_index( x,x0,tolerance= None):
    '''YG Octo 16,2017 copied from SAXS
     find index of x0 in x
     #find the position of P in a list (plist) with tolerance
    '''
 
    N=len(x)
    i=0
    if x0 > max(x):
        position= len(x) -1
    elif x0<min(x):
         position=0
    else:
         position = np.argmin( np.abs( x - x0 ) )
    return position


 


def show_data_series(  sample_list, center=None, w= 50, vmin=10, vmax= None, figsize=[10,16],
                    save=False, path=None, filename=None, logs=True,aspect=1, extent=None,
                     rotate=False, upside_down= True, verbose=False,  show_ticks =False,
                     data_dict=None,  extent_dict=None, sx=None):
    '''
    sample_list: list of str, a list of filename (each filename will be full path)
    center: the center of iamge (or direct beam pixel)
    w: the ROI half size in pixel
    vmin: the min intensity value for plot
    vmax: if None, will be max intensity value of the ROI  
    figsize: size of the plot (in inch)
    
    '''
    
    if center is not None:
        cy,cx = center
    if data_dict is None:    
        infs = sample_list
        #infs = sorted(sample_list)
        N = len(infs)
    else:
        ks = list( data_dict.keys() )
        N = len(  ks )
    if sx is None:
        sx = int( np.sqrt(N))
    sy = int( np.ceil( N/sx ) )
    
    fig = plt.figure(   figsize =figsize  ) 
    if filename is not None:
        plt.title(filename, fontsize= 14, y =0.98)
        plt.axis('off') 
    for i in range( N ):
        #print(i)
        ax = fig.add_subplot( sx, sy, i+1)
        if verbose:
            print(i, infs[i] )
        if data_dict is None:
            d = np.array(  PIL.Image.open( infs[i] ).convert('I') )
        else:
            k = ks[i]
            d = data_dict[ k ]
        if extent_dict is None:
            extenti=extent
        else:
            extenti=extent_dict[k]            
        if center is not None:
            d = d[ cy-w:cy+w, cx-w:cx+w   ]
        if rotate:
            d = d.T
        if upside_down:
            d = d[::-1]
        if vmax is None:
            vmax= np.max(d)
        #pritn(vmax)
        vmin= vmin         
        show_img( (d ), logs = logs, show_colorbar= False,show_ticks =show_ticks,
                 ax= [fig, ax], image_name= '%02d'%(i+1), cmap = cmap_vge_hdr, 
                 vmin= vmin, vmax= vmax,  
                 extent=extenti,
                aspect=aspect, save=False, path= None)
    if save:
        fig.tight_layout()
        fig.savefig( path +  filename +  '.png' )
   



def get_data_sub_bkg(  sam, bk, filename_dict, 
                     exclude_sam_index=None, exclude_bk_index=None,  scale=1, opt='avg'):
    '''Dev April 8, 2018 Average/Sumup images and substract backgound
    sam: sample name (should be in the keys of filename_dict)
    bk:  background sample name (should be in the keys of filename_dict)
    filename_dict: a dictionary containing key as sample name and value as the correponding filename list
    exclude_sam_index: if not None, should be list of bad sample spectra numbers (numbers start from 1 not 0)
    exclude_bk_index: if not None, should be list of bad background spectra numbers (numbers start from 1 not 0)
    opt: support avg or sum
    return:  sam_intensity - bk_intensity * scale
    '''
    infbk = sorted(filename_dict[bk]) 
    N = len(infbk)
    if exclude_bk_index is not None:
        infbk_ = []
        for i in range(N):
            if i not in (np.array(exclude_bk_index) -1):
                infbk_.append(  infbk[i] )
    else:
        infbk_ = infbk
    Nbk = float( len(infbk_)  )  
    avgbk = images_opt( infbk_, opt=opt, verbose= True )
    
    infsam = sorted(filename_dict[sam]) 
    N = len(infsam)
    if exclude_sam_index is not None:
        infsam_ = []
        for i in range(N):
            if i not in (np.array(exclude_sam_index) -1):
                infsam_.append(  infsam[i] )
    else:
        infsam_ = infsam
    avgsam = images_opt( infsam_, opt=opt, verbose= True )
    Nsam = len(infsam_)    
    
    return avgsam - avgbk * scale * Nsam/Nbk



def create_ring_mask( shape, r1, r2, center, mask=None):
    '''YG. Sep 20, 2017 Develop@CHX 
    Create 2D ring mask
    input:
        shape: two integer number  list, mask shape, e.g., [100,100]
        r1: the inner radius
        r2: the outer radius
        center: two integer number  list, [cx,cy], ring center, e.g., [30,50]
    output:
        2D numpy array, 0,1 type
    '''

    m = np.zeros( shape, dtype= bool) 
    rr,cc = circle(  center[1], center[0], r2, shape=shape  )
    m[rr,cc] = 1
    rr,cc = circle(  center[1], center[0], r1,shape=shape  )
    m[rr,cc] = 0 
    if mask is not None:
        m += mask
    return m








def images_opt( infiles, opt='sum',  verbose=False ):
    '''Do average/sum of images by giving infiles
    Input:
        infiles: list, a list of filename with full path        
        verbose: if True, print how many files to be processed.
    Output:
        return average image, float array
        
    '''
    if opt=='sum':
        return  sum_images( infiles,  verbose )
    elif opt=='avg':
        return average_images( infiles,  verbose )
    else:
        print('This func supports sum or avg .' )
    
    

def average_images( infiles,  verbose=False ):
    '''Do average of images by giving infiles
    Input:
        infiles: list, a list of filename with full path        
        verbose: if True, print how many files to be processed.
    Output:
        return average image, float array
        
    '''
    if verbose:
        print("There will be %s files to be averaged."%len(infiles) )
    for i, inf in enumerate(infiles): 
        if i==0:
            img0 = np.array(  PIL.Image.open( inf    ).convert('I') )
            shapes = img0.shape
            avg = np.zeros(shapes, dtype=np.float)
        avg += np.array(  PIL.Image.open( inf    ).convert('I') )
    return avg/(i+1)
    
    
def sum_images( infiles,  verbose=False ):
    '''Do sum of images by giving infiles
    Input:
        infiles: list, a list of filename with full path        
        verbose: if True, print how many files to be processed.
    Output:
        return sum image, int array
        
    '''
    if verbose:
        print("There will be %s files to be summed up."%len(infiles) )
    
    for i, inf in enumerate(infiles): 
        if i==0:
            img0 = np.array(  PIL.Image.open( inf    ).convert('I') )
            shapes = img0.shape
            avg = np.zeros(shapes, dtype=np.float)
        avg += np.array(  PIL.Image.open( inf    ).convert('I') )
    return avg    

 
 
    

    
def get_phi_spacing( width_overlap_pixel):
    '''get phi spacing by giving overlap pixel width for WAXS'''
    return 7 - np.degrees( np.arctan2( width_overlap_pixel * 172 * 10**(-3), 275 ) )
 

def get_pixel_overlap(phi_spacing):
    '''get  overlap pixel width by giving phi spacing in degrees'''
    #return 195  -  np.tan(np.radians(phi_spacing))/ (172 * 10**(-3)) * 275 
    ps = np.tan(  np.radians( 7 - phi_spacing))/ (172 * 10**(-3)) * 275 
    return int( np.ceil(  ps ) )
           






def save_array_to_tiff(array, filename ):
    #d = np.array(  PIL.Image.open( fp    ).convert('I') )
    img = PIL.Image.fromarray(array)
    #outp = ResDir + inf.split('/')[-1][:-4] + '_s.tif'
    img.save(  filename )   
    print( 'The data is save to: %s.'%(filename ))
    

def sgolay2d( z, window_size, order, derivative=None):
    """YG Octo 16, 2017
    Modified from http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    Procedure for sg2D:
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter#Two-dimensional_convolution_coefficients
    
        Two-dimensional smoothing and differentiation can also be applied to tables of data values, such as intensity values in a photographic image which is composed of a rectangular grid of pixels.[16] [17] The trick is to transform part of the table into a row by a simple ordering of the indices of the pixels. Whereas the one-dimensional filter coefficients are found by fitting a polynomial in the subsidiary variable, z to a set of m data points, the two-dimensional coefficients are found by fitting a polynomial in subsidiary variables v and w to a set of m × m data points. The following example, for a bicubic polynomial and m = 5, illustrates the process, which parallels the process for the one dimensional case, above.[18]

    enter image description here enter image description here

    The square of 25 data values, d1 − d25

    enter image description here

    becomes a vector when the rows are placed one after another.

    enter image description here

    The Jacobian has 10 columns, one for each of the parameters a00 − a03 and 25 rows, one for each pair of v and w values. Each row has the form

    enter image description here

    The convolution coefficients are calculated as

    enter image description here

    The first row of C contains 25 convolution coefficients which can be multiplied with the 25 data values to provide a smoothed value for the central data point (13) of the 25.

    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')
    
    
    
    
def extract_data_from_file(  filename, filepath, good_line_pattern, good_cols=None, labels=None,):
    '''YG Develop Octo 17, 2018 
        Extract data from a file
    Input:
        filename: str, filename of the data
        filepath: str, path of the data
        good_line_pattern: str, data will be extract below this good_line_pattern
        good_cols: list of integer, good index of cols
        lables: the label of the good_cols
        #save: False, if True will save the data into a csv file with filename appending csv ??
    Return:
        a pds.dataframe
    Example:
    filepath =  '/XF11ID/analysis/2017_3/lwiegart/Link_files/Exports/'
    filename = 'ANPES2 15-10-17 16-31-11-84Exported.txt'    
    good_cols = [ 1,2,4,6,8,10 ]
    labels = [  'time', 'temperature', 'force', 'distance', 'stress', 'strain'  ]
    good_line_pattern = "Index\tX\tY\tX\tY\tX\tY" 
    df =  extract_data_from_file(  filename, filepath, good_line_pattern, good_cols, labels)
    '''
    import pandas as pds
    with open( filepath + filename, 'r' ) as fin:
        p=fin.readlines()
        di = 1e20                
        for i, line in enumerate(p):
            if good_line_pattern in line:                
                di = i
            if i == di+1:
                els = line.split()  
                if good_cols is  None:
                    data = np.array( els, dtype=float  )
                else:
                    data = np.array( [els[j] for j in good_cols], dtype=float  )
            elif i > di:
                try:                    
                    els = line.split() 
                    if good_cols is  None:
                        temp = np.array( els, dtype=float  )
                    else:
                        temp=  np.array( [els[j] for j in good_cols], dtype=float  ) 
                    data=np.vstack( (data,temp))
                except:
                    pass
        if labels is None:
            labels = np.arange(data.shape[1])
        df = pds.DataFrame( data, index= np.arange(data.shape[0]), columns= labels  )    
    return df
    
    
    
def get_print_uids( start_time, stop_time):
    '''YG. Octo 3, 2017@CHX
    Get full uids and print uid plus Measurement contents by giving start_time, stop_time
    
    
    '''
    
    hdrs = list( db(start_time= start_time, stop_time = stop_time) )
    uids = np.zeros( len(hdrs),dtype=object)
    sids = np.zeros( len(hdrs), dtype=object)
    n=0
    for  i in range(len(hdrs)):
        uid = hdrs[i]['start']['uid'][:6]
        sid = hdrs[i]['start']['scan_id']
        uids[n]=uid
        sids[n]=sid
        n +=1
        try:
            m = hdrs[i]['start']['Measurement']
        except:
            m=''
        print( "   uid = '%s' #(scan num: %s (Measurement: %s        "%(uid,sid,m) )
    return uids, sids
    
    
def get_last_uids( n=-1 ):
    '''YG Sep 26, 2017
    A Convinient function to copy uid to jupyter for analysis'''
    uid = db[n]['start']['uid'][:6]
    sid = db[n]['start']['scan_id']
    m = db[n]['start']['Measurement']
    return "   uid = '%s' #(scan num: %s (Measurement: %s        "%(uid,sid,m) 


    
def get_base_all_filenames( inDir, base_filename_cut_length = -7  ):
    '''YG Sep 26, 2017
    Get base filenames and their related all filenames
    Input:
      inDir, str, input data dir
       base_filename_cut_length: to which length the base name is unique
    Output:
      dict: keys,  base filename
            vales, all realted filename
    '''
    from os import listdir
    from os.path import isfile, join
    tifs = np.array( [f for f in listdir(inDir) if isfile(join(inDir, f))] )
    tifsc = list(tifs.copy())    
    utifs = np.sort( np.unique( np.array([ f[:base_filename_cut_length] for f in tifs] ) )  )[::-1]
    files = {}
    for uf in utifs:     
        files[uf] = []
        i = 0
        reName = []
        for i in range(len(tifsc)): 
            if uf in tifsc[i]: 
                files[uf].append( inDir + tifsc[i] )            
                reName.append(tifsc[i])
        for fn in reName:
            tifsc.remove(fn)
    return files

    
def create_ring_mask2( shape, r1, r2, center, mask=None):
    '''YG. Sep 20, 2017 Develop@CHX 
    Create 2D ring mask
    input:
        shape: two integer number  list, mask shape, e.g., [100,100]
        r1: the inner radius
        r2: the outer radius
        center: two integer number  list, [cx,cy], ring center, e.g., [30,50]
    output:
        2D numpy array, 0,1 type
    '''

    m = np.zeros( shape, dtype= bool) 
    rr,cc = circle(  center[1], center[0], r2, shape=shape  )
    m[rr,cc] = True
    rr,cc = circle(  center[1], center[0], r1,shape=shape  )
    m[rr,cc] = False
    if mask is not None:
        m *= np.array(mask,dtype=bool)
    return m

def get_image_edge(img):
    '''
    Y.G. Developed at Sep 8, 2017 @CHX
    Get sharp edges of an image
    img: two-D array, e.g., a roi mask
    '''
    edg_ = prewitt(img/1.0)
    edg = np.zeros_like(edg_)
    w = np.where(edg_ > 1e-10)
    edg[w] = img[w]
    edg[np.where(edg==0)] = 1
    return edg      

def get_image_with_roi( img, roi_mask, scale_factor = 2):
    '''
    Y.G. Developed at Sep 8, 2017 @CHX
    Get image with edges of roi_mask by doing
        i)  get edges of roi_mask by function get_image_edge
        ii) scale img at region of interest (ROI) by scale_factor
    img: two-D array for image
    roi_mask: two-D array for ROI
    scale_factor: scaling factor of ROI in image
    '''
    edg = get_image_edge( roi_mask )
    img_ = img.copy()
    w = np.where(roi_mask)
    img_[w] = img[w] *  scale_factor
    return img_ * edg 





def get_today_date( ):
    from time import gmtime, strftime
    return strftime("%m-%d-%Y", gmtime() )


def move_beamstop( mask, xshift, yshift ):
    '''Y.G. Developed at July 18, 2017 @CHX
    Create new mask by shift the old one with xshift, yshift
    Input
    ---
    mask: 2D numpy array, 0 for bad pixels, 1 for good pixels
    xshift, integer, shift value along x direction
    yshift, integer, shift value along y direction
    
    Output
    ---
    mask, 2D numpy array,
    '''
    m = np.ones_like(mask)
    W,H = mask.shape
    w = np.where(mask==0)
    nx, ny = w[0]+ int(yshift), w[1]+ int(xshift    )
    gw = np.where( (nx >= 0) & (nx<W) & (ny >= 0) & (ny<H) )
    nx = nx[  gw  ]
    ny = ny[  gw  ] 
    m[ nx,ny ] = 0
    return m



def validate_uid(uid):
    '''check uid whether be able to load data'''
    try:
        sud = get_sid_filenames(db[uid])
        print(sud)
        md = get_meta_data( uid )
        imgs = load_data( uid, md['detector'], reverse= True  )
        print(imgs)
        return 1
    except:
        print("Can't load this uid=%s!"%uid)
        return 0

def validate_uid_dict(  uid_dict  ):
    ''' Y.G. developed July 17, 2017 @CHX
    Check each uid in a dict can load data or not
    uids: dict, val: meaningful decription, key: a list of uids 
 
    ''' 
    badn = 0
    badlist=[]
    for k in list(uids.keys()):
        for uid in uids[k]:  
            flag = validate_uid(uid)
            if not flag:
                badn += 1
                badlist.append( uid )
    print( 'There are %s bad uids:%s in this uid_dict.'%(badn, badlist))

def get_mass_center_one_roi(FD, roi_mask, roi_ind):
    '''Get the mass center (in pixel unit) of one roi in a time series FD
    FD: handler for a compressed time series
    roi_mask: the roi array
    roi_ind: the interest index of the roi  
    
    '''
    import scipy
    m =  (roi_mask == roi_ind)
    cx, cy = np.zeros(    int( ( FD.end - FD.beg)/1 )  ), np.zeros(    int( ( FD.end - FD.beg)/1 )  ) 
    n =0 
    for  i in tqdm(range( FD.beg, FD.end, 1  ), desc= 'Get mass center of one ROI of each frame' ):
        img = FD.rdframe(i) * m
        c = scipy.ndimage.measurements.center_of_mass(img)
        cx[n], cy[n] = int(c[0]), int(c[1])
        n +=1
    return cx,cy



def get_current_pipeline_filename(NOTEBOOK_FULL_PATH):
    '''Y.G. April 25, 2017
       Get the current running pipeline filename and path 
       Assume the piple is located in /XF11ID/
       Return, path and filename
    '''
    from IPython.core.magics.display import Javascript
    if False:
        Javascript( '''
        var nb = IPython.notebook;
        var kernel = IPython.notebook.kernel;
        var command = "NOTEBOOK_FULL_PATH = '" + nb.base_url + nb.notebook_path + "'";
        kernel.execute(command);
        ''' ) 
        print(NOTEBOOK_FULL_PATH)
    filename   = NOTEBOOK_FULL_PATH.split('/')[-1]
    path = '/XF11ID/'
    for s in NOTEBOOK_FULL_PATH.split('/')[3:-1]:
        path +=    s  + '/'
    return path, filename    
        
def get_current_pipeline_fullpath(NOTEBOOK_FULL_PATH):
    '''Y.G. April 25, 2017
       Get the current running pipeline full filepath 
       Assume the piple is located in /XF11ID/
       Return, the fullpath (path + filename)
    '''    
    p,f = get_current_pipeline_filename(NOTEBOOK_FULL_PATH)
    return p + f
    
def save_current_pipeline(NOTEBOOK_FULL_PATH, outDir):
    '''Y.G. April 25, 2017
       Save the current running pipeline to outDir
       The save pipeline should be the snapshot of the current state.    
    '''

    import  shutil
    path, fp = get_current_pipeline_filename(NOTEBOOK_FULL_PATH)
    shutil.copyfile(  path + fp, outDir + fp    ) 
    
    print('This pipeline: %s is saved in %s.'%(fp, outDir))
    
    

def plot_g1( taus, g2, g2_fit_paras, qr=None, ylim=[0,1], title=''):
    '''Dev Apr 19, 2017,
       Plot one-time correlation, giving taus, g2, g2_fit'''
    noqs = g2.shape[1]
    fig,ax=plt.subplots()
    if qr is None:
        qr = np.arange(noqs)
    for i in range(noqs):
        b =  g2_fit_paras['baseline'][i]
        beta = g2_fit_paras['beta'][i]
        y= np.sqrt( np.abs(g2[1:,i] - b)/beta )
        plot1D( x =  taus[1:], y= y, ax=ax, legend= 'q=%s'%qr[i],  ls='-', lw=2, 
           m=markers[i], c= colors[i], title=title, ylim=ylim,
           logx=True,  legend_size= 8 )
    ax.set_ylabel( r"$g_1$" + '(' + r'$\tau$' + ')' ) 
    ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)   
    return ax 



def filter_roi_mask( filter_dict, roi_mask, avg_img, filter_type= 'ylim' ):
    '''Remove bad pixels in roi_mask. The bad pixel is defined by the filter_dict, 
       if filter_type ='ylim', the filter_dict wit key as q and each value gives a high and low limit thresholds. The value of the pixels in avg_img above or below the limit are considered as bad pixels.
       if filter_type='badpix': the filter_dict wit key as q and each value gives a list of bad pixel.
       
    avg_img, the averaged image
    roi_mask: two-d array, the same shape as image, the roi mask, value is integer, e.g.,   1 ,2 ,...
    filter_dict: keys, as roi_mask integer, value, by default is [None,None], is the limit,
                      example, {2:[4,5], 10:[0.1,1.1]}
    NOTE: first q = 1 (not 0)
    '''
    rm =   roi_mask.copy()
    rf =   np.ravel(rm)    
    for k in list(filter_dict.keys()):        
        pixel = roi.roi_pixel_values(avg_img, roi_mask, [k] )[0][0]
        #print(   np.max(pixel), np.min(pixel)   )
        if filter_type == 'ylim':
            xmin,xmax = filter_dict[k]
            badp =np.where( (pixel>= xmax) | ( pixel <= xmin) )[0]
        else:
            badp = filter_dict[k]
        if len(badp)!=0:    
            pls = np.where([rf==k])[1]
            rf[ pls[badp] ] = 0       
    return rm

        
##
#Dev at March 31 for create Eiger chip mask
def create_chip_edges_mask( det='1M' ):
    ''' Create a chip edge mask for Eiger detector
    
    '''
    if det == '1M':
        shape = [1065, 1030]
        w = 4
        mask = np.ones( shape , dtype = np.int32)
        cx = [ 1030//4 *i for i in range(1,4) ]
        #cy = [ 1065//4 *i for i in range(1,4) ]
        cy =  [808, 257 ]
        #print (cx, cy )
        for c in cx:
            mask[:, c-w//2:c+w//2  ] = 0        
        for c in cy:
            mask[ c-w//2:c+w//2, :  ] = 0        
    
    return mask
     
def create_ellipse_donut(  cx, cy , wx_inner, wy_inner, wx_outer, wy_outer, roi_mask, gap=0):
    Nmax = np.max( np.unique( roi_mask ) )
    rr1, cc1 = ellipse( cy,cx,  wy_inner, wx_inner  )    
    rr2, cc2 = ellipse( cy, cx,  wy_inner + gap, wx_inner +gap ) 
    rr3, cc3 = ellipse( cy, cx,  wy_outer,wx_outer ) 
    roi_mask[rr3,cc3] = 2 + Nmax
    roi_mask[rr2,cc2] = 0
    roi_mask[rr1,cc1] = 1 + Nmax
    return roi_mask
    
def create_box( cx, cy, wx, wy, roi_mask):
    Nmax = np.max( np.unique( roi_mask ) )
    for i, [cx_,cy_] in enumerate(list( zip( cx,cy  ))):  #create boxes
        x = np.array( [ cx_-wx, cx_+wx,  cx_+wx, cx_-wx])  
        y = np.array( [ cy_-wy, cy_-wy, cy_+wy, cy_+wy])
        rr, cc = polygon( y,x)         
        roi_mask[rr,cc] = i +1 + Nmax
    return roi_mask








    
    
def create_user_folder( CYCLE, username = None ):
    '''
    Crate a folder for saving user data analysis result
    Input:
        CYCLE: run cycle
        username: if None, get username from the jupyter username
    Return:
        Created folder name
    '''
    if username is None:
        username = getpass.getuser() 
    data_dir0 = os.path.join('/XF11ID/analysis/', CYCLE, username, 'Results/')
    ##Or define data_dir here, e.g.,#data_dir = '/XF11ID/analysis/2016_2/rheadric/test/'
    os.makedirs(data_dir0, exist_ok=True)
    print('Results from this analysis will be stashed in the directory %s' % data_dir0) 
    return data_dir0    
    
    
    
    
    
    
##################################
#########For dose analysis #######
##################################
def get_fra_num_by_dose( exp_dose, exp_time, att=1, dead_time =2 ):
    '''
    Calculate the frame number to be correlated by giving a X-ray exposure dose
    
    Paramters:
        exp_dose: a list, the exposed dose, e.g., in unit of exp_time(ms)*N(fram num)*att( attenuation)
        exp_time: float, the exposure time for a xpcs time sereies
        dead_time: dead time for the fast shutter reponse time, CHX = 2ms
    Return:
        noframes: the frame number to be correlated, exp_dose/( exp_time + dead_time )  
    e.g.,
    
    no_dose_fra = get_fra_num_by_dose(  exp_dose = [ 3.34* 20, 3.34*50, 3.34*100, 3.34*502, 3.34*505 ],
                                   exp_time = 1.34, dead_time = 2)
                                   
    --> no_dose_fra  will be array([ 20,  50, 100, 502, 504])     
    '''
    return np.int_(    np.array( exp_dose )/( exp_time + dead_time)/ att )


def get_multi_tau_lag_steps( fra_max, num_bufs = 8 ):
    '''
    Get taus in log steps ( a multi-taus defined taus ) for a time series with max frame number as fra_max
    Parameters:
        fra_max: integer, the maximun frame number          
        buf_num (default=8),               
    Return:
        taus_in_log, a list 
        
    e.g., 
    get_multi_tau_lag_steps(  20, 8   )  -->  array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16])
    
    '''        
    num_levels = int(np.log( fra_max/(num_bufs-1))/np.log(2) +1) +1
    tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)    
    return lag_steps[lag_steps < fra_max]
    


def get_series_g2_taus( fra_max_list, acq_time=1, max_fra_num=None, log_taus = True, 
                        num_bufs = 8):
    '''
    Get taus for dose dependent analysis
    Parameters:
        fra_max_list: a list, a lsit of largest available frame number        
        acq_time: acquistion time for each frame
        log_taus: if true, will use the multi-tau defined taus bu using buf_num (default=8),
               otherwise, use deltau =1        
    Return:
        tausd, a dict, with keys as taus_max_list items  
    e.g., 
    get_series_g2_taus( fra_max_list=[20,30,40], acq_time=1, max_fra_num=None, log_taus = True,  num_bufs = 8)
    --> 
    {20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16]),
     30: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28]),
     40: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 32])
    }
     
    '''
    tausd = {}
    for n in fra_max_list:
        if max_fra_num is not None:
            L = max_fra_num
        else:
            L = np.infty            
        if n>L:
            warnings.warn("Warning: the dose value is too large, and please" 
                          "check the maxium dose in this data set and give a smaller dose value."
                          "We will use the maxium dose of the data.") 
            n = L 
        if log_taus:
            lag_steps = get_multi_tau_lag_steps(n,  num_bufs)
        else:
            lag_steps = np.arange( n )
        tausd[n] = lag_steps * acq_time
    return tausd




def check_lost_metadata(md, Nimg=None, inc_x0 =None, inc_y0= None, pixelsize=7.5*10*(-5) ):
    '''Y.G. Dec 31, 2016, check lost metadata
    
    Parameter:
        md: dict, meta data dictionay
        Nimg: number of frames for this uid metadata
        inc_x0/y0: incident beam center x0/y0, if None, will over-write the md['beam_center_x/y']
        pixelsize: if md don't have ['x_pixel_size'], the pixelsize will add it
    Return:
        dpix: pixelsize, in mm
        lambda_: wavelegth of the X-rays in Angstroms
        exposuretime:  exposure time in sec
        timeperframe:  acquisition time is sec 
        center:  list, [x,y], incident beam center in pixel
     Will also update md    
    '''
    
    if 'number of images'  not in list(md.keys()):
        md['number of images']  = Nimg
    if 'x_pixel_size' not in list(md.keys()):
        md['x_pixel_size'] = 7.5000004e-05
    dpix = md['x_pixel_size'] * 1000.  #in mm, eiger 4m is 0.075 mm
    lambda_ =md['incident_wavelength']    # wavelegth of the X-rays in Angstroms
    try:
        Ldet = md['det_distance']
        if Ldet<=1000:
            Ldet *=1000
            md['det_distance'] = Ldet
    except:
        Ldet = md['detector_distance']
        if Ldet<=1000:
            Ldet *=1000
            md['detector_distance'] = Ldet        

    
    try:#try exp time from detector
        exposuretime= md['count_time']     #exposure time in sec        
    except:    
        exposuretime= md['cam_acquire_time']     #exposure time in sec
    try:#try acq time from detector
        acquisition_period = md['frame_time']  
    except:
        try:
            acquisition_period = md['acquire period']
        except: 
            uid = md['uid']
            acquisition_period = float( db[uid]['start']['acquire period'] )
    timeperframe = acquisition_period 
    if inc_x0 is not None:
        md['beam_center_x']= inc_y0
        print( 'The metadata: beam_center_x has been changed to %s.'%inc_y0)
    if inc_y0 is not None:
        md['beam_center_y']= inc_x0         
        print( 'The metadata: beam_center_y has been changed to %s.'%inc_x0)
    center = [  int(md['beam_center_x']),int( md['beam_center_y'] ) ]  #beam center [y,x] for python image
    center=[center[1], center[0]]
    
    return dpix, lambda_, Ldet, exposuretime, timeperframe, center


def combine_images( filenames, outputfile, outsize=(2000, 2400)):
    '''Y.G. Dec 31, 2016
    Combine images together to one image using PIL.Image
    Input:
        filenames: list, the images names to be combined
        outputfile: str, the filename to generate
        outsize: the combined image size
    Output:
        save a combined image file
    '''
    N = len( filenames)
    #nx = np.int( np.ceil( np.sqrt(N)) )
    #ny = np.int( np.ceil( N / float(nx)  ) )

    ny = np.int( np.ceil( np.sqrt(N)) )
    nx = np.int( np.ceil( N / float(ny)  ) )
    
    #print(nx,ny)
    result = Image.new("RGB", outsize, color=(255,255,255,0))    
    basewidth = int( outsize[0]/nx )     
    hsize = int( outsize[1]/ny )         
    for index, file in enumerate(filenames):
        path = os.path.expanduser(file)
        img = Image.open(path)
        bands = img.split()
        ratio = img.size[1]/ img.size[0]  #h/w        
        if hsize > basewidth * ratio:
            basewidth_ = basewidth 
            hsize_ = int( basewidth * ratio )
        else:
            basewidth_ =  int( hsize/ratio )
            hsize_ =  hsize 
        #print( index, file, basewidth, hsize )
        size = (basewidth_,hsize_)
        bands = [b.resize(size, Image.LINEAR) for b in bands]
        img = Image.merge('RGBA', bands)  
        x = index % nx * basewidth
        y = index // nx * hsize
        w, h = img.size
        #print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h  ))
    result.save( outputfile,quality=100, optimize=True )
    print( 'The combined image is saved as: %s'%outputfile)
    

def get_qval_dict( qr_center, qz_center=None, qval_dict = None,  multi_qr_for_one_qz= True,
                 one_qz_multi_qr = True):
    '''Y.G. Dec 27, 2016
    Map the roi label array with qr or (qr,qz) or (q//, q|-) values
    Parameters:
        qr_center: list, a list of qr
        qz_center: list, a list of qz, 
        multi_qr_for_one_qz: by default=True, 
            if one_qz_multi_qr:                
                one qz_center corresponds to  all qr_center, in other words, there are totally,  len(qr_center)* len(qz) qs
            else:
                one qr_center corresponds to  all qz_center, 
            else: one qr with one qz
        qval_dict: if not None, will append the new dict to the qval_dict
    Return:
        qval_dict, a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
         
    '''
    
    if qval_dict is None:
        qval_dict = {}
        maxN = 0
    else:
        maxN = np.max( list( qval_dict.keys() ) ) +1
        
    if qz_center is not None:
        if multi_qr_for_one_qz:
            if one_qz_multi_qr:
                for qzind in range( len( qz_center)):
                    for qrind in range( len( qr_center)):    
                        qval_dict[ maxN + qzind* len( qr_center) + qrind ] = np.array( [qr_center[qrind], qz_center[qzind]  ] )
            else:
                for qrind in range( len( qr_center)):
                    for qzind in range( len( qz_center)):    
                        qval_dict[ maxN + qrind* len( qz_center) + qzind ] = np.array( [qr_center[qrind], qz_center[qzind]  ] )
       
    
        else:
            for i, [qr, qz] in enumerate(zip( qr_center, qz_center)):     
                qval_dict[ maxN + i  ] = np.array( [ qr, qz  ] )            
    else:
        for qrind in range( len( qr_center)):    
            qval_dict[ maxN +  qrind ] = np.array( [ qr_center[qrind] ] )        
    return qval_dict   


def update_qval_dict(  qval_dict1, qval_dict2 ):
    ''' Y.G. Dec 31, 2016
    Update qval_dict1 with qval_dict2
    Input:
        qval_dict1, a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
        qval_dict2, a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
    Output:
        qval_dict, a dict,  with the same key as dict1, and all key in dict2 but which key plus max(dict1.keys())
    '''
    maxN = np.max( list( qval_dict1.keys() ) ) +1
    qval_dict = {}
    qval_dict.update( qval_dict1 )
    for k in list( qval_dict2.keys() ):
        qval_dict[k + maxN ] = qval_dict2[k]
    return qval_dict

def update_roi_mask(  roi_mask1, roi_mask2 ):
    ''' Y.G. Dec 31, 2016
    Update qval_dict1 with qval_dict2
    Input:
        roi_mask1, 2d-array, label array,  same shape as xpcs frame, 
        roi_mask2, 2d-array, label array,  same shape as xpcs frame,
    Output:
        roi_mask, 2d-array, label array,  same shape as xpcs frame, update roi_mask1 with roi_mask2
    '''
    roi_mask = roi_mask1.copy()
    w= np.where( roi_mask2 )   
    roi_mask[w]  = roi_mask2[w] + np.max( roi_mask ) 
    return roi_mask


def check_bad_uids(uids, mask, img_choice_N = 10, bad_uids_index = None ):
    '''Y.G. Dec 22, 2016
        Find bad uids by checking the average intensity by a selection of the number img_choice_N of frames for the uid. If the average intensity is zeros, the uid will be considered as bad uid.
        Parameters:
            uids: list, a list of uid
            mask: array, bool type numpy.array
            img_choice_N: random select number of the uid
            bad_uids_index: a list of known bad uid list, default is None
        Return:
            guids: list, good uids
            buids, list, bad uids    
    '''   
    import random
    buids = []
    guids = list( uids )
    #print( guids )
    if bad_uids_index is None:
        bad_uids_index = []
    for i, uid in enumerate(uids):
        #print( i, uid )
        if i not in bad_uids_index:
            detector = get_detector( db[uid ] )
            imgs = load_data( uid, detector  )
            img_samp_index = random.sample( range(len(imgs)), img_choice_N)
            imgsa = apply_mask( imgs, mask )
            avg_img =  get_avg_img( imgsa, img_samp_index, plot_ = False, uid =uid)
            if avg_img.max() == 0:
                buids.append( uid ) 
                guids.pop(  list( np.where( np.array(guids) == uid)[0] )[0]  )
                print( 'The bad uid is: %s'%uid )
        else:
            guids.pop(  list( np.where( np.array(guids) == uid)[0] )[0]  )
            buids.append( uid )
            print( 'The bad uid is: %s'%uid )            
    print( 'The total and bad uids number are %s and %s, repsectively.'%( len(uids), len(buids) ) )        
    return guids, buids          
    


def find_uids(start_time, stop_time ):
    '''Y.G. Dec 22, 2016
        A wrap funciton to find uids by giving start and end time
        Return:
        sids: list, scan id
        uids: list, uid with 8 character length
        fuids: list, uid with full length
    
    '''
    hdrs = db(start_time= start_time, stop_time = stop_time)
    try:
        print ('Totally %s uids are found.'%(len(list(hdrs))))
    except:
        pass
    sids=[]
    uids=[]
    fuids=[]
    for hdr in hdrs:
        s= get_sid_filenames( hdr)
        #print (s[1][:8])
        sids.append( s[0] )
        uids.append( s[1][:8] )
        fuids.append( s[1] )
    sids=sids[::-1]
    uids=uids[::-1]
    fuids=fuids[::-1]
    return np.array(sids), np.array(uids), np.array(fuids)


def ployfit( y, x=None, order = 20 ):
    '''
    fit data (one-d array) by a ploynominal function
    return the fitted one-d array
    '''
    if x is None:
        x = range(len(y))
    pol = np.polyfit(x, y, order)
    return np.polyval(pol, x)
    
def check_bad_data_points( data, fit=True, polyfit_order = 30, legend_size = 12,
                       plot=True, scale=1.0, good_start=None, good_end=None, path=None, return_ylim=False  ):
    '''
    data: 1D array
    scale: the scale of deviation
    fit: if True, use a ploynominal function to fit the imgsum, to get a mean-inten(array), then use the scale to get low and high threshold, it's good to remove bad frames/pixels on top of not-flatten curve
         else: use the mean (a value) of imgsum and scale to get low and high threshold, it's good to remove bad frames/pixels on top of  flatten curve
     
    '''
    if good_start is  None:
        good_start=0
    if good_end is None:
        good_end = len( data )
    bd1 = [i for i in range(0, good_start)]
    bd3 = [i for i in range(good_end,len( data ) )]
    
    d_ = data[good_start:good_end]
    
    if fit:
        pfit = ployfit( d_, order = polyfit_order)
        d = d_ - pfit 
    else:
        d = d_ 
        pfit = np.ones_like(d) * data.mean()
        
    ymin = d.mean()-scale *d.std()
    ymax =  d.mean()+scale *d.std() 
    
    if plot:
        fig = plt.figure( )            
        ax = fig.add_subplot(2,1,1 )             
        plot1D( d_, ax = ax, color='k', legend='data',legend_size=legend_size  )
        plot1D( pfit,ax=ax, color='b', legend='ploy-fit', title='Find Bad Points',legend_size=legend_size  )

        ax2 = fig.add_subplot(2,1,2 )      
        plot1D( d, ax = ax2,legend='difference',marker='s', color='b', )          

        #print('here')
        plot1D(x=[0,len(d_)], y=[ymin,ymin], ax = ax2, ls='--',lw= 3, marker='o', color='r', legend='low_thresh', legend_size=legend_size  )

        plot1D(x=[0,len(d_)], y=[ymax,ymax], ax = ax2 , ls='--', lw= 3,marker='o', color='r',legend='high_thresh',title='',legend_size=legend_size  )

        if path is not None:
            fp = path + '%s'%( uid ) + '_find_bad_points'  + '.png' 
            plt.savefig( fp, dpi=fig.dpi)         
    bd2=  list(   np.where( np.abs(d -d.mean()) > scale *d.std() )[0] + good_start )
     
    if return_ylim:
        return np.array( bd1 + bd2 + bd3 ), ymin, ymax,pfit
    else:
        return np.array( bd1 + bd2 + bd3 ), pfit
    
    
    

def get_bad_frame_list( imgsum, fit=True, polyfit_order = 30,legend_size = 12,
                       plot=True, scale=1.0, good_start=None, good_end=None, uid='uid',path=None,
                        
                      return_ylim=False):
    '''
    imgsum: the sum intensity of a time series
    scale: the scale of deviation
    fit: if True, use a ploynominal function to fit the imgsum, to get a mean-inten(array), then use the scale to get low and high threshold, it's good to remove bad frames/pixels on top of not-flatten curve
         else: use the mean (a value) of imgsum and scale to get low and high threshold, it's good to remove bad frames/pixels on top of  flatten curve
     
    '''
    if good_start is  None:
        good_start=0
    if good_end is None:
        good_end = len( imgsum )
    bd1 = [i for i in range(0, good_start)]
    bd3 = [i for i in range(good_end,len( imgsum ) )]
    
    imgsum_ = imgsum[good_start:good_end]
    
    if fit:
        pfit = ployfit( imgsum_, order = polyfit_order)
        data = imgsum_ - pfit 
    else:
        data = imgsum_ 
        pfit = np.ones_like(data) * data.mean()
        
    ymin = data.mean()-scale *data.std()
    ymax =  data.mean()+scale *data.std() 
    
    if plot:
        fig = plt.figure( )            
        ax = fig.add_subplot(2,1,1 )             
        plot1D( imgsum_, ax = ax, color='k', legend='data',legend_size=legend_size  )
        plot1D( pfit,ax=ax, color='b', legend='ploy-fit', title=uid + '_imgsum',legend_size=legend_size  )

        ax2 = fig.add_subplot(2,1,2 )      
        plot1D( data, ax = ax2,legend='difference',marker='s', color='b', )          

        #print('here')
        plot1D(x=[0,len(imgsum_)], y=[ymin,ymin], ax = ax2, ls='--',lw= 3, marker='o', color='r', legend='low_thresh', legend_size=legend_size  )

        plot1D(x=[0,len(imgsum_)], y=[ymax,ymax], ax = ax2 , ls='--', lw= 3,marker='o', color='r',legend='high_thresh',title='imgsum_to_find_bad_frame',legend_size=legend_size  )

        if path is not None:
            fp = path + '%s'%( uid ) + '_imgsum_analysis'  + '.png' 
            plt.savefig( fp, dpi=fig.dpi)            
        

        
    bd2=  list(   np.where( np.abs(data -data.mean()) > scale *data.std() )[0] + good_start )
     
    if return_ylim:
        return np.array( bd1 + bd2 + bd3 ), ymin, ymax
    else:
        return np.array( bd1 + bd2 + bd3 )

def save_dict_csv( mydict, filename, mode='w'):
    import csv
    with open(filename, mode) as csv_file:
        spamwriter = csv.writer(csv_file)        
        for key, value in mydict.items():  
            spamwriter.writerow([key, value])
            
            

def read_dict_csv( filename ): 
    import csv
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
    return mydict


def find_bad_pixels( FD, bad_frame_list, uid='uid'):
    bpx = []
    bpy=[]
    for n in bad_frame_list:
        if n>= FD.beg and n<=FD.end:
            f = FD.rdframe(n)
            w = np.where(  f == f.max())
            if len(w[0])==1:
                bpx.append(   w[0][0]  )
                bpy.append(   w[1][0]  )
    
    
    return trans_data_to_pd( [bpx,bpy], label=[ uid+'_x', uid +'_y' ], dtype='list')





def mask_exclude_badpixel( bp,  mask, uid ):
    
    for i in range( len(bp)):
        mask[ int( bp[bp.columns[0]][i] ), int( bp[bp.columns[1]][i] )]=0 
    return mask



def print_dict( dicts, keys=None):
    '''
    print keys: values in a dicts
    if keys is None: print all the keys
    '''
    if keys is None:
        keys = list( dicts.keys())
    for k in keys: 
        try:
            print('%s--> %s'%(k, dicts[k]) )
        except:
            pass
        
        
def get_meta_data( uid,*argv,**kwargs ):
    '''
    Y.G. Dev Dec 8, 2016
    
    Get metadata from a uid
    Parameters:
        uid: the unique data acquisition id
        kwargs: overwrite the meta data, for example 
            get_meta_data( uid = uid, sample = 'test') --> will overwrtie the meta's sample to test
    return:
        meta data of the uid: a dictionay
        with keys:
            detector            
            suid: the simple given uid
            uid: full uid
            filename: the full path of the data
            start_time: the data acquisition starting time in a human readable manner
        And all the input metadata
        
            
    
    '''
    import time    
    md ={}
    md['detector'] = get_detector( db[uid ] )    
    md['suid'] = uid  #short uid
    md['filename'] = get_sid_filenames(db[uid])[2][0]
    #print( md )        
    ev, = get_events(db[uid], [md['detector']], fill= False) 
    dec =  list( ev['descriptor']['configuration'].keys() )[0]
    for k,v in ev['descriptor']['configuration'][dec]['data'].items():
        md[ k[len(dec)+1:] ]= v
    for k,v in ev['descriptor']['run_start'].items():
        if k!= 'plan_args':
            md[k]= v 
    md['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(md['time']))    
    md['stop_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime( ev['time']   ))  
    md['img_shape'] = ev['descriptor']['data_keys'][md['detector']]['shape'][:2][::-1]
    for k,v in kwargs.items():
        md[k] =v  
    return md



def get_max_countc(FD, labeled_array ):
    """YG. 2016, Nov 18
    Compute the max intensity of ROIs in the compressed file (FD)

    Parameters
    ----------
    FD: Multifile class
        compressed file
    labeled_array : array
        labeled array; 0 is background.
        Each ROI is represented by a nonzero integer. It is not required that
        the ROI labels are contiguous
    index : int, list, optional
        The ROI's to use. If None, this function will extract averages for all
        ROIs

    Returns
    -------
    max_intensity : a float
    index : list
        The labels for each element of the `mean_intensity` list
    """
    
    qind, pixelist = roi.extract_label_indices(  labeled_array  ) 
    timg = np.zeros(    FD.md['ncols'] * FD.md['nrows']   , dtype=np.int32   ) 
    timg[pixelist] =   np.arange( 1, len(pixelist) + 1  ) 
    
    if labeled_array.shape != ( FD.md['ncols'],FD.md['nrows']):
        raise ValueError(
            " `image` shape (%d, %d) in FD is not equal to the labeled_array shape (%d, %d)" %( FD.md['ncols'],FD.md['nrows'], labeled_array.shape[0], labeled_array.shape[1]) )

    max_inten =0 
    for  i in tqdm(range( FD.beg, FD.end, 1  ), desc= 'Get max intensity of ROIs in all frames' ):
        try:
            (p,v) = FD.rdrawframe(i)
            w = np.where( timg[p] )[0]
            max_inten = max( max_inten, np.max(v[w]) )  
        except:
            pass
    return max_inten


def create_polygon_mask(  image, xcorners, ycorners   ):
    '''
    Give image and x/y coners to create a polygon mask    
    image: 2d array
    xcorners, list, points of x coners
    ycorners, list, points of y coners
    Return:
    the polygon mask: 2d array, the polygon pixels with values 1 and others with 0
    
    Example:
    
    
    '''
    from skimage.draw import line_aa, line, polygon, circle    
    imy, imx = image.shape 
    bst_mask = np.zeros_like( image , dtype = bool)   
    rr, cc = polygon( ycorners,xcorners)
    bst_mask[rr,cc] =1    
    #full_mask= ~bst_mask    
    return bst_mask
    
    
def create_rectangle_mask(  image, xcorners, ycorners   ):
    '''
    Give image and x/y coners to create a rectangle mask    
    image: 2d array
    xcorners, list, points of x coners
    ycorners, list, points of y coners
    Return:
    the polygon mask: 2d array, the polygon pixels with values 1 and others with 0
    
    Example:
    
    
    '''
    from skimage.draw import line_aa, line, polygon, circle    
    imy, imx = image.shape 
    bst_mask = np.zeros_like( image , dtype = bool)   
    rr, cc = polygon( ycorners,xcorners)
    bst_mask[rr,cc] =1    
    #full_mask= ~bst_mask    
    return bst_mask
       
    
def create_multi_rotated_rectangle_mask(  image, center=None, length=100, width=50, angles=[0] ):
    ''' Developed at July 10, 2017 by Y.G.@CHX, NSLS2
     Create multi rectangle-shaped mask by rotating a rectangle with a list of angles
     The original rectangle is defined by four corners, i.e.,  
         [   (center[1] - width//2, center[0]),
             (center[1] + width//2, center[0]), 
             (center[1] + width//2, center[0] + length),
             (center[1] - width//2, center[0] + length)
          ]
     
     Parameters:
         image: 2D numpy array, to give mask shape 
         center: integer list, if None, will be the center of the image
         length: integer, the length of the non-ratoted rectangle 
         width: integer,  the width of the non-ratoted rectangle 
         angles: integer list, a list of rotated angles

    Return:
        mask: 2D bool-type numpy array
    '''
    
    from skimage.draw import  polygon
    from skimage.transform import rotate 
    cx,cy = center
    imy, imx = image.shape   
    mask = np.zeros( image.shape, dtype = bool) 
    wy =  length
    wx =   width
    x = np.array( [ max(0, cx - wx//2), min(imx, cx+wx//2), min(imx, cx+wx//2), max(0,cx-wx//2 ) ])  
    y = np.array( [ cy, cy, min( imy, cy + wy) , min(imy, cy + wy) ])
    rr, cc = polygon( y,x)
    mask[rr,cc] =1
    mask_rot=  np.zeros( image.shape, dtype = bool) 
    for angle in angles:
        mask_rot +=  np.array( rotate( mask, angle, center= center ), dtype=bool) #, preserve_range=True)        
    return  ~mask_rot
    
def create_wedge(  image, center, radius, wcors,  acute_angle=True) :
    '''YG develop at June 18, 2017, @CHX
        Create a wedge by a combination of circle and a triangle defined by center and wcors
        wcors: [ [x1,x2,x3...], [y1,y2,y3..]
    
    '''
    from skimage.draw import line_aa, line, polygon, circle
    imy, imx = image.shape   
    cy,cx = center
    x  = [cx] + list(wcors[0])
    y =  [cy] + list(wcors[1])
    
    maskc = np.zeros_like( image , dtype = bool) 
    rr, cc = circle( cy, cx, radius, shape = image.shape)
    maskc[rr,cc] =1  
    
    maskp = np.zeros_like( image , dtype = bool)
    x = np.array( x )  
    y = np.array( y ) 
    print(x,y)
    rr, cc = polygon( y,x)
    maskp[rr,cc] =1
    if acute_angle:
        return maskc*maskp
    else:
        return maskc*~maskp
        
                

def create_cross_mask(  image, center, wy_left=4, wy_right=4, wx_up=4, wx_down=4,
                     center_circle = True, center_radius=10
                     ):
    '''
    Give image and the beam center to create a cross-shaped mask
    wy_left: the width of left h-line
    wy_right: the width of rigth h-line
    wx_up: the width of up v-line
    wx_down: the width of down v-line
    center_circle: if True, create a circle with center and center_radius
    
    Return:
    the cross mask
    '''
    from skimage.draw import line_aa, line, polygon, circle
    
    imy, imx = image.shape   
    cx,cy = center
    bst_mask = np.zeros_like( image , dtype = bool)   
    ###
    #for right part    
    wy = wy_right
    x = np.array( [ cx, imx, imx, cx  ])  
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1
    
    ###
    #for left part    
    wy = wy_left
    x = np.array( [0,  cx, cx,0  ])  
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1    
    
    ###
    #for up part    
    wx = wx_up
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])  
    y = np.array( [ cy, cy, imy, imy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1    
    
    ###
    #for low part    
    wx = wx_down
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])  
    y = np.array( [ 0,0, cy, cy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1   
    
    if center_radius!=0:
        rr, cc = circle( cy, cx, center_radius, shape = bst_mask.shape)
        bst_mask[rr,cc] =1   
    
    
    full_mask= ~bst_mask
    
    return full_mask
    
    
    


def generate_edge( centers, width):
    '''YG. 10/14/2016
    give centers and width (number or list) to get edges'''
    edges = np.zeros( [ len(centers),2])
    edges[:,0] =  centers - width
    edges[:,1] =  centers + width
    return edges


def export_scan_scalar( uid, x='dcm_b', y= ['xray_eye1_stats1_total'],
                       path='/XF11ID/analysis/2016_3/commissioning/Results/' ):
    '''YG. 10/17/2016
    export uid data to a txt file
    uid: unique scan id
    x: the x-col 
    y: the y-cols
    path: save path
    Example:
        data = export_scan_scalar( uid, x='dcm_b', y= ['xray_eye1_stats1_total'],
                       path='/XF11ID/analysis/2016_3/commissioning/Results/exported/' )
    A plot for the data:
        d.plot(x='dcm_b', y = 'xray_eye1_stats1_total', marker='o', ls='-', color='r')
        
    '''
    from databroker import DataBroker as db, get_images, get_table, get_events, get_fields 
    from chxanalys.chx_generic_functions import  trans_data_to_pd
    
    hdr = db[uid]
    print( get_fields( hdr ) )
    data = get_table( db[uid] )
    xp = data[x]
    datap = np.zeros(  [len(xp), len(y)+1])
    datap[:,0] = xp
    for i, yi in enumerate(y):
        datap[:,i+1] = data[yi]
        
    datap = trans_data_to_pd( datap, label=[x] + [yi for yi in y])   
    datap.to_csv( path + 'uid=%s.csv'%uid)
    return datap




#####
#load data by databroker   

def get_flatfield( uid, reverse=False ):
    import h5py
    detector = get_detector( db[uid ] )
    sud = get_sid_filenames(db[uid])
    master_path = '%s_master.h5'%(sud[2][0])
    print( master_path)
    f= h5py.File(master_path, 'r')
    k= 'entry/instrument/detector/detectorSpecific/' #data_collection_date'
    d= np.array( f[ k]['flatfield'] )
    f.close()
    if reverse:        
        d = reverse_updown( d )
        
    return d



def get_detector( header ):
    keys = [k for k, v in header.descriptors[0]['data_keys'].items()     if 'external' in v]
    return keys[0]

    
def get_sid_filenames(header, fill=True):
    """get a bluesky scan_id, unique_id, filename by giveing uid and detector
        
    Parameters
    ----------
    header: a header of a bluesky scan, e.g. db[-1]
        
    Returns
    -------
    scan_id: integer
    unique_id: string, a full string of a uid
    filename: sring
    
    Usuage:
    sid,uid, filenames   = get_sid_filenames(db[uid])
    
    """   
    
    keys = [k for k, v in header.descriptors[0]['data_keys'].items()     if 'external' in v]
    events = get_events( header, keys, handler_overrides={key: RawHandler for key in keys}, fill=fill)
    key, = keys   
    try:
        filenames =  [  str( ev['data'][key][0]) + '_'+ str(ev['data'][key][2]['seq_id']) for ev in events]     
    except:
        filenames='unknown'
    sid = header['start']['scan_id']
    uid= header['start']['uid']
    
    return sid,uid, filenames   


def load_data( uid , detector = 'eiger4m_single_image', fill=True, reverse=False):
    """load bluesky scan data by giveing uid and detector
        
    Parameters
    ----------
    uid: unique ID of a bluesky scan
    detector: the used area detector
    fill: True to fill data
    reverse: if True, reverse the image upside down to match the "real" image geometry (should always be True in the future)
    
    Returns
    -------
    image data: a pims frames series
    if not success read the uid, will return image data as 0
    
    Usuage:
    imgs = load_data( uid, detector  )
    md = imgs.md
    """   
    hdr = db[uid]
    flag =1
    while flag<2 and flag !=0:    
        try:
            ev, = get_events(hdr, [detector], fill=fill) 
            flag = 0 
            
        except:
            flag += 1        
            print ('Trying again ...!')
    if flag:
        try:
            imgs = get_images( hdr, detector)
            #print( 'here')
            print('This should be dscan data.')
            print('You also can use get_images( hdr, detector) function to retrive data.')
            if len(imgs[0])>=1:
                md = imgs[0].md
                imgs = pims.pipeline(lambda img:  img[0])(imgs)
                imgs.md = md
        except:            
            print ("Can't Load Data!")
            uid = '00000'  #in case of failling load data
            imgs = 0
    else:
        imgs = ev['data'][detector]
    #print (imgs)
    if reverse:
        md=imgs.md
        imgs = reverse_updown( imgs )
        imgs.md = md
    return imgs


def mask_badpixels( mask, detector ):
    '''
    Mask known bad pixel from the giveing mask
    
    '''
    if detector =='eiger1m_single_image':
        #to be determined
        mask = mask
    elif detector =='eiger4m_single_image' or detector == 'image':    
        mask[513:552,:] =0 
        mask[1064:1103,:] =0 
        mask[1615:1654,:] =0     
        mask[:,1029:1041] = 0     
        mask[:, 0] =0 
        mask[0:, 2069] =0 
        mask[0] =0 
        mask[2166] =0 

    elif detector =='eiger500K_single_image':
        #to be determined
        mask = mask
    else:
        mask = mask        
    return mask
    
    
    


def load_data2( uid , detector = 'eiger4m_single_image'  ):
    """load bluesky scan data by giveing uid and detector
        
    Parameters
    ----------
    uid: unique ID of a bluesky scan
    detector: the used area detector
    
    Returns
    -------
    image data: a pims frames series
    if not success read the uid, will return image data as 0
    
    Usuage:
    imgs = load_data( uid, detector  )
    md = imgs.md
    """   
    hdr = db[uid]
    flag =1
    while flag<4 and flag !=0:    
        try:
            ev, = get_events(hdr, [detector]) 
            flag =0 
        except:
            flag += 1        
            print ('Trying again ...!')

    if flag:
        print ("Can't Load Data!")
        uid = '00000'  #in case of failling load data
        imgs = 0
    else:
        imgs = ev['data'][detector]

    #print (imgs)
    return imgs



def psave_obj(obj, filename ):
    '''save an object with filename by pickle.dump method
       This function automatically add '.pkl' as filename extension
    Input:
        obj: the object to be saved
        filename: filename (with full path) to be saved
    Return:
        None    
    '''
    with open( filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pload_obj(filename ):
    '''load a pickled filename 
        This function automatically add '.pkl' to filename extension
    Input:        
        filename: filename (with full path) to be saved
    Return:
        load the object by pickle.load method
    '''
    with open( filename + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
    
def load_mask( path, mask_name, plot_ = False, reverse=False, *argv,**kwargs): 
    
    """load a mask file
    the mask is a numpy binary file (.npy) 
        
    Parameters
    ----------
    path: the path of the mask file
    mask_name: the name of the mask file
    plot_: a boolen type
    reverse: if True, reverse the image upside down to match the "real" image geometry (should always be True in the future)    
    Returns
    -------
    mask: array
    if plot_ =True, will show the mask   
    
    Usuage:
    mask = load_mask( path, mask_name, plot_ =  True )
    """
    
    mask = np.load(    path +   mask_name )
    mask = np.array(mask, dtype = np.int32)
    if reverse:
        mask = mask[::-1,:]  
    if plot_:
        show_img( mask, *argv,**kwargs)   
    return mask



def create_hot_pixel_mask(img, threshold, center=None, center_radius=300, outer_radius=0 ):
    '''create a hot pixel mask by giving threshold
       Input:
           img: the image to create hot pixel mask
           threshold: the threshold above which will be considered as hot pixels
           center: optional, default=None
                             else, as a two-element list (beam center), i.e., [center_x, center_y]
           if center is not None, the hot pixel will not include a circle region 
                           which is defined by center and center_radius ( in unit of pixel)
       Output:
           a bool types numpy array (mask), 1 is good and 0 is excluded   
    
    '''
    bst_mask = np.ones_like( img , dtype = bool)    
    if center is not None:    
        from skimage.draw import  circle    
        imy, imx = img.shape   
        cy,cx = center        
        rr, cc = circle( cy, cx, center_radius,shape=img.shape )
        bst_mask[rr,cc] =0 
        if outer_radius:
            bst_mask = np.zeros_like( img , dtype = bool)   
            rr2, cc2 = circle( cy, cx, outer_radius,shape=img.shape )
            bst_mask[rr2,cc2] =1
            bst_mask[rr,cc] =0 
    hmask = np.ones_like( img )
    hmask[np.where( img * bst_mask  > threshold)]=0
    return hmask




def apply_mask( imgs, mask):
    '''apply mask to imgs to produce a generator
    
    Usuages:
    imgsa = apply_mask( imgs, mask )
    good_series = apply_mask( imgs[good_start:], mask )
    
    '''
    return pims.pipeline(lambda img: np.int_(mask) * img)(imgs)  # lazily apply mask


def reverse_updown( imgs):
    '''reverse imgs upside down to produce a generator
    
    Usuages:
    imgsr = reverse_updown( imgs)
     
    
    '''
    return pims.pipeline(lambda img: img[::-1,:])(imgs)  # lazily apply mask


def RemoveHot( img,threshold= 1E7, plot_=True ):
    '''Remove hot pixel from img'''
    
    mask = np.ones_like( np.array( img )    )
    badp = np.where(  np.array(img) >= threshold )
    if len(badp[0])!=0:                
        mask[badp] = 0  
    if plot_:
        show_img( mask )
    return mask


############
###plot data

def show_img( image, ax=None,label_array=None, alpha=0.5, interpolation='nearest',
             xlim=None, ylim=None, save=False,image_name=None,path=None, 
             aspect=None, logs=False,vmin=None,vmax=None,return_fig=False,cmap='viridis', 
             show_time= False, file_name =None, ylabel=None, xlabel=None, extent=None,
             show_colorbar=True, tight=True, show_ticks=True, save_format = 'png', dpi= None,
             center=None, center_size=None,origin='lower', lab_fontsize = 16,  tick_size = 12, colorbar_fontsize = 8, 
             *argv,**kwargs ):    
    """YG. Sep26, 2017 Add label_array/alpha option to show a mask on top of image
    
    a simple function to show image by using matplotlib.plt imshow
    pass *argv,**kwargs to imshow
    
    Parameters
    ----------
    image : array
        Image to show
    Returns
    -------
    None
    """ 
    if ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
    else:
        fig, ax=ax      
    if center is not None:        
        plot1D(center[1],center[0],markersize=center_size,ax=ax, c='b', m='o', legend='')
    if not logs:
        im=imshow(ax, image, origin=origin,cmap=cmap,interpolation=interpolation, vmin=vmin,vmax=vmax,
                    extent=extent)  #vmin=0,vmax=1,
    else:
        im=imshow(ax, image, origin=origin,cmap=cmap,
                interpolation=interpolation, norm=LogNorm(vmin,  vmax),extent=extent)   
    if label_array is not None:
        im2=show_label_array(ax, label_array, alpha= alpha, cmap=cmap, interpolation=interpolation )  
        
    ax.set_title( image_name )
    if xlim is not None:
        ax.set_xlim(   xlim  )        
    if ylim is not None:
        ax.set_ylim(   ylim )
    
    if not show_ticks:
        ax.set_yticks([])
        ax.set_xticks([])        
    else:   
        
        ax.tick_params(axis='both', which='major', labelsize=tick_size )
        ax.tick_params(axis='both', which='minor', labelsize=tick_size )
        #mpl.rcParams['xtick.labelsize'] = tick_size 
        #mpl.rcParams['ytick.labelsize'] = tick_size
        #print(tick_size)
    
    if ylabel is not None:
        #ax.set_ylabel(ylabel)#, fontsize = 9)
        ax.set_ylabel(  ylabel , fontsize = lab_fontsize    )
    if xlabel is not None:
        ax.set_xlabel(xlabel , fontsize = lab_fontsize )          
        
    if aspect is not None:
        #aspect = image.shape[1]/float( image.shape[0] )
        ax.set_aspect(aspect)
    else:
        ax.set_aspect(aspect='auto')
        
    if show_colorbar:
        try:
            cbar = fig.colorbar(im, extend='neither', spacing='proportional',
                orientation='vertical' )
            cbar.ax.tick_params(labelsize=colorbar_fontsize)        
        except:
            pass
    fig.set_tight_layout(tight) 
    if save:
        if show_time:
            dt =datetime.now()
            CurTime = '_%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)         
            fp = path + '%s'%( file_name ) + CurTime + '.' + save_format
        else:
            fp = path + '%s'%( image_name ) + '.' + save_format
        if dpi is None:
            dpi = fig.dpi
        plt.savefig( fp, dpi= dpi)          
    #fig.set_tight_layout(tight)     
    if return_fig:
        return im #fig
    

    
    
def plot1D( y,x=None, yerr=None, ax=None,return_fig=False, ls='-', figsize=None,
           legend_size=None, lw=None, markersize=None, tick_size=8, *argv,**kwargs):    
    """a simple function to plot two-column data by using matplotlib.plot
    pass *argv,**kwargs to plot
    
    Parameters
    ----------
    y: column-y
    x: column-x, by default x=None, the plot will use index of y as x-axis
    Returns
    -------
    None
    """     
    if ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
        
    if 'legend' in kwargs.keys():
        legend =  kwargs['legend']  
    else:
        legend = ' '

    try:
         logx = kwargs['logx']
    except:
        logx=False
    try:
         logy = kwargs['logy']
    except:
        logy=False
        
    try:
         logxy = kwargs['logxy']
    except:
        logxy= False        

    if logx==True and logy==True:
        logxy = True
        
    try:
        marker = kwargs['marker']         
    except:
        try:
            marker = kwargs['m'] 
        except:            
            marker= next(  markers_    )
    try:
        color =  kwargs['color']
    except:    
        try:
            color =  kwargs['c']
        except: 
            color = next(  colors_    ) 
            
    if x is None:
        x=range(len(y))
    if yerr is None:    
        ax.plot(x,y, marker=marker,color=color,ls=ls,label= legend, lw=lw,
                markersize=markersize, )#,*argv,**kwargs)
    else:
        ax.errorbar(x,y,yerr, marker=marker,color=color,ls=ls,label= legend, 
                    lw=lw,markersize=markersize,)#,*argv,**kwargs)    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if logxy:
        ax.set_xscale('log')
        ax.set_yscale('log')
         
 
    ax.tick_params(axis='both', which='major', labelsize=tick_size )
    ax.tick_params(axis='both', which='minor', labelsize=tick_size )
    
    if 'xlim' in kwargs.keys():
         ax.set_xlim(    kwargs['xlim']  )    
    if 'ylim' in kwargs.keys():
         ax.set_ylim(    kwargs['ylim']  )
    if 'xlabel' in kwargs.keys():            
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys():            
        ax.set_ylabel(kwargs['ylabel'])
        
    if 'title' in kwargs.keys():
        title =  kwargs['title']
    else:
        title =  'plot'
    ax.set_title( title ) 
    #ax.set_xlabel("$Log(q)$"r'($\AA^{-1}$)')    
    if (legend!='') and (legend!=None):
        ax.legend(loc = 'best', fontsize=legend_size )
    if 'save' in kwargs.keys():
        if  kwargs['save']: 
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)         
            #fp = kwargs['path'] + '%s'%( title ) + CurTime + '.png'  
            fp = kwargs['path'] + '%s'%( title )   + '.png' 
            plt.savefig( fp, dpi=fig.dpi)         
    if return_fig:
        return fig      

        
###

def check_shutter_open( data_series,  min_inten=0, time_edge = [0,10], plot_ = False,  *argv,**kwargs):     
      
    '''Check the first frame with shutter open
    
        Parameters        
        ----------
        data_series: a image series
        min_inten: the total intensity lower than min_inten is defined as shtter close
        time_edge: the searching frame number range
        
        return:
        shutter_open_frame: a integer, the first frame number with open shutter     
        
        Usuage:
        good_start = check_shutter_open( imgsa,  min_inten=5, time_edge = [0,20], plot_ = False )
       
    '''
    imgsum =  np.array(  [np.sum(img ) for img in data_series[time_edge[0]:time_edge[1]:1]]  ) 
    if plot_:
        fig, ax = plt.subplots()  
        ax.plot(imgsum,'bo')
        ax.set_title('uid=%s--imgsum'%uid)
        ax.set_xlabel( 'Frame' )
        ax.set_ylabel( 'Total_Intensity' ) 
        #plt.show()       
    shutter_open_frame = np.where( np.array(imgsum) > min_inten )[0][0]
    print ('The first frame with open shutter is : %s'%shutter_open_frame )
    return shutter_open_frame



def get_each_frame_intensity( data_series, sampling = 50, 
                             bad_pixel_threshold=1e10,                              
                             plot_ = False, save= False, *argv,**kwargs):   
    '''Get the total intensity of each frame by sampling every N frames
       Also get bad_frame_list by check whether above  bad_pixel_threshold  
       
       Usuage:
       imgsum, bad_frame_list = get_each_frame_intensity(good_series ,sampling = 1000, 
                                bad_pixel_threshold=1e10,  plot_ = True)
    '''
    
    #print ( argv, kwargs )
    imgsum =  np.array(  [np.sum(img ) for img in tqdm( data_series[::sampling] , leave = True ) ] )  
    if plot_:
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid']        
        fig, ax = plt.subplots()  
        ax.plot(imgsum,'bo')
        ax.set_title('uid= %s--imgsum'%uid)
        ax.set_xlabel( 'Frame_bin_%s'%sampling )
        ax.set_ylabel( 'Total_Intensity' )
        if save:
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
            path = kwargs['path'] 
            if 'uid' in kwargs:
                uid = kwargs['uid']
            else:
                uid = 'uid'
            #fp = path + "Uid= %s--Waterfall-"%uid + CurTime + '.png'     
            fp = path + "uid=%s--imgsum-"%uid  + '.png'    
            fig.savefig( fp, dpi=fig.dpi)        
        #plt.show()  
        
    bad_frame_list = np.where( np.array(imgsum) > bad_pixel_threshold )[0]
    if len(bad_frame_list):
        print ('Bad frame list are: %s' %bad_frame_list)
    else:
        print ('No bad frames are involved.')
    return imgsum,bad_frame_list


 

def create_time_slice( N, slice_num, slice_width, edges=None ):
    '''create a ROI time regions '''
    if edges is not None:
        time_edge = edges
    else:
        if slice_num==1:
            time_edge =  [ [0,N] ]
        else:
            tstep = N // slice_num
            te = np.arange( 0, slice_num +1   ) * tstep
            tc = np.int_( (te[:-1] + te[1:])/2 )[1:-1]
            if slice_width%2:
                sw = slice_width//2 +1
                time_edge =   [ [0,slice_width],  ] + [  [s-sw+1,s+sw] for s in tc  ] +  [ [N-slice_width,N]]
            else:
                sw= slice_width//2
                time_edge =   [ [0,slice_width],  ] + [  [s-sw,s+sw] for s in tc  ] +  [ [N-slice_width,N]]
                                 
            

    return np.array(time_edge)


def show_label_array(ax, label_array, cmap=None, aspect=None,interpolation='nearest',**kwargs):
    """
    YG. Sep 26, 2017
    Modified show_label_array(ax, label_array, cmap=None, **kwargs)
        from https://github.com/Nikea/xray-vision/blob/master/xray_vision/mpl_plotting/roi.py
    Display a labeled array nicely
    Additional kwargs are passed through to `ax.imshow`.
    If `vmin` is in kwargs, it is clipped to minimum of 0.5.
    Parameters
    ----------
    ax : Axes
        The `Axes` object to add the artist too
    label_array: ndarray
        Expected to be an unsigned integer array.  0 is background,
        positive integers label region of interest
    cmap : str or colormap, optional
        Color map to use, defaults to 'Paired'
    Returns
    -------
    img : AxesImage
        The artist added to the axes
    """
    if cmap is None:
        cmap = 'viridis'
    #print(cmap)
    _cmap = copy.copy((mcm.get_cmap(cmap)))
    _cmap.set_under('w', 0)
    vmin = max(.5, kwargs.pop('vmin', .5))
    im = ax.imshow(label_array, cmap=cmap,
                   interpolation=interpolation,
                   vmin=vmin,
                   **kwargs)    
    if aspect is None:
        ax.set_aspect(aspect='auto')
        #ax.set_aspect('equal')
    return im



def show_label_array_on_image(ax, image, label_array, cmap=None,norm=None, log_img=True,alpha=0.3, vmin=0.1, vmax=5,
                              imshow_cmap='gray', **kwargs):  #norm=LogNorm(), 
    """
    This will plot the required ROI's(labeled array) on the image
    
    Additional kwargs are passed through to `ax.imshow`.
    If `vmin` is in kwargs, it is clipped to minimum of 0.5.
    Parameters
    ----------
    ax : Axes
        The `Axes` object to add the artist too
    image : array
        The image array
    label_array : array
        Expected to be an unsigned integer array.  0 is background,
        positive integers label region of interest
    cmap : str or colormap, optional
        Color map to use for plotting the label_array, defaults to 'None'
    imshow_cmap : str or colormap, optional
        Color map to use for plotting the image, defaults to 'gray'
    norm : str, optional
        Normalize scale data, defaults to 'Lognorm()'
    Returns
    -------
    im : AxesImage
        The artist added to the axes
    im_label : AxesImage
        The artist added to the axes
    """
    ax.set_aspect('equal')
    
    #print (vmin, vmax )
    if log_img:
        im = ax.imshow(image, cmap=imshow_cmap, interpolation='none',norm=LogNorm(vmin,  vmax),**kwargs)  #norm=norm,
    else:
        im = ax.imshow(image, cmap=imshow_cmap, interpolation='none',vmin=vmin, vmax=vmax,**kwargs)  #norm=norm,
        
    im_label = mpl_plot.show_label_array(ax, label_array, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                                **kwargs)  # norm=norm,
    
    
    return im, im_label    
    
    
    
def show_ROI_on_image( image, ROI, center=None, rwidth=400,alpha=0.3,  label_on = True,
                       save=False, return_fig = False, rect_reqion=None, log_img = True, vmin=0.01, vmax=5, 
                      show_ang_cor = False,
                      uid='uid', path='',  aspect = 1, show_colorbar=True, show_roi_edge=False, *argv,**kwargs):
    
    '''show ROI on an image
        image: the data frame
        ROI: the interested region
        center: the plot center
        rwidth: the plot range around the center  
    
    '''

        
    if RUN_GUI:
        fig = Figure(figsize=(8,8))
        axes = fig.add_subplot(111)
    else:
        fig, axes = plt.subplots( ) #plt.subplots(figsize=(8,8))
    
    #print( vmin, vmax)
    #norm=LogNorm(vmin,  vmax)
    
    axes.set_title(  "%s_ROI_on_Image"%uid  )
    if log_img:
        if vmin==0:
            vmin += 1e-10
    
    vmax = max(1, vmax ) 
    if not show_roi_edge:    
        im,im_label = show_label_array_on_image(axes, image, ROI, imshow_cmap='viridis',
                            cmap='Paired',alpha=alpha, log_img=log_img,
                             vmin=vmin, vmax=vmax,  origin="lower")
    else:
        edg = get_image_edge( ROI )
        image_ = get_image_with_roi( image, ROI, scale_factor = 2) 
        #fig, axes = plt.subplots( )        
        show_img(  image_,  ax=[fig,axes], vmin=vmin, vmax=vmax,  
                  logs= log_img, image_name=  "%s_ROI_on_Image"%uid,
                 cmap = cmap_albula )
    
    
    if rect_reqion is  None:
        if center is not None:
            x1,x2 = [center[1] - rwidth, center[1] + rwidth]
            y1,y2 = [center[0] - rwidth, center[0] + rwidth]
            axes.set_xlim( [x1,x2])
            axes.set_ylim( [y1,y2])
    else:
        x1,x2,y1,y2= rect_reqion
        axes.set_xlim( [x1,x2])
        axes.set_ylim( [y1,y2])
    
    if label_on:
        num_qzr = len(np.unique( ROI )) -1        
        for i in range( 1, num_qzr + 1 ):
            ind =  np.where( ROI == i)[1]
            indz =  np.where( ROI == i)[0]
            c = '%i'%i
            y_val = int( indz.mean() )
            x_val = int( ind.mean() )
            #print (xval, y)
            axes.text(x_val, y_val, c, color='b',va='center', ha='center')    
    if show_ang_cor:
        axes.text(-0.0, 0.5, '-/+180' + r'$^0$', color='r', va='center', ha='center',transform=axes.transAxes)
        axes.text(1.0, 0.5, '0' + r'$^0$', color='r', va='center', ha='center',transform=axes.transAxes)
        axes.text(0.5, -0.0, '-90'+  r'$^0$', color='r', va='center', ha='center',transform=axes.transAxes)
        axes.text(0.5, 1.0, '90' +  r'$^0$', color='r', va='center', ha='center',transform=axes.transAxes)  

    axes.set_aspect(aspect)
    #fig.colorbar(im_label)
    if show_colorbar:
        if not show_roi_edge:   
            fig.colorbar(im)
    if save:   
        fp = path + "%s_ROI_on_Image"%uid  + '.png'    
        plt.savefig( fp, dpi=fig.dpi)      
    #plt.show()
    if return_fig:
        return fig, axes, im 

        

        
def crop_image(  image,  crop_mask  ):
    
    ''' Crop the non_zeros pixels of an image  to a new image 
         
         
    '''
    from skimage.util import crop, pad     
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]   #dimension in y, but in plot being x
    imgwidthx = dims[0]   #dimension in x, but in plot being y
    #x and y are flipped???
    #matrix notation!!!
    pixely = pxlst%imgwidthy
    pixelx = pxlst//imgwidthy

    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely) 
    crops = crop_mask*image
    img_crop = crop(  crops, ((minpixelx, imgwidthx -  maxpixelx -1 ),
                                (minpixely, imgwidthy -  maxpixely -1 )) )
    return img_crop
            

def get_avg_img( data_series,  img_samp_index=None, sampling = 100, plot_ = False , save=False, *argv,**kwargs):   
    '''Get average imagef from a data_series by every sampling number to save time'''
    if img_samp_index is None:
        avg_img = np.average(data_series[:: sampling], axis=0)
    else:
        avg_img = np.zeros_like( data_series[0] )
        n=0
        for i in img_samp_index:
            avg_img += data_series[i]
            n +=1
        avg_img = np.array( avg_img) / n
    
    if plot_:
        fig, ax = plt.subplots()
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
            
        im = ax.imshow(avg_img , cmap='viridis',origin='lower',
                   norm= LogNorm(vmin=0.001, vmax=1e2))
        #ax.set_title("Masked Averaged Image")
        ax.set_title('uid= %s--Masked Averaged Image'%uid)
        fig.colorbar(im)
        
        if save:
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
            path = kwargs['path'] 
            if 'uid' in kwargs:
                uid = kwargs['uid']
            else:
                uid = 'uid'
            #fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'     
            fp = path + "uid=%s--avg-img-"%uid  + '.png'    
            fig.savefig( fp, dpi=fig.dpi)        
        #plt.show()

    return avg_img



def check_ROI_intensity( avg_img, ring_mask, ring_number=3 , save=False, plot=True, *argv,**kwargs):
    
    """plot intensity versus pixel of a ring        
    Parameters
    ----------
    avg_img: 2D-array, the image
    ring_mask: 2D-array  
    ring_number: which ring to plot
    
    Returns
    -------

     
    """   
    #print('here')
    
    uid = 'uid'
    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
    pixel = roi.roi_pixel_values(avg_img, ring_mask, [ring_number] )
    
    if plot:
        fig, ax = plt.subplots()
        ax.set_title('%s--check-RIO-%s-intensity'%(uid, ring_number) )
        ax.plot( pixel[0][0] ,'bo', ls='-' )
        ax.set_ylabel('Intensity')
        ax.set_xlabel('pixel')
        if save: 
            path = kwargs['path']  
            fp = path + "%s_Mean_intensity_of_one_ROI"%uid  + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
    if save:
        path = kwargs['path']  
        save_lists( [range( len(  pixel[0][0] )), pixel[0][0]], label=['pixel_list', 'roi_intensity'],
                filename="%s_Mean_intensity_of_one_ROI"%uid, path= path)  
    #plt.show()
    return pixel[0][0]

#from tqdm import tqdm

def cal_g2( image_series, ring_mask, bad_image_process,
           bad_frame_list=None,good_start=0, num_buf = 8, num_lev = None ):
    '''calculation g2 by using a multi-tau algorithm'''
    
    noframes = len( image_series)  # number of frames, not "no frames"
    #num_buf = 8  # number of buffers

    if bad_image_process:   
        import skbeam.core.mask as mask_image       
        bad_img_list = np.array( bad_frame_list) - good_start
        new_imgs = mask_image.bad_to_nan_gen( image_series, bad_img_list)        

        if num_lev is None:
            num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        print ('In this g2 calculation, the buf and lev number are: %s--%s--'%(num_buf,num_lev))
        print ('%s frames will be processed...'%(noframes))
        print( 'Bad Frames involved!')

        g2, lag_steps = corr.multi_tau_auto_corr(num_lev, num_buf,  ring_mask, tqdm( new_imgs) )
        print( 'G2 calculation DONE!')

    else:

        if num_lev is None:
            num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        print ('In this g2 calculation, the buf and lev number are: %s--%s--'%(num_buf,num_lev))
        print ('%s frames will be processed...'%(noframes))
        g2, lag_steps = corr.multi_tau_auto_corr(num_lev, num_buf,   ring_mask, tqdm(image_series) )
        print( 'G2 calculation DONE!')
        
    return g2, lag_steps



def run_time(t0):
    '''Calculate running time of a program
    Parameters
    ----------
    t0: time_string, t0=time.time()
        The start time
    Returns
    -------
    Print the running time 
    
    One usage
    ---------
    t0=time.time()
    .....(the running code)
    run_time(t0)
    '''    
    
    elapsed_time = time.time() - t0
    if elapsed_time<60:
        print ('Total time: %.3f sec' %(elapsed_time ))
    else:
        print ('Total time: %.3f min' %(elapsed_time/60.)) 
    
    
def trans_data_to_pd(data, label=None,dtype='array'):
    '''
    convert data into pandas.DataFrame
    Input:
        data: list or np.array
        label: the coloum label of the data
        dtype: list or array [[NOT WORK or dict (for dict only save the scalar not arrays values)]]
    Output:
        a pandas.DataFrame
    '''
    #lists a [ list1, list2...] all the list have the same length
    from numpy import arange,array
    import pandas as pd,sys    
    if dtype == 'list':
        data=array(data).T    
        N,M=data.shape
    elif dtype == 'array':
        data=array(data)   
        N,M=data.shape        
    else:
        print("Wrong data type! Now only support 'list' and 'array' tpye")        
       
    
    index =  arange( N )
    if label is None:label=['data%s'%i for i in range(M)]
    #print label
    df = pd.DataFrame( data, index=index, columns= label  )
    return df


def save_lists( data, label=None,  filename=None, path=None, return_res = False):    
    '''
    save_lists( data, label=None,  filename=None, path=None)
    
    save lists to a CSV file with filename in path
    Parameters
    ----------
    data: list
    label: the column name, the length should be equal to the column number of list
    filename: the filename to be saved
    path: the filepath to be saved
    
    Example:    
    save_arrays(  [q,iq], label= ['q_A-1', 'Iq'], filename='uid=%s-q-Iq'%uid, path= data_dir  )    
    '''
    
    M,N = len(data[0]),len(data)
    d = np.zeros( [N,M] )
    for i in range(N):
        d[i] = data[i]        
    
    df = trans_data_to_pd(d.T, label, 'array')  
    #dt =datetime.now()
    #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    if filename is None:
        filename = 'data'
    filename = os.path.join(path, filename )#+'.csv')
    df.to_csv(filename)
    if return_res:
        return df

def get_pos_val_overlap( p1, v1, p2,v2, Nl):
    '''get the overlap of v1 and v2
        p1: the index of array1 in array with total length as Nl
        v1: the corresponding value of p1
        p2: the index of array2 in array with total length as Nl
        v2: the corresponding value of p2
        Return:
         The values in v1 with the position in overlap of p1 and p2
         The values in v2 with the position in overlap of p1 and p2
         
         An example:
            Nl =10
            p1= np.array( [1,3,4,6,8] )
            v1 = np.array( [10,20,30,40,50])
            p2= np.array( [ 0,2,3,5,7,8])
            v2=np.array( [10,20,30,40,50,60,70])
            
            get_pos_val_overlap( p1, v1, p2,v2, Nl)
         
    '''
    ind = np.zeros( Nl, dtype=np.int32 )
    ind[p1] = np.arange( len(p1) ) +1 
    w2 = np.where( ind[p2] )[0]
    w1 = ind[ p2[w2]] -1
    return v1[w1], v2[w2]
    
    
    
def save_arrays( data, label=None, dtype='array', filename=None, path=None, return_res = False):      
    '''
    July 10, 2016, Y.G.@CHX
    save_arrays( data, label=None, dtype='array', filename=None, path=None): 
    save data to a CSV file with filename in path
    Parameters
    ----------
    data: arrays
    label: the column name, the length should be equal to the column number of data
    dtype: array or list
    filename: the filename to be saved
    path: the filepath to be saved
    
    Example:
    
    save_arrays(  qiq, label= ['q_A-1', 'Iq'], dtype='array', filename='uid=%s-q-Iq'%uid, path= data_dir  )

    
    '''
    df = trans_data_to_pd(data, label,dtype)  
    #dt =datetime.now()
    #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    if filename is None:
        filename = 'data'
    filename_ = os.path.join(path, filename)# +'.csv')
    df.to_csv(filename_)
    print( 'The file: %s is saved in %s'%(filename, path) )
    #print( 'The g2 of uid= %s is saved in %s with filename as g2-%s-%s.csv'%(uid, path, uid, CurTime))
    if return_res:
        return df
        

def get_diffusion_coefficient( visocity, radius, T=298):
    '''July 10, 2016, Y.G.@CHX
        get diffusion_coefficient of a Brownian motion particle with radius in fuild with visocity
        visocity: N*s/m^2  (water at 25K = 8.9*10^(-4) )
        radius: m
        T: K
        k: 1.38064852(79)×10−23 J/T, Boltzmann constant   
        
        return diffusion_coefficient in unit of A^2/s
        e.g., for a 250 nm sphere in glycerol/water (90:10) at RT (298K) gives:
       1.38064852*10**(−23) *298  / ( 6*np.pi* 0.20871 * 250 *10**(-9)) * 10**20 /1e5 = 4.18*10^5 A2/s
       
       get_diffusion_coefficient( 0.20871, 250 *10**(-9), T=298) 
       
    '''
    
    k=  1.38064852*10**(-23)    
    return k*T / ( 6*np.pi* visocity * radius) * 10**20 


def ring_edges(inner_radius, width, spacing=0, num_rings=None):
    """
    Aug 02, 2016, Y.G.@CHX
    ring_edges(inner_radius, width, spacing=0, num_rings=None)
    
    Calculate the inner and outer radius of a set of rings.

    The number of rings, their widths, and any spacing between rings can be
    specified. They can be uniform or varied.

    Parameters
    ----------
    inner_radius : float
        inner radius of the inner-most ring

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    spacing : float or list of floats, optional
        margin between rings, 0 by default
        If a float, all rings will have the same spacing. If a list,
        the length of the list must be one less than the number of
        rings.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring

    Example
    -------
    # Make two rings starting at r=1px, each 5px wide
    >>> ring_edges(inner_radius=1, width=5, num_rings=2)
    [(1, 6), (6, 11)]
    # Make three rings of different widths and spacings.
    # Since the width and spacings are given individually, the number of
    # rings here is simply inferred.
    >>> ring_edges(inner_radius=1, width=(5, 4, 3), spacing=(1, 2))
    [(1, 6), (7, 11), (13, 16)]
    
    """
    # All of this input validation merely checks that width, spacing, and
    # num_rings are self-consistent and complete.
    width_is_list = isinstance(width, collections.Iterable)
    spacing_is_list = isinstance(spacing, collections.Iterable)
    if (width_is_list and spacing_is_list):
        if len(width) != len(spacing) + 1:
            raise ValueError("List of spacings must be one less than list "
                             "of widths.")
    if num_rings is None:
        try:
            num_rings = len(width)
        except TypeError:
            try:
                num_rings = len(spacing) + 1
            except TypeError:
                raise ValueError("Since width and spacing are constant, "
                                 "num_rings cannot be inferred and must be "
                                 "specified.")
    else:        
        if width_is_list:
            if num_rings != len(width):
                raise ValueError("num_rings does not match width list")
        if spacing_is_list:
            if num_rings-1 != len(spacing):
                raise ValueError("num_rings does not match spacing list")
    # Now regularlize the input.
    if not width_is_list:
        width = np.ones(num_rings) * width
        
    if spacing is None:
        spacing = [] 
    else:    
        if not spacing_is_list:
            spacing = np.ones(num_rings - 1) * spacing
    # The inner radius is the first "spacing."
    all_spacings = np.insert(spacing, 0, inner_radius)     
    steps = np.array([all_spacings, width]).T.ravel()
    edges = np.cumsum(steps).reshape(-1, 2)
    return edges

 

def get_non_uniform_edges(  centers, width = 4, number_rings=1, spacing=0,  ):
    '''
    YG CHX Spe 6
    get_non_uniform_edges(  centers, width = 4, number_rings=3 )
    
    Calculate the inner and outer radius of a set of non uniform distributed
    rings by giving ring centers
    For each center, there are number_rings with each of width

    Parameters
    ----------
    centers : float
        the center of the rings

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring    
    '''
    
    if number_rings is  None:    
        number_rings = 1
    edges = np.zeros( [len(centers)*number_rings, 2]  )
    #print( width )
    
    if not isinstance(width, collections.Iterable):
        width = np.ones_like( centers ) * width               
    for i, c in enumerate(centers):       
        edges[i*number_rings:(i+1)*number_rings,:] = ring_edges( inner_radius =  c - width[i]*number_rings/2,  
                      width= width[i], spacing=  spacing, num_rings=number_rings)
    return edges   



def trans_tf_to_td(tf, dtype = 'dframe'):
    '''July 02, 2015, Y.G.@CHX
    Translate epoch time to string
    '''
    import pandas as pd
    import numpy as np
    import datetime
    '''translate time.float to time.date,
       td.type dframe: a dataframe
       td.type list,   a list
    ''' 
    if dtype is 'dframe':ind = tf.index
    else:ind = range(len(tf))    
    td = np.array([ datetime.datetime.fromtimestamp(tf[i]) for i in ind ])
    return td



def trans_td_to_tf(td, dtype = 'dframe'):
    '''July 02, 2015, Y.G.@CHX
    Translate string to epoch time
    
    '''
    import time
    import numpy as np
    '''translate time.date to time.float,
       td.type dframe: a dataframe
       td.type list,   a list
    ''' 
    if dtype is 'dframe':ind = td.index
    else:ind = range(len(td))
    #tf = np.array([ time.mktime(td[i].timetuple()) for i in range(len(td)) ])
    tf = np.array([ time.mktime(td[i].timetuple()) for i in ind])
    return tf



def get_averaged_data_from_multi_res( multi_res, keystr='g2', different_length= True  ):
    '''Y.G. Dec 22, 2016
        get average data from multi-run analysis result
        Parameters:
            multi_res: dict, generated by function run_xpcs_xsvs_single
                       each key is a uid, inside each uid are also dict with key as 'g2','g4' et.al.
            keystr: string, get the averaged keystr
            different_length: if True, do careful average for different length results
        return:
            array, averaged results       
    
    '''
    maxM = 0
    mkeys = multi_res.keys()
    if not different_length:
        n=0
        for i, key in enumerate( list( mkeys) ):
            keystri = multi_res[key][keystr]
            if i ==0: 
                keystr_average = keystri
            else:
                keystr_average += keystri
            n +=1
        keystr_average /=n
    
    else:
        length_dict = {}  
        D= 1
        
        for i, key in enumerate( list( mkeys) ):
            shapes = multi_res[key][keystr].shape
            M=shapes[0]    
            if i ==0:                         
                if len(shapes)==2:
                    D=2                    
                    maxN = shapes[1] 
                elif len(shapes)==3:
                    D=3                   
                    maxN = shapes[2]    #in case of two-time correlation                
            if (M) not in length_dict:
                length_dict[(M) ] =1
            else:
                length_dict[(M) ] += 1
            maxM = max( maxM, M )   
        #print( length_dict )
        avg_count = {}
        sk = np.array( sorted(length_dict) ) 
        for i, k in enumerate( sk ):
            avg_count[k] = np.sum(  np.array( [ length_dict[k] for k in sk[i:] ] )   )            
        #print(length_dict, avg_count)        
        if D==2:
            keystr_average = np.zeros( [maxM, maxN] )
        elif D==3:
            keystr_average = np.zeros( [maxM, maxM, maxN ] )           
        else:
            keystr_average = np.zeros( [maxM] )
        for i, key in enumerate( list( mkeys) ):
            keystri = multi_res[key][keystr]
            Mi = keystri.shape[0]   
            if D!=3:
                keystr_average[:Mi] +=   keystri
            else:
                keystr_average[:Mi,:Mi,:] +=   keystri
        if D!=3:        
            keystr_average[:sk[0]] /= avg_count[sk[0]]  
        else:
            keystr_average[:sk[0],:sk[0], : ] /= avg_count[sk[0]] 
        for i in range( 0, len(sk)-1 ):
            if D!=3:   
                keystr_average[sk[i]:sk[i+1]] /= avg_count[sk[i+1]] 
            else:
                keystr_average[sk[i]:sk[i+1],sk[i]:sk[i+1],:] /= avg_count[sk[i+1]] 
            
    return keystr_average


def save_g2_general(  g2, taus, qr=None, qz=None, uid='uid', path=None, return_res= False ):
    
    '''Y.G. Dec 29, 2016
    
        save g2 results, 
       res_pargs should contain
           g2: one-time correlation function
           taus, lags of g2
           qr: the qr center, same length as g2
           qz: the qz or angle center, same length as g2
           path:
           uid:
      
      '''    
    
    df = DataFrame(     np.hstack( [ (taus).reshape( len(g2),1) ,  g2] )  ) 
    t,qs = g2.shape
    if qr is None:
        qr = range( qs )
    if qz is None:        
        df.columns = (   ['tau'] + [str(qr_)  for qr_ in qr ]    )
    else:
        df.columns = (   ['tau'] + [ str(qr_) +'_'+ str(qz_)  for (qr_,qz_) in zip(qr,qz) ]    )
    
    #dt =datetime.now()
    #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    
    #if filename is None:
      
    filename = uid
    #filename = 'uid=%s--g2.csv' % (uid)
    #filename += '-uid=%s-%s.csv' % (uid,CurTime)   
    #filename += '-uid=%s.csv' % (uid) 
    filename1 = os.path.join(path, filename)
    df.to_csv(filename1)
    print( 'The correlation function is saved in %s with filename as %s'%( path, filename))
    if return_res:
        return df
    

###########
#*for g2 fit and plot

def stretched_auto_corr_scat_factor(x, beta, relaxation_rate, alpha=1.0, baseline=1):
    return beta * np.exp(-2 * (relaxation_rate * x)**alpha ) + baseline

def simple_exponential(x, beta, relaxation_rate,  baseline=1):
    return beta * np.exp(-2 * relaxation_rate * x) + baseline


def simple_exponential_with_vibration(x, beta, relaxation_rate, freq, amp,  baseline=1):
    return beta * (1 + amp*np.cos(  2*np.pi*freq* x) )* np.exp(-2 * relaxation_rate * x) + baseline

def stretched_auto_corr_scat_factor_with_vibration(x, beta, relaxation_rate, alpha, freq, amp,  baseline=1):
    return beta * (1 + amp*np.cos(  2*np.pi*freq* x) )* np.exp(-2 *  (relaxation_rate * x)**alpha ) + baseline


def flow_para_function_with_vibration( x, beta, relaxation_rate, flow_velocity, freq, amp, baseline=1):     
    vibration_part = (1 + amp*np.cos(  2*np.pi*freq* x) )    
    Diff_part=    np.exp(-2 * relaxation_rate * x)
    Flow_part =  np.pi**2/(16*x*flow_velocity) *  abs(  erf(  np.sqrt(   4/np.pi * 1j* x * flow_velocity ) ) )**2    
    return  beta* vibration_part* Diff_part * Flow_part + baseline

def flow_para_function( x, beta, relaxation_rate, flow_velocity, baseline=1):
    '''flow_velocity: q.v (q vector dot v vector = q*v*cos(angle) )'''
    
    Diff_part=    np.exp(-2 * relaxation_rate * x)
    Flow_part =  np.pi**2/(16*x*flow_velocity) *  abs(  erf(  np.sqrt(   4/np.pi * 1j* x * flow_velocity ) ) )**2    
    return  beta*Diff_part * Flow_part + baseline

def get_flow_velocity( average_velocity, shape_factor):  
    
    return average_velocity * (1- shape_factor)/(1+  shape_factor)

def stretched_flow_para_function( x, beta, relaxation_rate, alpha, flow_velocity, baseline=1):  
    '''
    flow_velocity: q.v (q vector dot v vector = q*v*cos(angle) )
    '''
    Diff_part=    np.exp(-2 *  (relaxation_rate * x)**alpha   )
    Flow_part =  np.pi**2/(16*x*flow_velocity) *  abs(  erf(  np.sqrt(   4/np.pi * 1j* x * flow_velocity ) ) )**2    
    return  beta*Diff_part * Flow_part + baseline


def get_g2_fit_general_two_steps( g2, taus,  function='simple_exponential', 
                                 second_fit_range=[0,20],  
                       sequential_fit=False, *argv,**kwargs):
    '''
    Fit g2 in two steps,
    i)  Using the "function" to fit whole g2 to get baseline and beta (contrast)
    ii) Then using the obtained baseline and beta to fit g2 in a "second_fit_range" by using simple_exponential function
    '''
    g2_fit_result, taus_fit, g2_fit =  get_g2_fit_general( g2, taus,  function, sequential_fit, *argv,**kwargs)     
    guess_values = {}
    for k in list (g2_fit_result[0].params.keys()):
        guess_values[k] =   np.array( [ g2_fit_result[i].params[k].value
                                   for i in range(  g2.shape[1]  )      ]) 
        
    if 'guess_limits' in kwargs:         
        guess_limits  = kwargs['guess_limits'] 
    else:
         guess_limits = dict( baseline =[1, 1.8], alpha=[0, 2],
                            beta = [0., 1], relaxation_rate= [0.001, 10000])
            
    g2_fit_result, taus_fit, g2_fit =  get_g2_fit_general( g2, taus,  function ='simple_exponential', 
                                    sequential_fit= sequential_fit, fit_range=second_fit_range,
            fit_variables={'baseline':False, 'beta': False, 'alpha':False,'relaxation_rate':True},
                guess_values= guess_values,  guess_limits = guess_limits )         
        
    return g2_fit_result, taus_fit, g2_fit


def get_g2_fit_general( g2, taus,  function='simple_exponential', 
                       sequential_fit=False, *argv,**kwargs):
    '''
    Dec 29,2016, Y.G.@CHX
    
    Fit one-time correlation function
    
    The support functions include simple exponential and stretched/compressed exponential
    Parameters
    ---------- 
        g2: one-time correlation function for fit, with shape as [taus, qs]
        taus: the time delay
        sequential_fit: if True, will use the low-q fit result as initial value to fit the higher Qs
        function: 
            supported function include:
                'simple_exponential' (or 'simple'): fit by a simple exponential function, defined as  
                        beta * np.exp(-2 * relaxation_rate * lags) + baseline
                'streched_exponential'(or 'streched'): fit by a streched exponential function, defined as  
                        beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline
                 'stretched_vibration':   fit by a streched exponential function with vibration, defined as            
                     beta * (1 + amp*np.cos(  2*np.pi*60* x) )* np.exp(-2 * relaxation_rate * x)**alpha + baseline
                 'flow_para_function' (or flow): fit by a flow function
         
                    
    kwargs:
        could contains:
            'fit_variables': a dict, for vary or not, 
                                keys are fitting para, including 
                                    beta, relaxation_rate , alpha ,baseline
                                values: a False or True, False for not vary
            'guess_values': a dict, for initial value of the fitting para,
                            the defalut values are 
                                dict( beta=.1, alpha=1.0, relaxation_rate =0.005, baseline=1.0)
                                
            'guess_limits': a dict, for the limits of the fittting para, for example:
                                dict( beta=[0, 10],, alpha=[0,100] )
                            the default is:
                                dict( baseline =[0.5, 2.5], alpha=[0, inf] ,beta = [0, 1], relaxation_rate= [0.0,1000]  )
    Returns
    -------        
    fit resutls: a instance in limfit
    tau_fit
    fit_data by the model, it has the q number of g2
         
    an example:
        fit_g2_func = 'stretched'
        g2_fit_result, taus_fit, g2_fit = get_g2_fit_general( g2,  taus, 
                        function = fit_g2_func,  vlim=[0.95, 1.05], fit_range= None,  
                        fit_variables={'baseline':True, 'beta':True, 'alpha':True,'relaxation_rate':True},
                        guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,}) 
            
        g2_fit_paras = save_g2_fit_para_tocsv(g2_fit_result,  filename= uid_  +'_g2_fit_paras.csv', path=data_dir )        
        
        
    '''      
    
    if 'fit_range' in kwargs.keys():
        fit_range = kwargs['fit_range'] 
    else:
        fit_range=None    

        
    num_rings = g2.shape[1]    
    if 'fit_variables' in kwargs:
        additional_var  = kwargs['fit_variables']        
        _vars =[ k for k in list( additional_var.keys()) if additional_var[k] is False]
    else:
        _vars = []        
    if function=='simple_exponential' or function=='simple':
        _vars = np.unique ( _vars + ['alpha']) 
        mod = Model(stretched_auto_corr_scat_factor)#,  independent_vars= list( _vars)   )        
    elif function=='stretched_exponential' or function=='stretched':        
        mod = Model(stretched_auto_corr_scat_factor)#,  independent_vars=  _vars)       
    elif function=='stretched_vibration':        
        mod = Model(stretched_auto_corr_scat_factor_with_vibration)#,  independent_vars=  _vars)        
    elif function=='flow_para_function' or  function=='flow_para':         
        mod = Model(flow_para_function)#,  independent_vars=  _vars) 
    elif function=='flow_para_function_with_vibration' or  function=='flow_vibration':            
        mod = Model( flow_para_function_with_vibration )
        
    else:
        print ("The %s is not supported.The supported functions include simple_exponential and stretched_exponential"%function)    
        
    mod.set_param_hint( 'baseline',   min=0.5, max= 2.5 )
    mod.set_param_hint( 'beta',   min=0.0,  max=1.0 )
    mod.set_param_hint( 'alpha',   min=0.0 )
    mod.set_param_hint( 'relaxation_rate',   min=0.0,  max= 1000  )  
    
    if 'guess_limits' in kwargs:         
        guess_limits  = kwargs['guess_limits']         
        for k in list(  guess_limits.keys() ):
            mod.set_param_hint( k,   min=   guess_limits[k][0], max= guess_limits[k][1] )           
    
    if function=='flow_para_function' or  function=='flow_para' or function=='flow_vibration': 
        mod.set_param_hint( 'flow_velocity', min=0)        
    if function=='stretched_vibration' or function=='flow_vibration':     
        mod.set_param_hint( 'freq', min=0)
        mod.set_param_hint( 'amp', min=0)
        
    _guess_val = dict( beta=.1, alpha=1.0, relaxation_rate =0.005, baseline=1.0)    
    if 'guess_values' in kwargs:         
        guess_values  = kwargs['guess_values']         
        _guess_val.update( guess_values )  
   
    _beta=_guess_val['beta']
    _alpha=_guess_val['alpha']
    _relaxation_rate = _guess_val['relaxation_rate']
    _baseline= _guess_val['baseline']    
    pars  = mod.make_params( beta=_beta, alpha=_alpha, relaxation_rate =_relaxation_rate, baseline= _baseline)
    if function=='flow_para_function' or  function=='flow_para':
        _flow_velocity =_guess_val['flow_velocity']    
        pars  = mod.make_params( beta=_beta, alpha=_alpha, flow_velocity=_flow_velocity,
                                relaxation_rate =_relaxation_rate, baseline= _baseline)
        
    if function=='stretched_vibration':
        _freq =_guess_val['freq'] 
        _amp = _guess_val['amp'] 
        pars  = mod.make_params( beta=_beta, alpha=_alpha, freq=_freq, amp = _amp,
                                relaxation_rate =_relaxation_rate, baseline= _baseline)
        
    if  function=='flow_vibration':
        _flow_velocity =_guess_val['flow_velocity']    
        _freq =_guess_val['freq'] 
        _amp = _guess_val['amp'] 
        pars  = mod.make_params( beta=_beta,  freq=_freq, amp = _amp,flow_velocity=_flow_velocity,
                                relaxation_rate =_relaxation_rate, baseline= _baseline)        
        
    for v in _vars:
        pars['%s'%v].vary = False
    #print( pars )
    fit_res = []
    model_data = []    
    for i in range(num_rings):  
        if fit_range is not None:
            y=g2[1:, i][fit_range[0]:fit_range[1]]
            lags=taus[1:][fit_range[0]:fit_range[1]] 
        else:
            y=g2[1:, i]
            lags=taus[1:]     
        #print( _relaxation_rate )
        for k in list(pars.keys()):
            #print(k, _guess_val[k]  )
            if isinstance( _guess_val[k], (np.ndarray, list) ):
                pars[k].value = _guess_val[k][i]  
        
        if False:
            if isinstance( _beta, (np.ndarray, list) ):
                 pars['beta'].value = _guess_val['beta'][i]      
            if isinstance( _baseline, (np.ndarray, list) ):
                 pars['baseline'].value = _guess_val['baseline'][i]                
            if isinstance( _relaxation_rate, (np.ndarray, list) ):
                 pars['relaxation_rate'].value = _guess_val['relaxation_rate'][i]               
            if isinstance( _alpha, (np.ndarray, list) ):
                 pars['alpha'].value = _guess_val['alpha'][i] 
                
            #for k in list(pars.keys()):
                #print(k, _guess_val[k]  )
            # pars[k].value = _guess_val[k][i]        
        
        result1 = mod.fit(y, pars, x =lags ) 
        if sequential_fit:
            for k in list(pars.keys()):
                #print( pars )
                if k in list(result1.best_values.keys()):
                    pars[k].value = result1.best_values[k]  
                
        fit_res.append( result1) 
        model_data.append(  result1.best_fit )
    return fit_res, lags, np.array( model_data ).T




def get_short_long_labels_from_qval_dict(qval_dict, geometry='saxs'):
    '''Y.G. 2016, Dec 26
        Get short/long labels from a qval_dict
        Parameters
        ----------  
        qval_dict, dict, with key as roi number,
                        format as {1: [qr1, qz1], 2: [qr2,qz2] ...} for gi-saxs
                        format as {1: [qr1], 2: [qr2] ...} for saxs
                        format as {1: [qr1, qa1], 2: [qr2,qa2], ...] for ang-saxs
        geometry:
            'saxs':  a saxs with Qr partition
            'ang_saxs': a saxs with Qr and angular partition
            'gi_saxs': gisaxs with Qz, Qr            
    '''

    Nqs = len( qval_dict.keys())
    len_qrz = len( list( qval_dict.values() )[0] )
    qr_label = np.array( list( qval_dict.values() ) )[:,0]
    if geometry=='gi_saxs' or geometry=='ang_saxs':# or geometry=='gi_waxs':
        if len_qrz < 2:
            print( "please give qz or qang for the q-label")
        else:
            qz_label = np.array( list( qval_dict.values() ) )[:,1] 
    else:
        qz_label = np.array(   [0]    ) 
        
    uqz_label = np.unique( qz_label )
    num_qz = len( uqz_label)
    
    uqr_label = np.unique( qr_label )
    num_qr = len( uqr_label)       
    
    #print( uqr_label, uqz_label )
    if len( uqr_label ) >=  len( uqz_label ):
        master_plot= 'qz'  #one qz for many sub plots of each qr 
    else:
        master_plot= 'qr' 

    mastp=  master_plot    
    if geometry == 'ang_saxs':
        mastp= 'ang'   
    num_short = min(num_qz, num_qr)
    num_long =  max(num_qz, num_qr)
    
    #print( mastp, num_short, num_long)
    if num_qz != num_qr:
        short_label = [qz_label,qr_label][ np.argmin( [num_qz, num_qr]    ) ]
        long_label  = [qz_label,qr_label][ np.argmax( [num_qz, num_qr]    ) ]
        short_ulabel = [uqz_label,uqr_label][ np.argmin( [num_qz, num_qr]    ) ]
        long_ulabel  = [uqz_label,uqr_label][ np.argmax( [num_qz, num_qr]    ) ]
    else:
        short_label = qz_label
        long_label  = qr_label
        short_ulabel = uqz_label
        long_ulabel  = uqr_label        
    #print( long_ulabel )    
    #print( qz_label,qr_label )
    #print( short_label, long_label ) 
        
    if geometry == 'saxs' or geometry == 'gi_waxs':
        ind_long = [ range( num_long )  ] 
    else:
        ind_long = [ np.where( short_label == i)[0] for i in short_ulabel ] 
        
        
    if Nqs  == 1:
        long_ulabel = list( qval_dict.values() )[0]
        long_label = list( qval_dict.values() )[0]
    return qr_label, qz_label, num_qz, num_qr, num_short,num_long, short_label, long_label,short_ulabel,long_ulabel, ind_long, master_plot, mastp
        
        
############################################
##a good func to plot g2 for all types of geogmetries
############################################ 
 
    
    

def plot_g2_general( g2_dict, taus_dict, qval_dict, fit_res=None,  geometry='saxs',filename='g2', 
                    path=None, function='simple_exponential',  g2_labels=None, 
                    fig_ysize= 12, qth_interest = None,
                    ylabel='g2',  return_fig=False, append_name='', outsize=(2000, 2400), 
                    max_plotnum_fig=16, figsize=(10, 12), show_average_ang_saxs=True,
                    qphi_analysis = False,
                    *argv,**kwargs):    
    '''
    Dec 26,2016, Y.G.@CHX
    
    Plot one/four-time correlation function (with fit) for different geometry
    
    The support functions include simple exponential and stretched/compressed exponential
    Parameters
    ---------- 
    g2_dict: dict, format as {1: g2_1, 2: g2_2, 3: g2_3...} one-time correlation function, g1,g2, g3,...must have the same shape
    taus_dict, dict, format {1: tau_1, 2: tau_2, 3: tau_3...}, tau1,tau2, tau3,...must have the same shape
    qval_dict, dict, with key as roi number,
                    format as {1: [qr1, qz1], 2: [qr2,qz2] ...} for gi-saxs
                    format as {1: [qr1], 2: [qr2] ...} for saxs
                    format as {1: [qr1, qa1], 2: [qr2,qa2], ...] for ang-saxs
                    
    fit_res: give all the fitting parameters for showing in the plot    

    filename: for the title of plot
    append_name: if not None, will save as filename + append_name as filename
    path: the path to save data        
    outsize: for gi/ang_saxs, will combine all the different qz images together with outsize    
    function: 
        'simple_exponential': fit by a simple exponential function, defined as  
                    beta * np.exp(-2 * relaxation_rate * lags) + baseline
        'streched_exponential': fit by a streched exponential function, defined as  
                    beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline  
    geometry:
        'saxs':  a saxs with Qr partition
        'ang_saxs': a saxs with Qr and angular partition
        'gi_saxs': gisaxs with Qz, Qr
                  
    one_plot: if True, plot all images in one pannel 
    kwargs:
                    
    Returns
    -------        
    None

    ToDoList: plot an average g2 for ang_saxs for each q
    
    '''    

    if ylabel=='g2':
        ylabel='g_2'
    if ylabel=='g4':
        ylabel='g_4' 
        
    if geometry =='saxs':
        if qphi_analysis:
            geometry = 'ang_saxs'
    (qr_label, qz_label, num_qz, num_qr, num_short,
     num_long, short_label, long_label,short_ulabel,
     long_ulabel,ind_long, master_plot,
     mastp) = get_short_long_labels_from_qval_dict(qval_dict, geometry=geometry)  
    fps = [] 
    
    #$print( num_short, num_long )
    
    for s_ind in range( num_short  ):
        ind_long_i = ind_long[ s_ind ]
        num_long_i = len( ind_long_i )
        #if show_average_ang_saxs:
        #    if geometry=='ang_saxs':
        #        num_long_i += 1        
        if RUN_GUI:
            fig = Figure(figsize=(10, 12))            
        else:
            #fig = plt.figure( )
            if num_long_i <=4:
                if master_plot != 'qz':
                    fig = plt.figure(figsize=(8, 6))   
                else:
                    if num_short>1:
                        fig = plt.figure(figsize=(8, 4))
                    else:
                        fig = plt.figure(figsize=(10, 6))
                    #print('Here')
            elif num_long_i > max_plotnum_fig:
                num_fig = int(np.ceil(num_long_i/max_plotnum_fig)) #num_long_i //16
                fig = [ plt.figure(figsize=figsize)  for i in range(num_fig) ]
                #print( figsize )
            else:
                #print('Here')
                if master_plot != 'qz':
                    fig = plt.figure(figsize=figsize)
                else:
                    fig = plt.figure(figsize=(10, 10))
        
        if master_plot == 'qz':
            if geometry=='ang_saxs':
                title_short = 'Angle= %.2f'%( short_ulabel[s_ind] )  + r'$^\circ$'                               
            elif geometry=='gi_saxs':
                title_short = r'$Q_z= $' + '%.4f'%( short_ulabel[s_ind] ) + r'$\AA^{-1}$'
            else:
                title_short = ''            
        else: #qr
            if geometry=='ang_saxs' or geometry=='gi_saxs':
                title_short =   r'$Q_r= $' + '%.5f  '%( short_ulabel[s_ind] ) + r'$\AA^{-1}$'            
            else:
                title_short=''        
                
        #filename =''
        til = '%s:--->%s'%(filename,  title_short )

        if num_long_i <=4:            
            plt.title( til,fontsize= 14, y =1.15)  
        else:
            plt.title( til,fontsize=20, y =1.06)                 
        
        #print( num_long )
        
        if num_long!=1:   
            #print( 'here')
            plt.axis('off')             
            #sy =   min(num_long_i,4) 
            sy =   min(num_long_i,  int( np.ceil( min(max_plotnum_fig,num_long_i)/4))   ) 
            #fig.set_size_inches(10, 12)
            #fig.set_size_inches(10, fig_ysize )
        else: 
            sy =1
            #fig.set_size_inches(8,6)
            
            #plt.axis('off') 
        sx = min(4, int( np.ceil( min(max_plotnum_fig,num_long_i)/float(sy) ) ))
        
        temp = sy
        sy = sx
        sx = temp
        
        #print( num_long_i, sx, sy )
        #print( master_plot )
        #print(ind_long_i, len(ind_long_i) )
        
        for i, l_ind in enumerate( ind_long_i ):  
            if num_long_i <= max_plotnum_fig:
                #print('Here')
                #print(i, l_ind,long_label[l_ind] )                
                ax = fig.add_subplot(sx,sy, i + 1 ) 
            else:
                #fig_subnum = l_ind//max_plotnum_fig
                #ax = fig[fig_subnum].add_subplot(sx,sy, i + 1 - fig_subnum*max_plotnum_fig) 
                fig_subnum = i//max_plotnum_fig
                #print(  i, sx,sy, fig_subnum, max_plotnum_fig, i + 1 - fig_subnum*max_plotnum_fig )
                ax = fig[fig_subnum].add_subplot(sx,sy, i + 1 - fig_subnum*max_plotnum_fig) 
                
                  
            ax.set_ylabel( r"$%s$"%ylabel + '(' + r'$\tau$' + ')' ) 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)         
            if master_plot == 'qz' or master_plot == 'angle':                 
                title_long =  r'$Q_r= $'+'%.5f  '%( long_label[l_ind]  ) + r'$\AA^{-1}$'  
                #print(  long_label   )
            else:             
                if geometry=='ang_saxs':
                    #title_long = 'Ang= ' + '%.2f'%(  long_label[l_ind] ) + r'$^\circ$' + '( %d )'%(l_ind)
                    title_long = 'Ang= ' + '%.2f'%(  long_label[l_ind] ) #+ r'$^\circ$' + '( %d )'%(l_ind)
                elif geometry=='gi_saxs':
                    title_long =   r'$Q_z= $'+ '%.5f  '%( long_label[l_ind]  ) + r'$\AA^{-1}$'                  
                else:
                    title_long = ''    

            if master_plot != 'qz':
                ax.set_title(title_long + ' (%s  )'%(1+l_ind), y =1.1, fontsize=12) 
            else:                
                ax.set_title(title_long + ' (%s  )'%(1+l_ind), y =1.05, fontsize=12) 
            
            
            for ki, k in enumerate( list(g2_dict.keys()) ):                
                if ki==0:
                    c='b'
                    if fit_res is None:
                        m='-o'                        
                    else:
                        m='o'                        
                elif ki==1:
                    c='r'
                    if fit_res is None:
                        m='s'                        
                    else:
                        m='-'                
                elif ki==2:
                    c='g'
                    m='-D'
                else:
                    c = colors[ki+2]
                    m= '-%s'%markers[ki+2] 
                try:
                    dumy = g2_dict[k].shape
                    #print( 'here is the shape' )
                    islist = False 
                except:
                    islist_n = len( g2_dict[k] )
                    islist = True
                    #print( 'here is the list' )                    
                if islist:
                    for nlst in range( islist_n ):
                        m = '-%s'%markers[ nlst ]  
                        #print(m)
                        y=g2_dict[k][nlst][:, l_ind ]
                        x = taus_dict[k][nlst]
                        if ki==0:
                            ymin,ymax = min(y), max(y[1:])
                        if g2_labels is None:                             
                            ax.semilogx(x, y, m, color=c,  markersize=6) 
                        else:
                            #print('here ki ={} nlst = {}'.format( ki, nlst ))
                            if nlst==0:
                                ax.semilogx(x, y, m,  color=c,markersize=6, label=g2_labels[ki]) 
                            else:
                                ax.semilogx(x, y, m,  color=c,markersize=6)
                                
                        if nlst==0:
                            if l_ind==0:
                                ax.legend(loc='best', fontsize = 8, fancybox=True, framealpha=0.5)             
                
                else:    
                    y=g2_dict[k][:, l_ind ]    
                    x = taus_dict[k]
                    if ki==0:
                        ymin,ymax = min(y), max(y[1:])
                    if g2_labels is None:    
                        ax.semilogx(x, y, m, color=c,  markersize=6) 
                    else:
                        ax.semilogx(x, y, m,  color=c,markersize=6, label=g2_labels[ki]) 
                        if l_ind==0:
                            ax.legend(loc='best', fontsize = 8, fancybox=True, framealpha=0.5)                   

            if fit_res is not None:
                result1 = fit_res[l_ind]    
                #print (result1.best_values)
                rate = result1.best_values['relaxation_rate']
                beta = result1.best_values['beta'] 
                baseline =  result1.best_values['baseline'] 
                if function=='simple_exponential' or function=='simple':
                    alpha =1.0 
                elif function=='stretched_exponential' or function=='stretched':
                    alpha = result1.best_values['alpha']
                elif function=='stretched_vibration': 
                    alpha = result1.best_values['alpha']
                    freq = result1.best_values['freq'] 
                elif function=='flow_vibration': 
                    freq = result1.best_values['freq'] 
                if function=='flow_para_function' or  function=='flow_para' or  function=='flow_vibration': 
                    flow = result1.best_values['flow_velocity']                          

                if rate!=0:
                    txts = r'$\tau_0$' + r'$ = %.3f$'%(1/rate) +  r'$ s$'
                else:
                    txts = r'$\tau_0$' + r'$ = inf$' +  r'$ s$'
                x=0.25
                y0=0.9
                fontsize = 12
                ax.text(x =x, y= y0, s=txts, fontsize=fontsize, transform=ax.transAxes) 
                #print(function)
                dt=0
                if function!='flow_para_function' and  function!='flow_para' and function!='flow_vibration':
                    txts = r'$\alpha$' + r'$ = %.3f$'%(alpha)  
                    dt +=0.1
                    #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
                    ax.text(x =x, y= y0-dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                    
                txts = r'$baseline$' + r'$ = %.3f$'%( baseline) 
                dt +=0.1
                ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                
                if function=='flow_para_function' or  function=='flow_para' or  function=='flow_vibration': 
                    txts = r'$flow_v$' + r'$ = %.3f$'%( flow) 
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)                        
                if function=='stretched_vibration'  or  function=='flow_vibration': 
                    txts = r'$vibration$' + r'$ = %.1f Hz$'%( freq) 
                    dt += 0.1
                    ax.text(x =x, y= y0-dt, s=txts, fontsize=fontsize, transform=ax.transAxes)  
                
                txts = r'$\beta$' + r'$ = %.3f$'%( beta ) 
                dt +=0.1
                ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                

            if 'ylim' in kwargs:
                ax.set_ylim( kwargs['ylim'])
            elif 'vlim' in kwargs:
                vmin, vmax =kwargs['vlim']
                ax.set_ylim([ymin*vmin, ymax*vmax ]) 
                
            else:
                pass
            if 'xlim' in kwargs:
                ax.set_xlim( kwargs['xlim'])
        if   num_short == 1: 
            fp = path + filename 
        else:
            fp = path + filename + '_%s_%s'%(mastp, s_ind)   
            
        if append_name is not '':
            fp = fp + append_name
        fps.append( fp  + '.png' )  
        #if num_long_i <= 16:
        if num_long_i <= max_plotnum_fig:            
            fig.set_tight_layout(True)    
            #fig.tight_layout()
            plt.savefig( fp + '.png', dpi=fig.dpi) 
        else:
            fps=[]
            for fn, f in enumerate(fig):
                f.set_tight_layout(True)
                fp = path + filename + '_q_%s_%s'%(fn*16, (fn+1)*16) 
                if append_name is not '':
                    fp = fp + append_name
                fps.append( fp  + '.png' )  
                f.savefig( fp + '.png', dpi=f.dpi)
        #plt.savefig( fp + '.png', dpi=fig.dpi)        
    #combine each saved images together
    
    if (num_short !=1) or (num_long_i > 16):
        outputfile =  path + filename + '.png'
        if append_name is not '':
            outputfile =  path + filename  + append_name + '__joint.png'
        else:
            outputfile =  path + filename   + '__joint.png'
        combine_images( fps, outputfile, outsize= outsize )    
    if return_fig:
        return fig   
    

def power_func(x, D0, power=2):
    return D0 * x**power


def get_q_rate_fit_general( qval_dict, rate, geometry ='saxs', weights=None, *argv,**kwargs): 
    '''
    Dec 26,2016, Y.G.@CHX
    
    Fit q~rate by a power law function and fit curve pass  (0,0)   
     
    Parameters
    ----------  
    qval_dict, dict, with key as roi number,
                    format as {1: [qr1, qz1], 2: [qr2,qz2] ...} for gi-saxs
                    format as {1: [qr1], 2: [qr2] ...} for saxs
                    format as {1: [qr1, qa1], 2: [qr2,qa2], ...] for ang-saxs
    rate: relaxation_rate

    Option:
    if power_variable = False, power =2 to fit q^2~rate, 
                Otherwise, power is variable.
    Return:
    D0
    qrate_fit_res
    '''
    
    power_variable=False
                
    if 'fit_range' in kwargs.keys():
        fit_range = kwargs['fit_range']         
    else:    
        fit_range= None
        
    mod = Model( power_func )
    #mod.set_param_hint( 'power',   min=0.5, max= 10 )
    #mod.set_param_hint( 'D0',   min=0 )
    pars  = mod.make_params( power = 2, D0=1*10^(-5) )
    if power_variable:
        pars['power'].vary = True
    else:
        pars['power'].vary = False
        
    (qr_label, qz_label, num_qz, num_qr, num_short,
     num_long, short_label, long_label,short_ulabel,
     long_ulabel,ind_long, master_plot,
     mastp) = get_short_long_labels_from_qval_dict(qval_dict, geometry=geometry)
        
    Nqr = num_long
    Nqz = num_short    
    D0= np.zeros( Nqz )
    power= 2 #np.zeros( Nqz )
    qrate_fit_res=[]    
    #print(Nqz)    
    for i  in range(Nqz):        
        ind_long_i = ind_long[ i ]
        y = np.array( rate  )[ind_long_i]        
        x =   long_label[ind_long_i] 
        #print(y,x)
        if fit_range is not None:
            y=y[fit_range[0]:fit_range[1]]
            x=x[fit_range[0]:fit_range[1]]             
        #print (i, y,x)          
        _result = mod.fit(y, pars, x = x ,weights=weights )
        qrate_fit_res.append(  _result )
        D0[i]  = _result.best_values['D0']
        #power[i] = _result.best_values['power']  
        print ('The fitted diffusion coefficient D0 is:  %.3e   A^2S-1'%D0[i])
    return D0, qrate_fit_res


def plot_q_rate_fit_general( qval_dict, rate, qrate_fit_res, geometry ='saxs',  ylim = None, 
                            plot_all_range=True, plot_index_range = None, show_text=True,return_fig=False,
                            show_fit=True,
                            *argv,**kwargs): 
    '''
    Dec 26,2016, Y.G.@CHX
    
    plot q~rate fitted by a power law function and fit curve pass  (0,0)   
     
    Parameters
    ----------  
    qval_dict, dict, with key as roi number,
                    format as {1: [qr1, qz1], 2: [qr2,qz2] ...} for gi-saxs
                    format as {1: [qr1], 2: [qr2] ...} for saxs
                    format as {1: [qr1, qa1], 2: [qr2,qa2], ...] for ang-saxs
    rate: relaxation_rate
    plot_index_range:  
    Option:
    if power_variable = False, power =2 to fit q^2~rate, 
                Otherwise, power is variable.
    show_fit:, bool, if False, not show the fit
    
    ''' 
    
    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
    else:
        uid = 'uid' 
    if 'path' in kwargs.keys():
        path = kwargs['path'] 
    else:
        path = ''         
    (qr_label, qz_label, num_qz, num_qr, num_short,
     num_long, short_label, long_label,short_ulabel,
     long_ulabel,ind_long, master_plot,
     mastp) = get_short_long_labels_from_qval_dict(qval_dict, geometry=geometry)
         
    power = 2
    fig,ax = plt.subplots()
    plt.title(r'$Q^%s$'%(power) + '-Rate-%s_Fit'%(uid),fontsize=20, y =1.06)
    Nqz = num_short  
    if Nqz!=1:
        ls = '--'
    else:
        ls=''    
    for i  in range(Nqz):
        ind_long_i = ind_long[ i ]        
        y = np.array( rate  )[ind_long_i]        
        x =   long_label[ind_long_i]
        D0  = qrate_fit_res[i].best_values['D0']         
        #print(i,  x, y, D0 )        
        if Nqz!=1:
            label=r'$q_z=%.5f$'%short_ulabel[i]
        else:
            label=''
        ax.plot(x**power,  y, marker = 'o', ls =ls, label=label)
        yfit = qrate_fit_res[i].best_fit

        if show_fit:  
            if plot_all_range:
                ax.plot(x**power, x**power*D0,  '-r') 
            else:        
                ax.plot( (x**power)[:len(yfit) ], yfit,  '-r')  
            
            if show_text:
                txts = r'$D0: %.3e$'%D0 + r' $A^2$' + r'$s^{-1}$'
                dy=0.1
                ax.text(x =0.15, y=.65 -dy *i, s=txts, fontsize=14, transform=ax.transAxes) 
        if Nqz!=1:legend = ax.legend(loc='best')

    if plot_index_range is not None:
        d1,d2 = plot_index_range
        d2 = min( len(x)-1, d2 )             
        ax.set_xlim(  (x**power)[d1], (x**power)[d2]  )
        ax.set_ylim( y[d1],y[d2])
    if ylim is not None:
        ax.set_ylim( ylim )
        
    ax.set_ylabel('Relaxation rate 'r'$\gamma$'"($s^{-1}$)")
    ax.set_xlabel("$q^%s$"r'($\AA^{-2}$)'%power)
    fp = path + '%s_Q_Rate'%(uid) + '_fit.png'
    fig.savefig( fp, dpi=fig.dpi)
    fig.tight_layout()
    if return_fig:
        return fig,ax
  

def save_g2_fit_para_tocsv( fit_res, filename, path):
    '''Y.G. Dec 29, 2016, 
    save g2 fitted parameter to csv file
    '''
    col = list( fit_res[0].best_values.keys() )
    m,n = len( fit_res ), len( col )
    data = np.zeros( [m,n] )
    for i in range( m ):
        data[i] = list( fit_res[i].best_values.values() )
    df = DataFrame( data ) 
    df.columns = col    
    filename1 = os.path.join(path, filename) # + '.csv')
    df.to_csv(filename1)
    print( "The g2 fitting parameters are saved in %s"%filename1)
    return df
    
            
