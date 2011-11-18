def findlines(x, y, fwhm, smoothwindow = 'hanning', sigma_threshold = 3.):

    '''
    Several things here and I am not quite sure yet what turn out to be  useful
    - smoothing: show real peaks and not just noise
    - maximum_filter = array: will find the peaks
    - sigma_clipping = are the peaks large enough to be relevant?
    
    y could be e.g. the flux, the residual flux after subtracting the
    model or als res_flux / error (where is significant stuff missing?)
    
    '''
    fwhminpix = int(fwhm / np.diff(x).mean())
    if smoothwindow:
        flux = smooth(y, window_len = fwhminpix, window = smoothwindow)

    maxindex = (maximum_filter1d(y, max(fwhminpix,3)) == y)
    maxindex = maxindex & (y > (y.mean() + sigma_threshold * y.std()))
    # sigma_clipping works only if there is plenty of continuum
    #clipped_y = sigma_clipping(y, threshold = sigma_threshold)
    # believe only peaks which are so large, that the get clipped by sigma_clipping
    #maxindex = maxindex & (clipped_y.mask == False)

    return maxindex.nonzero()[0]

def smooth(x,window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    window could accept even number 
       
    from http://www.scipy.org/Cookbook/SignalSmooth
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x

    # make it an odd number, so that reflection of values is same on each side
    if np.mod(window_len,2) != 1:
        window_len +=1

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[(window_len-1)/2:0:-1],x,x[-1:-window_len/2:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

