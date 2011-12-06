import time
import numpy as np
from scipy.ndimage import maximum_filter1d
import matplotlib.pylab as plt
#from sigma_clip import sigma_clipping
# TBD: Should this be separated in code that requires sherpa and code that does not?
import sherpa.astro.ui as ui

import shmodelshelper as smh

def findlines(x, y, fwhm, smoothwindow = 'hanning', sigma_threshold = 3.):

    '''
    Several things here and I am not quite sure yet what turn out to be  useful
    - smoothing: show real peaks and not just noise
    - maximum_filter = array: will find the peaks
    - sigma_clipping = are the peaks large enough to be relevant?
    
    Parameters
    ----------
    x : ndarray
        x values, e.g. wavelength
    y : ndarray
        y values, e.g. flux or res_flux / error
    fwhm : float
        estimate for FWHM of lines. Used as smoothing scale
    smoothwindow : string or None
        if `smoothwindow` is on of `['flat', 'hanning', 'hamming',
        'bartlett', 'blackman']` a correspondig window function 
        will be used to smooth the signal before line detection.
    
    Returns
    -------
    peaks : ndarray
        index numbers for peaks found
    '''
    fwhminpix = int(fwhm / np.diff(x).mean())
    if smoothwindow is not None:
        #print smoothwindow
        #print fwhminpix
        y = smooth(y, window_len = 3*fwhminpix, window = smoothwindow)

    maxindex = (maximum_filter1d(y, max(fwhminpix,3)) == y)
    maxindex = maxindex & (y > (y.mean() + sigma_threshold * y.std()))
    # sigma_clipping works only if there is plenty of continuum
    #clipped_y = sigma_clipping(y, threshold = sigma_threshold)
    # believe only peaks which are so large, that the get clipped by sigma_clipping
    #maxindex = maxindex & (clipped_y.mask == False)

    return np.flatnonzero(maxindex)

def smooth(x,window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
    x: ndarray
        the input signal 
    window_len: integer , optional
        The dimension of the smoothing window; should be an odd integer
    window: string, optional
        The type of window from `['flat', 'hanning', 'hamming', 'bartlett',
        'blackman']`. A 'flat' window will produce a moving average
        smoothing.

    Returns
    -------
    y : ndarray
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    See also
    --------
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

smoothwindow = 'hanning'
sigma_threshold = 2.

def mainloop(mymodel, fwhm, id = None, maxiter = 5, mindist = 0., do_plots = 0):
    
    if id is None:
        id = ui.get_default_id()
    data = ui.get_data(id)
    wave = data.get_indep()[0]
    error = data.get_error()[0]
    
    # model could habe been initalized with arbitrary values
    ui.fit(id) 

    for i in range(maxiter):
        oldmodel = smh.get_model_parts(id)
        res_flux = ui.get_resid_plot(id).y
        if smoothwindow is not None:
            fwhminpix = int(fwhm / np.diff(wave).mean())
            y = smooth(res_flux/error, window_len = 3*fwhminpix, window = smoothwindow)
        else:
            y = res_flux/error
        peaks = findlines(wave, y, fwhm, smoothwindow = None, sigma_threshold = sigma_threshold)
        if do_plots > 2:
            plt.figure()
            plt.plot(wave, res_flux/error, 's')
            for pos in mymodel.line_value_list('pos'):
                plt.plot([pos, pos], plt.ylim(),'k:')
            for peak in peaks:
                plt.plot([wave[peak], wave[peak]], plt.ylim())
            plt.plot(wave, y)
            plt.draw()
            
        for peak in peaks:
            if (len(mymodel.line_value_list('pos')) == 0) or (min(np.abs(mymodel.line_value_list('pos') - wave[peak])) >= mindist):
                mymodel.add_line(**mymodel.guess(wave, smooth(res_flux, window_len = 3*fwhminpix, window = smoothwindow), peak, fwhm = fwhm))
        newmodel = smh.get_model_parts(id)
        print 'Iteration {0:3n}: {1:3n} lines added'.format(i, len(newmodel) - len(oldmodel))
        
        if set(newmodel) == set(oldmodel):
            print 'No new lines added this step - fitting finished'
            break
        # Now do the fitting in Sherpa
        #ui.set_method('simplex')
        ui.fit(id)
        #ui.set_method('moncar')
        #ui.fit(id)
        
        if do_plots > 0:
            if do_plots > 1:
                plt.figure()
            else:
                plt.clf()
            ui.plot_fit(id)
            for pos in mymodel.line_value_list('pos'):
                plt.plot([pos, pos], plt.ylim(),'k:')
            for peak in peaks:
                plt.plot([wave[peak], wave[peak]], plt.ylim())
            plt.plot(wave, res_flux)
            plt.draw()
        

    else:
        print 'Max number of iterations reached'
    #model.cleanup() #remove lines running to 0 etc.
    return mymodel
