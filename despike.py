import numpy as np

def despike_background(y, length, threshold=3.5, order=6, verbose=False):
    """
    Despike the vector y using background estimation.
    
    Inputs:
        y
            (vector)
            Evenly spaced vector            
        length
            (int>0)
            Minimum size of real peaks
        threshold
            (float>0) 
            Threshold for detecting potential bad peaks (units of noise)            
        order
            (int>=0) 
            Order of the polynomial to use as background model  
        verbose
            (bool) 
            Print additional informations            
    """
    noise, INear, IFar, yBack = calculateNoise(y, threshold=threshold, order=order, cost_function='stq', verbose=verbose)

    I1 = IFar[0]
    loop_over_peaks = True
    i = 0

    while loop_over_peaks:
        while True:  # search the last point in this group
            i += 1
            if i > len(IFar)-1:  # we reached the end of the vector
                i = len(IFar)
                break
            if IFar[i] > IFar[i-1] + 1:  # the next point is not adjacent (it is in another group)
                break
        I2 = IFar[i-1] 

        if I2 - I1 < length:  # only keep small enough groups
            # index of points at left to fit bad points
            Ibegin = np.where(np.logical_and(INear < I1, INear > I1 - length/2))
            # index of points at right to fit bad points
            Iend = np.where(np.logical_and(INear > I2, INear < I2 + length/2))
            Ifit = np.concatenate((INear[Ibegin], INear[Iend]))  # index of the points used for fitting

            # points for fitting
            x_fit = np.array(Ifit)
            y_fit = np.array(y[Ifit])

            # order of the polynomial for fitting
            order = 2
            if len(x_fit) == 1:  # it not enough points, use lower order polynomials
                order = 0
            elif len(x_fit) == 2:
                order = 1
                
            # polynomial fitting
            poly = np.poly1d(np.polyfit(x_fit, y_fit, order))

            x_new = np.arange(I1, I2+1)  # position of the points to replace
            y[x_new] = poly(x_new)  # replace the bad points with fitted ones
    
        if i > len(IFar)-1:
            break

        I1 = IFar[i]
    

def calculateNoise(y, threshold=1, order=6, cost_function='atq', verbose=False):
    """
    Calculate the noise, the background, points close and far for vector y.
    
    We suppose that y is for evenly spaced position y(x).
    
    Inputs:
        y
            (vector)
            Evenly spaced vector
        threshold
            (float>0) 
            Threshold for Far points (units of noise)
        order
            (int>=0) 
            Order of the polynomial to use as background model 
        cost_function
            'sh', 'ah', 'stq', 'atq' 
            Cost function for background calculation. See backcor.backcor
        verbose
            (bool) 
            Print additional informations
    Outputs:
        noise
            (float) 
            Estimation of the noise of y. Using median absolute deviation (MAD) of y-yBack.
        INear
            (vector, int)
            Index of points near from yBack
        IFar
            (vector, int)
            Index of points far from yBAck
        yBack
            (vector)
            Background of y (polynomial)
    """
    from backcor import backcor

    # Calculate the noise (first estimation)
    MAD = np.median(np.abs(y - np.median(y)))
    if verbose:
        print("MAD (first estimation)")
        print(MAD)

    # Calculate the background
    yBack, poly, it = backcor(np.arange(len(y)), y, order, MAD, fct=cost_function)
    
    if verbose:
        pl.plot(y)
        pl.plot(yBack)
        pl.show()
        
    deviation = y - yBack

    # Calculate the noise
    noise = np.median(np.abs(deviation - np.median(deviation)))
    if verbose:
        print("MAD")
        print(noise)
        
    if verbose:
        print("threshold level:")
        print(threshold*noise)
              
    # detect the peaks far from the background
    INear = np.where(np.logical_and(deviation>=-threshold*noise, deviation<=threshold*noise))[0]
    IFar = np.where(np.logical_or(deviation<-threshold*noise, deviation>threshold*noise))[0]
    
    return noise, INear, IFar, yBack
