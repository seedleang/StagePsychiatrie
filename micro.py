# //////////////////////////////////////////////////////////////////////////////////

# Classical libraries
import pandas as pd
import numpy as np
import math
# Display tools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import mplot3d
# Handling the drift
import quat, vector, rotmat
from scipy import constants
g = constants.g
# Signal correlation
from scipy.ndimage import interpolation as ip
from scipy import integrate, interpolate as ipd
from scipy.signal import correlate
# Fourier Transforms
import numpy.fft as fft
colors = [(126/255,154/255,11/255 ), (127/255,155/255,12/255 ),
          (128/255,156/255,13/255 ), (129/255,157/255,14/255 ),
          (1      ,200/255,207/255),
          (1      ,162/255,      0), (120/255,      0,29/255 )] 
spectre = LinearSegmentedColormap.from_list('spectre', colors, N=100)
    
# //////////////////////////////////////////////////////////////////////////////////

'''1. Common tools for signal processing'''

# //////////////////////////////////////////////////////////////////////////////////

def kalmanFilter(data, att, eps, disp=False, afflim=None):

####################################################################################
#                                                                                  #
#  Using Kalman's approach to filter a fixed column in a DataFrame. This should    #
#  rule out any kind of noise that would be brought in by the measuring devices.   #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  data : pd.DataFrame --------------------------------- The DataFrame envelope.   #
#  att : int or string ---------------------------- The column we're looking at.   # 
#  eps : float ----------- A subjective estimate of the device's std. deviation.   #
#  disp : bool ----------------- Deciding whether to display the results or not.   # 
#  afflim : tuple ----------- Additional x- and y-limits for the display window.   #
#  Returns                                                                         #
#  ------------------------------------------------------------------------------  #
#  xhat : ndarray(N) -------------------------------------- The filtered column.   #
#  Examples                                                                        #
#  ------------------------------------------------------------------------------  #
#  >>> knct['MwX'] = kalmanFilter(knct,'MwX',0.2) ----------- filtering a signal   #
#  >>> knct['MwX'] -= kalmanFilter(knct,'MwX',2) -------- straightening a signal   #
#                                                                                  #
####################################################################################

    # Initialisation. --------------------------------------------------------------
    nIter = len(data[att])
    szArray = (nIter,)
    z = data[att] 
    x = -0.02   # truth value -----------------------------
    Q = 1e-5    # process variance ------------------------
    R = eps**2  # personal estimate of measurement variance
    
    # Allocating space. ------------------------------------------------------------
    xhat = np.zeros(szArray)      # post. estimate --------
    P = np.zeros(szArray)         # post. error estimate --
    xhatminus = np.zeros(szArray) # prior estimate --------
    Pminus = np.zeros(szArray)    # prior error estimate --
    K = np.zeros(szArray)         # gain or blending factor

    # Starting analysis. -----------------------------------------------------------
    xhat[0] = 0.0
    P[0] = 1.0
    for k in range(1,nIter):
        # ----- Time update --------------------------------------------------------
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
        # ----- Measurement update -------------------------------------------------
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    
    # Publishing the results. ------------------------------------------------------
    if disp:
        plt.figure(figsize=(8,4))
        plt.plot(data[att],'k+',label=att+' according to device')
        plt.plot(xhat, label='Post-Kalman '+att, color='indianred')
        plt.legend()
        if afflim : plt.ylim(afflim[0], afflim[1])
        plt.title('Kalman filtering on '+att)
        plt.xlabel('Iteration n°')
        plt.ylabel(att)
        plt.show()

    # Done. ------------------------------------------------------------------------
    return xhat

# //////////////////////////////////////////////////////////////////////////////////

'''2. Reading the Physilog devices' output'''

# //////////////////////////////////////////////////////////////////////////////////
    
def guessCalib(acci,settings,idtf):

####################################################################################
# Helps the user launch the analysis by displaying a preview plot.                 #
####################################################################################
   
    plt.figure(figsize=(16,3))
    ax = plt.axes()
    ax.set_title('This is a preview. Can you guess where the relevant signal starts?')
    if len(acci) < 20000:
        ax.set_xticks(np.arange(0, len(acci), 500))
        ax.set_xticks(np.arange(0, len(acci), 100),minor=True)
    else:
        ax.set_xticks(np.arange(0, len(acci), 2000))
        ax.set_xticks(np.arange(0, len(acci), 500),minor=True)
    ax.grid(which='both',axis='x')
    plt.plot(acci['AccX'],color='grey')
    plt.tight_layout()
    plt.show()
    lenCalib = int(input("\nPlease provide the length of the standby phase.         ")) 
    settings.insert(len(list(settings.columns)),idtf,lenCalib)
    return settings, lenCalib
    
# //////////////////////////////////////////////////////////////////////////////////

def readAcclValues(filename):

####################################################################################
# Reads and saves a CSV-shaped accelerometer record to a DataFrame.                #
####################################################################################
    
    acci = pd.read_csv(""+filename,skiprows=[0,1,2,3,4,6])
    acci = acci.rename(columns={'Gyro X':'GyrX', 'Gyro Y':'GyrY', 'Gyro Z':'GyrZ',
                                'Accel X':'AccX', 'Accel Y':'AccY', 'Accel Z':'AccZ',
                                'Quat W':'QuaW', 'Quat X':'QuaX', 'Quat Y':'QuaY', 'Quat Z':'QuaZ'})
    acci = acci.drop(['Event','Unnamed: 12'], axis=1)
    return acci
    
# //////////////////////////////////////////////////////////////////////////////////

def showSteps(acls):

####################################################################################
# Shows a snippet from the acclmtr. signals - the walking pattern should be clear. #
####################################################################################
    
    intg = acls.copy()
    intg = intg.iloc[range(3500,4000),[0,1,2,6,7,8]]   
    time = acls['Time'].loc[range(3500,4000)]
    intg = intg.apply(integrate.cumtrapz, axis=0, args=(intg.index,time))
    intg = intg.apply(integrate.cumtrapz, axis=0, args=(intg.index,time))
    intg.columns = ['LpsX','LpsY','LpsZ','RpsX','RpsY','RpsZ']
    fig = plt.figure(figsize = (16,5))
    ax = fig.add_subplot(projection='3d')
    ax.plot(intg['LpsX'], intg['LpsY'], intg['LpsZ'], color='olive')  
    ax.plot(intg['RpsX'], intg['RpsY'], intg['RpsZ'], color='olive')
    ax.set_title("Showing the walking motion", pad = 20)    
    plt.tight_layout()
    plt.show() 
    
# //////////////////////////////////////////////////////////////////////////////////

'''3. Improving the Physilog devices' output'''

# //////////////////////////////////////////////////////////////////////////////////

def analytical(R_initialOrientation=np.eye(3), 
               omega=np.zeros((5,3)),
               initialPosition=np.zeros(3),
               accMeasured=np.column_stack((np.zeros((5,2)), g*np.ones(5))),
               rate=128):
    
####################################################################################
#                                                                                  #
#  github.com/thomas-haslwanter/scikit-kinematics/blob/master/skinematics/imus.py  #
#  Analytically reconstructing accmtr. position and orientation, using angular     #
#  velocity and linear acceleration. Assumes a start in a stationary position.     #
#  Needs auxiliary libraries - quat.py, vector.py, rotmat.py.                      #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  R_initialOrientation : ndarray(3,3) --------- Rotation matrix describing        #
#  the sensor's initial orientation, except for a mis-orientation w/rt gravity.    #
#  omega : ndarray(N,3) ------------------------ Angular velocity, in [rad/s]      #
#  initialPosition : ndarray(3,) --------------- Initial position, in [m]          #
#  accMeasured : ndarray(N,3) ------------------ Linear acceleration, in [m/s^2]   #
#  rate : float -------------------------------- Sampling rate, in [Hz]            #
#  Returns                                                                         #
#  ------------------------------------------------------------------------------  #
#  q : ndarray(N,3) ---------------------------- Orientation - quaternion vector   #
#  pos : ndarray(N,3) -------------------------- Position in space [m]             #
#  vel : ndarray(N,3) -------------------------- Velocity in space [m/s]           #
#                                                                                  #
####################################################################################
    
    # Transform recordings to angVel/acceleration in space -------------------------
    # ----- Find gravity's orientation on the sensor in "R_initialOrientation" -----
    g0 = np.linalg.inv(R_initialOrientation).dot(np.r_[0,0,g])
    # ----- For the remaining deviation, assume the shortest rotation to there. ----
    q0 = vector.q_shortest_rotation(accMeasured[0], g0)    
    q_initial = rotmat.convert(R_initialOrientation, to='quat')
    # ----- Combine the two to form a reference orientation. -----------------------
    q_ref = quat.q_mult(q_initial, q0)
    
    # Compute orientation q by "integrating" omega ---------------------------------
    q = quat.calc_quat(omega, q_ref, rate, 'bf')
    
    # Acceleration, velocity, and position -----------------------------------------
    # ----- Using q and the measured acceleration, get the \frac{d^2x}{dt^2} -------
    g_v = np.r_[0,0,g] 
    accReSensor = accMeasured - vector.rotate_vector(g_v, quat.q_inv(q))
    accReSpace = vector.rotate_vector(accReSensor, q)
    # ----- Make the first position the reference position -------------------------
    q = quat.q_mult(q, quat.q_inv(q[0]))
    
    # Done. ------------------------------------------------------------------------
    return q
    
# //////////////////////////////////////////////////////////////////////////////////
  
def acceleration(df, lenCalib):
    
####################################################################################
#                                                                                  #
#  Amend all acceleration values in order to compensate for the accelerometers'    #
#  default drifting. Requires the signal to start with a calibration phase.        #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  df : pd.DataFrame --------------------------------- The data being processed.   #
#  lenCalib : int --------------------------- The duration of the standby phase.   #
#  Returns                                                                         #
#  ------------------------------------------------------------------------------  #
#  accl : ndarray(N,3) ---------------- Time vs. acceleration - dim. X, Y and Z.   #
#  gyro : ndarray(N,4) ----------- Time vs. rotation angle - dim. X, Y, Z and W.   #
#  Example                                                                         #
#  ------------------------------------------------------------------------------  #
#  >>> trueAccl = pd.DataFrame(np.array(acceleration(accl,lenCalib))[lenCalib:])   #
#                                                                                  #
####################################################################################
    
    # Zero-ing out the calibration phase. ------------------------------------------
    df['GyrX'] -= df['GyrX'][0:lenCalib].mean(axis=0)
    df['GyrY'] -= df['GyrY'][0:lenCalib].mean(axis=0)
    df['GyrZ'] -= df['GyrZ'][0:lenCalib].mean(axis=0)
    df['AccX'] -= df['AccX'][0:lenCalib].mean(axis=0)
    df['AccY'] -= df['AccY'][0:lenCalib].mean(axis=0)
    df['AccZ'] -= (df['AccZ'][0:lenCalib]-1).mean(axis=0)
    
    # Creating proper DataFrames. --------------------------------------------------
    gyroDf = pd.DataFrame({'GyrX' : df['GyrX'].apply(math.radians), 
                           'GyrY' : df['GyrY'].apply(math.radians), 
                           'GyrZ' : df['GyrZ'].apply(math.radians)})
    gyroDf.index = df['Time']
    gyro = gyroDf.values
    acclDf = pd.DataFrame({'AccX' : df['AccX']*g, 
                           'AccY' : df['AccY']*g, 
                           'AccZ' : df['AccZ']*g})
    acclDf.index = df['Time']
    accl = acclDf.values
    
    # Unwinding the analytical transformation. -------------------------------------
    q = analytical(omega=gyro, initialPosition=(0,0,0), accMeasured=accl)
    rotmatrix = quat.convert(q)
    # ----- Computing the rotation matrix: -----------------------------------------
    rotation = []
    for i in range(len(rotmatrix)): rotation.append(np.reshape(rotmatrix[i],(3,3)))
    # ----- Finding out the real acceleration values: ------------------------------
    trueAccl = []
    for i in range(len(df)): trueAccl.append(np.dot(rotation[i],accl[i]))
    
    # Done. ------------------------------------------------------------------------
    return trueAccl
    
# //////////////////////////////////////////////////////////////////////////////////

'''4. Cleaning the Kinect sensor data'''
    
# //////////////////////////////////////////////////////////////////////////////////

def consecutiveSeries(data, stepsize=1):

####################################################################################
# Separates all of the consecutive integer series occurring in a DataFrame column  #
# and yields a list of their start and end indices.                                #
####################################################################################

    return [0]+list(np.where(np.diff(data) != stepsize)[0]+1)+[len(data)]
    
# //////////////////////////////////////////////////////////////////////////////////
    
def cleanupRef(data,splitref,time):

####################################################################################
# Gets rid of the pauses occurring during a test, including mandatory calibration. #
# Uses the previous listing of all the consecutively indexed sequences.            #
####################################################################################
    
    evit = []
    drop = 0
    # A bit of tidying. ------------------------------------------------------------
    for i in range(len(splitref)-1):
        diff = splitref[i+1]-splitref[i]
        if not np.any(list(data[time][splitref[i]-drop:splitref[i+1]-drop].astype(float))):
            # Getting rid of data sequences in which the time column stays null. ---
            data = data.drop(list(np.arange(splitref[i]-drop,splitref[i+1]-drop)),axis=0)
            drop += diff
            evit += [splitref[i]]
            data = data.reset_index(drop=True)
        elif splitref[i]>1000 and diff<1000 :
            # Rarer - ignoring smaller gaps in the continuum. ----------------------
            evit += [splitref[i]]
    
    # Cleaning up the sequencing reference list. -----------------------------------
    splitref = list(set(splitref)-set(evit))
    splitref.sort()
    splitref -= drop
    splitref[1:] += 1
    data.loc[-1] = 0
    data.index += 1
    data.sort_index(inplace=True) 
    
    # Done. ------------------------------------------------------------------------
    return data, splitref[:-1], drop
    
# //////////////////////////////////////////////////////////////////////////////////

def timeRestore(zerolist,knct):

####################################################################################
# The original clocking column in the Kinect DF starts over after each pause.      #
# Replace it to take these pauses into account and keep the timeflow continuous.   #
####################################################################################
    
    lenpause = {}
    # Finding the approximate length of the pauses, --------------------------------
    # according to the jumps in the Frame numbers. ---------------------------------
    for i in range(1,len(zerolist)):
        curr = int(knct['Unity Frame'].loc[zerolist[i]])
        ante = int(knct['Unity Frame'].loc[zerolist[i]-1])
        lenpause[zerolist[i]] = float(knct['AnimationTime'].loc[curr-ante])
    
    # Reconstructing the timeflow. -------------------------------------------------
    time = [0]
    offset = 0
    for i in range(1,len(knct)):
        if i in zerolist: 
            offset += float(knct['AnimationTime'][i-1])
            offset += lenpause[i]
        time.append(float(knct['AnimationTime'][i])+offset)
    knct['RealTime'] = time
    
    # Done. ------------------------------------------------------------------------
    return knct
    
# //////////////////////////////////////////////////////////////////////////////////
    
def kinectWristData(df):

####################################################################################
# Retrieves the positions yielded by the Kinect for the waist and both wrists.     #
# Returns one DataFrame per anchor point.                                          #
####################################################################################
    
    df.loc[0] = '(0.0, 0.0, 0.0)'
    # Retrieving left wrist data. --------------------------------------------------
    leftWristDf = df['Left wrist position'].str[1:-1].str.split(',', expand=True)
    leftWristDf.columns = ['LpsX', 'LpsY', 'LpsZ']   
    leftWristDf['LpsX'] = pd.to_numeric(leftWristDf['LpsX'])
    leftWristDf['LpsY'] = pd.to_numeric(leftWristDf['LpsY'])  
    leftWristDf['LpsZ'] = pd.to_numeric(leftWristDf['LpsZ'])
    leftWristDf['LpsX'] -= leftWristDf['LpsX'][0]
    leftWristDf['LpsY'] -= leftWristDf['LpsY'][0]
    leftWristDf['LpsZ'] -= leftWristDf['LpsZ'][0]
    
    # Retrieving waist data. -------------------------------------------------------
    midWaistDf = df['Center hip position'].str[1:-1].str.split(',', expand=True)
    midWaistDf.columns = ['MpsX', 'MpsY', 'MpsZ']   
    midWaistDf['MpsX'] = pd.to_numeric(midWaistDf['MpsX'])
    midWaistDf['MpsY'] = pd.to_numeric(midWaistDf['MpsY'])  
    midWaistDf['MpsZ'] = pd.to_numeric(midWaistDf['MpsZ'])
    midWaistDf['MpsX'] -= midWaistDf['MpsX'][0]
    midWaistDf['MpsY'] -= midWaistDf['MpsY'][0]
    midWaistDf['MpsZ'] -= midWaistDf['MpsZ'][0]
    
    # Retrieving right wrist data. -------------------------------------------------
    rightWristDf = df['Right wrist position'].str[1:-1].str.split(',', expand=True)
    rightWristDf.columns = ['RpsX', 'RpsY', 'RpsZ']
    rightWristDf['RpsX'] = pd.to_numeric(rightWristDf['RpsX'])
    rightWristDf['RpsY'] = pd.to_numeric(rightWristDf['RpsY'])  
    rightWristDf['RpsZ'] = pd.to_numeric(rightWristDf['RpsZ'])
    rightWristDf['RpsX'] -= rightWristDf['RpsX'][0]
    rightWristDf['RpsY'] -= rightWristDf['RpsY'][0]
    rightWristDf['RpsZ'] -= rightWristDf['RpsZ'][0]
    
    # Done. ------------------------------------------------------------------------
    return leftWristDf, midWaistDf, rightWristDf
    
# //////////////////////////////////////////////////////////////////////////////////

'''5. Reducing the sensors' misalignment and improving precision'''

# //////////////////////////////////////////////////////////////////////////////////
    
def shareTime(knct,acls,start):

####################################################################################
# Correlating two signals requires that they share the exact same length.          #
# Step 1: arrange their durations so that they only differ in their frequency.     #
####################################################################################
    
    # Starting both signals at the same point in time. -----------------------------
    start = knct['Time'].loc[start]
    topa = np.max(np.where(acls['Time'].astype(float) <= start))
    acls = acls.drop(range(topa), axis=0)
    acls = acls.reset_index(drop = True)
    acls['Time'] -= acls['Time'].loc[0]
    # Truncating both signals to the same endpoint. --------------------------------
    threshold = np.min(np.where(acls['Time']>=knct['Time'][len(knct['Time'])-1]))
    acls = acls.drop(range(threshold,len(acls)),axis=0)
    aAcc = acls['RacX']
    
    # Converting the Kinect positions to acceleration values - right hand X. -------
    kAcc = np.gradient(np.gradient(knct['RpsX'], knct['Time']),knct['Time'])
    # Stretching the indexes for the Kinect sensor, --------------------------------
    # which has lower sampling frequency, hence fewer values over the same timeframe
    faclen = len(aAcc)/len(kAcc)
    time = ip.zoom(knct['Time'].astype(float),faclen)
    kAcc = ip.zoom(kAcc,faclen)
    
    # Sampling the accelerometer signal down to share the new scale. ---------------
    func = ipd.interp1d(acls['Time'][:threshold],aAcc)
    aAcc = func(time[10:-10])
    
    # Arranging both in a single DataFrame. ----------------------------------------
    aAcc = (aAcc - np.min(aAcc))/np.ptp(aAcc)
    kAcc = (kAcc - np.min(kAcc))/np.ptp(kAcc)
    finl = pd.DataFrame({'Knct':kAcc[10:-10], 'Acls':aAcc, 'Time':time[10:-10]})
    finl['Acls'] -= np.mean(finl['Acls'])
    finl['Knct'] -= np.mean(finl['Knct'])
    
    # Done. ------------------------------------------------------------------------
    return acls, finl
    
# //////////////////////////////////////////////////////////////////////////////////
    
def correlData(idtf,finl,precomputed):

####################################################################################
# Step 2: do the actual correlation. The signals need to be filtered and rescaled  #
# so that they share a maximum of corresponding datapoints.                        #
####################################################################################
    
    # Smoothing the sharpest edges, deleting any outliers. -------------------------
    finl['Acls'] = kalmanFilter(finl, 'Acls', 0.01)
    finl['Knct'] = kalmanFilter(finl, 'Knct', 0.2)
    aAcc = np.sort(finl['Acls'])
    finl['Acls'] = np.where((finl['Acls']>np.percentile(aAcc,99.5)), np.percentile(aAcc,99.5), finl['Acls'])
    finl['Acls'] = np.where((finl['Acls']<np.percentile(aAcc,.5)), np.percentile(aAcc,.5), finl['Acls'])
    kAcc = np.sort(finl['Knct'])
    finl['Knct'] = np.where((finl['Knct']>np.percentile(kAcc,99.5)), np.percentile(kAcc,99.5), finl['Knct'])
    finl['Knct'] = np.where((finl['Knct']<np.percentile(kAcc,.5)), np.percentile(kAcc,.5), finl['Knct'])
    
    # Rescaling the values to get similar signal widths. ---------------------------
    # Since percentile-based smoothing discarded all extreme values, ---------------
    # this code really helps achieving comparable widths. --------------------------
    amplia = np.max(finl['Acls'])-np.min(finl['Acls'])
    amplik = np.max(finl['Knct'])-np.min(finl['Knct'])
    finl['Knct'] *= -amplia/amplik
    
    # We can finally compute the suitable shift to align the signals precisely. ----
    # Some of the shifts are already known since we computed them by hand: ---------
    # that's what the "precomputed" argument means. --------------------------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!! A SIDE NOTE ON THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # We updated the experiment's protocol so that scipy.correlate works perfectly.
    # However, it wasn't secure at all to use it on the tests, i.e. on the current 
    # database. We hence "cheat" using a pre-filled CSV file.
    if precomputed: shift = precomputed
    else: shift = 0
    #    dx = np.mean(np.diff(finl['Time']))
    #    shift = (np.argmax(correlate(finl['Acls'],finl['Knct'],"full"))-len(finl))*dx
    
    # Showing results. -------------------------------------------------------------
    plt.figure(figsize=(40,5))
    plt.plot(finl['Time']+shift, finl['Knct'], color='sandybrown')
    plt.plot(finl['Time'], finl['Acls'], color='papayawhip')
    plt.show()
    
    # Done. ------------------------------------------------------------------------
    return shift
    
# //////////////////////////////////////////////////////////////////////////////////
    
def integrateData(knct,acls,shift):    

####################################################################################
# Step 3: merge the sources. Kinect-based data are kept as is. The accelerometer   #
# values count as local variations. Returns the final DataFrame for analysis.      #
####################################################################################

    # Identifying the data source - 0 for K, 1 for A. ------------------------------
    knct['Type'] = np.zeros(len(knct))
    acls['Type'] = np.ones(len(acls))
    
    # Shifting the Kinect signal. --------------------------------------------------
    knct['Time'] += shift
    acls.index = list(range(len(knct),len(acls)+len(knct)))
    acls.loc[len(knct)] = np.zeros((acls.shape[1]))
    
    # A few typing shortcuts. ------------------------------------------------------
    # The waist data from the Kinect (3,4,5, 14,15,16...) is considered dismissable,
    # since it does not show any variation. ----------------------------------------
    acccols = [20,21,22,26,27,28]
    vitcols = [11,12,13,17,18,19]
    poscols = [0,1,2,6,7,8]
    
    # Extracting accl. variations from the accelerometers. -------------------------
    avar = acls.copy(deep = True)
    for col in acccols:
        avar.iloc[col] = acls.iloc[col].diff()

    # Storing the Kinect-based speed... --------------------------------------------
    knct['LviX'] = np.gradient(knct['LpsX'],knct['Time'])
    knct['LviY'] = np.gradient(knct['LpsY'],knct['Time'])
    knct['LviZ'] = np.gradient(knct['LpsZ'],knct['Time'])
    knct['MviX'] = np.gradient(knct['MpsX'],knct['Time'])
    knct['MviY'] = np.gradient(knct['MpsY'],knct['Time'])
    knct['MviZ'] = np.gradient(knct['MpsZ'],knct['Time'])  
    knct['RviX'] = np.gradient(knct['RpsX'],knct['Time'])
    knct['RviY'] = np.gradient(knct['RpsY'],knct['Time'])
    knct['RviZ'] = np.gradient(knct['RpsZ'],knct['Time'])
    
    # ...and the Kinect-based acceleration at the same address. --------------------
    knct['LacX'] = np.gradient(knct['LviX'],knct['Time'])
    knct['LacY'] = np.gradient(knct['LviY'],knct['Time'])
    knct['LacZ'] = np.gradient(knct['LviZ'],knct['Time'])
    knct['MacX'] = np.gradient(knct['MviX'],knct['Time'])
    knct['MacY'] = np.gradient(knct['MviY'],knct['Time'])
    knct['MacZ'] = np.gradient(knct['MviZ'],knct['Time'])   
    knct['RacX'] = np.gradient(knct['RviX'],knct['Time'])
    knct['RacY'] = np.gradient(knct['RviY'],knct['Time'])
    knct['RacZ'] = np.gradient(knct['RviZ'],knct['Time'])
    
    # Launching the fusion. --------------------------------------------------------
    # Special care is needed for the axes matching (X = -X ; Z = -Y ; Y = -Z). -----
    fuse = pd.concat([knct,avar])
    under = fuse.loc[range(len(knct),len(fuse)),fuse.columns[-9:]]
    under = under.applymap(lambda x: x*-1)
    under = under.rename(columns={'LacY':'LacZZ','LacZ':'LacYY','RacY':'RacZZ','RacZ':'RacYY'})
    under = under.rename(columns={'LacZZ':'LacZ','LacYY':'LacY','RacZZ':'RacZ','RacYY':'RacY'})
    fuse.loc[range(len(knct),len(fuse)),fuse.columns[-9:]] = under
    fuse = fuse.sort_values(by = 'Time')
    fuse = fuse.reset_index(drop = True)
    # The Time column is ready now.
    
    # The next phases are highly time-consuming. -----------------------------------
    # We hence discard most of the data to keep the central rows only. -------------
    # They should be sufficient for the frequency analysis. ------------------------
    before = list(fuse['Type'].loc[range(15000)].value_counts())[1]
    fuse = fuse.drop(range(15000),axis=0)
    after = list(fuse['Type'].loc[range(22000,len(fuse)+15000)].value_counts())[1]
    fuse = fuse.drop(range(22000,len(fuse)+15000),axis=0)
    knct = knct.loc[range(before-1,len(knct)-after+1),:]
    fuse = fuse.reset_index(drop = True)
    knct = knct.reset_index(drop = True)
    
    pd.options.mode.chained_assignment = None
    
    # Saving the acc. values from the Kinect and adding the variations. ------------
    temp = fuse.copy(deep = True)
    for i in list(temp.index):
        if temp['Type'].loc[i] == 1.0:
            temp.iloc[i,acccols] = fuse.iloc[i,acccols] - temp.iloc[i-1,acccols]
    fuse = temp
    
    # Integrating these reconstructed accelerations to get the speed. --------------
    for col in vitcols:
        fuse.iloc[:,col] = integrate.cumtrapz(fuse.iloc[:,col+9],fuse['Time'],initial=0)
    # The issue is that Pythonic integration systematically results in drifting. ---
    # It can ruin everything. If we are to obtain a truthful position plot, --------
    # we need to keep on with the preprocessing. -----------------------------------
    # Confronting this speed to the Kinect-based speed. ---------------------------- 
    c = 0
    temp = fuse.copy(deep = True)
    for i in list(temp.index):
        if temp['Type'].loc[i] == 1.0:
            temp.iloc[i,vitcols] = fuse.iloc[i,vitcols] - fuse.iloc[i-1,vitcols]            
        elif temp['Type'].loc[i] == 0.0:
            temp.iloc[i,vitcols] = knct.iloc[c,vitcols]            
            c += 1
    fuse = temp
    for i in list(fuse.index):
        if fuse['Type'].loc[i] == 1.0:
            fuse.iloc[i,vitcols] += fuse.iloc[i-1,vitcols]   
    
    # Retrieving the final 128-Hz positions after all the mess. --------------------
    for col in poscols:
        fuse.iloc[:,col] = integrate.cumtrapz(fuse.iloc[:,col+11],fuse['Time'],initial=0)
    
    # Showing results. -------------------------------------------------------------
    fig = plt.figure(figsize = (14,6))
    intervals = [knct['RpsX'].loc[0]-knct['LpsX'].loc[0], knct['RpsY'].loc[0]-knct['LpsY'].loc[0], knct['RpsZ'].loc[0]-knct['LpsZ'].loc[0]]
    # Setting up the original plot. ------------------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(knct['LpsX'], knct['LpsY'], knct['LpsZ'], color='firebrick')  
    ax1.plot(knct['RpsX'], knct['RpsY'], knct['RpsZ'], color='firebrick')        
    ax1.set_title("Original Kinect plot for both arms", pad=25)   
    # Setting up the recomputed plot. ----------------------------------------------
    ax2 = fig.add_subplot(122, projection='3d')      
    ax2.plot(fuse['LpsX'], fuse['LpsY'], fuse['LpsZ'], color='olive')  
    ax2.plot(fuse['RpsX']+intervals[0], fuse['RpsY']+intervals[1], fuse['RpsZ']+intervals[2], color='olive')
    ax2.set_title("Reconstructed 128-Hz plot for both arms", pad=25)   
    plt.tight_layout()
    plt.show()
    
    # Done. ------------------------------------------------------------------------               
    return fuse
    
# //////////////////////////////////////////////////////////////////////////////////

'''6. Evaluating the handwriting'''

# //////////////////////////////////////////////////////////////////////////////////

def readWriting(filename):

####################################################################################
# Retrieves the pen's positions from a CSV file.                                   #
####################################################################################

    posd = pd.read_csv(filename)
    posd['pressure'] = np.abs(posd['pressure']) / 0.5
    return posd
    
# //////////////////////////////////////////////////////////////////////////////////

def showWriting(dictfilenames):  

####################################################################################
# Displays a set of writing exercises (x-y position and pressure variation).       #
####################################################################################
  
    # Fancy display assets ---------------------------------------------------------
    colorsfair = [(1      ,219/255,163/255),(1      ,196/255,102/255), 
                  (244/255,189/255,96/255 ),(244/255,164/255, 96/255)]
    colorsdark = [(120/255,120/255,120/255),(80/255 ,80/255 ,80/255 ),
                  (40/255 ,40/255 ,40/255 ),(0      ,0      ,0      )]
    spectrefair = LinearSegmentedColormap.from_list('spectre', colorsfair, N=100)
    spectredark = LinearSegmentedColormap.from_list('spectre', colorsdark, N=100)
    
    # Going through the CSV collection and extracting the data. --------------------
    allWritings = {}
    
    # The "easy" tasks come first. Click to enlarge the display. -------------------
    glyphs = plt.figure(figsize=(40,20))
    for i in range(6):
        filename = list(dictfilenames.keys())[i]
        qualify = dictfilenames[filename]
        posd = readWriting(filename)
        allWritings[qualify] = posd
        ax = glyphs.add_subplot(6,6,i+1)
        # The pressure variation is shown using the colormap. ----------------------
        # Notice how increasing the speed loosens the pen's pressure. --------------
        ax.scatter(posd['x'],posd['y'],c=posd['pressure'],cmap=spectrefair,marker='.',s=0.05)
        ax.set_title("Writing task - "+qualify,fontsize=20,pad=12)
        ax.invert_yaxis() # to compensate for the mirroring effect
        # The children tend to use various areas from the tablet : -----------------
        ax.set_ylim(200,1000)
        if i > 2:
            ax.set_ylim(-900,-400)
        ax.set_xlim(200,1800)
        plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    # We turn to harder tasks where the child has to trace full letters and words. -
    glyphs = plt.figure(figsize=(27,31))
    for i in range(6,10):
        filename = list(dictfilenames.keys())[i]
        qualify = dictfilenames[filename]
        posd = readWriting(filename)
        ax = glyphs.add_subplot(4,2,i-5)
        # Some of these exercises are never recorded for younger children, ----------
        # we take that fact into account --------------------------------------------
        if len(posd) > 5: # 5 instead of 0, just in case the DF is not totally empty
            ax.scatter(posd['x'],posd['y'],c=posd['pressure'],cmap=spectredark,marker='.',s=0.05)
            ax.set_title("Writing task - "+qualify,fontsize=14,pad=7)
            ax.set_ylim(0,1000)
            ax.invert_yaxis()
            ax.set_xlim(0,2000)
            allWritings[qualify] = posd
        else:
            ax.fill_between([0,1],[1,1], hatch='\\', linewidth=0, facecolor='white', edgecolor='lightgrey')
        plt.xticks([]),plt.yticks([])
    plt.subplots_adjust(wspace=7, hspace=50)
    plt.tight_layout()
    plt.show()
    
    return allWritings

# //////////////////////////////////////////////////////////////////////////////////

'''7. Performing frequency-based tests and other computations'''

# //////////////////////////////////////////////////////////////////////////////////

def normAccel(df):

####################################################################################
# Filters and normalises acceleration values before computing the spectres.        #
####################################################################################

    kf = pd.DataFrame({ 'Time':df['Time'], 
                        'LacX':kalmanFilter(df,'LacX',0.01), 
                        'LacY':kalmanFilter(df,'LacY',0.01), 
                        'LacZ':kalmanFilter(df,'LacZ',0.01),
                        'MacX':kalmanFilter(df,'MacX',0.01), 
                        'MacY':kalmanFilter(df,'MacY',0.01), 
                        'MacZ':kalmanFilter(df,'MacZ',0.01), 
                        'RacX':kalmanFilter(df,'RacX',0.01), 
                        'RacY':kalmanFilter(df,'RacY',0.01), 
                        'RacZ':kalmanFilter(df,'RacZ',0.01),
                        'NrmL':(df['LacX']**2+df['LacY']**2+df['LacZ']**2).apply(math.sqrt),
                        'NrmM':(df['MacX']**2+df['MacY']**2+df['MacZ']**2).apply(math.sqrt),
                        'NrmR':(df['RacX']**2+df['RacY']**2+df['RacZ']**2).apply(math.sqrt)})
    return kf
    
# //////////////////////////////////////////////////////////////////////////////////  

def diffT(df,w):

####################################################################################
# Computes a difference between two vectors.                                       #
####################################################################################

    return df[w]-df[0]
    
# //////////////////////////////////////////////////////////////////////////////////
    
def defWindows(df, w=128, o=64):

####################################################################################
# Defines a frequency window for each column undergoing the transformation.        #
####################################################################################
    
    windows = []
    for column in list(df.columns):
        a = np.array(df[column])
        sh = (a.size-w+1, w)
        st = a.strides*2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        windows.append(view.copy())
    return windows
    
# //////////////////////////////////////////////////////////////////////////////////

def transformAll(windows, diff):

####################################################################################
# Computes the Fourier transforms.                                                 #
####################################################################################

    transforms = []
    for window in windows:
        transforms.append(1/(diff)*fft.fft(window, axis=1))
    return transforms
    
# //////////////////////////////////////////////////////////////////////////////////
    
def baseSpectre(transform,diff): 

####################################################################################
# Sets up the basic structure for the FT-spectrogram plots.                        #
####################################################################################

    N = len(transform[0])
    fe = N/(diff)
    f = np.arange(-fe/2.0, +fe/2.0, fe/N)
    FwindowShift = fft.fftshift(transform)
    liste = np.arange(0,len(transform)/2,0.5)
    freq = np.arange(0, fe, fe/N)
    X,Y = np.meshgrid(liste, freq)
    return X, Y, f, FwindowShift
    
# //////////////////////////////////////////////////////////////////////////////////

def dispSpectre(translate,transforms,diff):  

####################################################################################
# Computes and displays all the spectrograms for a single task and a single child. #
####################################################################################
    
    # Preparing the layout. --------------------------------------------------------
    i_spectr = [1,2,3, 7,8,9, 13,14,15   ]
    i_transf = [4,5,6, 10,11,12, 16,17,18]
    
    # We're plotting the basic data first. -----------------------------------------
    fig = plt.figure(figsize=(16,28))
    i_window = 0           
    while i_window < 9:
        transform = transforms[i_window]
        X,Y,f,FwindowShift = baseSpectre(transform,diff)
        # Setting up the spectrogram display. --------------------------------------
        fig.add_subplot(6,3, i_spectr[i_window])
        plt.contourf(X, Y, np.log(transform.T), 128, cmap=spectre, vmin=-10, vmax=5)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title("Spectrogram - "+translate[i_window])
        plt.ylim(0,64)   
        # Setting up the Fourier Transform display. --------------------------------
        ax = fig.add_subplot(6,3, i_transf[i_window])
        ax.plot(f, np.log(np.mean(np.absolute(FwindowShift.T),axis=1)), color='firebrick')
        ax.set_xticks(np.arange(0, 30, 1),minor=True)
        ax.grid(which='both',axis='x')
        ax.set_yticks(np.arange(-0.5, 9.5, 1),minor=True)
        ax.grid(which='minor',axis='y')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title("Fourier Transform - "+translate[i_window])
        plt.ylim(0,6.5)  
        plt.xlim(0,30)
        i_window += 1
    plt.tight_layout()
    plt.show()
    
    # Getting on with the normalised data (which are more informative). ------------
    fig = plt.figure(figsize=(16,28))
    for i_window in range(3):
        transform = transforms[i_window+9]
        X,Y,f,FwindowShift = baseSpectre(transform,diff)
        # Setting up the spectrogram display. --------------------------------------
        fig.add_subplot(6,3, i_spectr[i_window])
        plt.contourf(X, Y, np.log(transform.T), 128, cmap=spectre, vmin=-10, vmax=5)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title("Spectrogram - "+translate[i_window+9])
        plt.ylim(0,64)
        # Setting up the Fourier Transform display. --------------------------------
        ax = fig.add_subplot(6,3, i_transf[i_window])
        ax.plot(f, np.log(np.mean(np.absolute(FwindowShift.T),axis=1)), color='olive')
        ax.set_xticks(np.arange(0, 30, 1),minor=True)
        ax.grid(which='both',axis='x')
        ax.set_yticks(np.arange(-0.5, 9.5, 1),minor=True)
        ax.grid(which='minor',axis='y')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title("Fourier Transform - "+translate[i_window+9])
        plt.xlim(0,30)
        plt.ylim(0,9.5)
    plt.tight_layout()
    plt.show()
    
# //////////////////////////////////////////////////////////////////////////////////

'''7 bis. Performing frequency-based tests on the handwriting'''

# //////////////////////////////////////////////////////////////////////////////////

def normWrite(df):

####################################################################################
# Filters and normalises pen position values before computing the spectres.        #
####################################################################################

    kf = pd.DataFrame({ 'Time':df['time'], 
                        'Nrm':(df['x']**2+df['y']**2).apply(math.sqrt)})
    return kf

# //////////////////////////////////////////////////////////////////////////////////

def directSpectre(norm,translate):  

####################################################################################
# A specific shortcut for the writing task. UNDER CONSTRUCTION                     #
####################################################################################
    
    # # Plotting the normalised data. ------------------------------------------------
    # plt.figure(figsize=(16,4))
    # # Setting up the spectrogram display. ------------------------------------------
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.title("Spectrogram - "+translate[0])
    # sp,f,t,im = plt.specgram(norm, Fs=128, NFFT=256, noverlap=255, cmap=spectre, vmin=-88, vmax=+60)
    # # scipy.signal.spectrogram(x, fs=1.0, window='tukey', 0.25, noverlap=255, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=- 1, mode='complex')
    # plt.tight_layout()
    # plt.show()
    return 

# //////////////////////////////////////////////////////////////////////////////////