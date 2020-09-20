# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import micro

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''This file is dedicated to higher level functions. 
These functions use the tools featured in micro.py. They represent the major steps to be taken throughout the full analysis.

Since we're dealing with real-life situations, we have to start with a complimentary step : each child has a unique behaviour, 
and some of the settings that are related to this behaviour cannot be analytically computed. 
We need to retrieve the right settings values from a separate logbook.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def calibAccl(settingsfile,acclfile,idtf):

#################################################################################################################################
#                                                                                                                               #
#  Making sure that all the child-specific data are available in the logbook. (If not, we fill them up using a preview plot.)   #
#  Parameters                                                                                                                   #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  settingsfile : string -------------------------------------------------- A common CSV logbook where the values are stored.   #
#  acclfile : string ------------------------------------------ An accelerometer-sourced CSV file - used for the visual hint.   #
#  idtf : string -------------------------------------------------- The child's anon. identifier - used to save their values.   #
#  Returns                                                                                                                      #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  lenCalib : int ---------------------------------------- The (critically useful) length of the standby phase for this task.   #
#                                                                                                                               #
#################################################################################################################################

    print("\n                                              \033[1m📎 (ENTERING PROCEDURE) 📎\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    print("\033[1mRetrieving calibration settings, might need to ask you.\033[0m") #---------------------------------------------
    settings = pd.read_csv(settingsfile)
    
    if idtf in list(settings.columns): # The value is already known and saved in the logbook. -----------------------------------
        lenCalib = settings[idtf].loc[0]
        print("...Impressive! "+idtf+"'s settings are already on fleek - all shiny and new.\n\n") # -----------------------------
        
    else: # The value was never stated before and we must fill it in now. -------------------------------------------------------
        print('Found no data for '+idtf+'. We need a little help.\n')
        acci = micro.readAcclValues(acclfile)
        settings, lenCalib = micro.guessCalib(acci,settings,idtf)
        settings = settings.reindex(sorted(settings.columns), axis=1)
        settings.to_csv(settingsfile,index=False)
        print('This value is now saved for '+idtf+' and will be used for any forthcoming analyses.\n') # ------------------------
        
        print("\n                                                        \033[1mDone.\033[0m")
        print("------------------------------------------------------------------------------------------------------------------\n")
    
    return lenCalib
    
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''Accelerometers usually suffer from long-term drifting. A quick web search shows that this is a common issue.
Using them for several minutes hence results in increasing imprecision.
The longest and the most difficult part of the work was to get ours to deliver trustworthy values.
See micro.py for the analytical solution (cf. GitHub : thomas-haslwanter)

The goal here is to get a relatively straight signal 
- whatever its variations may be in the middle, it should start at a null value and end there as well.
This procedure is used for the walking and the mimicking tasks, which depend on three sensors attached 
resp. to the left ankle, right ankle and waist  ////  and to the left wrist, right wrist and waist.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
def acclValues(limb,dictfilenames,lenCalib,i):

#################################################################################################################################
#                                                                                                                               #
#  Reading, editing and plotting the values yielded on a given exercise by all three accelerometers.                            #
#  Parameters                                                                                                                   #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  limb : string -----------------------------------------------------------------------  An identifier for the current task.   #
#  dictfilenames : dict ---------------------------------------------------------------- Qualifying the CSV files to be read.   #
#  lenCalib : int --------------------------------------------------------- The synchronised start index for the actual data.   #
#  Returns                                                                                                                      #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  true : DataFrame -------------------------------------------------------------- The amended, isolated acceleration values.   #
#                                                                                                                               #
#################################################################################################################################
    
    print("\n                                       \033[1m🔮 %d. Taking a look at the sensors. 🔮\033[0m"%(i))
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    allthree = {}
    for filename, qualify in dictfilenames.items():
        if qualify != 'waist': title = qualify+limb
        else: title = qualify
        print("\033[1mConsidering the "+title+" signal.\033[0m") # --------------------------------------------------------------
        print("Reading and recomputing the accelerometers' outputs.") # ---------------------------------------------------------
        print("Retrieving CSV file...") # ---------------------------------------------------------------------------------------
        acci = micro.readAcclValues(filename)
        
        print("Computing compensated accelerations...") # -----------------------------------------------------------------------
        trac = pd.DataFrame(np.array(micro.acceleration(acci,lenCalib))[lenCalib:])
        trac.columns = ['AccX','AccY','AccZ']
        
        print("Straightening things out...") # ----------------------------------------------------------------------------------
        trueAccelX = trac['AccX'] - micro.kalmanFilter(trac, 'AccX', 0.25)
        trueAccelY = trac['AccY'] - micro.kalmanFilter(trac, 'AccY', 0.25)
        trueAccelZ = trac['AccZ'] - micro.kalmanFilter(trac, 'AccZ', 0.25)
        acci = acci.drop(range(lenCalib), axis=0)
        acci = acci.reset_index(drop = True)
        true = pd.DataFrame({'AccX':trueAccelX, 'AccY':trueAccelY, 'AccZ':trueAccelZ, 'Time':acci['Time']})
        
        print("Showing results.") # ---------------------------------------------------------------------------------------------
        accs = plt.figure(figsize=(16,4))
        ax0 = accs.add_subplot(131)
        plt.plot(trac['AccX'],color='papayawhip')
        plt.plot(true['AccX'],color='sandybrown')
        plt.plot(acci['AccX'],color='floralwhite')
        ax0.set_title('Successive renderings of x-axis acceleration ('+title+').')
        ax1 = accs.add_subplot(132)
        plt.plot(trac['AccY'],color='lightcoral')
        plt.plot(true['AccY'],color='firebrick')
        plt.plot(acci['AccY'],color='mistyrose')
        ax1.set_title('Successive renderings of y-axis acceleration ('+title+').')
        ax2 = accs.add_subplot(133)
        plt.plot(trac['AccZ'],color='darkkhaki')
        plt.plot(true['AccZ'],color='olive')
        plt.plot(acci['AccZ'],color='beige')
        ax2.set_title('Successive renderings of z-axis acceleration ('+title+').')
        plt.tight_layout()
        plt.show()
        
        print("Done with the "+title+".\n") # -----------------------------------------------------------------------------------
        true['Time'] -= true['Time'][0]
        allthree[qualify] = true
    
    # Renaming for better handling. ---------------------------------------------------------------------------------------------
    allthree['left'].columns=['LacX','LacY','LacZ','Time']
    allthree['waist'].columns=['MacX','MacY','MacZ','Time']
    allthree['right'].columns=['RacX','RacY','RacZ','Time']
    
    # Adding a merged DataFrame as a summary. -----------------------------------------------------------------------------------
    acls = pd.concat([allthree['left'],allthree['waist'],allthree['right']], axis=1, sort=False)
    acls = acls.drop(range(min(len(allthree['left']),len(allthree['waist']),len(allthree['right'])),len(acls)),axis=0)
    cols = [x for x in range(acls.shape[1])] 
    cols.remove(3)
    cols.remove(7)
    acls = acls.iloc[:,cols]
    acls = acls.reset_index(drop=True)
    
    # Making sure everything looks as it should - if needed. --------------------------------------------------------------------
    # micro.showSteps(acls)
    
    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    i += 1
    return list(allthree.values())+[acls]+[i]
    
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''The Kinect signal is only used for the "Funambule" mimicking task, where we draw our eyes to the exact arms' position. 
Its reliability should compensate for the accelerometers' residual drift, although its sampling frequency isn't satisfactory.
This part prepares for the upcoming fusion of both sensors (resulting in a precise AND reliable signal).

The Kinect records the full exercise, but splits it into six walks and a calibrating phase. There are pauses in between 
where it enters sleep mode, so a bit of cleaning is required. 
See micro.py to dive into the details.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def knctValues(filename,i):

#################################################################################################################################
#                                                                                                                               #
#  Reading and cleaning the values provided by the Kinect sensor.                                                               #
#  Parameters                                                                                                                   #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  filename : string ------------------------------------------------------------------------------- The CSV file being read.   #
#  Returns                                                                                                                      #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  knct : DataFrame -------------------------------------------------------------- The amended, isolated acceleration values.   #  
#  start : int ------------------------------------------------------------ The amount of rows subtracted from the DataFrame.   # 
#                                                                                                                               #
#################################################################################################################################

    print("\n                                 \033[1m📡 %d. Turning to the Kinect RGB-depth camera. 📡\033[0m"%(i))
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    print("\033[1mTidying the Kinect data.\033[0m") # ---------------------------------------------------------------------------
    # Skipping non-numerical data. ----------------------------------------------------------------------------------------------
    init = pd.read_csv(""+filename,skiprows=[0,1,2,3,4])
    init = init[init['Kinect FrameNumber'].apply(lambda x: x.isnumeric())]
    # Pauses are now invisible. -------------------------------------------------------------------------------------------------
    
    print("Separating the walks...") # ------------------------------------------------------------------------------------------
    # Getting the start and stop indexes for the six "Funambule" series. --------------------------------------------------------
    reft = micro.consecutiveSeries(init['Kinect FrameNumber'].astype(int))
    # Cleaning the DataFrame by skipping any untimed (or badly timed) rows. -----------------------------------------------------
    clen, zerolist, start = micro.cleanupRef(init,reft,'AnimationTime')
    
    print("Restoring the timeflow...") # ----------------------------------------------------------------------------------------
    clen = micro.timeRestore(zerolist,clen)
    
    print("Saving the data...") # -----------------------------------------------------------------------------------------------
    right, left, waist = micro.kinectWristData(clen)
    # Arranging it in a single DataFrame. ---------------------------------------------------------------------------------------
    knct = pd.concat([right, left, waist], axis=1, sort=False)
    knct['Time'] = clen['RealTime']
    knct.loc[0,'Time'] = 0.0
    
    print("Showing results.") # -------------------------------------------------------------------------------------------------
    accs = plt.figure(figsize=(16,4))
    ax0 = accs.add_subplot(131)
    plt.plot(knct['LpsZ'][1:]-1180,color='olive')
    plt.plot(knct['LpsY'][1:],color='firebrick')
    plt.plot(knct['LpsX'][1:],color='sandybrown')
    ax0.set_title('Kinect-based x, y, and z-axis acceleration (left wrist).')
    ax1 = accs.add_subplot(132)
    plt.plot(knct['MpsZ'][1:]-1180,color='olive')
    plt.plot(knct['MpsY'][1:],color='firebrick')
    plt.plot(knct['MpsX'][1:],color='sandybrown')
    ax1.set_title('Kinect-based x, y, and z-axis acceleration (waist).')
    ax2 = accs.add_subplot(133)
    plt.plot(knct['RpsZ'][1:]-1180,color='olive')
    plt.plot(knct['RpsY'][1:],color='firebrick')
    plt.plot(knct['RpsX'][1:],color='sandybrown')
    ax2.set_title('Kinect-based x, y, and z-axis acceleration (right wrist).')
    plt.tight_layout()
    plt.show()
    
    print("\n                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    i += 1
    return knct, start, i   

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''This is the critical part where the data are fused. 
The actual *values* from the Kinect are used as a support for accelerometer-based *variations*.
We trim the signals so that they share the same real-world duration. We stretch them to account for diverging sampling frequencies.
Slight mismatches are mended through np.correlate, before the fusion is eventually done. A 3D model of the movement is displayed.

That's it -- we return one last DataFrame - exhaustive, precise, and reliable.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
def alignSensors(settingsfile,idtf,knct,acls,start,i):
    print("\n                                    \033[1m🕒 %d. An endeavour to align the sources. 🕒\033[0m"%(i))
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    print("\033[1mTaking advantage of the sampling frequency gap.\033[0m") # ----------------------------------------------------
    print("Preparing interpolation...") # --------------------------------------------------------------------------------------- 
    reacls,finl = micro.shareTime(knct,acls,start)
    print("Computing the shift...") # -------------------------------------------------------------------------------------------
    settings = pd.read_csv(settingsfile)
    if idtf in list(settings.columns): precomputed = settings[idtf].loc[0]
    else: precomputed = None
    shift = micro.correlData(idtf,finl,precomputed)
    print("Improving precision...") # -------------------------------------------------------------------------------------------
    print("Please wait.") # -----------------------------------------------------------------------------------------------------
    fuse = micro.integrateData(knct,reacls,shift)

    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    i += 1
    return fuse, i
    
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''Proudly introducing the final spectrogram plots. They're self-explanatory for a machine-based diagnosis.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
def computeSpectre(limb,acls,i,recouvr=64,):
	
#################################################################################################################################
#                                                                                                                               #
#  Computing and displaying the spectrograms for all the signals in a task.                                                     #
#  Parameters                                                                                                                   #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  df : pd.DataFrame --------------------------------------------------------------------- The rearranged DataFrame envelope.   #
#                                                                                                                               #
#################################################################################################################################
    
    print("\n                                             \033[1m🔬 %d. Spectral analysis. 🔬\033[0m"%(i))
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    print("\033[1mGenerating clear evidence for the diagnosis.\033[0m") # -------------------------------------------------------
    norm = micro.normAccel(acls)
    
    print("Computing the transforms...") # --------------------------------------------------------------------------------------
    diff = micro.diffT(norm['Time'],128)
    windows = micro.defWindows(norm.iloc[:,range(1,13)])
    transforms = micro.transformAll(windows,diff)
    
    print("Showing results.") # -------------------------------------------------------------------------------------------------
    print("Please wait.") # -----------------------------------------------------------------------------------------------------
    translate = [ 'Left'+limb+' X acceleration',
                  'Left'+limb+' Y acceleration',
                  'Left'+limb+' Z acceleration',
                  'Waist X acceleration',
                  'Waist Y acceleration',
                  'Waist Z acceleration',
                  'Right'+limb+' X acceleration',
                  'Right'+limb+' Y acceleration',
                  'Right'+limb+' Z acceleration',
                  'Normalised left'+limb+' acceleration',
                  'Normalised waist acceleration',
                  'Normalised right'+limb+' acceleration' ]   
    micro.dispSpectre(translate,transforms,diff)

    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''A separate analysis for the writing task, 
where the sensor's specificity (a Wacom pen, as opposed to accelerometers) implies distinct handling.'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def wacomAnalysis(dictfilenames):

#################################################################################################################################
#                                                                                                                               #
#  Reading, editing and plotting the values yielded on the writing exercises.                                                   #
#  Parameters                                                                                                                   #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  dictfilenames : dict ---------------------------------------------------------------- Qualifying the CSV files to be read.   #
#  Returns                                                                                                                      #
#  ---------------------------------------------------------------------------------------------------------------------------  #
#  glyphs : DataFrame ------------------------------------------------------------ The amended, isolated acceleration values.   #
#                                                                                                                               #
#################################################################################################################################
    
    print("\n                                       \033[1m✒️ Previewing the handwritten data. ✒️\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    # Saving one plot/DataFrame per task. ---------------------------------------------------------------------------------------
    glyphs = micro.showWriting(dictfilenames)
    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    print("\n                                              \033[1m🔬️ Spectral analysis. 🔬\033[0m")
    print("------------------------------------------------------------------------------------------------------------------")
    # Lauching the analysis for each DataFrame. ---------------------------------------------------------------------------------
    for quality, df in glyphs.items():
        translate = [quality+" - normalised"]
        norm = micro.normWrite(df)
        # We're using a shortcut - no need for the separate windows, etc. -------------------------------------------------------
        micro.directSpectre(norm,translate)
    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    return
    
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////