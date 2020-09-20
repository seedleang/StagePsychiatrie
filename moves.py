import utils
# Dealing with well-known warnings 
import warnings

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''This script and its attachments are supposed to help analyse multiple series of measurements provided by three kinds of sensors.
We mainly use three Gait'Up Physilog (c) accelerometers, punctually entwined with a Kinect (c) camera and a Wacom (c) tablet.

This is preliminary research. 
The final goal is to deliver proper diagnosis for Autistic Spectrum Disorders, along with dyspraxia, before the age of 12. 
Through a series of filterings and corrections, and eventual frequency analysis, the code would reveal any unseen,
but typical micromovements (slight trembling) occurring in the recorded signals.

The results on previously diagnosed children have all been conclusive for now.

Each child would undergo three tests, including
- a walking task (approx. 2 minutes, walking straight up to a given landmark and back),
- a mimicking task (approx. 6 minutes, imitating an animated "funambulist" shown on screen),
- and a writing task (approx. 45 minutes, tracing and copying lines, symbols, letters and words).
All the data about the legs, the arms and the handwriting shall be processed using this single function.
DISCLAIMER : this is a toy example. No actual data will be published for the sake of privacy.

Find out more at *choisir article*
and an explanation of the initial hypothesis at *choisir article*

         -    Hôpital Pitié-Salpêtrière, APHP, 2020 // Service de Psychiatrie de l'Enfant et de l'Adolescent
              under supervision from Soizic Gauthier, PhD, and Salvatore Anzalone, PhD'''

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def analyseThis():
    idtf = ((str(input("\n\033[1mEnter the child's ID.\033[0m\nWe'll search for their files in the current repository.\n\033[1m>>>>>>\033[0m "))).upper()).strip()
    print("Let's go.") 
    warnings.filterwarnings("ignore")
    
    # Analysing the lower limbs' movements ----------------------------------------------------------------------------------------
    i = 1
    print("\n\n------------------------------------------------------------------------------------------------------------------")
    print("\033[1mKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKING WALKI\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    limb = ' foot'
    leftfile = "./runway/"+idtf+"/marche_gauche_"+idtf+".csv"
    waistfile = "./runway/"+idtf+"/marche_ceinture_"+idtf+".csv"
    rightfile = "./runway/"+idtf+"/marche_droite_"+idtf+".csv"
    acclfiles = {leftfile:'left',waistfile:'waist',rightfile:'right'}
    settings = "./runway/calib.csv"
    lenCalib = utils.calibAccl(settings,leftfile,idtf)
    accL,accM,accR,acls,i = utils.acclValues(limb,acclfiles,lenCalib,i)
    utils.computeSpectre(limb,acls,i)
    
    # Analysing the upper limbs' movements ----------------------------------------------------------------------------------------
    i = 1
    print("\n\n------------------------------------------------------------------------------------------------------------------")
    print("\033[1mE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FUNAMBULE FU\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    limb = ' wrist'
    leftfile = "./mirror/"+idtf+"/funambule_gauche_"+idtf+".csv"
    waistfile = "./mirror/"+idtf+"/funambule_ceinture_"+idtf+".csv"
    rightfile = "./mirror/"+idtf+"/funambule_droite_"+idtf+".csv"
    kinctfile = "./mirror/"+idtf+"/kinect_"+idtf+".csv"
    acclfiles = {leftfile:'left',waistfile:'waist',rightfile:'right'}
    settings = "./mirror/calib.csv"
    extrashift = "./mirror/shifts.csv"
    lenCalib = utils.calibAccl(settings,leftfile,idtf)
    accL,accM,accR,acls,i = utils.acclValues(limb,acclfiles,lenCalib,i)
    knct,start,i = utils.knctValues(kinctfile,i)
    fuse,i = utils.alignSensors(extrashift,idtf,knct,acls,start,i)
    utils.computeSpectre(limb,fuse,i)
        
    # Analysing the handwriting ---------------------------------------------------------------------------------------------------
    print("\n\n---------------------------------------------------------------------------------------------------------------------")
    print("\033[1mRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITING WRITIN\033[0m")
    print("---------------------------------------------------------------------------------------------------------------------\n")
    bhkfile = "./poetry/"+idtf+"/"+idtf+"_BHK.csv"
    bouclefile = "./poetry/"+idtf+"/"+idtf+"_Boucle.csv"
    dicteefile = "./poetry/"+idtf+"/"+idtf+"_Dictee.csv"
    geo1file = "./poetry/"+idtf+"/"+idtf+"_Geometrique_1.csv"
    geo2file = "./poetry/"+idtf+"/"+idtf+"_Geometrique_2.csv"
    lunefile = "./poetry/"+idtf+"/"+idtf+"_Lune.csv"
    motsfile = "./poetry/"+idtf+"/"+idtf+"_Mots.csv"
    traitfile = "./poetry/"+idtf+"/"+idtf+"_Trait.csv"
    coudefile = "./poetry/"+idtf+"/"+idtf+"_Trajet_coude.csv"
    ellipsefile = "./poetry/"+idtf+"/"+idtf+"_Trajet_ellipse.csv"
    
    wacomfiles = {traitfile:'Lines',coudefile:'Angular path',ellipsefile:'Elliptic path',geo1file:'Shapes 1',geo2file:'Shapes 2',bouclefile:'Curly loops',dicteefile:'Dictation',motsfile:'Random words',lunefile:'Lune',bhkfile:'BHK'}
    utils.wacomAnalysis(wacomfiles)
    
    return
    
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////