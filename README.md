# StagePsychiatrie
Prémices d'un projet codé pour un laboratoire de recherche.  
  > Hôpital Pitié-Salpêtrière, APHP, 2020 // Service de Psychiatrie de l'Enfant et de l'Adolescent  
  > sous la supervision interdisciplinaire des Drs. S. Gauthier et S. Anzalone  

**Portée du code**  
Ces scripts servent à analyser des séries de mesures faites avec trois types de capteurs : des accéléromètres, une Kinect, une tablette graphique.  
L'objectif est d'étudier les mouvements des enfants résidents de l'hôpital de jour, et de mettre au point à terme une procédure de diagnostic simple et peu fatigante pour le patient. Il s'agit d'y trouver d'éventuels micromouvements à haute fréquence pour une détection précoce de troubles du développement.  
A travers une série de filtrages et de corrections, et une phase d'analyse fréquentielle, le code révèle ces micromouvements invisibles pour l'oeil humain.  
Le projet n'en est qu'à ses débuts et le code sera modifié par les intervenants suivants.

**Mode d'emploi**  
Au cours des passations pour cette recherche, des enfants témoins et des enfants du service s'essaient à trois exercices :  
- un exercice de marche (environ 2 minutes, trois allers-retours définis par un repère),  
- un exercice d'imitation (environ 6 minutes, sur l'exemple d'un "Funambule" animé qui apparaît à l'écran),  
- et un exercice d'écriture (environ 45 minutes, tracé et copie de traits, de symboles, de lettres puis de mots, jusqu'au test BHK selon l'âge et la maîtrise de l'enfant). 

Ceci est un exemple factice qui montre comment les données recueillies au sujet de la démarche, des mouvements des bras puis des mains sont analysées et interprétées.
Lancer le Notebook .ipynb dans l'environnement défini pour un aperçu d'un examen complet.

**Description des fichiers**  
*moves.py* contient la fonction globale, analyseThis(), et une introduction au projet. moves.py appelle  
> *utils.py*, qui contient, résume et explique les diverses grandes étapes entreprises au cours d'une analyse. utils.py appelle  
>> *micro.py*, qui contient toutes les fonctions "petites mains" plus ou moins complexes qui se chargent de rendre cette analyse possible.  
>>> *quat.py*, *vector.py* et *rotmat.py* sont les annexes d'un petit extrait de scipy.kinematics qui permet une meilleur gestion des capteurs.  

Les dossiers runway, mirror et poetry contiennent resp. les données de marche, d'imitation et d'écriture d'un enfant ici isolé et anonyme.

**En savoir plus ?**  
Les articles suivants donneront plus de contexte :  
https://www.frontiersin.org/articles/10.3389/fpsyg.2018.01467/full  
https://www.sciencedirect.com/science/article/pii/S0013700618301891  
https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00676/full  

*******************************************************************

**ENGLISH VERSION**

  > Hôpital Pitié-Salpêtrière, APHP, 2020 // Service de Psychiatrie de l'Enfant et de l'Adolescent  
  > under supervision from Soizic Gauthier, PhD, and Salvatore Anzalone, PhD
              
This script and its attachments are supposed to help analyse multiple series of measurements provided by three kinds of sensors.  
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

All the data about the legs, the arms and the handwriting shall be processed using the Notebook attached.  
DISCLAIMER : this is a toy example. No actual data will be published for the sake of privacy.  

Find out more at  
https://www.frontiersin.org/articles/10.3389/fpsyg.2018.01467/full  
https://www.sciencedirect.com/science/article/pii/S0013700618301891  
https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00676/full
