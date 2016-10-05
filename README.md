# Bag-of-Features Acoustic Event Classification and Detection with multi-channel fusion

This is Python code for Bag-of-Features representations for acoustic event detection and classification.
The method for multi-channel fusion is described in [3] and the single-channel method in [1,2].
It also includes a two demos:
* Classifying single-channel acoustic events on the FINCA acoustic event dataset 
* Classifying multi-channel events on the FINCA multi-channel dataset 
For details please read the 'Getting started' section

Note that this is a updated version of the code used in the publications ([1,2,3]).
Some dependencies have been replaced by python libraries that are easier to handle.
If you have used one of our previous - non github - released, the results may differ slightly.

References
============
The code was used in:

[1] A Bag-of-Features Approach to Acoustic Event Detection
Axel Plinge, Rene Grzeszick, Gernot A. Fink.
Int. Conf. on Acoustics, Speech and Signal Processing, 2014.

[2] Temporal Acoustic Words for Online Acoustic Event Detection
Rene Grzeszick, Axel Plinge, Gernot A. Fink.
German Conf. Pattern Recognition, Aachen, Germany, 2015.

[3] Bag-of-Features Acoustic Event Detection for Sensor Networks
Julian Kürby, Rene Grzeszick, Axel Plinge, and Gernot A. Fink.
Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop,
Budapest, Hungary, 2016.

If you use this code in your scientific work please cite the corresponding paper.

System Requirements
===================
* Python, including Numpy, Scipy, Sklearn and pysoundfile
The software was tested on Debian 64bit.

Getting started
===============
1. Either add the src folder to your PYTHONPATH 
   or use a Python IDE of your liking (e.g. Eclipse and pydev)
3. Use the multi_channel_experiments or the single_channel_experiment sample file to include our code in your experiments

For running the sample experiments:
4. Download the FINCA "Multi-channel acoustic event dataset" for multi-channel experiments
   or the FINCA "Acoustic event dataset" for single-channel experiments from
   http://patrec.cs.tu-dortmund.de/cms/en/home/Resources/index.html
5. Set the respective FINCA_DATASET_PATH in the demo file
6. Run the experiment 

Contact Authors
===============
Rene Grzesick
Department of Computer Science
TU-Dortmund, Germany.
Email: rene.grzeszick@udo.edu

Axel Plinge
Department of Computer Science
TU-Dortmund, Germany.
Email: axel.plinge@udo.edu

Julian Kürby
Department of Computer Science
TU-Dortmund, Germany.
Email: julian.kuerby@udo.edu
  


