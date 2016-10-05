'''
This file demonstrates the use of the classifier.

The FINCA Acoustic Event dataset is used, available at
http://patrec.cs.tu-dortmund.de/cms/en/home/Resources/index.html

Adjust the finca dataset path below to the location of the files.

Copyright (C) 2015 Axel Plinge and Rene Grzeszick. All rights reserved.
This file made available under the terms of the BSD license (see the COPYING file).
For any publication please cite the corresponding paper (see the README file).
'''

FINCA_DATASET_PATH = '/vol/corpora/patrec-corpora/icassp2014aed_dortmund/'

import numpy as np
import sys
import os
import glob
from aed.features import FeatureCalculator, FeatureNormalizer
from aed.classifiers import get_classifier
from aed.eval import Evaluator


class AedTask(object):
    '''
    Abstract class for an Acoustic Event Task
    (Implemented by FincaClassification)
    '''

    def __init__(self):
        '''
        Initializes an AED task
        '''
        self.featurecalculator = FeatureCalculator()
        self.featurenormalizer = FeatureNormalizer()
        self.frame_advance = 1
        return

    def _read_annotations(self, annnotationspath):
        '''
        Read annotations from CSV.
        @param path: Path to annotation file 
        '''
        with open(annnotationspath) as annofile:
            annotations = annofile.readlines()
        annotations = annotations[1:]
        result = []
        for  anno in annotations:
            result.append(anno.strip().split(','))
        return result

    def _read_windows(self, soundfile, annotations, features):
        '''
        Read all annotated windows from a file and add to features dictionary.
        @param soundfile: Filename of a soundfile
        @param annotations: Annotations as read from annotation file
        @param features: Feature dictionary. New features from each sound file will be appended
        '''
        classname = os.path.basename(soundfile).split('.')[0]
        allfeatures, framerate, _ = self.featurecalculator.compute(soundfile)
        # The feature calculator returns the features in a multi-channel structure.
        # Here, we only have single-channel data, therefore, we only use the first channel.
        allfeatures = allfeatures[0]
        length = int(framerate * self.window)
        for filename, start, end, classname2 in annotations:
            if filename == classname and classname2 in self.classes:
                slices = np.arange(int(float(start) * framerate),
                                   int((float(end) - self.window) * framerate), self.frame_advance)
                for frameindex in slices:
                    data = allfeatures[frameindex:frameindex + length]
                    features[classname2].append(data)
        return

    def read_files(self, datapath, annotations):
        '''
        Read all files in the datapath and create features dictionary.
        @param datapath: Path containing the soundfiles
        @param annotations: Annotations as read from annotation file
        @return: A dictionary containing a feature matrix [windows x features] with the classnames as keys
        '''
        files = glob.glob(datapath + '*.wav')

        features = {}
        for classname in self.classes:
            features[classname] = []

        for filename in files:
            print >> sys.stderr, filename
            self._read_windows(filename, annotations, features)

        return features

class FincaClassification(AedTask):
    '''
    classification task on the  FINCA Acoustic Event Dataset
    '''

    def __init__(self, basepath, classifiertype, tiles=6, frame_advance=1):
        '''
        @param basepath: path to dataset
        @param classifiertype: on of 'super', 'pyramid', 'temporal'
                    for 'super', 'pyramid', cf. 
                        A Bag-of-Features Approach to Acoustic Event Detection
                        Axel Plinge, Rene Grzeszick, Gernot A. Fink.
                        Int. Conf. on Acoustics, Speech and Signal Processing, 2014.
                    
                    for all, especially 'temporal'  cf.
                        Temporal Acoustic Words for Online Acoustic Event Detection
                        Rene Grzeszick, Axel Plinge, Gernot A. Fink.
                        German Conf. Pattern Recognition, Aachen, Germany, 2015.   
        @param tiles: number of temporal tiles 
        '''

        super(FincaClassification, self).__init__()
        self.basepath = basepath
        self.window = 0.6
        self.frame_advance = frame_advance
        self.classes = ['chairs', 'door', 'laptopkeys', 'pouring', 'silence', 'steps',
                        'cups', 'keyboard', 'paper', 'rolling', 'speech' ]

        self.classifier = get_classifier(classifiertype, 30 * len(self.classes), tiles)
        return

    def train(self):
        '''
        Trains the classifier
        '''
        print 'training...'
        # Read annotations and features
        annnotationspath = self.basepath + 'training/annotations.csv'
        annotations = self._read_annotations(annnotationspath)
        features = self.read_files(self.basepath + 'training/', annotations)

        for classname, data in features.iteritems():
            print >> sys.stderr, 'Training class', classname, 'with', len(data), 'windows.'

        # Train classifier
        self.classifier.train(features)
        return


    def test(self):
        '''
        Tests the classifier
        '''
        print 'testing...'
        # Read annotations and features
        annnotationspath = self.basepath + 'test/annotations.csv'
        annotations = self._read_annotations(annnotationspath)
        features = self.read_files(self.basepath + 'test/', annotations)

        # Initialize confusion matrix
        confusion = np.zeros((len(self.classes), len(self.classes)), dtype=np.int)

        #
        # Evaluate classifier
        #
        for classname, data in features.iteritems():
            for window in data:
                classification = self.classifier.classify(window)
                confusion[self.classes.index(classname), self.classes.index(classification)] += 1

        # Print results ...
        print
        print 'confusion matrix (frame counts)'
        print

        print '%-14s' % ' ',
        for detectionindex, detectionname in enumerate(self.classes):
            print  ('%8s' % detectionname[:7]),
        print

        for classindex, classname in enumerate(self.classes):
            print '%-14s' % classname,
            for detectionindex, detectionname in enumerate(self.classes):
                print  ('%8d' % confusion[classindex, detectionindex]),
            print

        # Compute precision recall and f-measure
        total = np.sum(confusion)
        tp = 0
        for classindex in range(len(self.classes)):
            tp += confusion[classindex, classindex]
        print
        print '%.2f%% error' % (100.0 * (1.0 - tp / float(total)))
        evaluator = Evaluator()
        precision, recall, fmeasure = evaluator.calculate_precision_recall(self.classes, confusion, 'silence')
        print
        print 'P %.2f%% R %.2f%% F-Score %.2f%%' % (precision, recall, fmeasure)
        return

if __name__ == '__main__':
    test = FincaClassification(FINCA_DATASET_PATH, 'temporal', 6)
    test.train()
    test.test()
