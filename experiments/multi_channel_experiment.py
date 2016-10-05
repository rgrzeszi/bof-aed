'''
The FINCA multi-channel Acoustic Event dataset is used, available at
http://patrec.cs.tu-dortmund.de/cms/en/home/Resources/index.html

Adjust the FINCA_DATASET_PATH below to the location of the files.

Copyright (C) 2016 Julian Kuerby. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''
import os
from _collections import defaultdict
import random
import numpy as np

from aed.features import FeatureCalculator
from aed.classifiers import ClassificationML
from aed.models import get_model
from aed.eval import Evaluator
from aed.multichannel_classifier import MultichannelClassifier
from aed.fusion import StackingFusion

####################################################
# Set the path to the multi-channel FINCA dataset. #
####################################################
FINCA_DATASET_PATH = ''

CLASSES = ['applause',
           'chairs',
           'cups',
           'door',
           'doorbell',
           'doorknock',
           'keyboard',
           'knock',
           'music',
           'paper',
           'phonering',
           'phonevibration',
           'pouring',
           'screen',
           'speech',
           'steps',
           'streetnoise',
           'touching',
           'ventilator',
           'silence']

class FincaClassification(object):
    '''
    classification task on the  FINCA Acoustic Event Dataset
    '''

    def __init__(self, basepath, modeltype, tiles=6):
        '''
        @param basepath: path to dataset
        @param modeltype: on of 'super', 'pyramid', 'temporal'
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
        self.basepath = basepath
        self.window = .6
        self.frame_advance = 1
        self.featurecalculator = FeatureCalculator()

        self.classes = CLASSES

        self.modeltype = modeltype
        self.modeltiles = tiles
        self.codebooksize_per_class = 30
        self.codebooksize = self.codebooksize_per_class * len(self.classes)

        # Prepare list of classifiers for each of the 32 channel
        self.classifiers = [None] * 32

        print 'Experiment on FINCA'
        print 'window:', self.window
        print 'frame_advance', self.frame_advance
        print 'num classes:', len(self.classes)
        print 'modeltype:', self.modeltype
        print 'codebooksize per class:', self.codebooksize_per_class
        print 'tiles:', self.modeltiles
        print 'classes:\n\t', '\n\t'.join(self.classes)

    def _read_annotations(self, annnotationspath):
        '''
        Read annotations from CSV.
        @param path: Path to annotation file 
        '''
        with open(annnotationspath) as annofile:
            annotations = annofile.readlines()
        result = []
        for anno in annotations:
            anno = anno.strip()
            if anno != '' and not anno.startswith('#'):
                result.append(anno.split(','))
        return result

    def _read_windows(self, soundfile, annotations, features_windows, features_frames, channels):
        '''
        Read all annotated features from a file and add to features_frames dictionary.
        Additionally add all extracted windows of given length to features_windows.
        @param soundfile: Filename of a soundfile. Should be multi channel
        @param annotations: Annotations as read from annotation file
        @param features_windows: Feature dictionary. Extracted windows will be appended
        @param features_frames: Feature dictionary. Features of all frames will be appended
        @param channels: 1D numpy array of channel indices to use.
        '''
        filename = os.path.basename(soundfile)
        allfeatures, framerate, _ = self.featurecalculator.compute(soundfile, channels)

        length = int(framerate * self.window)

        for classname, start, end, filename_anno in annotations:
            if filename == filename_anno and classname in self.classes:
                data = allfeatures[:, int(float(start) * framerate):int(float(end) * framerate)]
                features_frames[classname].extend(list(data))
                slices = np.arange(int(float(start) * framerate),
                                   int((float(end) - self.window) * framerate), self.frame_advance)
                for frameindex in slices:
                    data = allfeatures[:, frameindex:frameindex + length]
                    features_windows[classname].extend(list(data))

        return

    def read_files(self, annotations, channels):
        '''
        Read all files in the datapath and create features_windows dictionary.
        @param annotations: Annotations as read from annotation file
        @param channels: 1D numpy array of channel indices to use.
        @return: A dictionary containing a feature matrix [windows x features] with the classnames as keys
        '''

        if type(channels) == int or type(channels) == np.int64:
            channels = np.array([channels])
        elif type(channels) == list:
            channels = np.array(channels)

        features_frames = {}
        for classname in self.classes:
            features_frames[classname] = []

        features_windows = {}
        for classname in self.classes:
            features_windows[classname] = []

        annotation_dict = defaultdict(list)
        for anno in annotations:
            annotation_dict[anno[3]].append(anno)

        for filename, annos in annotation_dict.items():
            path = self.basepath + '/audio/' + filename
            self._read_windows(path, annos, features_windows, features_frames, channels)

        return features_windows, features_frames

    def calc_log_prob_for_files(self, annotations):
        '''
        Calculate the logprobs for the classification windows given in annotations
        @param annotations: Annotations as read from annotation file
        @return: tuple (features, labels). features is a list of logprobs-matrices for the windows.
                        labels is numpy-array of the labels for the respective windows.
        '''

        features = []
        labels = []

        annotation_dict = defaultdict(list)
        for anno in annotations:
            annotation_dict[anno[3]].append(anno)

        for filename, annos in annotation_dict.items():
            path = self.basepath + '/audio/' + filename
            self._calc_log_probs_for_windows(path, annos, features, labels)

        return features, np.array(labels)

    def _calc_log_probs_for_windows(self, soundfile, annotations, features, labels):
        '''
        Read all annotated windows from a file and calculate the probabilities for each class and channel.
        @param soundfile: Filename of a soundfile. Should be multi channel
        @param annotations: Annotations as read from annotation file
        @param features: Feature list. New features from each sound file will be appended
        @param labels: label list. The labels of the windows will be appended
        '''
        filename = os.path.basename(soundfile)
        allfeatures, framerate, _ = self.featurecalculator.compute(soundfile)

        length = int(framerate * self.window)

        for classname, start, end, filename_anno in annotations:
            if filename == filename_anno and classname in self.classes:
                slices = np.arange(int(float(start) * framerate),
                                   int((float(end) - self.window) * framerate), self.frame_advance)
                for frameindex in slices:
                    data = allfeatures[:, frameindex:frameindex + length]
                    data = self.estimate_log_probs(data)
                    features.append(data)
                    labels.append(self.classes.index(classname))

        return

    def estimate_log_probs(self, window):
        '''
        Estimate the log probabilities for all channels of the given window.
        @param window: features of the window
        @return: log probabilities for all channels
        '''
        log_probs = np.zeros((len(self.classifiers), len(self.classes)))
        for channel_id, classifier in enumerate(self.classifiers):
            log_probs[channel_id, :] = classifier.predict_log_prob(window[channel_id])

        return log_probs

    def read_test_files(self, annotation_file):
        '''
        Read files for testing
        '''
        features_test = []
        labels_test = []

        annotation_file = self.basepath + '/annotations/general/' + annotation_file
        annotations = self._read_annotations(annotation_file)

        annotation_dict = defaultdict(list)
        for anno in annotations:
            annotation_dict[anno[3]].append(anno)

        for filename, annos in annotation_dict.items():
            path = self.basepath + '/audio/' + filename
            features, labels = self._read_test_windows(path, annos)
            features_test.extend(features)
            labels_test.extend(labels)

        return features_test, labels_test

    def _read_test_windows(self, soundfile, annotations):
        '''
        Read all annotated windows from a file.
        @param soundfile: Filename of a soundfile. Should be multi channel
        @param annotations: Annotations as read from annotation file
        '''
        features = []
        labels = []

        filename = os.path.basename(soundfile)
        allfeatures, framerate, _ = self.featurecalculator.compute(soundfile)

        length = int(framerate * self.window)

        for classname, start, end, filename_anno in annotations:
            if filename == filename_anno and classname in self.classes:
                slices = np.arange(int(float(start) * framerate),
                                   int((float(end) - self.window) * framerate), self.frame_advance)
                for frameindex in slices:
                    data = allfeatures[:, frameindex:frameindex + length]
                    features.append(data)
                    labels.append(classname)

        return features, labels

    def train(self, annotationfile, channels):
        '''
        Trains the classifiers
        '''
        print 'training set:\t' + os.path.basename(annotationfile)
        annotationfile = self.basepath + '/annotations/general/' + annotationfile

        # read annotations
        annotations = self._read_annotations(annotationfile)
        # split annotations for training of base and stacking classifiers
        random.shuffle(annotations)
        split_index = len(annotations) / 3
        annotations_stacking = annotations[:split_index]
        annotations_base = annotations[split_index:]

        # train the classifiers for the channels
        print 'Train %d separate base classifiers' % channels.shape[0]
        for channel_ids in channels:
            print 'training channels', channel_ids
            # Calculate features
            features_windows, features_frames = self.read_files(annotations_base, channel_ids)

            # Train model
            model = get_model(self.modeltype, self.codebooksize, self.modeltiles)
            model.train(features_frames, self.classes)

            # Train classifier
            classifier = ClassificationML(model)
            classifier.train(features_windows, self.classes)

            for i in channel_ids:
                self.classifiers[i] = classifier

        del features_windows
        del features_frames

        # train the stacking classifier
        features, labels = self.calc_log_prob_for_files(annotations_stacking)
        fusion = StackingFusion()
        fusion.train(features, labels)

        multi_classifier = MultichannelClassifier(self.classifiers, fusion, self.classes)

        return multi_classifier

    def test(self, features, labels, multi_classifier):
        '''
        Tests the classifier
        '''
        print
        print 'testing...'

        # Initialize confusion matrix
        confusion = np.zeros((len(self.classes), len(self.classes)), dtype=np.int)

        #
        # Evaluate classifier
        #
        for classname, window in zip(labels, features):
            classification = multi_classifier.classify(window)
            confusion[self.classes.index(classname), self.classes.index(classification)] += 1

        precision, recall, fmeasure, error = self._evaluate_confusion_matrix(confusion)
        print
        print 'P %.2f%% R %.2f%% F-Score %.2f%%' % (precision, recall, fmeasure)

        return precision, recall, fmeasure, error

    def _evaluate_confusion_matrix(self, confusion):
        # Print results ...
        print
        print 'confusion matrix (frame counts)'
        print

        print '%-14s' % ' ',
        for detectionindex, detectionname in enumerate(self.classes):
            print  ('%8s' % detectionname[:7]),
        print

        for classindex, classname in enumerate(self.classes):
            print '%-14s' % classname[:14],
            for detectionindex, detectionname in enumerate(self.classes):
                print  ('%8d' % confusion[classindex, detectionindex]),
            print

        # Compute precision recall and f-measure
        total = np.sum(confusion)
        tp = 0
        for classindex in range(len(self.classes)):
            tp += confusion[classindex, classindex]
        print
        error = 100.0 * (1.0 - tp / float(total))
        evaluator = Evaluator()
        precision, recall, fmeasure = evaluator.calculate_precision_recall(self.classes, confusion, 'silence')
        return precision, recall, fmeasure, error

if __name__ == '__main__':
    # Use one separate classifier for each channel
    channels = np.arange(32).reshape((32, 1))

    results = np.zeros((5, 4))

    # Run experiment for all five splits
    for i in range(1, 6):
        print '\nSplit %d' % i
        experiment = FincaClassification(FINCA_DATASET_PATH, 'super')

        # train classifiers
        multi_classifier = experiment.train('training_%d.csv' % i, channels)

        # evaluate
        features, labels = experiment.read_test_files('test_%d.csv' % i)
        results[i - 1, :] = experiment.test(features, labels, multi_classifier)
        print 'Precision %.2f%%\nRecall %.2f%%\nF-Score %.2f%%\nError %.2f%%' % tuple(results[i - 1])
        print 
    
    res_mean = results.mean(axis=0)
    print '\nMean results:'
    print 'Precision %.2f%%\nRecall %.2f%%\nF-Score %.2f%%\nError %.2f%%' % tuple(res_mean)

