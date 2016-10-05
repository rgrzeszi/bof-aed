'''
Copyright (C) 2016 Julian Kuerby. All rights reserved.
This file made available under the terms of the BSD license (see the COPYING file).
For any publication please cite the corresponding paper (see the README file).
'''
import numpy as np

class MultichannelClassifier(object):
    '''
    This multi-channel classifier uses a single-channel classifier for each
    channel to estimate posterior probabilities for each channel and class.
    These probabilities will then be combined by a FusionStrategy to predict an
    overall class for the multi-channel signal.
    '''

    def __init__(self, classifiers, fusion_strategy, labels):
        '''
        Constructor
        @param classifiers: list of classifiers for each channel. The
                            classifiers must be trained beforehand.
        @param fusion_strategy: The fusion strategy to use.
        @param labels: list of labels
        '''
        self.classifiers = classifiers
        self.fusion_strategy = fusion_strategy
        self.labels = labels

    def classify(self, features):
        '''
        
        Estimate the posterior probabilities for each channels separately.
        Then, apply the fusion strategy to classify the multi-channel signal.
        @param features: Features of one window. List of features for channels
        @return: Class label of predicted class.
        '''
        if len(self.classifiers) != len(features):
            raise RuntimeError("channel count doesn't match for classifiers and features")

        # predict_log_probs for each channel
        # log probabilities for each channel and class in shape (channel, class)
        probs = np.zeros((len(self.classifiers), len(self.labels)))
        for i in range(len(self.classifiers)):
            probs[i, :] = self.classifiers[i].predict_log_prob(features[i])

        # fusion
        classindex = self.fusion_strategy.apply(probs)

        return self.labels[classindex]

