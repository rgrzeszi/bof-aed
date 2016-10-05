'''
Copyright (C) 2016 Julian Kuerby. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier

class FusionStrategy(object):
    '''
    Abstract base class for all fusion strategies.
    A fusion strategy predicts an overall class using the posterior
    probabilities estimated for each channel separately.
    '''

    def apply(self, log_probs):
        '''
        Apply fusion strategy to classifier probabilities
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: Class index for the predicted class
        '''
        raise NotImplementedError


###################
# MAXIMUM FUSION  #
###################
class MaximumFusion(FusionStrategy):
    '''
    The MaximumFusion selects the class with the highest probability over all
    classes and channels.
    '''

    def apply(self, log_probs):
        '''
        Apply fusion strategy to classifier probabilities
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: Class index for the predicted class
        '''
        index = np.unravel_index(log_probs.argmax(), log_probs.shape)
        return index[1]


###################
#   VOTE FUSION   #
###################
class VoteFusion(FusionStrategy):
    '''
    The VoteFusion predicts a class for each channel according to the highest
    probability. Then, a majority voting is applied.
    '''

    def apply(self, log_probs):
        '''
        Apply fusion strategy to classifier probabilities
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: Class index for the predicted class
        '''
        indices = log_probs.argmax(axis=1)
        count = np.bincount(indices)
        return count.argmax()


###################
# PRODUCT FUSION  #
###################
class ProductFusion(FusionStrategy):
    '''
    The ProductFusion accumulates the class probabilities over all channels by
    multiplication. After that the class with the highest probability product is
    chosen.
    '''

    def apply(self, log_probs):
        '''
        Apply fusion strategy to classifier probabilities
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: Class index for the predicted class
        '''
        log_probs = log_probs.sum(axis=0)
        return log_probs.argmax()


###################
# STACKING FUSION #
###################
class ChannelSortStrategy(object):
    '''
    Abstract base class for all channel sorting strategies.
    A sorting strategy sorts an array of posterior probabilities along the
    channels. The ordering of the classes is not changed.
    '''

    def sort(self, log_probs):
        '''
        Sort the log_probs along the channels and flatten the array.
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: The sorted and flattened log_probs array.
        '''
        raise NotImplementedError


class ChannelSortNone(ChannelSortStrategy):
    '''
    In the ChannelSortNone strategy the ordering of the channels remains the
    same. 
    '''

    def sort(self, log_probs):
        '''
        Flatten the array log_probs. No sorting is performed.
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: The flattened log_probs array.
        '''
        return log_probs.T.flatten()


class ChannelSortProb(ChannelSortStrategy):
    '''
    The ChannelSortProb strategy sorts the channels according the their highest
    probability. Additionally only the channels with the highest probability may
    be chosen.
    '''

    def __init__(self, num_channels=None):
        '''
        Constructor
        @param num_channels: Number of channels to return in sort. If None, all
                             channels will be returned
        '''
        self.num_channels = num_channels

    def sort(self, log_probs):
        '''
        Sort the log_probs along the channels according the maximum probability.
        If self.num_channels is not None, only the num_channels best channels
        will be used. Then the array is flattened.
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: The sorted and flattened log_probs array.
        '''
        maximum = np.max(log_probs, axis=1)
        sorted_ids = np.argsort(maximum)[::-1]
        if self.num_channels is not None:
            sorted_ids = sorted_ids[:self.num_channels]
        log_probs = log_probs[sorted_ids, :]
        return log_probs.T.flatten()


class StackingFusion(FusionStrategy):
    '''
    The StackingFusion learns a fusion strategy from training data.
    A classifier is trained that uses the posterior probabilities from all
    microphones in the sensor network as input features.
    '''

    def __init__(self, channel_sort=ChannelSortNone()):
        '''
        Constructor
        @param channel_sort: An object of type ChannelSortStrategy. 
        '''
        self.stacked_classifier = None
        self.channel_sort = channel_sort

    def train(self, log_probs, labels):
        '''
        Train the stacked classifier
        @param log_probs: list of probability matrices (channels, label)
        @param labels: label for each feature-vector
        '''
        print 'Train stacked classifier with %d windows' % labels.shape[0]
        log_probs = [self.channel_sort.sort(f) for f in log_probs]
        log_probs = np.vstack(log_probs)

        # TODO: classifier as Parameter
        self.stacked_classifier = RandomForestClassifier(n_estimators=10)
        self.stacked_classifier.fit(log_probs, labels)

    def apply(self, log_probs):
        '''
        Apply fusion strategy to classifier probabilities
        @param log_probs: log probabilities for each channel and class in shape (channel, class)
        @return: Class index for the predicted class
        '''
        log_probs = self.channel_sort.sort(log_probs)
        # return the classindex as a scalar not as an array
        return self.stacked_classifier.predict(log_probs)[0]

