'''
Copyright (C) 2015 Axel Plinge and Rene Grzeszick. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''

import sys 
import numpy as np
import  models
import sklearn.naive_bayes as bayes


class ClassificationML(object):
    '''
    Implements the ML Classifier
    '''
    
    def __init__(self, model, verbouse=True):
        '''
        Initializes the ML classifier
        @param model:  The feature repr. model (e.g. normal, temporal, pyramid)
        @param verbouse: Optional verbosity parameter 
        '''
        self.model = model
        self.verbouse = verbouse
        self.labels = None
        self.bay = None
                
        return
    
    
    def _parse_dict(self, datadict, labels):
        '''
        Parses the data dictionary into a set of feature representations 
        and corresponding labels using the given feature repr. model.
        @param datadict: dictionary containing the train data (for each class??)
        @param labels:  labels for the datadict entries
        @return: features, labels
        '''
        # Get the feature repr. model        
        model = self.model
        
        # Init variables
        samplecount = 0
        inputcount = 0
        
        # Compute feature and sample counts
        for featurelist in datadict.values():
            samplecount += len(featurelist)
            for features in featurelist:
                inputcount += len(features)
                
        if self.verbouse:
            print >> sys.stderr, samplecount,' samples out of ', inputcount, 'frames'
        
        # Compute the feature representations from the training data
        bofindex=0
        print >> sys.stderr, 'size of bofs (%d, %d)' % (samplecount, model.getfeaturesize())
        bofl = np.zeros(samplecount)
        bofs = np.zeros((samplecount, model.getfeaturesize()))
        for label in labels:
            featurelist = datadict[label]
            if self.verbouse:
                print >> sys.stderr, 'Calculating', model.__class__.__name__ ,'for', label, 'based on', len(featurelist), 'samples'
            classindex = labels.index(label)                                
            for features in featurelist:
                # Compute features
                bof = model.getfeatures(features)
                # Set features and labels in matrix
                bofs[bofindex,:] = bof
                bofl[bofindex] = classindex                       
                bofindex+=1
                
        return bofs, bofl
    
    
    def train(self, datadict, labels=None):
        '''
        Runs the classifier training using the dictionary of label, features
        @param datadict: dictonary of label, features
        @param labels: (optional) list of labels. If given the order of labels is used from this list.
        '''
        
        # Set labels from data dict
        if labels is None:
            self.labels = datadict.keys()
        else:
            self.labels = labels
        # Train the GMM for BoF computation
        if self.model.gmm is None:
            print >> sys.stderr, 'Model not trained yet.'
            self.model.train(datadict, self.labels)
        
        print >> sys.stderr,'Computing',self.model.__class__.__name__,'...'
        # Parse dictionary into BoF representations and labels
        bofs, bofl = self._parse_dict(datadict, self.labels)
                        
        #Create Multinomial Bayes
        print >> sys.stderr,'Training Multinomial Bayes ...'
        self.bay = bayes.MultinomialNB(alpha=0.5, fit_prior=False)
        self.bay.fit(bofs, bofl)
        return


    def classify(self, features):    
        '''
        Classify the given features
        @return: Label for the most likely class
        '''
        
        # Use model to compute a feature vec. from the acoustic features
        feature_vec = self.model.getfeatures(features)
        if feature_vec == None:
            return None
        
        # Predict label
        classindex  = self.bay.predict(feature_vec).astype(int)
        classindex = classindex[0]
        return self.labels[classindex]
 
    
    def predict_log_prob(self, features):
        '''
        Get the log probabilty for all classes.
        @return numpy array with log probability for each class
                where the index matches the class index, i.e. 
                self.labels[0] corresponds to  predict_log_prob(f)[0]        
        '''
        probs  = np.zeros((len(self.labels))) - 100.0
        thefeature = self.model.getfeatures(features)
        if thefeature == None:
            return probs
        allpr = self.bay.predict_log_proba(thefeature)
        for index, logprob in enumerate(allpr[0]):
            classindex = int(self.bay.classes_[index])
            probs[classindex]=logprob
        return probs
        
        
    def classify_proba(self, features):
        '''
        Returns a class wise probability for the features
        '''        
        probs  = np.exp(self.score(features))
        probs *= 1.0/np.sum(probs)
        return probs
    
    
    def score(self, features):
        '''
        Returns raw probability for the features
        '''    
        thefeature = self.model.getfeatures(features)
        if thefeature == None:
            return None
        if len(features.shape)>1 and features.shape[0]>1:
            raise RuntimeError('score can take only a single frame!')
        probs  = np.zeros((len(self.labels))) - 100.0        
        allpr = self.bay.predict_log_proba(thefeature)
        for index, logprob in enumerate(allpr[0]):
            if index<len(self.bay.classes_):
                classindex = int(self.bay.classes_[index])
            probs[classindex]=logprob
        return probs
   

def get_classifier(typename, cbsize, num_tiles):
    '''
    Returns a classifier object as a combination of a classifier with a model.
    @param typename: type  of temporal augmentation 
                    'super', 'pyramid', cf. 
                        A Bag-of-Features Approach to Acoustic Event Detection
                        Axel Plinge, Rene Grzeszick, Gernot A. Fink.
                        Int. Conf. on Acoustics, Speech and Signal Processing, 2014.
                    
                    'temporal'  cf.
                        Temporal Acoustic Words for Online Acoustic Event Detection
                        Rene Grzeszick, Axel Plinge, Gernot A. Fink.
                        German Conf. Pattern Recognition, Aachen, Germany, 2015. 
                             
    @param cbsize: total codebook size
    
    @param num_tiles: number of temporal subdivisions    
    '''
    if typename == 'pyramid':
        return ClassificationML(models.BoFModelPyramid(cbsize, num_tiles))
    
    if typename == 'temporal':        
        return ClassificationML(models.BoFModelTemporal(cbsize, -num_tiles))
    
    if typename == 'super':
        return ClassificationML(models.BoFModelSuper(cbsize))
    
    return None

