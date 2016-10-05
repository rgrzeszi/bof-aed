'''
Copyright (C) 2015 Axel Plinge and Rene Grzeszick. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''
import sys
import numpy as np

from sklearn.mixture import GMM
from features import FeatureNormalizer


class BoFModelTemporal(object):
    '''
    Soft quantization using an supervised GMM using input features with temporal augmentation.
    Computes normalized/pooled probabilities of Gaussians.
    '''

    _OFFSET_VALUE = 10

    def __init__(self, vocab_size, frames):
        '''
        Initializes the temporal model
        @param vocab_size: size of the overall vocabulary (vocab_size/num_classes per class) 
        @param frames: number of frames per temporal step
                       if negative, number of temporal steps == coordinates
        '''

        self.vocab_size = vocab_size
        self.gmm = None
        self.gmms = []
        self.frames = frames
        self.featuresize = vocab_size
        return

    def __augment_temporal(self, features):
        '''
        Add quantized temporal information to the features.
        @param features: input features 
        @return: the augmented features
        '''

        # No temporal information
        temp_features = []
        if self.frames == 0:
            return features

        # number of frames to get the same temporal coordinate
        if self.frames > 0:
            div = self.frames
        else:
            div = max(1, (1 + len(features)) / -self.frames)

        # Append temporal coordinates
        for t_i, feat_i in enumerate(features):
            temp_feat = np.hstack([int(t_i / div) * self._OFFSET_VALUE, feat_i])
            temp_features.append(temp_feat)
        temp_features = np.array(temp_features)
        return temp_features

    def train(self, datadict, labels):
        '''
        Trains the model using the given data
        @param datadict: dictonary of label,features 
        @param labels: the labels of the datadict in a given order
        '''
        # Stack the features
        allfeatures = np.vstack(list([np.vstack(x) for x in datadict.values()]))

        # Determine the normalisation statistics and remember them
        self.norm = FeatureNormalizer()
        self.norm.setup(allfeatures)

        # Get number of classes
        ncl = len(labels)
        # Compute vocabsize per class
        vocab_size_per_cl = max(1, self.vocab_size / ncl)
        # Update vocabsize to account for rounding errors
        self.vocab_size = vocab_size_per_cl * ncl


        #
        # Initialize complete GMM (used for supercodebook)
        # This will later on be overwritten and is
        # a workaround to pre-initialize an sklearn GMM
        #
        self.gmms = []
        print >> sys.stderr, 'Initialising GMM with', self.vocab_size, 'components,', vocab_size_per_cl, 'per class.'
        self.gmm = GMM(self.vocab_size, n_iter=2, params='w')

        # Initialize by fitting with ones
        self.gmm.fit(np.ones((self.vocab_size, allfeatures.shape[1] + 1)))

        #
        # For each class train a GMM
        #
        index = 0
        for label in labels:
            # Compute feature representations
            temp_feat_reprs = []
            for feat in datadict[label]:
                feat = self.norm.normalize(feat)
                feat = self.__augment_temporal(feat)
                temp_feat_reprs.append(feat)
            temp_feat_reprs = np.vstack(temp_feat_reprs)

            print >> sys.stderr, ("Training a GMM for label %s with %d densities, using data of shape %s"
                                 % (label, vocab_size_per_cl, str(np.shape(temp_feat_reprs))))

            # Train the GMM
            gmm = GMM(vocab_size_per_cl, covariance_type='diag')
            gmm.fit(temp_feat_reprs)

            # Overwrite values from supervised codebook GMM by class GMMs
            self.gmm.means_  [index * vocab_size_per_cl:(index + 1) * vocab_size_per_cl, :] = gmm.means_
            self.gmm.covars_ [index * vocab_size_per_cl:(index + 1) * vocab_size_per_cl, :] = gmm.covars_
            self.gmm.weights_[index * vocab_size_per_cl:(index + 1) * vocab_size_per_cl] = gmm.weights_ / float(ncl)
            # Append the GMM
            self.gmms.append(gmm)
            index += 1

        # Set uniform GMM weights
        self.gmm.weights_ = np.ones(self.vocab_size) / self.vocab_size
        # Set feature size
        self.featuresize = self.vocab_size
        return

    def getfeaturesize(self):
        '''
        Returns the feature size (i.e. vocab_size)
        '''
        return self.featuresize

    def getfeatures(self, features):
        '''
        Returns the Temporal Bag-of-Super-Features representation
        for the feature matrix [frames x features]
        '''
        # Normalize features
        norm_feat = self.norm.normalize(features)
        # Append temporal information
        temp_feat = self.__augment_temporal(norm_feat)
        # Get probabilities
        probas = self.gmm.predict_proba(temp_feat)
        # Compute mean for BoF as frequencies
        bof_hist = np.mean(probas, axis=0)
        return bof_hist


class BoFGMMBase(object):
    '''
    Abstract base class for the GMM based models.
    Used by the Pyramid and supervised codebook model
    '''

    def __init__(self, vocab_size):
        '''
        Initialization
        '''
        self.vocab_size = vocab_size
        self.normalize = True
        self.norm = FeatureNormalizer()
        self.gmm = None
        return

    def compute_super_codebook(self, feature_size):
        '''
        Merges the GMMs that were computed for each class
        into one GMM (super codebook).
        @param feature_size: dimensionality of a feature vector 
        '''

        # Get number of classes
        ncl = len(self.labels)
        # Compute vocabsize per class
        vocab_size_per_cl = max(1, self.vocab_size / ncl)
        # Create GMM for overall repr
        print >> sys.stderr, 'Using GMM with', self.vocab_size, 'and', vocab_size_per_cl , 'per class.'
        self.gmm = GMM(self.vocab_size, n_iter=1, params='w', covariance_type='diag')
        # Init by fitting with ones
        self.gmm.fit(np.ones((self.vocab_size, feature_size)))

        # Overwrite values from supervised codebook GMM by class GMMs
        index = 0
        for _, sgmm in self.gmms.iteritems():
            vocab_size_per_cl = len(sgmm.means_)
            self.gmm.means_  [index:index + vocab_size_per_cl, :] = sgmm.means_
            self.gmm.covars_ [index:index + vocab_size_per_cl, :] = sgmm.covars_
            index += vocab_size_per_cl

        # Set uniform GMM weights
        self.gmm.weights_ = np.ones(self.vocab_size) / float(self.vocab_size)
        return

    def train(self, datadict, labels, rand_features=True):
        '''
        Trains a scipy GMM for each class, joins them into a super codebook.
        @param datadict: Dictionary of class labels. 
        Inside each label there is a list of feature matrices for each window [frames x feature]
        @param labels: the labels of the datadict in a given order
        @param rand_features: Shuffles the samples before running the GMM
        '''
        self.criterion = []
        # Stack the features
        allfeatures = np.vstack(list([np.vstack(x) for x in datadict.values()]))

        # Determine the normalisation statistics and remember them
        self.norm = FeatureNormalizer()
        self.norm.setup(allfeatures)


        # Get number of classes
        ncl = len(labels)
        # Compute vocabsize per class
        vocab_size_per_cl = max(1, self.vocab_size / ncl)
        # Update vocabsize to account for rounding errors
        self.vocab_size = vocab_size_per_cl * ncl

        #
        # Train GMMs for each class
        #
        self.gmms = {}
        self.labels = labels
        for label in labels:
            # Compute feature representations
            feats = np.vstack(datadict[label])
            if rand_features:
                np.random.shuffle(feats)
            if self.normalize:
                norm_features = self.norm.normalize(feats)
            else:
                norm_features = (feats)
            print >> sys.stderr, ("Training a GMM for label %s, using scipy and data of shape %s"
                                 % (label, str(np.shape(norm_features))))
            # Train the gmm
            sub_gmm = GMM(vocab_size_per_cl, covariance_type='diag', n_iter=100)
            sub_gmm.fit(norm_features)
            # Set GMM for class
            self.gmms[label] = sub_gmm
        #
        # Combine GMMs to super codebook
        #
        self.compute_super_codebook(allfeatures.shape[1])
        return

    def classify_proba(self, features):
        '''
        Returns the GMM predictions for the features of each component
        '''
        return self.gmm.predict_proba(features)

    def score_samples(self, features):
        '''
        Return the GMM scores for the features
        '''
        return self.gmm.score_samples(features)


class BoFModelSuper(BoFGMMBase):
    '''
    Soft quantization using an supervised GMM.
    Computes normalized/pooled probabilities of Gaussians.
    '''

    def __init__(self, vocab_size=100):
        '''
        Initializes the supervised Bag-of-Features model
        '''
        super(BoFModelSuper, self).__init__(vocab_size)
        self.featuresize = vocab_size
        return


    def train(self, datadict, labels):
        '''
        Trains the GMM Model with supervised codebooks 
        and soft quantization
        '''
        super(BoFModelSuper, self).train(datadict, labels, rand_features=True)
        self.featuresize = self.vocab_size
        return


    def getfeaturesize(self):
        '''
        Returns the feature size (i.e. vocab_size)
        '''
        return self.featuresize


    def getfeatures(self, features):
        '''
        Returns the Bag-of-Super-Features representation
        for the feature matrix [frames x features]
        '''
        norm_features = self.norm.normalize(features)
        probas = self.gmm.predict_proba(norm_features)
        bof_hist = np.mean(probas, axis=0)
        return bof_hist



class BoFModelPyramid(BoFGMMBase):
    '''
    Soft quantization using an supervised GMM using a pyramid.
    Computes normalized/pooled probabilities of Gaussians on multiple tiles.
    '''

    def __init__(self, vocab_size=100, tiles=2):
        '''
        Initializes the pyramid model
        @param vocab_size: Size of the overall vocabulary (vocab_size/num_classes per class) 
        @param tiles:     Number of tiles for base level of the pyramid
        '''
        super(BoFModelPyramid, self).__init__(vocab_size)
        self.tiles = tiles
        return


    def _pyramid2(self, features):
        '''
        Returns a 2 level pyramid
        @param features: input features 
        '''
        # Error case: Frame too small for pyramid, just assume symmetric frame
        if len(features) < 2:
            features = np.vstack((features, features))

        # Compute 2 level pyramid
        l_2 = int(len(features) / 2)
        p_left = self.gmm.predict_proba(features[:l_2])
        p_right = self.gmm.predict_proba(features[l_2:])

        # Compute means
        p_left = np.mean(p_left, axis=0)
        p_right = np.mean(p_right, axis=0)
        # Compute max pooling and stacking
        term_vector = np.max((p_left, p_right), axis=0)
        term_vector = np.hstack((term_vector, p_left, p_right))
        return term_vector


    def _pyramidn(self, features, num_tiles):
        '''
        Returns a n level pyramid at ground
        with an additional top level
        @param features: input features
        @param num_tiles: number of pyramid tiles 
        '''
        # Error case: Frame too small for pyramid, just assume symmetric frame
        if len(features) < num_tiles:
            stacked_feat = []
            for _ in range(num_tiles):
                stacked_feat.append(features)
            features = np.vstack(stacked_feat)

        # Compute tile size
        l_n = len(features) / float(num_tiles)

        # Compute probabilities for each tile
        tiles = []
        for i in range(num_tiles):
            probas = self.gmm.predict_proba(features[int(i * l_n):int((i + 1) * l_n)])
            probas = np.mean(probas, axis=0)
            tiles.append(probas)
        # Compute the top level
        top_level = np.vstack(tiles)
        top_level = np.max(top_level, axis=0)
        # Unfold into term vector
        tiles.append(top_level)
        # Create overall feature representation
        term_vector = np.hstack(tiles)
        return term_vector


    def train(self, datadict, labels):
        '''
        Trains the Pyramid Model with supervised codebooks 
        and soft quantization
        '''
        super(BoFModelPyramid, self).train(datadict, labels, rand_features=True)
        self.featuresize = self.vocab_size
        return


    def getfeaturesize(self):
        '''
        Returns the feature size
        '''
        feat_size = (self.tiles + 1) * self.vocab_size
        return feat_size


    def getfeatures(self, features):
        '''
        Returns the Pyramid Bag-of-Super-Features representation
        for the feature matrix [frames x features]
        '''
        # Normalize features
        norm_features = self.norm.normalize(features)
        # Compute pyramid representations
        if self.tiles == 2:
            return self._pyramid2(norm_features)
        elif self.tiles > 2:
            return self._pyramidn(norm_features, self.tiles)
        else:
            raise RuntimeError("No Pyramid level specified")
        return None

def get_model(typename, cbsize, num_tiles):
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
        return BoFModelPyramid(cbsize, num_tiles)

    if typename == 'temporal':
        return BoFModelTemporal(cbsize, -num_tiles)

    if typename == 'super':
        return BoFModelSuper(cbsize)

    return None

