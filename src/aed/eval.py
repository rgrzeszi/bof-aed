'''
Copyright (C) 2015 Axel Plinge and Rene Grzeszick. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''

import numpy as np

class Evaluator(object):
    '''
    Basic Evaluation Measures
    '''

    def __init__(self):
        '''
        Constructor
        '''
        return
        
    def f_value(self, precision, recall):
        '''
        Computes the f1-measure
        @param precision: Precision value
        @param recall: Recall value
        @return: f1-measure
        '''
        if precision is None or recall is None:
            return None
        measure = 0.0
        if precision + recall > 0:
            measure = 2.0 * precision * recall / (precision + recall)
        return measure


    def calculate_precision_recall(self, classes, confusion, otherslabel = 'other'):
        '''
        Calculates precision, recall and f-measure given a confusion matrix
        @param classes:       vector of classes
        @param confusion:     confusion matrix
        @param otherslabel:   background labels that are omitted in the frame-wise evaluation
        @return:  precision, recall, f-measure
        '''
        gtcount=0
        for true_index, true_label in enumerate(classes):
            if true_label != otherslabel:
                gtcount += np.sum(confusion[true_index, :])
                            
        estimated = 0
        true_events = 0
        for true_index, true_label in enumerate(classes):        
            for det_index in range(len(classes)):
                if classes[det_index] != otherslabel:
                    estimated += confusion[true_index, det_index]
            if true_label != otherslabel:            
                true_events += confusion[true_index, true_index]
        
        if gtcount>0:
            precision = float(100.0 * true_events) / float(estimated)
        else:
            precision = 0
        if estimated>0:
            recall    = float(100.0 * true_events) / float(gtcount)
        else:
            recall = 0
        fmeasure =  self.f_value(precision, recall) 
        return [precision,  recall, fmeasure] 

