import numpy as np


class ContigencyMatrix:
    def __init__(self, y_true, y_pred):
        """
        Contigency matrix calculated as intersection of indices
       
        Matrix: 
        rows: prediction labels
        cols: ground truth labels

        Example:
        ground truth labels: [0,0,0,1,1,1]
        prediction labels  : [0,0,1,0,1,1]

        TP = sum([F,F,F,T,T,T] *
                 [F,F,T,F,T,T]) = 2
        FP = sum([T,T,T,F,F,F] *
                 [F,F,T,F,T,T]) = 1
        """
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Sort labels by frequency. Base class equals the most frequent class
        self.vals, self.cnts = np.unique(y_true, return_counts=True)
        if len(self.vals) == 2:
            self.binary = True
            if 0 in self.vals:
                self.labels = np.concatenate((self.vals[self.vals!=0], [0]))
        else:
            self.binary = False
            self.labels = tuple(self.vals[np.argsort(self.cnts)])
        
        # Sanity check
        if len(self.labels) < 2:
            raise ValueError('Only one class in the train')

        # Indices of Real and Prediction values by label
        self.true_idx = {label: y_true==label for label in self.labels}
        self.pred_idx = {label: y_pred==label for label in self.labels}

        self.mtx = np.zeros((len(self.labels), len(self.labels)))
        for row, row_label in enumerate(self.labels):
            for col, col_label in enumerate(self.labels):
                self.mtx[row, col] = np.sum(self.true_idx[col_label]*self.pred_idx[row_label])

    def accuracy(self):
        return np.trace(self.mtx)/np.sum(self.mtx)

    def precision(self):
        if self.binary:
            self.precision_score = self.mtx[0,0]/sum(self.mtx[0,:])
        else:
            self.precision_score = np.array([self.mtx[i,i]/sum(self.mtx[i,:]) for i in range(self.mtx.shape[0])])
        return np.average(self.precision_score)

    def recall(self):
        if self.binary:
            self.recall_score = self.mtx[0,0]/sum(self.mtx[:,0])
        else:
            self.recall_score = np.array([self.mtx[i,i]/sum(self.mtx[:,i]) for i in range(self.mtx.shape[0])])
        return np.average(self.recall_score)

    def f1(self):
        self.f1_score = 2*np.average(self.recall_score)*np.average(self.precision_score)/(
            np.average(self.recall_score+self.precision_score))
        return self.f1_score


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    cm = ContigencyMatrix(ground_truth, prediction)
    
    return cm.precision(), cm.recall(), cm.f1(), cm.accuracy()


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    cm = ContigencyMatrix(ground_truth, prediction)
    return cm.accuracy()


if __name__=='__main__':
    y_true = np.array([ True,  True,  True, False, False,  True,  True,  True,  True, 
        False,  True, False,  True,  True,  True,  True])
    y_pred = np.array([ True,  True,  True,  True,  True,  True, False,  True,  True,
        False, False,  True,  True,  True, False, False])

    cm = ContigencyMatrix(y_true, y_pred)
    print('Precision=%2.2f\nRecall=%2.2f\nF1=%2.2f\nAcc=%2.2f' %(cm.precision(), cm.recall(), cm.f1(), cm.accuracy()))
