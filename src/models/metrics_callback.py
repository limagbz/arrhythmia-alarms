import os
import keras
import numpy as np

from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, precision_recall_fscore_support,accuracy_score

class metrics_callback(keras.callbacks.Callback):
    
    def __init__(self):
        self.hasValidData   = False
        self.hasEarlyStop   = False
    
    def setTrainValidData(self, training_data, validation_data):
        self.x_train, self.y_train = training_data[0], training_data[1]
        
        if (validation_data == ([], [])):
            self.hasValidData = False 
        else:    
            self.hasValidData = True 
            self.x_valid = validation_data[0]
            self.y_valid = validation_data[1]
    
    def setEarlyStopping(self, metric="aucroc", patience=0, min_delta=0, classEval = 0):
        self.hasEarlyStop = True
        self.metric       = metric
        self.patience     = patience
        self.min_delta    = min_delta
        self.class_eval   = classEval
    
    def on_train_begin(self, logs={}):
        
        # Early Stopping
        if (self.hasEarlyStop == True):
            self.best       = -np.Infinity
            self.epoch_stop = 0
            self.wait       = 0
            self.bestModel  = None
 
        # Metrics calculations
        if (self.hasValidData != True): raise RuntimeError("Training and validation data were not set")
            
        self.results = dict(
            train_accuracy     = [], train_roc_auc      = [], 
            train_precision_C0 = [], train_precision_C1 = [],  
            train_recall_C0    = [], train_recall_C1    = [],
            train_f1_C0        = [], train_f1_C1        = [],
            valid_accuracy     = [], valid_roc_auc      = [],
            valid_precision_C0 = [], valid_precision_C1 = [],
            valid_recall_C0    = [], valid_recall_C1    = [],
            valid_f1_C0        = [], valid_f1_C1        = [],
        )
        
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred_train_prob = self.model.predict(self.x_train, batch_size=1)
        y_pred_train      = (y_pred_train_prob > 0.5).astype('int32')
        
        accuracy_train                                         = accuracy_score(self.y_train, y_pred_train)
        roc_train                                              = roc_auc_score(self.y_train, y_pred_train_prob.ravel())
        precision_train, recall_train, f1_train, support_train = precision_recall_fscore_support(self.y_train, y_pred_train, beta=1.0) #pylint: disable=W0612
        
        self.results["train_accuracy"].append(accuracy_train)
        self.results["train_roc_auc"].append(roc_train)
        self.results["train_precision_C0"].append(precision_train[0])
        self.results["train_precision_C1"].append(precision_train[1])
        self.results["train_recall_C0"].append(recall_train[0])
        self.results["train_recall_C1"].append(recall_train[1])
        self.results["train_f1_C0"].append(f1_train[0])
        self.results["train_f1_C1"].append(f1_train[1])  
        
        if (self.hasValidData == True):
            y_pred_valid_prob = self.model.predict(self.x_valid)
            y_pred_valid      = (y_pred_valid_prob > 0.5).astype('int32')
            
            # Confusion Matrix
            accuracy_valid                                         = accuracy_score(self.y_valid, y_pred_valid)
            roc_valid                                              = roc_auc_score(self.y_valid, y_pred_valid_prob.ravel())
            precision_valid, recall_valid, f1_valid, support_valid = precision_recall_fscore_support(self.y_valid, y_pred_valid, beta=1.0)  #pylint: disable=W0612
            
            # Appending Values
            self.results["valid_accuracy"].append(accuracy_valid)
            self.results["valid_roc_auc"].append(roc_valid)
            self.results["valid_precision_C0"].append(precision_valid[0])
            self.results["valid_precision_C1"].append(precision_valid[1])
            self.results["valid_recall_C0"].append(recall_valid[0])
            self.results["valid_recall_C1"].append(recall_valid[1])
            self.results["valid_f1_C0"].append(f1_valid[0])
            self.results["valid_f1_C1"].append(f1_valid[1])
                
        if (self.hasEarlyStop == True):
            # Selecting metric
            if (self.metric == "aucroc"):      value = roc_valid
            elif (self.metric == "f1"):        value = f1_valid[self.class_eval]
            elif (self.metric == "precision"): value = precision_valid[self.class_eval]
            else: raise ValueError("Metric: " + self.metric + "is not a valid metric")
            
            # Early Stopping 
            if ((value - self.min_delta) > self.best):
                self.best       = value
                self.wait       = 0
                self.bestModel  = self.model
            else:
                self.wait  += 1
            
            if (self.wait >= self.patience):
                self.epoch_stop          = epoch
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        self.hasValidData   = False