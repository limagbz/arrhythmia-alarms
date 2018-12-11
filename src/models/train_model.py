

import click
import logging
import numpy           as np
import pandas          as pd


from tqdm              import tqdm
from .models           import ModelSVM
from keras             import optimizers
from .metrics_callback import metrics_callback
from sklearn.metrics   import confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score

def train_model(data_dir, result_dir, metric="AUC", patience=100, min_delta=0, evaluation_class=0, **params):
    
    train_dirs = [x for x in (data_dir / "train").iterdir() if x.is_dir()]
    valid_dirs = [x for x in (data_dir / "valid").iterdir() if x.is_dir()]
    
    test_results_dict     = dict(accuracy = [], roc_auc = [], precision_C0 = [], precision_C1 = [], recall_C0 = [], recall_C1 = [], f1_C0 = [], f1_C1 = [])
    confusion_matrix_dict = dict(
        train_C00 = [], train_C01 = [], train_C10 = [], train_C11 = [],
        valid_C00 = [], valid_C01 = [], valid_C10 = [], valid_C11 = [],
        test_C00  = [], test_C01  = [], test_C10  = [], test_C11  = [],
    )
    
    test_data    = np.load(data_dir / "test" / (str(params["batch_size"]) +  "_features.npy"))
    test_classes = np.load(data_dir / "test" / (str(params["batch_size"]) +  "_classes.npy"))
    

    fold_n = 0
    for t, v in tqdm(list(zip(train_dirs, valid_dirs)), position=1, desc="Folds Bar", dynamic_ncols=True):

        logging.info('Started Fold %s' %str(fold_n))

        train_data    = np.load(t / (str(params["batch_size"]) + "_features.npy"))
        valid_data    = np.load(v / (str(params["batch_size"]) + "_features.npy"))
        train_classes = np.load(t / (str(params["batch_size"]) + "_classes.npy"))
        valid_classes = np.load(v / (str(params["batch_size"]) + "_classes.npy"))
        
        callback = metrics_callback()
        callback.setTrainValidData((train_data, train_classes), (valid_data, valid_classes))
        callback.setEarlyStopping(metric, patience, min_delta, evaluation_class)
        
        optimizer = optimizers.SGD(params["lr"], params["momentum"], params["decay"], params["nesterov"])
        model     = ModelSVM(train_data.shape[1:], optimizer, params["hidden_layers"], params["dropout"])        
        history   = model.fit(x=train_data, y=train_classes, verbose=0, batch_size=1, epochs=2000, callbacks=[callback], validation_data=(valid_data, valid_classes), shuffle=True) #pylint: disable=W0612

        ### Metrics ###
        # Train
        y_pred_train_prob                           = model.predict(train_data, batch_size=1)
        y_pred_train                                = (y_pred_train_prob > 0.5).astype('int32')
        train_confusion_matrix                      = confusion_matrix(train_classes, y_pred_train, labels=[0, 1])
        train_roc_fpr, train_roc_tpr, train_roc_thr = roc_curve(train_classes, y_pred_train)
        
        # Valid
        y_pred_valid_prob                           = model.predict(valid_data, batch_size=1)
        y_pred_valid                                = (y_pred_valid_prob > 0.5).astype('int32')
        valid_confusion_matrix                      = confusion_matrix(valid_classes, y_pred_valid, labels=[0, 1])
        valid_roc_fpr, valid_roc_tpr, valid_roc_thr = roc_curve(valid_classes, y_pred_valid)
        
        # Test Set
        y_pred_test_prob                                   = model.predict(test_data, batch_size=1)
        y_pred_test                                        = (y_pred_test_prob > 0.5).astype('int32')
        accuracy_test                                      = accuracy_score(test_classes, y_pred_test)
        roc_test                                           = roc_auc_score(test_classes, y_pred_test_prob.ravel())
        precision_test, recall_test, f1_test, support_test = precision_recall_fscore_support(test_classes, y_pred_test, beta=1.0) #pylint: disable=W0612
        test_confusion_matrix                              = confusion_matrix(test_classes, y_pred_test, labels=[0, 1])
        test_roc_fpr, test_roc_tpr, test_roc_thr           = roc_curve(test_classes, y_pred_test)
        
        ### Results ###
        test_results_dict["accuracy"].append(accuracy_test)
        test_results_dict["roc_auc"].append(roc_test)
        test_results_dict["precision_C0"].append(precision_test[0])
        test_results_dict["precision_C1"].append(precision_test[1])
        test_results_dict["recall_C0"].append(recall_test[0])
        test_results_dict["recall_C1"].append(recall_test[1])
        test_results_dict["f1_C0"].append(f1_test[0])
        test_results_dict["f1_C1"].append(f1_test[1])
        
        confusion_matrix_dict["train_C00"].append(train_confusion_matrix[0][0])
        confusion_matrix_dict["train_C01"].append(train_confusion_matrix[0][1])
        confusion_matrix_dict["train_C10"].append(train_confusion_matrix[1][0])
        confusion_matrix_dict["train_C11"].append(train_confusion_matrix[1][1])
        confusion_matrix_dict["valid_C00"].append(valid_confusion_matrix[0][0])
        confusion_matrix_dict["valid_C01"].append(valid_confusion_matrix[0][1])
        confusion_matrix_dict["valid_C10"].append(valid_confusion_matrix[1][0])
        confusion_matrix_dict["valid_C11"].append(valid_confusion_matrix[1][1])
        confusion_matrix_dict["test_C00"].append(test_confusion_matrix[0][0])
        confusion_matrix_dict["test_C01"].append(test_confusion_matrix[0][1])
        confusion_matrix_dict["test_C10"].append(test_confusion_matrix[1][0])
        confusion_matrix_dict["test_C11"].append(test_confusion_matrix[1][1])
        
        # Saving Results
        epoch_results_df = pd.DataFrame(callback.results)
        train_roc_df     = pd.DataFrame({"fpr": train_roc_fpr, "tpr": train_roc_tpr, "thr": train_roc_thr})
        valid_roc_df     = pd.DataFrame({"fpr": valid_roc_fpr, "tpr": valid_roc_tpr,"thr": valid_roc_thr})
        test_roc_df      = pd.DataFrame({"fpr": test_roc_fpr, "tpr": test_roc_tpr,"thr": test_roc_thr})
        
        result_dir.mkdir(exist_ok=True, parents=True)
        file_prefix = str(result_dir / ("fold" + str(fold_n) + "_"))
        
        epoch_results_df.to_csv(file_prefix + "epoch_results.csv")
        train_roc_df.to_csv(file_prefix     + "train_roc.csv")
        valid_roc_df.to_csv(file_prefix     + "valid_roc.csv")
        test_roc_df.to_csv(file_prefix      + "test_roc.csv")

        logging.info('Ended Fold %s' %str(fold_n))
        
        fold_n += 1
        
    # Test results
    test_results_df = pd.DataFrame(test_results_dict)
    conf_results_df = pd.DataFrame(confusion_matrix_dict)
    
    test_results_df.to_csv(str(result_dir / "test_results.csv"))
    conf_results_df.to_csv(str(result_dir / "confusion_matrices.csv"))