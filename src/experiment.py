
import warnings
warnings.simplefilter("ignore")

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

import gc
from keras             import backend as K

import logging
import click

import pandas                  as pd

from tqdm                      import tqdm
from pathlib                   import Path
from itertools                 import product
from keras.applications.vgg16  import VGG16

from models.train_model        import train_model

PROJECT_DIR              = Path(__file__).parents[1]
RAW_DIR                  = PROJECT_DIR / "data/raw"
INTERIM_DIR              = PROJECT_DIR / "data/interim"
PROCESSED_DIR            = PROJECT_DIR / "data/processed"

SIGNAL_PROCESSED_DIR     = PROCESSED_DIR / "bottleneck/signal"
RECURRENCE_PROCESSED_DIR = PROCESSED_DIR / "bottleneck/recurrence"

RESULTS_SIGNAL_DIR       = PROJECT_DIR / "reports/results/signal"
RESULTS_RECURRENCE_DIR   = PROJECT_DIR / "reports/results/recurrence"

logging.basicConfig(filename=str(PROJECT_DIR / 'experiment_log.log'),level=logging.DEBUG, format='%(asctime)s: %(levelname)s - %(message)s')

EXP_NUMBER = 0

def clear_mem():
    c = gc.collect()
    g = K.clear_session()
    logging.info("Clearing Memory: GC: %s GPU: %s" %(str(c), str(g)))
    pass
    
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def grid_search_values(**params):
    # https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/model_selection/_search.py#L779
    items = sorted(params.items())
    if not items: yield {}
    else:  
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params
        
def grid_search_experiment(data_dir, results_dir, metric="AUC", patience=100, min_delta=0, evaluation_class=0, **params):
    
    results_dir.mkdir(exist_ok=True, parents=True)
    exp_number    = EXP_NUMBER
    
    logging.info("Started %s experiment with parameters %s" %(results_dir.parts[-2], str(params)))
    
    for gs in tqdm(list(grid_search_values(**params)), position=0, desc="Grid Search Bar", dynamic_ncols=True):

        logging.info('Started GridSearch %s' %(str(gs)))
           
        save_dir      = results_dir / ("E" + str(exp_number)) 
        train_model(data_dir, save_dir, metric, patience, min_delta, evaluation_class, **gs)
        exp_number += 1

        logging.info('Ended GridSearch %s' %(str(gs)))

        try: 
            parameters_df = pd.read_csv(str(results_dir / "parameters.csv"), index_col=None)
        except:
            pd.DataFrame().to_csv(str(results_dir / "parameters.csv"))
            parameters_df = pd.read_csv(str(results_dir / "parameters.csv"), index_col = None)

        parameters_df = parameters_df.append(gs, ignore_index=True)[list(gs.keys())]
        parameters_df.to_csv(str(results_dir / "parameters.csv"))

        clear_mem()

    logging.info("Ended %s experiment with parameters %s" %(results_dir.parts[-2], str(params)))

if __name__ == "__main__":

    logging.info('Script Started')

    experiment_grid_search = {
        # EXP 01
        "hidden_layers" : [[128], [256], [128, 128], [128, 256], [256, 128], [256, 256]],
        "dropout":        [0.2, 0.35, 0.5],

        # EXP 02
        "lr":             [0.0001, 0.001, 0.025, 0.01],
        "batch_size":     [8, 16, 32],
        
        # EXP 03
        "nesterov":       [True, False],
        "momentum":       [0.5, 0.7, 0.9],
        
        # Constants
        "decay":          [0.0]         
    }

    # # EXP 01
    
    # step = "step1"
    # exp_params               = experiment_grid_search.copy()
    # exp_params["lr"]         = [experiment_grid_search["lr"][1]]
    # exp_params["batch_size"] = [experiment_grid_search["batch_size"][1]]
    # exp_params["momentum"]   = [experiment_grid_search["momentum"][1]]
    # exp_params["nesterov"]   = [experiment_grid_search["nesterov"][1]]

    # EXP 02
    # step = "step2"
    # exp_params                  = experiment_grid_search.copy()
    # exp_params["hidden_layers"] = [experiment_grid_search["hidden_layers"][1]]
    # exp_params["dropout"]       = [experiment_grid_search["dropout"][0]]
    # exp_params["momentum"]      = [experiment_grid_search["momentum"][1]]
    # exp_params["nesterov"]      = [experiment_grid_search["nesterov"][1]]

    # # EXP 03
    # step = "step3"
    # exp_params                  = experiment_grid_search.copy()
    # exp_params["hidden_layers"] = [experiment_grid_search["hidden_layers"][1]]
    # exp_params["dropout"]       = [experiment_grid_search["dropout"][0]]
    # exp_params["batch_size"]    = [experiment_grid_search["batch_size"][0]]
    # exp_params["lr"]            = [experiment_grid_search["lr"][1]]


    try:

#         EXP_NUMBER = 0
#         step = "step3"
       
#         exp_params                  = experiment_grid_search.copy()
#         exp_params["hidden_layers"] = [experiment_grid_search["hidden_layers"][1]]
#         exp_params["dropout"]       = [experiment_grid_search["dropout"][0]]
#         exp_params["batch_size"]    = [experiment_grid_search["batch_size"][1]]
#         exp_params["lr"]            = [experiment_grid_search["lr"][0]]
#         grid_search_experiment(SIGNAL_PROCESSED_DIR, RESULTS_SIGNAL_DIR / step, metric="aucroc", patience=100, min_delta=0, evaluation_class=0, **exp_params)


        EXP_NUMBER = 4
        step = "step3"
        exp_params                  = experiment_grid_search.copy()
        exp_params["hidden_layers"] = [experiment_grid_search["hidden_layers"][1]]
        exp_params["dropout"]       = [experiment_grid_search["dropout"][0]]
        exp_params["batch_size"]    = [experiment_grid_search["batch_size"][0]]
        exp_params["lr"]            = [experiment_grid_search["lr"][0]]

        exp_params["nesterov"] = [True, False]
        exp_params["momentum"] = [0.9]

        grid_search_experiment(RECURRENCE_PROCESSED_DIR, RESULTS_RECURRENCE_DIR / step,  metric="aucroc", patience=100, min_delta=0, evaluation_class=0, **exp_params)

        # logging.warning('Shutting down system')
        # click.secho("Shutting down system", fg='yellow')
        # os.system('shutdown -s')
    
    except Exception as e:
        import traceback
        traceback.print_exc()
         
        pass
        # logging.error('System found an error: %s' %(e))
        # click.secho("System found an error %s" %(e), fg='red')

        # logging.warning('Shutting down system')
        # click.secho("Shutting down system", fg='red')

        # os.system('shutdown -s')