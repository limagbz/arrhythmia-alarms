# -*- coding: utf-8 -*-
import os
import wfdb
import click

import numpy  as np
import pandas as pd

from shutil                     import copy
from tqdm                       import tqdm
from numpy                      import isnan
from pathlib                    import Path
from wfdb.io                    import dl_database
from sklearn.preprocessing      import minmax_scale 
from sklearn.model_selection    import StratifiedKFold, train_test_split                           

PROJECT_DIR         = Path.joinpath(Path(__file__).parents[2])
RAW_DATA_DIR        = PROJECT_DIR / "data/raw"

def download_db(save_dir):
    
    """
        Downloads data from Physionet's 2015 Challenge Reducing False Arrhythmia Alarms in the ICU

        Parameters:
            save_dir -- Path to save the WFDB files
                type: patlib.Path

    """
    WFDB_NAME = 'challenge/2015/training'

    click.echo()
    click.secho("Downloading " + WFDB_NAME + " dataset on " + str(save_dir), fg="green")

    # Create Folder if it not exists
    if not save_dir.is_dir():       
        click.secho("Folder " + str(save_dir) + "do not exists. Creating Folder...", fg="yellow")
        save_dir.mkdir(exist_ok=True)
    
    # Download database from Physionet
    dl_database(db_dir = WFDB_NAME, dl_dir = str(save_dir)) 

    click.secho("Done!", fg="green")

def tabulate_scale_data(data_dir, save_dir, scale=False):
    
    """
       Get list of records that respect's the filter. 

        Parameters:
            records_list -- List of records to tabulate
                type:       Array of Strings

            save_dir     -- Path to save the tabulated data
                type:       patlib.Path

            scale    -- If data needs to be scaled
                type:       boolean
                default:    False
    """
    click.echo()
    if (scale): click.secho("Scaling data...", fg="green")
    else:       click.secho("Tabulating data...", fg="green")

    # Creates the folder if it not exists
    if not save_dir.is_dir():       
        click.secho("Folder " + str(save_dir) + " do not exists. Creating Folder...", fg="yellow")
        save_dir.mkdir(exist_ok=True)
    
    for r in tqdm(list(data_dir.glob('*.hea'))):

        rec_name   = r.name[0:5]
        csv_signal = save_dir / (rec_name + ".csv")
        
        # Skipping if file already exists
        if csv_signal.exists():
            click.secho("File %s already exists. Skipping..." %str(csv_signal), fg="yellow")  
            continue
        
        # Gettiing Signal Info from WFDB records
        record  = wfdb.rdsamp(str(r)[:-4])
        pvalues = record[0]
        info    = record[1]

        n_signals   = info['n_sig']
        sig_len     = info["sig_len"]
        sig_names   = info["sig_name"]

        # Passing Record to a CSV
        signals = {}
        for i in range(n_signals):
            signals[sig_names[i]] = []
        
        for i in range(sig_len):
            for j in range(n_signals):
                signals[sig_names[j]].append(pvalues[i][j])

        signal_db = pd.DataFrame(signals)

        # scale Data
        if (scale == True):
            for i in signal_db:
                if not (isnan(signal_db[i]).any()):
                    signal_db[i] = minmax_scale(signal_db[i], feature_range=(-1, 1), axis=0, copy=True)

        signal_db.to_csv(str(csv_signal))

    click.secho("Done!", fg="green")

def dataset_split(data_dir, save_dir, record_db_path, test_size=0.2, n_folds=5, shuffle=True, random_seed=42):

    """
       Splits the dataset into train, validation and test set on the K-Fold format
       based on records_db.csv

        Parameters:
            data_dir    -- Folder containing the image data
                type:      pathlib.Path
            
            save_dir    -- Where to save the results
                type:      pathlib.Path

            test_size   -- Size of the test set in percentage
                type:      float

            test_size   -- Number of folds for the experiment
                type:      int

            shuffle     -- If data will be shuffled before splitting
                type:      boolean

            random_seed -- Random seed
                type:      int               
    """
    
    # Reading Records db to get names and classes of the records
    rec_db          = pd.read_csv(str(record_db_path), index_col=0)
    records_names   = list(rec_db.name)
    records_classes = list(rec_db.alarm)
    
    # Dataset Splitting
    click.secho("Train and Test Splitting...", fg="green")
    rec_train, rec_test, class_train, class_test = train_test_split(records_names, records_classes, test_size=test_size, shuffle=shuffle, stratify=records_classes, random_state=random_seed)
    
    # Test Set generation
    true_test_dir       = save_dir / "test/true"
    false_test_dir      = save_dir / "test/false"
    
    true_test_dir.mkdir(exist_ok=True, parents=True)
    false_test_dir.mkdir(exist_ok=True, parents=True)
    
    click.secho("Copying Files for test set...", fg="green")
    # Copying Files
    for record, cls in tqdm(zip(rec_test, class_test)):
        
        rec_name = (record + ".png")
        
        src_path = data_dir / rec_name
        dst_path = true_test_dir if cls == True else false_test_dir
        dst_path /= rec_name
        
        try:
            copy(str(src_path), str(dst_path))
        except FileNotFoundError:
            click.secho("File not found: " + str(src_path), fg="yellow")
    
    click.secho("Done!", fg="green")
            
    # Train Set Generation
    click.secho("Copying Files for training/validation sets...", fg="green")
    fold_n = 1
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
    
    # Converting to np array for easy indexing
    rec_train   = np.array(rec_train) 
    class_train = np.array(class_train)
    
    for train_index, valid_index in tqdm(skf.split(rec_train, class_train)):
        
        true_train_dir  = save_dir / "train" / ("fold" + str(fold_n)) / "true"
        false_train_dir = save_dir / "train" / ("fold" + str(fold_n)) / "false"
        true_valid_dir  = save_dir / "valid" / ("fold" + str(fold_n)) / "true"
        false_valid_dir = save_dir / "valid" / ("fold" + str(fold_n)) / "false"
        
        true_train_dir.mkdir(exist_ok=True, parents=True)
        false_train_dir.mkdir(exist_ok=True, parents=True)
        true_valid_dir.mkdir(exist_ok=True, parents=True)
        false_valid_dir.mkdir(exist_ok=True, parents=True)
        
        # Train Files
        for record, cls in zip(rec_train[train_index], class_train[train_index]):
            
            rec_name = (record + ".png")

            src_path = data_dir / rec_name
            dst_path = true_train_dir if cls == True else false_train_dir
            dst_path /= rec_name
            
            try:
                copy(str(src_path), str(dst_path))
            except FileNotFoundError:
               click.secho("[Train] File not found: " + str(src_path), fg="yellow")
        
        # Valid Files
        for record, cls in zip(rec_train[valid_index], class_train[valid_index]):
            
            rec_name = (record + ".png") 

            src_path = data_dir / rec_name
            dst_path = true_valid_dir if cls == True else false_valid_dir
            dst_path /= rec_name
            
            try:
                copy(str(src_path), str(dst_path))
            except FileNotFoundError:
                click.secho("[Valid]File not found: " + str(src_path), fg="yellow", blink=True, bold=True)

        fold_n += 1

    click.secho("Done!", fg="green")

def generate_records_db(data_dir, save_dir):

    """ 
        Generates a .csv file named records_db.csv that contains all the information 
        about each record. 

        Parameters:
            save_dir -- Path to save the WFDB files 
                type: patlib.Path

    """
    click.secho("Generating records DB...", fg="green")
    
    # Skip the generation if the database is already created
    save_path = save_dir / "records_db.csv"
    if save_path.is_file():
        click.secho("File already exists. Skipping this step...", fg="yellow", blink=True, bold=True)
        return

    # Getting records names
    files   = os.listdir(str(data_dir))
    records = [f[:5] for f in files if f[6:] == "hea"]

    # Creating dict of signal info
    df_dict = {
        "name":         [],
        "n_signals":    [],
        "disease":      [],
        "sig_names":    [],
        "sig_len":      [],
        "alarm":        [],
        "problem":      []
    }

    # Generates records_db.csv
    for r in tqdm(records):

        record         = wfdb.rdsamp(str(data_dir / r))
        record_info    = record[1]                              # Get info from record

        df_dict["name"].append(r)
        df_dict["n_signals"].append(record_info['n_sig'])
        df_dict["disease"].append(record_info['comments'][0])
        df_dict["sig_names"].append(record_info["sig_name"])
        df_dict["sig_len"].append(record_info["sig_len"])
        df_dict["alarm"].append(True if record_info['comments'][1] == "True alarm" else False)
        df_dict["problem"].append("realtime" if record_info["sig_len"] == 75000 else "retrospective")

    # Creates the folder if it not exists
    if not save_dir.is_dir():       
        click.secho("Folder " + str(save_dir) + " do not exists. Creating Folder...", fg="yellow", blink=True, bold=True)
        save_dir.mkdir(exist_ok=True)

    # Saving Dataframe on disk
    records_db = pd.DataFrame(df_dict)
    records_db.to_csv(save_dir / "records_db.csv")
    
    click.secho("Done!", fg="green")