import click

import numpy as np

from tqdm          import tqdm
from shutil        import copyfile
from pathlib       import Path
from collections   import Counter
from sklearn.utils import check_random_state

def random_oversampling(src_dir, n_folds=5, random_state=42):

    """
        Perform's random oversampling over the entire data folder

        Parameters:
            src_dir    -- Folder with data to be oversampled
                type:     pathlib.PATH
            
            n_folds    -- Number of folds 
                type:     int              
    """

    click.secho("Performing random oversampling...", fg="green")
    for fold in tqdm(range(n_folds)):

        fold_dir  = src_dir / ("fold" + str(fold + 1))
        true_dir  = fold_dir / "true"
        false_dir = fold_dir / "false"

        # Saving data on arrays
        X, Y = [], []
        true_x  = list(true_dir.iterdir())
        true_y  = [True] * len(true_x)

        X += true_x
        Y += true_y

        false_x = list(false_dir.iterdir())
        false_y = [False] * len(false_x)

        X += false_x
        Y += false_y

        # Select elements to be oversampled
        cls, values = binary_random_oversampler(np.array(X), np.array(Y), random_state)

        # Saves the file on the folder with a "o_" suffix and avoid colisions by renaming
        for file in values:

            dst = file.with_name("o_" + str(file.name))
            
            file_index = 1
            while(dst.is_file()):
                name_split  = str(file.name).split(".")
                dst         = file.with_name("o_" + str(name_split[0]) + "_" + str(file_index) + "." + name_split[1])
                file_index  +=1

            copyfile(str(file), str(dst))


    click.secho("Done!", fg="green")

def binary_random_oversampler(X, Y, random_state=42):

    """
        Perform's random oversampling over arrays of data

        Parameters:
            X          -- Data to be oversampled
                type:     numpy.array
            
            Y           -- Classes of X
                type:      numpy.array

            random_state -- Random state
                type:      int               
    """
    
    # Count number of elements in each class
    dist       = dict(Counter(Y))
    over_count = dist[True] - dist[False]
    
    # Selects the correct elements to be oversampled
    over_class = True if over_count < 0 else False
    over_count = abs(over_count)
    
    # Selects only the elements in the data that needs to be oversampled
    target = np.array([i for i, j in zip(X, Y) if j == over_class])
    
    # Generates the indexes of the data (already oversampled)
    random_state = check_random_state(random_state)
    sample_index = random_state.randint(low=0, high=len(target), size=over_count)
    
    sampling = list(target[sample_index])
    
    return over_class, sampling