import click

import numpy as np

from tqdm                      import tqdm
from pathlib                   import Path, PureWindowsPath
from keras.preprocessing.image import ImageDataGenerator
                               
PROJECT_DIR         = Path.joinpath(Path(__file__).parents[2])
INTERIM_DATA_DIR    = PROJECT_DIR / "data/interim"

def extract_bottleneck_features(model, data_dir, save_dir, color_mode="rgb", batch_size=16):
    
    datagen = ImageDataGenerator(rescale= 1./255)
    
    # Train Bottleneck Features
    generator = datagen.flow_from_directory(
        directory   = str(data_dir), 
        target_size = (224, 224), 
        color_mode  = color_mode,
        class_mode  = None,        # Changed to get the bottleneck
        batch_size  = batch_size, 
        shuffle     = False
    )
     
    bottleneck_features = model.predict_generator(generator)
    
    # Saving bottleneck features
    features_path = save_dir / (str(batch_size) + "_features.npy")
    classes_path  = save_dir / (str(batch_size) + "_classes.npy")
    
    np.save(open(str(features_path), 'wb'), bottleneck_features)
    np.save(open(str(classes_path), 'wb'), generator.classes)
    
def make_bottleneck_dataset(model, src_dir, dst_dir, n_folds=5, color_mode="rgb", batch_size=16):
    
    # Test Bottlencks
    src_test  = src_dir / "test"
    dst_test  = dst_dir / "test"
    
    src_test.mkdir(exist_ok=True, parents=True)
    dst_test.mkdir(exist_ok=True, parents=True)
    
    extract_bottleneck_features(model, src_test, dst_test, color_mode, batch_size)
    
    # Train/Validation (K-Fold) bottlenecks
    for fold in range(1, n_folds + 1):
        
        src_train = src_dir / "train" / ("fold" + str(fold))
        src_valid = src_dir / "valid" / ("fold" + str(fold))
        dst_train = dst_dir / "train" / ("fold" + str(fold))
        dst_valid = dst_dir / "valid" / ("fold" + str(fold))
        
        src_train.mkdir(exist_ok=True, parents=True)
        src_valid.mkdir(exist_ok=True, parents=True)
        dst_train.mkdir(exist_ok=True, parents=True)
        dst_valid.mkdir(exist_ok=True, parents=True)
        
        extract_bottleneck_features(model, src_train, dst_train, color_mode, batch_size)
        extract_bottleneck_features(model, src_valid, dst_valid, color_mode, batch_size)