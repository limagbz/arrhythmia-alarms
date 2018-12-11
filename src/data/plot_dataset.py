# -*- coding: utf-8 -*-
import os
import gc
import math
import click
import threading

import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec

from tqdm                   import tqdm
from numpy                  import nan
from scipy.spatial.distance import pdist, squareform

def plot_signals(data_dir, save_dir, plot_type):

    """
       Plot signals in the correct format to generate the bottleneck features. For the signals
       database the records are plotted stacked, this is, one below the other. For the recurrence
       the signals are plotted in a 3x3 grid. Note that to maintain a pattern, each type of signal
       has your place in the figure. For example, the ECG signals are the first ones in each plot. 

        Parameters:
            save_dir  -- Where to get the data
                type:    pathlib.Path

            save_dir  -- Where to save the plots
                type:    pathlib.Path

            plot_type -- Which type of plot 
                type:    String
                values:  signal, recurrence
             
    """

    PLOT_SIGNAL_LIST  = ["ECG1", "ECG2", "ABP", "PLETH", "RESP"]                 # List of Signals

    SIGNAL_TYPES = {
        "ECG":    ["I", "II", "III", "MCL", "V", "aVF", "aVL", "aVR"],           # Possible Signals encountered in physionet
        "PLETH":  "PLETH",
        "ABP":    "ABP",
        "RESP":   "RESP"
    }   

    click.secho("Plotting Data...", fg="green")

    # Creates the folder if it not exists
    if not save_dir.is_dir():       
        click.secho("Folder " + str(save_dir) + " do not exists. Creating Folder...", fg="yellow", blink=True, bold=True)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Reading Records
    gc.collect()                                            # Cleaning memory before start plotting
    for f in tqdm(os.listdir(str(data_dir))):
        
        # To do the pattern analysis in the models, we need to have a structured way to format the data
        # Since there is 4 types of signals that can be encountered (where a record have 2 ECG signals)
        # it is possible to have at most 5 inputs considering all the records. This snippet of code adds 
        # columns for the missing signals in each record and enforce an order to the output for the plots
        record_signals = pd.read_csv(str(data_dir / f) , index_col=0)

        # Skipping if file already exists
        image_path = save_dir / (str(f[0:5]) + ".png")
        
        if image_path.exists(): click.secho("File %s already exists. Skipping..." %str(image_path), fg="yellow")  
        
        ecg_ix_control = 0
        for x in record_signals:
            
            # Check if the signals is an ECG
            if x in SIGNAL_TYPES["ECG"]:
                ecg_ix_control += 1
                record_signals.rename(columns={x : "ECG" + str(ecg_ix_control)}, inplace=True)
            
            # Get the signals that are missing
            missList = list(set(PLOT_SIGNAL_LIST) - set(list(record_signals)))

        # Add signal to the record
        for m in missList:
            record_signals[m] = nan
    
        # Creating Images
        data_start = minuteToSample("4:40")
        data_end   = minuteToSample("5:00")
        
        if(plot_type == "signal"):
            threading.Thread(
                target=stacked_signal_plot(
                    df          = record_signals[PLOT_SIGNAL_LIST][data_start : data_end],
                    path        = str(image_path),
                    figure_dim  = 256,
                    dpi         = 5
                )
            ).start()
        
        else:
            threading.Thread(
                target=stacked_recurrence_plot(
                    df          = record_signals[PLOT_SIGNAL_LIST][data_start : data_end],
                    path        = str(image_path),
                    figure_dim  = 256,
                    dpi         = 5
                )
            ).start()

def minuteToSample(value, frequency=250):

    """
       Converts time in format mm:ss in number of samples based
       on the frequency. 

        Parameters:
            value     -- Time in format mm:ss
                type:    String

            frequency -- Sampling Frequency
                type:    int
                default: 250
    """

    if (value == None): return None
        
    time    = value.split(":")
    seconds = int(time[1])
    seconds += (int(time[0]) * 60)
    
    return seconds * frequency

def stacked_signal_plot(df, path, figure_dim=256, dpi=1, y_lim=(-1, 1)):

    """ 
        Plot all the columns of a dataframe as stacked line plots

        Parameters:
            df          -- dataframe containing the data  
                type:       pandas.DataFrame

            path        -- Path to save the figure 
                type:       String, 
                
            figure_dim  -- Width and Height of the image
                type:       int
                default:    512

            dpi         -- DPI of the image
                type:       int
                default:    1
            
            y_lim       -- Limit values for y axis
                type:       tuple (int, int)
                default:    (-1, 1)

    """
    plt.clf()
    
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, dpi = dpi, figsize=(figure_dim, figure_dim))
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, wspace = 0.1, hspace = 0.1)
 
    for ax, ix in zip(range(len(axes)), df):
        axes[ax].plot(df[ix], linewidth=5)
        axes[ax].set_frame_on(False)
        axes[ax].set_ymargin(0)
        axes[ax].set_xmargin(0)
        axes[ax].set_ylim(y_lim)
        axes[ax].lines[0].set_color("black")
        axes[ax].autoscale(enable=True, axis='x', tight=True)
        axes[ax].get_xaxis().set_visible(False)
        axes[ax].get_yaxis().set_visible(False)
        
    fig.savefig(path, pad_inches=0)    
    plt.close(fig)

def stacked_recurrence_plot(df, path, figure_dim=256, dpi=1, cmap='gray'):

    """ 
        Plot all the columns of a dataframe as a grid of recurrence plots

        Parameters:
            df          -- dataframe containing the data  
                type:       list

            path        -- Path to save the figure 
                type:       String, 
                
            figure_dim  -- Width and Height of the image
                type:       int
                default:    512

            dpi         -- DPI of the image
                type:       int
                default:    1

            cmap         -- Matplotlib's colormaps
                type:       string
                default:    gray
    """

    plt.clf()

    plt.figure(dpi=1, figsize = (figure_dim, figure_dim))
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.01, hspace=0.02, top=0.95, bottom=0.05, left=0.17, right=0.845)

    for i, ix in zip(range(6), df):
        
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_frame_on(False)
        ax1.set_ymargin(0)
        ax1.set_xmargin(0)
        
        if (math.isnan(df[ix].iloc[0])):
            ax1.plot()
        else:
            ax1.imshow(rec_plot(df[ix]), cmap=cmap)
        
    plt.savefig(path, pad_inches=0)    
    plt.close()

def rec_plot(s, eps=0.10, steps=10):

    """ 
        Plot recurrence plots
        OBS: Created by https://github.com/laszukdawid/recurrence-plot
        For more info about parameters, see https://en.wikipedia.org/wiki/Recurrence_plot

        Parameters:
            s   -- time series  
                type: pandas.DataFrame

            eps -- Epsilon Value
                type: float
                default: 0.1

            steps -- Steps to compute the plot
                type: int
                default: 10

        returns:
            Array (2 Dimensions) - Recurrence plot in an array format    

    """
    d = pdist(s[:,None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    Z.astype("float32")
    return Z