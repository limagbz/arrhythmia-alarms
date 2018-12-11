# -*- coding: utf-8 -*-
import click

from pathlib           import Path
from plot_dataset      import plot_signals
from resample_dataset  import random_oversampling
from prepare_dataset   import dataset_split, download_db, tabulate_scale_data, generate_records_db

PROJECT_DIR        = Path(__file__).resolve().parents[2]
RAW_DATA_DIR       = PROJECT_DIR / 'data/raw'
INTERIM_DATA_DIR   = PROJECT_DIR / 'data/interim'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data/processed'

@click.command()
@click.argument('steps', nargs=-1)
@click.option('--only',     is_flag=True,    help="Steps to perform")
@click.option("--exception",   is_flag=True, help="Steps to not Performe")
@click.option("--not-scaling",   is_flag=True, default=False, help="Use raw signals, without max-min scaling")
@click.option('--plot-type', default="all", type=click.Choice(['all', 'signal', 'recurrence']), help="Signals to Plot")
def make_dataset(steps, only, exception, not_scaling, plot_type):

    make_arguments = {
        "download":     True,
        "tabulate":     True, 
        "records":      True,
        "plot":         True,
        "split":        True, 
        "sampling":     True
    }

    # Parameters Validation
    if not (set(list(steps)) <= set(make_arguments.keys())):
        raise click.ClickException("Invalid arguments on input, only: " + ", ".join(list(make_arguments.keys())) + " are permitted")

    if only == True and exception == True:
        raise click.ClickException("--only and --except options are select, choose only one")
    
    # Setting Options
    if (only == True):
        for a in make_arguments.keys():
            make_arguments[a] = True if a in list(steps) else False
    elif (exception == True):
        for s in list(steps):
            make_arguments[s] = False
   
    # Making Dataset
    if make_arguments["download"]: download_db(RAW_DATA_DIR)
    if make_arguments["tabulate"]: tabulate_scale_data(RAW_DATA_DIR, INTERIM_DATA_DIR / "tabulated", not not_scaling)
    if make_arguments["records"]:  generate_records_db(RAW_DATA_DIR, INTERIM_DATA_DIR)
    
    if make_arguments["plot"]:

        data_src = INTERIM_DATA_DIR/"tabulated"
        data_dst = INTERIM_DATA_DIR/"plot"

        if plot_type == "all":
            plot_signals(data_src, data_dst/"signal",  "signal")
            plot_signals(data_src, data_dst/"recurrence", "recurrence")
        else:
            plot_signals(data_src, data_dst/plot_type, plot_type)

    if make_arguments["split"]:
        data_src = INTERIM_DATA_DIR/"plot"
        data_dst = PROCESSED_DATA_DIR/"plot"
        rec_path = INTERIM_DATA_DIR/"records_db.csv"

        if plot_type == "all":
            dataset_split(data_src/"signal", data_dst/"signal", rec_path)
            dataset_split(data_src/"recurrence", data_dst/"recurrence", rec_path)
        else:
            dataset_split(data_src/plot_type, data_dst/plot_type, rec_path)

    if make_arguments["sampling"]:

        data_src = PROCESSED_DATA_DIR/"plot"

        if plot_type == "all":
            random_oversampling(data_src/"signal"/"train")
            random_oversampling(data_src/"recurrence"/"train")
        else:
            random_oversampling(data_src/plot_type/"train")
        
if __name__ == "__main__":
    make_dataset() #pylint: disable=E1120