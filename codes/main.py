
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: main.py - python script with the main operations to run in cluster                            -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from os import environ
from data import ohlc_data as data
from data import iter_fold
from data import iter_exp
from datetime import datetime
from multiprocessing import cpu_count

import functions as fn
import data as dt
import multiprocessing as mp
import warnings
import random

# reproducible results
random.seed(123)

# ignore warnings
warnings.filterwarnings("ignore")

# Suppress console log messages from TensorFlow
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# main process to run from console
if __name__ == "__main__":

    print('                                                            ')
    print(' -- ---------------- ------------------- ---------------- --')
    print(' -- ----------------   Start execution   ---------------- --')
    print(' -- ---------------- ------------------- ---------------- --\n\n')

    # main loop to test all t-fold sizes
    for iteration in iter_fold:

        # debugging
        # iteration = iter_fold[0]
    
        # Measure the begining of the code execution process
        ini_time = datetime.now()

        # ------------------------------------------------------------- TIMESERIES FOLDS FOR DATA DIVISION  #
        # ----------------------------------------------------------- ---------------------------------- -- #

        # Timeseries data division in t-folds
        folds = fn.t_folds(p_data=data, p_period=iteration)

        # -- ------------------------------------------------------------------- FOLD EVALUATION PROCESS -- #
        # -- ------------------------------------------------------------------- ----------------------- -- #

        # List with the names of the models
        # ml_models = list(dt.models.keys())
        ml_models = ['ann-mlp']

        # Establish the number of workers with as many cores as the computer has
        workers = cpu_count()-1
        # workers = 1

        # create pool of workers for asyncronous parallelism
        pool = mp.Pool(workers)

        # configuration for tensorflow and CPU/GPU processing
        fn.tf_processing(p_option='cpu', p_cores=workers)
        
        # Parallel Asyncronous Process 
        fold_process = {'fold_' + str(iteration): pool.starmap(fn.fold_process,
                                                               [(folds, ml_models,
                                                                 exp[0], exp[1], exp[2], exp[3])
                                                               for exp in iter_exp])}

        # close pool
        pool.close()
        # rejoin sepparated resources
        pool.join()

        # -- ------------------------------------------------------------------------ PRINT ELAPSED TIME -- #
        # -- ------------------------------------------------------------------------ ------------------ -- #

        # Measure the end of the code execution process
        end_time = datetime.now()
        print('elapsed time for the whole process: ', str(end_time - ini_time))
        # ---------------------------------------------------------------------------------------------- -- #
