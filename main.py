
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

# project files
import functions as fn
import data as dt
from data import ohlc_data as data
from data import exec_models
from data import exec_exp

# support functions
from datetime import datetime
import multiprocessing as mp
import warnings
import random

# reproducible results
random.seed(123)

# ignore warnings
warnings.filterwarnings("ignore")

# main process to run from console
if __name__ == "__main__":

    print('                                                            ')
    print(' -- ---------------- ------------------- ---------------- --')
    print(' -- ----------------   Start execution   ---------------- --')
    print(' -- ---------------- ------------------- ---------------- --\n\n')

    # main loop to test all t-fold sizes
    for model in exec_models:
   
        # Measure the begining of the code execution process
        ini_time = datetime.now()

        # ------------------------------------------------------------- TIMESERIES FOLDS FOR DATA DIVISION  #
        # ----------------------------------------------------------- ---------------------------------- -- #

        # Timeseries data division in t-folds
        # folds = fn.t_folds(p_data=data, p_period=iteration)
        
        # -- ------------------------------------------------------------------- FOLD EVALUATION PROCESS -- #
        # -- ------------------------------------------------------------------- ----------------------- -- #

        # List with the names of the models
        ml_models = dt.exec_models
        
        # Establish the number of workers with as many cores as the computer has
        workers = dt.exec_workers

        # create pool of workers for asyncronous parallelism
        pool = mp.Pool(workers)

        # configuration for tensorflow and CPU/GPU processing
        fn.tf_processing(p_option='cpu', p_cores=workers)
        
        # Parallel Asyncronous Process 
        fold_process = {'fold_' + str(model): pool.starmap(fn.fold_process,
                                                           [(fn.t_folds(p_data=data, p_period=exp[0]),
                                                             ml_models, exp[1], exp[2], exp[3], exp[4])
                                                            for exp in exec_exp])}

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
        