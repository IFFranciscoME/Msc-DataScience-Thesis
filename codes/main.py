
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

# main process to run from console
if __name__ == "__main__":

    print('                                                            ')
    print(' -- ---------------- ------------------- ---------------- --')
    print(' -- ----------------   Start execution   ---------------- --')
    print(' -- ---------------- ------------------- ---------------- --')

    # main loop to test all t-fold sizes
    for iteration in iter_fold:
    
        print('                                                            ')
        print(' ***********************************************************')
        print(' ***************** ITERATION: ' + iteration + ' *********************')
        print(' ***********************************************************')

        # Measure the begining of the code execution process
        ini_time = datetime.now()

        # ------------------------------------------------------------- TIMESERIES FOLDS FOR DATA DIVISION  #
        # ----------------------------------------------------------- ---------------------------------- -- #

        # Timeseries data division in t-folds
        folds = fn.t_folds(p_data=data, p_period=iteration)

        # -- ------------------------------------------------------------------- FOLD EVALUATION PROCESS -- #
        # -- ------------------------------------------------------------------- ----------------------- -- #

        # List with the names of the models
        ml_models = list(dt.models.keys())

        # Create a pool of workers with as many cores as the computer has
        workers = cpu_count()-1
        # workers = 1
        pool = mp.Pool(workers)
        
        # Parallel Asyncronous Process 
        fold_process = {'fold_' + str(iteration): pool.starmap(fn.fold_process,
                                                               [(folds, ml_models, exp[0], exp[1], exp[2])
                                                               for exp in iter_exp])}
        # close pool
        pool.close()
        # rejoin sepparated
        pool.join()

        # -- ------------------------------------------------------------------------------- DATA BACKUP -- #
        # -- ------------------------------------------------------------------------------- ----------- -- #

        # File name to save the data
        file_name = 'files/pickle_rick/genetic_net_' + iteration + '.dat'

        # objects to be saved
        pickle_rick = {'data': dt.ohlc_data, 't_folds': folds, 'fold_process': fold_process}

        # pickle format function
        dt.data_save_load(p_data_objects=pickle_rick, p_data_file=file_name, p_data_action='save')

        # Measure the end of the code execution process
        end_time = datetime.now()
        print('elapsed time for: ', iteration, ' iteration was: ', str(end_time - ini_time))
        # ---------------------------------------------------------------------------------------------- -- #
