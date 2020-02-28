# Bismillahhirahmanirahim

import os
import joblib
import numpy as np
from datetime import datetime
from multiprocessing import  Pool
from sklearn.ensemble import RandomForestRegressor

datasetdir = '/casawaridisk/Share/Dataset/SUSY/pkl/'
outputdir = '/casawaridisk/qunox/ML/Sepohon2/Output/V2-Result/SuSy'  # <--- Change to your dir
outputfilename = 'RegAnal-RF-5f-SuSy%s.pkl' % datetime.now().strftime('%Y%m%d%H%M%S')

maxcycle = 10
regressionsamplesizeratio = 0.3
max_highfea_inuse = 5
clusternum = 100
similiarityfunc = 'euclidean'
scalescore = False
processnum = 7

def sumerror(truels, regls):
    return np.sum(np.abs(truels - regls), axis=1)

print('Start')
filelist = os.listdir(datasetdir)

for filename in filelist:
    datasetpath = os.path.join(datasetdir, filename)
    print('\n\n\nReading file: %s' % datasetpath)

    dt = joblib.load(datasetpath)
    trainnoi_1_df = dt['noisedf1']
    trainnoi_2_df = dt['noisedf2']
    datasetratio = dt['signaltonoiseratio']
    test_df = dt['testdf']
    lowfeals = dt['lowfeals']
    highfeals = dt['highfeals']
    samplelbl = test_df['label']

    outputfilename = 'RegAnal-RF-5f_%s_%s.pkl' % (datasetratio, datetime.now().strftime('%Y%m%d%H%M%S'))

    trainlow_mat = trainnoi_1_df[lowfeals].values
    testlow_mat = test_df[lowfeals].values
    reflow_mat = trainnoi_2_df[lowfeals].values

    ref_mse_mat = []
    test_mse_mat = []

    pool = Pool(processes=processnum)
    for cycle in range(maxcycle):
        print('Entering cycle: %s' % cycle)

        # chosing random highFea
        inuse_highfea_ls = np.random.choice(highfeals, size=max_highfea_inuse, replace=False)

        # Creating the highFea matrix
        trainhigh_mat = trainnoi_1_df[inuse_highfea_ls].values
        testhigh_mat = test_df[inuse_highfea_ls].values
        refhigh_mat = trainnoi_2_df[inuse_highfea_ls].values

        # Training the RegModel
        print('\tTraining the regmodel')
        regmodel = RandomForestRegressor(n_jobs=-1)
        regmodel.fit(trainlow_mat, trainhigh_mat)
        print('\tFinish training the regmodel')

        # Predicting the highFea
        print('\tTesting the regmodel')
        pre_test_highmat = regmodel.predict(testlow_mat)
        pre_ref_highmat = regmodel.predict(reflow_mat)
        print('\tFinish testing the regmodel')

        # Measuring the absolute error
        test_errsum = sumerror(testhigh_mat, pre_test_highmat)
        ref_errsum = sumerror(refhigh_mat, pre_ref_highmat)

        # Collectiong the mse result
        ref_mse_mat.append(ref_errsum)
        test_mse_mat.append(test_errsum)
        print('End of cycle: %s\n\n' % cycle)

    pool.close()
    print('Saving file')
    outputfile = {'ref_mse_mat': ref_mse_mat,
                  'test_mse_mat': test_mse_mat,
                  'maxcycle' : maxcycle,
                  'samplelbl': samplelbl}
    joblib.dump(outputfile, os.path.join(outputdir, outputfilename), compress=1)
print('Job Done')
