# Bismillahhirahmanirahim

import os
import numpy as np
from time import time
from numpy import mean
from datetime import datetime
from collections import Counter
from scipy.spatial.distance import euclidean

from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
# --------------------------------------------- Parameter --------------------------------------------------------------

datasetpath = '' # <--- .pkl file type
outputdir = ''  # <--- Change to your dir
outputfilename = 'Sepohon[ANN]-%s.pkl' % datetime.now().strftime('%Y%m%d%H%M%S')

maxcycle = 1000
regressionsamplesizeratio = 0.3
max_highfea_inuse = 5
clusternum = 100
similiarityfunc = 'euclidean'
scalescore = False
nodesizefactor = 2


def givescore(rawscore):
    scoremat = np.array(rawscore)
    scoremat = np.transpose(scoremat)
    scorels = np.sum(scoremat, axis=1)

    stdscorels = MinMaxScaler(feature_range=(0, 1)).fit_transform(scorels.reshape(-1, 1))
    alpha = 0.5 / np.median(stdscorels)
    f = lambda x: 1 / (1 + np.exp(-alpha * x))
    stdscorels = list(map(f, stdscorels))
    stdscorels = np.array(stdscorels).reshape(-1)

    return stdscorels


# -------------------------------------------------- Main --------------------------------------------------------------
print('Start')
print('Loading the dataset')
dt = joblib.load(datasetpath)
trainnoi_1_df = dt['noisedf1']
trainnoi_2_df = dt['noisedf2']
test_df = dt['mixdf1']
lowfeals = dt['lowfeals']
highfeals = dt['highfeals']
samplelbl = test_df['label']
print('Fininsh Loading')

regresstrainsampleindex_ls = trainnoi_1_df.index.values
clustertrainsampleindex_ls = trainnoi_2_df.index.values

testingsampleindex_ls = test_df.index.values

totaltestingsample = len(testingsampleindex_ls)
totalregsample = len(trainnoi_1_df)
totalclstrainsample = len(trainnoi_2_df)

regressionsamplesize = int(len(regresstrainsampleindex_ls) * regressionsamplesizeratio)
print('Starting the filter cycle')

print('Creating the regression model')
highfeanum = len(highfeals)

clusterlblls = list(range(100))
scaler = MinMaxScaler()
regmodel = Sequential()
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 10), input_dim=len(lowfeals), kernel_initializer='normal',
                   activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 10), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 10), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 5), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 5), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 1), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(int(max_highfea_inuse * nodesizefactor * 1), kernel_initializer='normal', activation='relu'))
regmodel.add(Dense(max_highfea_inuse, kernel_initializer='normal'))
regmodel.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
regmodel.summary()

cyclels = []
feausels = []
cycleperiodls = []
scorels = []
testsample_ratiols = []
trainsample_ratiols = []
testsample_distls = []
trrainsample_distls = []

for cycle in range(maxcycle):

    print('\n\n\n==> Starting cycle: %s (%.2f)' % (cycle, cycle / maxcycle))
    starttime = time()
    cyclels.append(cycle)

    print('==> Selecting the high feature')
    inuse_highfea_ls = np.random.choice(highfeals, size=max_highfea_inuse, replace=False)
    feausels.append(inuse_highfea_ls)
    print('==> High feature in use: %s' % ','.join(inuse_highfea_ls))

    # Creatinge the regression sample and model
    print('==> Creating regression matrix')
    regresstrainsampleindex_ls = np.random.choice(trainnoi_1_df.index.values, size=regressionsamplesize, replace=False)
    regdf = trainnoi_1_df.loc[regresstrainsampleindex_ls]
    reg_lowfea_mat = regdf[lowfeals].values
    reg_highfea_mat = regdf[inuse_highfea_ls].values

    clstraindf = trainnoi_2_df.loc[clustertrainsampleindex_ls]
    clstrain_lowfea_mat = clstraindf[lowfeals].values
    clstrain_highfea_mat = clstraindf[inuse_highfea_ls].values

    inusetestsample_df = test_df.loc[testingsampleindex_ls]
    inusetestsample_lowfea_mat = inusetestsample_df[lowfeals].values
    trueinusetestsample_highfea_mat = inusetestsample_df[inuse_highfea_ls].values

    print('==> Training the regression model')
    regmodel.fit(reg_lowfea_mat, reg_highfea_mat,
                 epochs=16,
                 batch_size=16384,
                 validation_split=0.2,
                 verbose = 0)
    print('==> Done training')

    # Regressing the sample
    print('==> Regressing the sample and clstrain matrix')
    predict_inusetestsample_highfea_mat = regmodel.predict(inusetestsample_lowfea_mat)
    predict_clstrain_highfea_mat = regmodel.predict(clstrain_lowfea_mat)
    predict_reg_highfea_mat = regmodel.predict(reg_lowfea_mat)
    print('==> Finish Regressing the sample and clstrain matrix')

    print('==> Calculating the regression error')
    regerr_inusetestsample_mat = trueinusetestsample_highfea_mat - predict_inusetestsample_highfea_mat
    regerr_clstrain_mat = clstrain_highfea_mat - predict_clstrain_highfea_mat

    # Clustering the sample
    print('==> Creating the cluster model')
    clsmodel = MiniBatchKMeans(n_clusters=clusternum, max_iter=1000)  # <--- Cluster model
    print('==> Training the cluster model')
    clsmodel.fit(regerr_inusetestsample_mat)
    print('==> Done training the cluster model')

    print('==> Clustering the testing sample')
    clslabel_inusetestsample_ls = clsmodel.predict(regerr_inusetestsample_mat)
    clslabel_clstrain_ls = clsmodel.predict(regerr_clstrain_mat)

    print('==> Calculating cluster distributions')
    counter_inusetestsample_dict = Counter(clslabel_inusetestsample_ls)
    counter_clstrain_dict = Counter(clslabel_clstrain_ls)

    print('==> Calculating the ratio')
    ratiols = []
    ils = []

    for i in counter_inusetestsample_dict.keys():
        clstraincount = counter_clstrain_dict[i]
        if clstraincount == 0: clstraincount = 1
        clsratio = counter_inusetestsample_dict[i] / clstraincount
        ratiols.append(clsratio)
        ils.append(i)

    if scalescore:
        ratiols = scaler.fit_transform(np.array(ratiols).reshape(-1, 1))
        ratiols = ratiols.reshape(-1)

    ratiodict = {}
    for i, ratio in zip(ils, ratiols):
        ratiodict[i] = {'ratio': ratio}

    for clslbl in clusterlblls:
        if clslbl not in ratiodict.keys():
            ratiodict[clslbl] = {'ratio': 0}

    testsample_cyclescore = [ratiodict[clslabel]['ratio'] for clslabel in clslabel_inusetestsample_ls]
    testsample_ratiols.append(testsample_cyclescore)

    trainsample_cyclescore = [ratiodict[clslabel]['ratio'] for clslabel in clslabel_clstrain_ls]
    trainsample_ratiols.append(trainsample_cyclescore)

    print('==> Calculating the intra-cluster sample distance distribution')
    intradistdict = {}
    for i in clusterlblls:
        intradistdict[i] = {'trainmat': [],
                            'testmat': [], }

    for i, vec in zip(clslabel_inusetestsample_ls, regerr_inusetestsample_mat):
        intradistdict[i]['testmat'].append(vec)

    for i, vec in zip(clslabel_clstrain_ls, regerr_clstrain_mat):
        intradistdict[i]['trainmat'].append(vec)

    ils = []
    meanshiftls = []
    for i in clusterlblls:

        trainmat = np.array(intradistdict[i]['trainmat'])
        testmat = np.array(intradistdict[i]['testmat'])

        if len(trainmat) < 1 or len(testmat) < 1:
            meanshift = 0
        else:
            train_meanvec = mean(trainmat, axis=0)
            test_meanvec = mean(testmat, axis=0)
            meanshift = euclidean(train_meanvec, test_meanvec)

        ils.append(i)
        meanshiftls.append(meanshift)

    if scalescore:
        meanshiftls = scaler.fit_transform(np.array(meanshiftls).reshape(-1, 1))
        meanshiftls = meanshiftls.reshape(-1)

    for i, dist in zip(ils, meanshiftls):
        intradistdict[i]['dist'] = dist

    testsample_cycledisls = [intradistdict[lbl]['dist'] for lbl in clslabel_inusetestsample_ls]
    trainsample_cycledisls = [intradistdict[lbl]['dist'] for lbl in clslabel_clstrain_ls]

    testsample_distls.append(testsample_cycledisls)
    trrainsample_distls.append(trainsample_cycledisls)

    cycletime = time() - starttime
    cycleperiodls.append(cycletime)
    print('==> Cycle period: %s second' % cycletime)
    eta = ((maxcycle - cycle) * cycletime) / 3600
    print('==> Eta: %.3f hours' % eta)

print('Post-processing')

testratios = givescore(testsample_ratiols)
trainratiols = givescore(trainsample_ratiols)
testdistls = givescore(testsample_distls)
traindistls = givescore(trrainsample_distls)

resultdic = {
    'cyclels': cyclels,
    'feausels': feausels,
    'processingtime': cycleperiodls,
    'testratiosls': testratios,
    'trainratiols': trainratiols,
    'testdistls': testdistls,
    'traindistls': traindistls,
    'distancefunc': 'euclidean',
    'scalescore' : scalescore,
    'samplelbl' : samplelbl
}

print('Saving the file')
joblib.dump(resultdic, os.path.join(outputdir, outputfilename), compress=1)
print('Alhamdulilah done')
