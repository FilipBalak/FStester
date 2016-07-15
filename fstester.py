import xml.etree.ElementTree as ET
import pprint
import os
import sys
import time
from time import gmtime, strftime
from sklearn.externals import joblib
import machinelearning
# Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import coefbugrepair
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis 
# Clustering
from sklearn import cluster
from sklearn import mixture
import numpy as np
np.set_printoptions(threshold=np.inf)


def do_task(task,source,target=None,model = None,test = None, dump_folder=False, fs_task=False):
    # task -- String with name of used machine learning algorithm.
    # source -- Path to the file that is used to train.
    # target -- Name of attribute in source that is considered as target.
    # model -- Object loaded from file with trained model.
    # test -- Path to the file that is used to test.
    # dump_folder - Path to folder where will be created files with trained model.
    # fs_task -- String with name of used feature selection algorithm.
    
    if model == None:
        # Classification
        if task == "RandomForestClassifier":
            tasktype = "classification"
            model = RandomForestClassifier(n_estimators=10)
        elif task == "KNeighborsClassifier":
            tasktype = "classification"
            model = KNeighborsClassifier(20)
        elif task == "SVC":
            tasktype = "classification"
            model = SVC(kernel="linear", C=0.025)
        elif task == "DecisionTreeClassifier":
            tasktype = "classification"
            model = coefbugrepair.DecisionTreeClassifierWithCoef(max_depth=5)
        elif task == "AdaBoostClassifier":
            tasktype = "classification"
            model = AdaBoostClassifier()
        elif task == "GaussianNB":
            tasktype = "classification"
            model = GaussianNB()
        elif task == "MultinomialNB":
            # For demonstration on text data
            # https://classes.soe.ucsc.edu/cmps290c/Spring12/lect/14/CEAS2006_corrected-naiveBayesSpam.pdf
            #
            # The term Multinomial Naive Bayes simply lets us know that each p(fi|c) is a multinomial distribution, rather than some other distribution. This works well for data which can easily be turned into counts, such as word counts in text.
            # http://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
            tasktype = "classification"
            model = MultinomialNB()
        elif task == "QDA":
            tasktype = "classification"
            model = QuadraticDiscriminantAnalysis()
        elif task == "LDA":
            tasktype = "classification"
            model = LinearDiscriminantAnalysis()
        # Clustering
        elif task == "KMeans":
            tasktype = "clustering"
            model = cluster.KMeans(n_clusters=4)
        elif task == "AF":
            tasktype = "clustering"
            model = cluster.AffinityPropagation(preference=-50)
        elif task == "MeanShift":
            tasktype = "clustering"
            model = cluster.MeanShift(bin_seeding=True)
        elif task == "Agglomerative":
            tasktype = "clustering"
            model = cluster.AgglomerativeClustering()
        elif task == "DBSCAN":
            tasktype = "clustering"
            model = cluster.DBSCAN(eps=0.3, min_samples=10)
        elif task == "Birch":
            tasktype = "clustering"
            model = cluster.Birch()
        elif task == "Gaussian":
            tasktype = "clustering"
            model = mixture.GMM(n_components=2, covariance_type='full')
        
    if model !=None:
        if tasktype == "classification":
            results = machinelearning.classification(source,model,target,test,fs_task)
        elif tasktype == "clustering":
            results = machinelearning.clustering(source,model,target,test,fs_task)
            
        print("Results")
        pprint.pprint(results["score"])

        if dump_folder != False:
            if dump_folder != "":
                directory = dump_folder
            else:
                directory = '../models/model'+str(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
            if not os.path.exists(directory):
                os.makedirs(directory)
            joblib.dump(results["model"], directory+'/model.pkl')
            print("Dump file of model was created: " + directory+'/model.pkl')
            
        return results

    else:
        print("On input is wrong task type")
        sys.exit(1)
        
# Gets individual experiments with their parameters
e = ET.parse('options.xml').getroot()

for child in e.findall('dataset'):
    print("Loading data...")
    task = child.find('task').text
    if child.find('test').text != None:
        test = child.find('test').text
    else:
        test=False
    if child.find('target').text != None:
        target = child.find('target').text
    print("Task processing begins...")
    start_time = time.time()
    if child.find('dumpfolder').text != None:
        dump_folder = child.find('dumpfolder').text
    else:
        dump_folder = False
    if child.find('loadfile').text != None:
        try:
            model = joblib.load(child.find('loadfile').text)
        except:
            print("Could not find a file to load: " + child.find('loadfile').text)
            sys.exit(1)
        results = do_task(task,child.find('source').text,target,model,test,dump_folder,child.find('fs').text)
    else:
        results = do_task(task,child.find('source').text,target,None,test,dump_folder,child.find('fs').text)
    print("Processing finished...")
    
    # Processing time in seconds
    elapsed_time = time.time() - start_time
    
    # Creation of log file or editing of existing one.
    if child.find('logfile').text != None:
        try:
            if not os.path.exists(os.path.dirname(child.find('logfile').text)):
                os.makedirs(os.path.dirname(child.find('logfile').text))
            logFile=open(child.find('logfile').text, 'a')
            logFile.write("Algorithm:\r\n")
            logFile.write(task+"\r\n")
            if child.find('fs').text is not None:
                logFile.write("Feature selection:\r\n")
                pprint.pprint(child.find('fs').text, logFile)
            logFile.write("Processing time:\r\n")
            logFile.write(str(elapsed_time)+"\r\n")
            logFile.write("Time of the end: "+strftime("%Y-%m-%d-%H-%M-%S", gmtime())+"\r\n")
            logFile.write("Score:\r\n")
            pprint.pprint(results["score"], logFile)
            if results["removed_features"]:
                logFile.write("Removed features:\r\n")
                pprint.pprint(results["removed_features"], logFile)
                
                if isinstance(results["removed_features"][0],list):
                    logFile.write("Removed features count:\r\n")
                    pprint.pprint(set(sum(results["removed_features"], [])), logFile)
                
            if results["metrics"]:
                pprint.pprint(results["metrics"], logFile)
                for metric in results["metrics"]:
                    if isinstance(results["metrics"][metric], list):
                        pprint.pprint(metric, logFile)
                        pprint.pprint(np.mean([x for x in results["metrics"][metric] if x is not None]), logFile)
            logFile.write("\r\n")
        except:
            print("There was an error in log file creation: ", sys.exc_info()[0]+" - "+sys.exc_info()[1])
            sys.exit(1)
    
    # Info in terminal.
    print("More info:")
    pprint.pprint(results["score"])
    if results["removed_features"]:
        print("Removed features:")
    if results["metrics"]:
        pprint.pprint(results["metrics"])