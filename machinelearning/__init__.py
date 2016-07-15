from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, linear_model, cluster, datasets, metrics as mx
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted, NotFittedError
import featureselection
import numpy as np
import csvhandling
import pprint
import featureselection

def classification(source, model, target_att, test_source = "", fs_task=False):
    # source -- Path to the file that is used to train.
    # model -- Object loaded from file with trained model.
    # target_att -- Name of attribute in source that is considered as target.
    # test_source -- Path to the file that is used to test.
    # fs_task -- String with name of used feature selection algorithm.  
    
    results = dict.fromkeys(["predictions", "score", "model", "features", "removed_features", "selected_features", "feature_importances", "measures"])
    results["predictions"] = []
    
    # Basic metrics used for classification and feature selection evaluation.
    metrics = dict.fromkeys(["accuracy","recall","precision","f_measure","f_beta"])
    metrics["accuracy"] = []
    metrics["recall"] = []
    metrics["precision"] = []
    metrics["f_measure"] = []
    results["removed_features"] = []
    results["selected_features"] = []
    results["feature_importances"] = []
    # http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html
    metrics["f_beta"] = []
    
    cfr = model
    print(model)
    
    # Object for reading train data and test data
    csv = csvhandling.CsvHandling()
    
    # Numpy array with values from source path without feature names and target values.
    train = csv.read_csv(source)
    
    # List of feature names
    features = csv.get_features(source)
    
    # Numpy array with target values
    target = csv.get_target(source,target_att)

    if test_source != False:
        
        # Numpy array with values from test_source path without feature names and target values.
        test = csv.read_csv(test_source)
        
        # Numpy array with test target values
        test_target = csv.get_target(test_source,target_att)
        
        if fs_task:
            # Pipeline with fitted model and feature selection filter or only fitted model.
            cfr = featureselection.get_fs_model(cfr, fs_task, train, target)
            
            original_features = features[:]
            if fs_task == "RFE":
                selected_features = []
            else:
                selected_features = featureselection.get_selected_features(cfr.named_steps["feature_selection"],original_features)
            removed_features = [i for i in features if not i in selected_features]
            results["removed_features"].append(removed_features)
            results["selected_features"].append(selected_features)
        else:
            cfr.fit(train, target)    
        prediction = cfr.predict(test)
        results["predictions"].append(prediction)
        metrics["accuracy"].append(mx.accuracy_score(test_target, prediction))
        metrics["precision"].append(mx.precision_score(test_target, prediction, average="macro"))
        metrics["recall"].append(mx.recall_score(test_target, prediction, average="macro"))
        metrics["f_measure"].append(mx.f1_score(test_target, prediction, average="macro"))
    else:
        # If there are no test data than there is cross-validation used for model evaluation.
        cv = cross_validation.KFold(len(train), n_folds=5, shuffle=False, random_state=None)
        
        if fs_task == "RFE":
            # Pipeline with fitted model and feature selection filter or only fitted model.
            cfr = featureselection.get_fs_model(cfr, fs_task+"CV", train, target, cv)
            
            original_features = features[:]
            selected_features = featureselection.get_selected_features(cfr,original_features)
            removed_features = [i for i in features if not i in selected_features]
            results["removed_features"].append(removed_features)
            results["selected_features"].append(selected_features)
            for traincv, testcv in cv:
                test = train[testcv]
                test_target = target[testcv]
                prediction = cfr.predict(test)
                results["predictions"].append(prediction)
                metrics["accuracy"].append(mx.accuracy_score(test_target, prediction))
                metrics["precision"].append(mx.precision_score(test_target, prediction))
                metrics["recall"].append(mx.recall_score(test_target, prediction))
                metrics["f_measure"].append(mx.f1_score(test_target, prediction))
                metrics["f_beta"].append(mx.fbeta_score(test_target, prediction, 0.5))        
        else:
            for traincv, testcv in cv:
                # Repaired bug from http://stackoverflow.com/questions/19265097/why-does-cross-validation-for-randomforestregressor-fail-in-scikit-learn
                if fs_task:
                    cfr = featureselection.get_fs_model(cfr, fs_task, train[traincv], target[traincv])
                    original_features = features[:]
                    if fs_task == "fromModel":
                        selected_features = featureselection.get_selected_features(cfr,original_features)
                    else:
                        selected_features = featureselection.get_selected_features(cfr.named_steps["feature_selection"],original_features)
                    removed_features = [i for i in features if not i in selected_features]
                    results["removed_features"].append(removed_features)
                    results["selected_features"].append(selected_features)
                else:
                    cfr.fit(train[traincv], target[traincv])
                
                test = train[testcv]
                test_target = target[testcv]
                prediction = cfr.predict(test)
                
                results["predictions"].append(prediction)
                metrics["accuracy"].append(mx.accuracy_score(test_target, prediction))
                metrics["precision"].append(mx.precision_score(test_target, prediction))
                metrics["recall"].append(mx.recall_score(test_target, prediction))
                metrics["f_measure"].append(mx.f1_score(test_target, prediction))
                metrics["f_beta"].append(mx.fbeta_score(test_target, prediction, 0.5))
    results["score"] = cfr.score(test, test_target)
    results["model"] = cfr
    results["metrics"] = metrics
    return results

def clustering(source, model, target_att=None, test_source = "", fs_task=False):
    # source -- Path to the file that is used to train.
    # model -- Object loaded from file with trained model.
    # target_att -- Name of attribute in source that is considered as target.
    # test_source -- Path to the file that is used to test.
    # fs_task -- String with name of used feature selection algorithm.
    
    results = dict.fromkeys(["predictions", "score", "model", "features", "removed_features", "selected_features", "measures"])
    results["predictions"] = []
    metrics = dict.fromkeys(["homogeneity", "completeness","v_measure"])
    metrics["homogeneity"] = []
    metrics["completeness"] = []
    metrics["v_measure"] = []
    cls = model
    
    # If there is already fitted model loaded from file.
    try:
        check_is_fitted(model, 'estimator_')
        fitted = True
    except NotFittedError as err:
        fitted = False
        
    
    # Object for reading train data and test data
    csv = csvhandling.CsvHandling()
    
    # Numpy array with values from source path without feature names and target values.
    train = csv.read_csv(source)
    
    # List of feature names
    features = csv.get_features(source)
    
    # Expected clusters
    if target_att is not None:
        # Numpy array with target values
        target = csv.get_target(source,target_att)
    else:
        target = False
        
    results["features"] = features
    if fitted == False:
        if fs_task:
            # Pipeline with fitted model and feature selection filter or only fitted model.
            cls = featureselection.get_fs_model(cls, fs_task, train,target)
            
            original_features = features[:]
            selected_features = featureselection.get_selected_features(cls.named_steps["feature_selection"],original_features)
            results["removed_features"] = [i for i in features if not i in selected_features]
            results["selected_features"] = selected_features
        else:
            if target is not False:
                cls.fit(train,target)
            else:
                cls.fit(train)
    
    if test_source != False:
        test = csv.readCsv(test_source)
        results["predictions"].append(cls.predict(test))
    else:
        if target_att is None:
            prediction = cls.predict(train)
            results["predictions"].append(prediction)
            metrics["homogeneity"].append(mx.homogeneity_score(test_target, prediction))
            metrics["completeness"].append(mx.completeness_score(test_target, prediction))
            metrics["v_measure"].append(mx.v_measure_score(test_target, prediction))
        
        else:
            # If there are no test data than there is cross-validation used for model evaluation.
            cv = cross_validation.KFold(len(train), n_folds=5, shuffle=False, random_state=None)
            for traincv, testcv in cv:
                
                # Repaired bug from http://stackoverflow.com/questions/19265097/why-does-cross-validation-for-randomforestregressor-fail-in-scikit-learn
                train = np.asarray(train)
                target = np.asarray(target)

                cls.fit(train[traincv], target[traincv])
                test = train[testcv]
                test_target = target[testcv]
                
                prediction = cls.predict(test)
                results["predictions"].append(prediction)
                metrics["homogeneity"].append(mx.homogeneity_score(test_target, prediction))
                metrics["completeness"].append(mx.completeness_score(test_target, prediction))
                metrics["v_measure"].append(mx.v_measure_score(test_target, prediction))
    
    results["model"] = cls
    results["metrics"] = metrics
    
    return results
