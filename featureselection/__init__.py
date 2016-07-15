from sklearn.svm import LinearSVC
import sklearn.feature_selection as fs_scikit
import sys
from sklearn.pipeline import Pipeline
import numpy as np

def get_selected_features(model,features):
    """
    Estimates support of given fitted model and compares it to given list of features.
    Returns list of features selected through feature selection.
    """
    
    # Different methods use different ways to get support.
    try:
        support = model.support_
    except:
        try:
            support = model.get_support()
        except:
            # The transform function is computed from attribute coef_ for linearSVC, where coef_ must be greater than mean(coef)
            # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
            try:
                support = [True if i >= np.mean(model.coef_[0]) else False for i in model.coef_[0]]
            except:
                print("There was an eror in estimating support")
                sys.exit(1)
    if support is not None:
        for idx, val in enumerate(support):
            if not val:
                features[idx] = None
    features = list(filter(None,features))
    return features

def get_fs_model(model, method, train, target=None, cv=None):
    """
    Connects given model with specified feature selection method and trains the final structure.
    """
    
    if method == "RFE":
        model = fs_scikit.RFE(model, 3, step=10)
        if target != None:
            return model.fit(train, target)
        else:
            return model.fit(train)
    if method == "RFECV":
        model = fs_scikit.RFECV(model, 3, cv=cv)
        if target != None:
            return model.fit(train, target)
        else:
            return model.fit(train)
    elif method == "linearSVC":
        sel = LinearSVC()
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "fromModel":
        if target != None:
            fs = fs_scikit.SelectFromModel(model)
            fs.fit(train, target)
            fs.estimator_.support_ = fs.get_support()
            return fs.estimator_
        else:
            fs = fs_scikit.SelectFromModel(model)
            fs.fit(train)
            fs.estimator_.support_ = fs.get_support()
            return fs.estimator_
        
    elif method == "Anova":
        # ANOVA SVM-C
        anova_filter = fs_scikit.SelectKBest(f_regression, k=5)
        model = Pipeline([
            ('feature_selection', anova_filter),
            ('data_mining', model)
        ])
    elif method == "VarianceThreshold":
        sel = fs_scikit.VarianceThreshold(threshold=(.8 * (1 - .8)))
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectPercentile":
        sel = fs_scikit.SelectPercentile(fs_scikit.f_classif, percentile=50)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFpr":
        sel = fs_scikit.SelectFpr()
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFdr":
        sel = fs_scikit.SelectFdr()
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFwe":
        sel = fs_scikit.SelectFwe()
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "ch2":
        sel = fs_scikit.SelectKBest(fs_scikit.chi2)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    else:
        print("Feature selection method was not found: "+method)
        sys.exit(1)
    if target != None:
        return model.fit(train, target)
    else:
        return model.fit(train)