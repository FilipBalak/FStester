from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Rapairment according to http://stackoverflow.com/questions/24123498/recursive-feature-elimination-on-random-forest-using-scikit-learn

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class DecisionTreeClassifierWithCoef(DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(DecisionTreeClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_