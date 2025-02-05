import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class EEGClassifier:
    """EEG分类器类"""
    def __init__(self, classifier_type='svm'):
        self.classifier_type = classifier_type
        self.classifier = self._create_classifier()
        
    def _create_classifier(self):
        if self.classifier_type == 'svm':
            return SVC(kernel='rbf')
        elif self.classifier_type == 'rf':
            return RandomForestClassifier()
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier()
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
            
    def fit(self, X, y):
        """训练分类器"""
        self.classifier.fit(X, y)
        
    def predict(self, X):
        """预测类别"""
        return self.classifier.predict(X)