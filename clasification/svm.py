import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
import joblib

class SVMClassifier:
    def __init__(self, dataset_file, target_column, label_column):
        self.dataset_file = dataset_file
        self.target_column = target_column
        self.label_column = label_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        data = pd.read_csv(self.dataset_file)
        self.X = data.drop([self.target_column, self.label_column], axis=1)
        self.y = data[self.target_column]

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self, kernel='linear'):
        self.model = SVC(kernel=kernel)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy:", accuracy)

    def save_model(self, filename='svm_model.joblib'):
        output_model_path = 'models'
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        # Export model dengan nama file dari params dijoin path dengan filename
        joblib.dump(self.model, os.path.join(output_model_path, filename))
