import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

class SVMClassifier:
    def __init__(self, dataset_file, label_column, feature_column=None, except_feature_column=None):
        self.dataset_file = dataset_file
        self.feature_column = feature_column
        self.except_feature_column = except_feature_column
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
        print(self.dataset_file)
        
        if self.feature_column is None and (self.except_feature_column is None or self.except_feature_column == [None]):
            raise ValueError("The 'feature_column' and 'except_feature_column' parameters are both empty. One of them must be provided.")
        
        if self.except_feature_column is not None and self.except_feature_column != [None]:
            self.X = data.drop(self.except_feature_column, axis=1).values
        elif self.feature_column is not None:
            self.X = data[self.feature_column].values
        
        self.y = LabelEncoder().fit_transform(data[self.label_column].values)  # Encode label
    def split_data(self, test_size=0.2, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Feature scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self, C=1, kernel='linear', gamma='scale', autoParams=False):
        if autoParams:
            # Low Range: 0.01, 0.1, 1
            # Medium Range: 1, 10, 100
            # High Range: 10, 100, 1000
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
            # Menggunakan GridSearchCV
            # grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
            # grid_search.fit(self.X_train, self.y_train)
            # self.model = grid_search.best_estimator_
            # print(f"Best parameters Grid Search: {grid_search.best_params_}")

            # Menggunakan Kombinasi Manual
            best_params = {
                "combination": {},
                "accuracy": 0
            }

            for C in param_grid['C']:
                for kernel in param_grid['kernel']:
                    for gamma in param_grid['gamma']:
                        # print(f"Evaluating combination: C={C}, kernel={kernel}, gamma={gamma}")
                        model = SVC(C=C, kernel=kernel, gamma=gamma)
                        model.fit(self.X_train, self.y_train)
                        predictions = model.predict(self.X_test)
                        accuracy = accuracy_score(self.y_test, predictions)
                        # print(f"Accuracy for combination C={C}, kernel={kernel}, gamma={gamma}: {accuracy}")

                        if accuracy > best_params["accuracy"] and accuracy != 1:
                            best_params["combination"] = {'C': C, 'kernel': kernel, 'gamma': gamma}
                            best_params["accuracy"] = accuracy

            print("\nBest combination found:")
            print(best_params["combination"])
            print(f"Best accuracy: {best_params['accuracy']}")

            # Train the model with the best parameters
            self.model = SVC(**best_params["combination"])
            self.model.fit(self.X_train, self.y_train)
        else:
            print("C:", C, "Kernel:", kernel, "Gamma:", gamma)
            self.model = SVC(C=C, kernel=kernel, gamma=gamma)
            self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)
        
        print("Accuracy:", accuracy)
        print("\nConfusion Matrix:")
        print(cm)

    def save_model(self, filename='svm_model.joblib'):
        output_model_path = 'models'
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        joblib.dump(self.model, os.path.join(output_model_path, filename))