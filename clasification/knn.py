import os
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    train_test_split,
    GridSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KNNClassifier:
    def __init__(
        self,
        dataset_file,
        label_column,
        feature_column=None,
        except_feature_column=None,
    ):
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
        self.label_encoder = LabelEncoder()

    def load_data(self):
        data = pd.read_csv(self.dataset_file)
        print(f"Loaded dataset from {self.dataset_file}\n")

        if self.feature_column is None and (
            self.except_feature_column is None or self.except_feature_column == [None]
        ):
            raise ValueError(
                "The 'feature_column' and 'except_feature_column' parameters are both empty. One of them must be provided."
            )

        if self.except_feature_column is not None and self.except_feature_column != [
            None
        ]:
            self.X = data.drop(self.except_feature_column, axis=1).values
        elif self.feature_column is not None:
            self.X = data[self.feature_column].values

        self.y = self.label_encoder.fit_transform(data[self.label_column].values)

    def split_data(self, test_size=0.2, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Feature scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self, n_neighbors=5, metric="minkowski", p=2, autoParams=False):
        if autoParams:
            param_grid = {
                "n_neighbors": np.arange(1, 4),
                # "metric": ["minkowski", "euclidean", "manhattan"],
                "metric": ["manhattan"],
                # use euclidean only
                # "p": [1, 2],
            }

            # gunakan KFold dengan k=10
            cv = KFold(n_splits=10, shuffle=True, random_state=0)

            grid_search = GridSearchCV(
                KNeighborsClassifier(), param_grid, cv=cv, n_jobs=-1
            )

            grid_search.fit(self.X_train, self.y_train)
            print(f"Best hyperparameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            print("Training model with specified parameters:")
            print(f"n_neighbors: {n_neighbors}, metric: {metric}, p: {p}")
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors, metric=metric, p=p
            )
            self.model.fit(self.X_train, self.y_train)

        # Print the number of neighbors used for training
        print(f"Number of neighbors (K): {self.model.n_neighbors}")

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)

        # Display the confusion matrix with proper formatting
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
        plt.show()

        # Print detailed evaluation metrics
        print("\nModel Evaluation:")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Display classification report
        report = classification_report(
            self.y_test, predictions, target_names=self.label_encoder.classes_
        )
        print("\nClassification Report:")
        print(report)

    def save_model(
        self, filename="knn_model.joblib", label_encoder_filename="label_encoder.joblib"
    ):
        output_model_path = "models"
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        joblib.dump(self.model, os.path.join(output_model_path, filename))
        joblib.dump(
            self.label_encoder, os.path.join(output_model_path, label_encoder_filename)
        )
        print(f"Model saved to {os.path.join(output_model_path, filename)}")
        print(
            f"Label encoder saved to {os.path.join(output_model_path, label_encoder_filename)}"
        )


# Example usage:
# knn_classifier = KNNClassifier(dataset_file='data.csv', label_column='target', feature_column=['feature1', 'feature2'])
# knn_classifier.load_data()
# knn_classifier.split_data()
# knn_classifier.train_model(autoParams=True)
# knn_classifier.evaluate_model()
# knn_classifier.save_model()
