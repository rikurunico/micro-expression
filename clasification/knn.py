import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


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

    def load_data(self):
        data = pd.read_csv(self.dataset_file)
        print(self.dataset_file)

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

        self.y = LabelEncoder().fit_transform(
            data[self.label_column].values
        )  # Encode label

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
                "n_neighbors": [3, 5, 7, 9, 11],
                "metric": ["minkowski", "euclidean", "manhattan"],
                "p": [1, 2],
            }

            best_params = {"combination": {}, "accuracy": 0}

            for n_neighbors in param_grid["n_neighbors"]:
                for metric in param_grid["metric"]:
                    for p in param_grid["p"]:
                        model = KNeighborsClassifier(
                            n_neighbors=n_neighbors, metric=metric, p=p
                        )
                        model.fit(self.X_train, self.y_train)
                        predictions = model.predict(self.X_test)
                        accuracy = accuracy_score(self.y_test, predictions)

                        if accuracy > best_params["accuracy"] and accuracy != 1:
                            best_params["combination"] = {
                                "n_neighbors": n_neighbors,
                                "metric": metric,
                                "p": p,
                            }
                            best_params["accuracy"] = accuracy

            print("\nBest combination found:")
            print(best_params["combination"])
            print(f"Best accuracy: {best_params['accuracy']}")

            self.model = KNeighborsClassifier(**best_params["combination"])
            self.model.fit(self.X_train, self.y_train)
        else:
            print("n_neighbors:", n_neighbors, "Metric:", metric, "P:", p)
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors, metric=metric, p=p
            )
            self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)

        print("Accuracy:", accuracy)
        print("\nConfusion Matrix:")
        print(cm)

    def save_model(self, filename="knn_model.joblib"):
        output_model_path = "models"
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        joblib.dump(self.model, os.path.join(output_model_path, filename))


# Example usage:
# knn_classifier = KNNClassifier(dataset_file='data.csv', label_column='target', feature_column=['feature1', 'feature2'])
# knn_classifier.load_data()
# knn_classifier.split_data()
# knn_classifier.train_model(autoParams=True)
# knn_classifier.evaluate_model()
# knn_classifier.save_model()
