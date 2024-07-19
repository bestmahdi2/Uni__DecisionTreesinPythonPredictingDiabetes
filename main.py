from time import time
from six import StringIO
from math import floor, log
from pandas import read_csv
from sklearn import metrics
from os import path as path_
from datetime import timedelta
from typing import Tuple, List
from IPython.display import Image
from pandas.core.frame import DataFrame
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class Tree:
    """
        Class for main
    """

    FILE = "diabetes.csv"
    SAVE_FILE = "diabetes.png"
    FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']

    def __init__(self) -> None:
        """
            Constructor for Main class,
        """

        self.x, self.y = self.read_file()

    @staticmethod
    def convert_size(size_bytes: int) -> str:
        """
            Method to convert byte size to KB - MB - GB,

            Parameters:
                size_bytes (int): The size in bytes

            Returns:
                The result of the conversion
        """

        if not size_bytes:
            return "0B"

        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")

        i = int(floor(log(size_bytes, 1024)))
        s = round(size_bytes / pow(1024, i), 2)

        return "%s %s" % (s, size_name[i])

    def read_file(self) -> Tuple[List, List]:
        """
            Method to read the file,

            Returns:
                The x and y lists
        """

        print("-" * 10 + " Train Reading " + "-" * 10 + "\n")

        # Read the CSV file including data and create dataframe
        dataframe = read_csv(self.FILE)

        # Change the 0 values with mean
        dataframe = self.fix_missing_values(dataframe)

        x = dataframe[self.FEATURES]  # input data
        y = dataframe.Outcome  # output data

        print("Done !\n")

        return x, y

    def create_model(self, param_grid: dict) -> None:
        """
            Method to create model from parameters,
        """

        print(f"Test Size: {int(param_grid['test_size'] * 100)}% - Random State: {param_grid['random_state']}")

        start_time = time()

        # splits dataset into train (100 - param_grid['test_size']) and test (param_grid['test_size']) dataset
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=param_grid['test_size'],
                                                            random_state=param_grid['random_state'])

        # create model
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(X_train, Y_train)

        # predict
        y_pred = classifier.predict(X_test)

        # save the image
        self.save_image(classifier)

        end_time = time()

        b = path_.getsize(f'{self.i}.{self.SAVE_FILE}')

        print(f"Time: {str(timedelta(seconds=end_time - start_time))} - Size: {self.convert_size(b)}")

        # accuracy
        print(f"The model is {metrics.accuracy_score(Y_test, y_pred) * 100}% accurate !")

    def fix_missing_values(self, dataframe: DataFrame):
        """
            Method to fix the 0 values

            Parameters:
                dataframe (DataFrame): The dataframe to fix
        """

        # print null values
        # print(dataframe.isnull().sum())

        for i in self.FEATURES:
            # Correcting missing values
            dataframe[i] = dataframe[i].replace(0, dataframe[i].mean())

        return dataframe

    def save_image(self, classifier: DecisionTreeClassifier) -> None:
        """
            Method to save the image,

            Parameters:
                classifier (DecisionTreeClassifier): The classifier
        """

        # Text I/O implementation using an in-memory buffer
        dot_data = StringIO()

        export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                        feature_names=self.FEATURES, class_names=['0', '1'])

        # create the graph
        graph = graph_from_dot_data(dot_data.getvalue())

        graph.write_png(f'{self.i}.{self.SAVE_FILE}')

        Image(graph.create_png())

    def run_tests(self) -> None:
        """
            Method to run tests,
        """

        keeper = [
            {'test_size': 0.20, 'random_state': 1},
            {'test_size': 0.20, 'random_state': 10},
            {'test_size': 0.50, 'random_state': 1},
            {'test_size': 0.50, 'random_state': 10},
        ]

        print("-" * 10 + " Create Models " + "-" * 10 + "\n")

        for i in keeper:
            self.i = keeper.index(i)
            self.create_model(i)
            print(f'\n{"-" * 30}\n')


if __name__ == "__main__":
    T = Tree()
    T.run_tests()
