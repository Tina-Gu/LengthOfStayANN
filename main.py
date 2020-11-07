import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def gendata():

    # Create dataframe from csv
    data = pd.read_csv('https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv')
    #Save the length of stay in a different variable
    labels = data['lengthofstay']
    # Drop columns that we dont need like specific dates, or the id of the patient
    data = data.drop(["eid", "vdate", "discharged", "lengthofstay"], axis=1)
    # for col in data.columns:
    #     print(col)
    data = pd.get_dummies(data, columns=['rcount'])
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['facid'])


    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    test_X = data.tail(n=20000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()
    test_Y = labels.tail(n=20000).to_numpy()
    return train_X, test_X, train_Y, test_Y


def main():
    train_X, test_X, train_Y, test_Y = gendata()

    model_sklearn = MLPClassifier(max_iter=10000, hidden_layer_sizes=(12,32,), activation='relu', learning_rate_init=0.001, )
    model_sklearn.fit(train_X, train_Y)

    pred_y_test_sklearn = model_sklearn.predict(test_X)

    cm1 = classification_report(test_Y, pred_y_test_sklearn)
    print(cm1)











if __name__ == '__main__':
    main()


