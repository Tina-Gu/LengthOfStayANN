import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor




pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def gendata(PCA):

    # Create dataframe from csv
    data = pd.read_csv('https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv')
    #Save the length of stay in a different variable
    labels = data['lengthofstay']
    # Drop columns that we dont need like specific dates, or the id of the patient
    data = data.drop(["eid", "vdate", "discharged", "lengthofstay"], axis=1)
    # Add dummy encoding for the object and type variables
    # For example, turn gender column into 2 columns, where a male will be 1 in the first column
    # and a 0 in the second column, and a female will be the inverse
    data = pd.get_dummies(data, columns=['rcount'])
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['facid'])

    hematocrit = data[['hematocrit']].values
    data['hematocrit'] = preprocessing.StandardScaler().fit_transform(hematocrit)

    bloodureanitro = data[['neutrophils']].values
    data['neutrophils'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    sodium = data[['sodium']].values
    data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)

    glucose = data[['glucose']].values
    data['glucose'] = preprocessing.StandardScaler().fit_transform(glucose)

    bloodureanitro = data[['bloodureanitro']].values
    data['bloodureanitro'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    creatinine = data[['creatinine']].values
    data['creatinine'] = preprocessing.StandardScaler().fit_transform(creatinine)

    bmi = data[['bmi']].values
    data['bmi'] = preprocessing.StandardScaler().fit_transform(bmi)

    pulse = data[['pulse']].values
    data['pulse'] = preprocessing.StandardScaler().fit_transform(pulse)

    respiration = data[['respiration']].values
    data['respiration'] = preprocessing.StandardScaler().fit_transform(respiration)



    # ADD PCA CONSIDERATIONS
    if PCA:
        pass

    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    test_X = data.tail(n=20000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()
    test_Y = labels.tail(n=20000).to_numpy()


    return train_X, test_X, train_Y, test_Y


def patrick(train_X, test_X, train_Y, test_Y, f,s,t,a,lr):

    model_sklearn = MLPClassifier(max_iter=100000, hidden_layer_sizes=(f, s, t), activation=a,
                                  learning_rate_init=lr, )
    model_sklearn.fit(train_X, train_Y)
    # print("Training")
    # pred_X_train_sklearn = model_sklearn.predict(train_X)
    # cm = classification_report(train_Y, pred_X_train_sklearn)
    # print(cm)
    print("Test")
    pred_y_test_sklearn = model_sklearn.predict(test_X)
    cm1 = classification_report(test_Y, pred_y_test_sklearn)
    print(cm1)


def tina(train_X, test_X, train_Y, test_Y):

    reg_model = RandomForestRegressor(random_state=0)
    reg_model.fit(train_X, train_Y)
    y_test_preds = reg_model.predict(test_X)
    results = r2_score(test_Y, y_test_preds)
    print('R2 score: ', results)
    mse = mean_squared_error(test_Y, y_test_preds)
    rmse = mse ** 0.5
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)


def aimee(train_X, test_X, train_Y, test_Y):
    pass

def calub(train_X, test_X, train_Y, test_Y):
    pass


def main():




    train_X, test_X, train_Y, test_Y = gendata(False)

    pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y = gendata(True)
    print("done preprocessing")

    print("Starting Patricks Code for 128, 64, 32, 'relu', 0.01 ")
    patrick(train_X, test_X, train_Y, test_Y, 128, 64, 32, 'relu', 0.01)





if __name__ == '__main__':
    main()


