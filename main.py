import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools




pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def gendata(doPCA):

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
    if doPCA:
        pca = PCA()   
        data=pca.fit_transform(data)
        train_X = np.array(data[:80000])
        train_Y = labels.head(n=80000).to_numpy()
        test_X = np.array(data[80000:])
        test_Y = labels.tail(n=20000).to_numpy()
        return train_X, test_X, train_Y, test_Y
        

    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    test_X = data.tail(n=20000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()
    test_Y = labels.tail(n=20000).to_numpy()


    return train_X, test_X, train_Y, test_Y

def plot_confusion_matrix(cm, classes, title,normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def brute_force_parameters():

    listOLists = [[128, 64, 32, 16], [128, 64, 32, 16], [128, 64, 32, 16], ['identity', 'logistic', 'tanh', 'relu'],
                  [0.001]]
    count = 0
    for list in itertools.product(*listOLists):
        temp = patrick(train_X, test_X, train_Y, test_Y, list[0], list[1], list[2], list[3], list[4])
        print(count, temp)
        myArray.append(temp)
        count = count + 1

    new = sorted(myArray, key=lambda x: x[-1])
    for i in new:
        print(i)


def patrick(train_X, test_X, train_Y, test_Y, f,s,t,a,lr):

    model_sklearn = MLPClassifier(max_iter=100000, hidden_layer_sizes=(f, s, t), activation=a,
                                  learning_rate_init=lr, )
    model_sklearn.fit(train_X, train_Y)
    pred_y_test_sklearn = model_sklearn.predict(test_X)


    results = r2_score(test_Y, pred_y_test_sklearn)
    print('R2 score: ', results)
    mse = mean_squared_error(test_Y, pred_y_test_sklearn)
    rmse = mse ** 0.5
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)
    cm1 = classification_report(test_Y, pred_y_test_sklearn)
    print(cm1)


    # cm = confusion_matrix(test_Y, pred_y_test_sklearn)
    # plt.figure()
    # plot_confusion_matrix(cm, classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'], title="Patrick's Code")
    # plt.show()


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

    # cm1 = classification_report(test_Y, y_test_preds)
    # print(cm1)


    # cm = confusion_matrix(test_Y.argmax, y_test_preds)
    # print(unique(test_Y))
    # plt.figure()
    # plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14'], title="Tinas's Code")
    # plt.show()

def aimee(train_X, test_X, train_Y, test_Y):
    model = Sequential()
    model.add(Dense(units = 68,  input_dim =34 ,  kernel_initializer =  'normal' ,  activation = 'sigmoid'))
    model.add(Dense(units = 17, kernel_initializer =  'normal',activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer =  'normal'))
    adam=optimizers.Adam(lr=0.001,  epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error') 
    model.fit(train_X, train_Y, epochs=200,verbose=2)
    loss = model.evaluate(test_X,  test_Y,verbose=2)
    print("The mean square error is:", loss)

def caleb(train_X, test_X, train_Y, test_Y):
    pass


def main():

    train_X, test_X, train_Y, test_Y = gendata(False)
    pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y = gendata(True)
    print("done preprocessing")

    print("Starting Patricks Code\n")
    print("----------------------Patrick Code Results----------------------------")
    patrick(train_X, test_X, train_Y, test_Y, 32, 16, 64, 'logistic', 0.01)

    print("Starting Aimee Code\n")
    print("----------------------Aimee Code Results----------------------------")
    aimee(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y )

    print("\nStarting Tina Code\n")
    print("----------------------Tina Code Results----------------------------")
    tina(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y)
    
    





if __name__ == '__main__':
    main()


