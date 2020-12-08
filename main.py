import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
import keras.backend as K
from sklearn.utils import class_weight




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

def brute_force_parameters(train_X, test_X, train_Y, test_Y,):
    myArray = []
    listOLists = [[256,128, 64, 32, 16], [256,128, 64, 32, 16], [256,128, 64, 32, 16], ['identity', 'logistic', 'tanh', 'relu'],
                  [0.1,0.01,0.001]]
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
    train_Y = train_Y - 1
    test_Y = test_Y - 1
    output_size = 17
    model = tf.keras.Sequential()
    #34input, 512hidden
    model.add(layers.Dense(512, input_shape=(34,), activation='relu'))
    model.add(layers.Dropout(0.5))
    #512input,1024 hidden
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    #32*32=1024
    model.add(tf.keras.layers.Reshape((32, 32, 1)))
    #64 kernel，64 feature map
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    #17 output
    model.add(layers.Dense(output_size, kernel_initializer='glorot_uniform'))
    model.summary()

    class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(train_Y),
                                                train_Y)
    lr=0.01
    epoch = 10000
    sgd = optimizers.SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy
                  (from_logits=True), optimizer=sgd, metrics=['accuracy'])

    # here we use the test_set as the validation_set    
    model.fit(train_X,train_Y,batch_size=64,epochs=epoch,verbose=2,
              validation_data=(test_X,test_Y), class_weight=class_weights)
    score = model.evaluate(test_X,test_Y,verbose=0)
    print(score)

    # cm1 = classification_report(test_Y, y_test_preds)
    # print(cm1)


    # cm = confusion_matrix(test_Y.argmax, y_test_preds)
    # print(unique(test_Y))
    # plt.figure()
    # plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14'], title="Tinas's Code")
    # plt.show()
    
def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def aimee(train_X, test_X, train_Y, test_Y):
    model = Sequential()
    model.add(Dense(units = 68,  input_dim =34 ,  kernel_initializer =  'normal' ,  activation = 'sigmoid'))
    model.add(Dense(units = 17, kernel_initializer =  'normal',activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer =  'normal'))
    adam=optimizers.Adam(lr=0.001,  epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[soft_acc])
    model.fit(train_X, train_Y, epochs=15000,verbose=2)
    loss,accuracy = model.evaluate(test_X,  test_Y,verbose=2)
    print("The mean square error is:", loss, "The accuracy is: ")

def caleb(train_X, test_X, train_Y, test_Y):
    train_X= np.reshape(train_X,(train_X.shape[0], 1, train_X.shape[1]))
    test_X= np.reshape(test_X,(test_X.shape[0], 1, test_X.shape[1]))
    
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"), input_shape=(1, 34,)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.1), input_shape=(1, 34)))
    model.add(layers.Dense(units = 18))

    #model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"])
    
    model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=30, batch_size = 50, verbose=0)
    results = model.evaluate(test_X, test_Y, verbose=0)
    vectors = model.predict(test_X)
    vectors= np.reshape(vectors,(vectors.shape[0], vectors.shape[2]))
    
    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, solver='adam', activation='logistic', hidden_layer_sizes=(32, 64, 17), batch_size=100)).fit(vectors, test_Y)
    pred = clf.predict(vectors)
    print("Accuracy of mlp", accuracy_score(pred, test_Y))
    print("MSE", mean_squared_error(pred, test_Y))


def main():

    train_X, test_X, train_Y, test_Y = gendata(False)
    pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y = gendata(True)
    print("done preprocessing")

    print("Starting Patricks Code\n")
    print("----------------------Patrick Code Results----------------------------")
    patrick(train_X, test_X, train_Y, test_Y, 32, 32, 32, 'tanh', 0.01)

    print("Starting Aimee Code\n")
    print("----------------------Aimee Code Results----------------------------")
    aimee(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y )

    print("\nStarting Tina Code\n")
    print("----------------------Tina Code Results----------------------------")
    tina(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y)

    print("Starting Caleb Code\n")
    print("----------------------Caleb Code Results----------------------------")
    caleb(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y )

    





if __name__ == '__main__':
    main()


