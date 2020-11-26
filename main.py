import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing  
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
from sklearn.decomposition import PCA




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
    train_X = tf.cast(train_X, tf.float32)
    test_X = tf.cast(test_X, tf.float32)

    train_db=tf.data.Dataset.from_tensor_slices((train_X,train_Y)).batch(2000)
    test_db=tf.data.Dataset.from_tensor_slices((test_X,test_Y)).batch(2000)
    w1 = tf.Variable(tf.random.truncated_normal([33, 17], stddev=0.1, seed=1))
    b1 = tf.Variable(tf.random.truncated_normal([17], stddev=0.1, seed=1))

    lr, epoch, loss_all = 0.1,15000,0
    for epoch in range(epoch):  
        for step, (train_X, train_Y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                y = tf.matmul(train_X, w1) + b1  
                y = tf.nn.softmax(y)  
                d = tf.one_hot(train_Y, depth=17)
                loss = tf.reduce_mean(tf.square(d - y)) 
                loss_all += loss.numpy()  
            
            grads = tape.gradient(loss, [w1, b1])
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
        loss_all = 0  
        
        total_correct, total_number = 0, 0
        for test_X, test_Y in test_db:
            y = tf.matmul(test_X, w1) + b1
            y = tf.nn.softmax(y)
            pred = tf.argmax(y, axis=1)  
            pred = tf.cast(pred, dtype=test_Y.dtype)
            correct = tf.cast(tf.equal(pred, test_Y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total_number += test_X.shape[0]
        acc = total_correct / total_number
    print("Test_acc:", acc)
    mse = mean_squared_error(d,y)
    rmse = mse** 0.5
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)
        



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

def calub(train_X, test_X, train_Y, test_Y):
    pass


def main():




    train_X, test_X, train_Y, test_Y = gendata(False)

    pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y = gendata(True)
    print("done preprocessing")

    print("Starting Patricks Code for 128, 64, 32, 'relu', 0.01 ")
    patrick(train_X, test_X, train_Y, test_Y, 128, 64, 32, 'relu', 0.01)
    
    print("Starting Aimee Code")
    aimee(pcaTrain_X, pcaTest_X, pcaTrain_Y, pcaTest_Y )
    
    





if __name__ == '__main__':
    main()


