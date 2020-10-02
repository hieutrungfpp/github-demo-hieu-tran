
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 

datatrain = pd.read_csv("Train.csv")
datatest = pd.read_csv("Test.csv")

datatrain.columns

Y_train = np.array(datatrain.iloc[:, 13].values)
Y_test = np.array(datatest.iloc[:, 13].values)

for i in range(len(datatrain.columns)-1):
   
    X_train = np.array(datatrain.iloc[:, i].values).reshape(-1, 1)
    X_test = np.array(datatest.iloc[:, i].values).reshape(-1, 1)
    i_new = datatrain.columns[i]
    
    plt.figure(figsize = (15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, Y_train, color = "red")
    plt.title(i_new + " vs MEDV")
    plt.xlabel(i_new)
    plt.ylabel("MEDV")
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    Y_train_pred = regressor.predict(X_train)
    plt.subplot(1, 3, 2)
    plt.scatter(X_train, Y_train, color = "red")
    plt.plot( X_train, Y_train_pred, color ="blue")
    plt.title(i_new + " vs MEDV (Training set)")
    plt.xlabel(i_new)
    plt.ylabel("MEDV")
    
    Y_test_pred = regressor.predict(X_test)
    plt.subplot(1, 3, 3)
    plt.scatter(X_test, Y_test, color ="red")
    plt.plot(X_test, Y_test_pred, color ="blue")
    plt.scatter(X_test, Y_test_pred, color ="black")
    plt.title(i_new + " vs MEDV (Testing set)")
    plt.xlabel(i_new)
    plt.ylabel("MEDV")
    r = sklearn.metrics.r2_score(Y_test, Y_test_pred)
    print("R square (" + i_new + ") = ",r)
    name = i_new
    plt.savefig(name)
    plt.show()
   
    
    

    
        
