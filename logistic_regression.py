from sklearn.linear_model import LogisticRegression
import numpy as np

def LogisticRegression(train_X, train_Y, test_X, test_Y):
    LogReg = LogisticRegression(solver = 'lbfgs')
    LogReg.fit(train_X, train_Y)
    predict_Y = LogReg.predict(test_X)
    # MAP
    map_list = np.zeros(len(predict_Y))
    map_list[predict_Y == test_Y] = 1
    MAP = sum(map_list) / len(map_list)
    print("The precision:%f"%(MAP))
