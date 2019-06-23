from sklearn.linear_model import LogisticRegression
import numpy as np

def my_LogisticRegression(train_X, train_Y, test_X, test_Y):
    LogReg = LogisticRegression(solver = 'lbfgs')
    LogReg.fit(train_X, train_Y)
    predict_Y = LogReg.predict(test_X)
    # MAP
    map_list = np.zeros(len(predict_Y))
    map_list[predict_Y == test_Y] = 1
    MAP = sum(map_list) / len(map_list)
    print("The precision of test is:%f"%(MAP))
if __name__ == "__main__":
    x = [1,2,5,4,6,8,7,9,2]
    x = [[item] for item in x]
    y = [0,0,1,1,0,0,1,0,1]
    my_LogisticRegression(x,y,x,y)