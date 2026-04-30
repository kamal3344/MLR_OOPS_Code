'''
In this file we are going to load the data and develop MLR code in oops concept
'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
class MLR:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            self.df['State'] = self.df['State'].map({'New York':0,'California':1,'Florida':2}).astype(int)
            self.X = self.df.iloc[: , :-1] # independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            print(f"Training dataset size : {len(self.X_train)} : {len(self.y_train)}")
            print(f"Testing dataset size : {len(self.X_test)} : {len(self.y_test)}")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")


    def training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            self.y_train_predictions = self.reg.predict(self.X_train)
            print(f"Train Accuracy : {r2_score(self.y_train,self.y_train_predictions)}")
            print(f"Train Loss : {root_mean_squared_error(self.y_train,self.y_train_predictions)}")

        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")


    def testing(self):
        try:
            self.y_test_predictions = self.reg.predict(self.X_test)
            print(f"Test Accuracy : {r2_score(self.y_test, self.y_test_predictions)}")
            print(f"Test Loss : {root_mean_squared_error(self.y_test, self.y_test_predictions)}")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

    def check_own_data(self):
        try:
            rd = 1200
            admin = 1800
            ms = 1900
            s = 1
            print(f"Test Point Predictions : {self.reg.predict([[rd,admin,ms,s]])[0]}")

        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

    def saving_model(self):
        try:
            with open("Model.pkl","wb") as f:
                pickle.dump(self.reg , f)

            print(f"----------Load and check-------------")
            with open("Model.pkl","rb") as t:
                model = pickle.load(t)
                rd = 1200
                admin = 1800
                ms = 1900
                s = 1
                print(f"Loaded Model Predictions : {model.predict([[rd, admin, ms, s]])[0]}")


        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

if __name__ == "__main__":
    try:
        path = "50_Startups.csv"
        obj = MLR(path)
        obj.training()
        obj.testing()
        obj.check_own_data()
        obj.saving_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")








