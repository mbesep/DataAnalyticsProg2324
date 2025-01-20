MY_UNIQUE_ID = "SalPlax"

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import load_model


from pytorch_tabular import TabularModel
import torch

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pickle



# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID



# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(df, clfName):
    clfName = clfName.lower().strip()
    if (clfName == "lr"):
        X = df.drop('Year',axis=1)
        y = df['Year']
        scaler = pickle.load(open("modelli/LR/normal_lr.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew
    elif (clfName == "rf"): 
        return df
    elif (clfName == "knr"):
        X = df.drop('Year',axis=1)
        y = df['Year']
        scaler = pickle.load(open("modelli/KNN/scaler_knn.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew
    elif (clfName == "svr"):
        X = df.drop('Year',axis=1)
        y = df['Year']
        scaler = pickle.load(open("modelli/SVR/normal_svr.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew
    elif (clfName == "ff"):
        X = df.drop('Year',axis=1)
        y = df['Year']
        scaler = pickle.load(open("modelli/NN/scaler_nn.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew
    elif (clfName == "tb"):
        return df
    elif (clfName == "tf"):
        return df
    else:
        return None
        





# Input: Regressor name ("lr": Linear Regression, "SVR": Support Vector Regressor)
# Output: Regressor object
def load(clfName):
    clfName = clfName.lower().strip()
    if (clfName == "lr"):
        clf = pickle.load(open("modelli/LR/model_lr.save", 'rb'))
        return clf
    elif (clfName == "svr"):
        clf = pickle.load(open("modelli/SVR/model_svr.save", 'rb'))
        return clf
    elif (clfName == "rf"):
        clf = pickle.load(open("modelli/RF/model_rf.save", 'rb'))
        return clf  
    elif (clfName == "knr"):
        clf = pickle.load(open("modelli/KNN/model_knn.save", 'rb'))
        return clf
    elif (clfName == "ff"):
        clf = load_model("modelli/NN/model")  
        return clf
    elif (clfName == "tb"):
        model = torch.load("modelli/TABNET/model_tabnet.save", map_location=torch.device('cpu'))
        return model
    elif (clfName == "tf"):
        model = torch.load("modelli/TABTRANSFORMER/model_tab_transformer.save", map_location=torch.device('cpu'))
        return model
    else:
        return None



# Input: PreProcessed dataset, Regressor Name, Regressor Object 
# Output: Performance dictionary
def predict(df, clfName, clf):
    clfName = clfName.lower().strip()
    if(clfName == "ff"):
        X = df.drop('Year',axis=1)
        y = df['Year']
        ypred = clf.predict(X)
        minimo = pickle.load(open("modelli/NN/minimo_nn.save", 'rb'))
        ypred = ypred + minimo
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
        perf = {"mse": mse, "mae": mae, "mape": mape, "r2score": r2}
        return perf
    elif (clfName == "tb" or clfName == "tf" ):
        y = df['Year']
        ypred = clf.predict(df)
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
        perf = {"mse": mse, "mae": mae, "mape": mape, "r2score": r2}
        return perf
    else:
        X = df.drop('Year',axis=1)
        y = df['Year']
        ypred = clf.predict(X)
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
        perf = {"mse": mse, "mae": mae, "mape": mape, "r2score": r2}
        return perf