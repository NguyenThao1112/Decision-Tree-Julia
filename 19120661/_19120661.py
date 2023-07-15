import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import model_selection as ms

# Doc file wine.csv vao dataframe
dataframe = pd.read_csv('wine.csv', sep = ';')

# Doc cac cot du lieu cua data_frame thanh cac dong cua ma tran xs, cot cuoi cung doc vao ys
def read_XY(data_frame):
    xs = []
    xs.append(data_frame['fixed acidity'].values)
    xs.append(data_frame['volatile acidity'].values)
    xs.append(data_frame['citric acid'].values)
    xs.append(data_frame['residual sugar'].values)
    xs.append(data_frame['chlorides'].values)
    xs.append(data_frame['free sulfur dioxide'].values)
    xs.append(data_frame['total sulfur dioxide'].values)
    xs.append(data_frame['density'].values)
    xs.append(data_frame['pH'].values)
    xs.append(data_frame['sulphates'].values)
    xs.append(data_frame['alcohol'].values)
    
    ys = data_frame['quality'].values

    return xs, ys

# Ma tran A cac cot gia tri cua xs them cot toan gia tri 1 o dau / b la vecto cac gia tri cua ys
def getAb(xs, ys):
    if (xs.shape != (1,)):
        A = []
        A.append(np.ones(len(xs[0])))
        for i in range(len(xs)):
            A.append(xs[i])
        A = np.transpose(A)
    else:
        col1 = np.ones(len(xs))
        colx = np.array(xs)
        A = np.array([col1, colx]).T

    b = np.array(ys).reshape(len(ys), 1)

    return A, b

#########################################################################################

# Cau 1
print('--------------------------------Cau 1--------------------------------')

X, Y = read_XY(dataframe)       # 11 dong dac trung
X = np.transpose(X)     # 11 cot dac trung
A, b = getAb(np.transpose(X), Y)

# Tinh cac tham so va chuan vector phan du
x_hat = np.linalg.inv(np.matmul(np.transpose(A),A)) @ np.matmul(np.transpose(A),b)
norm = np.linalg.norm(A@x_hat - b)
print('- Mo hinh tuyen tinh: y =', x_hat[0], '+', x_hat[1], 'x')
print('- Chuan cua vector phan du:', norm)

#########################################################################################

# Cau 2
print('\n--------------------------------Cau 2--------------------------------')

model = LinearRegression()
label = dataframe.columns.values
cross_validation = []

for name in label:
    if name == 'quality':
        continue

    # Tach bo du lieu thanh tap train va tap test
    X_train, X_test, y_train, y_test = ms.train_test_split(dataframe[name], dataframe.quality, random_state = 3)
    X_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)

    # K-fold Cross Validation
    cv = ms.cross_val_score(model, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)
    cv_error = -1*cv.mean()
    cross_validation.append([name, cv_error])

# Bang sai so cua cac dac trung
print(pd.DataFrame(cross_validation, columns = ['Dac trung', 'Sai so']))

# Tim ra dac trung co sai so nho nhat
min = cross_validation[0][1]
for i in range(len(cross_validation)):
    if (cross_validation[i][1] <= min):
        feature = cross_validation[i][0]
        error = cross_validation[i][1]

# Xay dung mo hinh tuyen tinh tu dac trung vua tim duoc
X_1 = dataframe[feature].values
X_2 = dataframe[feature].values
X_1 = X_1.reshape(1, -1)
A_1, b_1 = getAb(X_1, Y)

# Tinh cac tham so va chuan vector phan du
x_hat_1 = np.linalg.inv(np.matmul(np.transpose(A_1),A_1)) @ np.matmul(np.transpose(A_1),b_1)
norm_1 = np.linalg.norm(A_1@x_hat_1 - b_1)
print('\nDac trung cho sai so trung binh nho nhat la', feature, 'voi sai so la', error)
print('- Mo hinh tuyen tinh: y =', x_hat_1[0], '+', x_hat_1[1], 'x')
print('- Chuan cua vector phan du:', norm_1)

#########################################################################################

# Cau 3
print('\n--------------------------------Cau 3--------------------------------')

Y_2 = np.log(np.array(Y))
A_2, b_2 = getAb(np.transpose(X), Y_2)

# Tinh cac tham so va chuan vector phan du
x_hat_2 = np.linalg.inv(np.matmul(np.transpose(A_2),A_2)) @ np.matmul(np.transpose(A_2),b_2)
norm = np.linalg.norm(A_2@x_hat_2 - b_2)
print('- Mo hinh tuyen tinh: lny =', x_hat_2[0], '+', x_hat_2[1], 'x')
print('- Chuan cua vector phan du:', norm)