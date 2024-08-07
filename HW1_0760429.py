import pandas as pd
import numpy as np
import csv
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 移除特定 features
REMOVE_ITEM = [
    'RAINFALL            ',
    'THC                 ',
    # 'NOx                 '
]
# 取後 n 個小時的資料
LAST_HOURS = 7
FEATURE_NUM = 18 - len(REMOVE_ITEM)
data = pd.read_csv('train.csv')
for item in REMOVE_ITEM:
    data = data[data['ItemName                 '] != item]
data = data.iloc[:, 3:]
raw_data = data.to_numpy()

# 補齊Missing Data
def checkAllRowIsNan(data):
    count = 0
    for i in range(len(data)):
        if (np.isnan(data[i])):
            count = count + 1
    return count == len(data)

def getPreData(data, i, j):
    new_i = i
    new_j = j

    while (True):
        if (new_j < len(data[i]) - 1):
            new_j -= 1
        else:
            new_j = len(data[i]) - 1
            new_i -= FEATURE_NUM
        if not np.isnan(data[new_i][new_j]):
            return data[new_i][new_j]
        if (new_i == len(data) and new_j == len(data[i]) - 1):
            break
    return np.nanmean(data[i])

def getNextData(data, i, j):
    new_i = i
    new_j = j

    while (True):
        if (new_j < len(data[i]) - 1):
            new_j += 1
        else:
            new_j = 0
            new_i += FEATURE_NUM
        if not np.isnan(data[new_i][new_j]):
            return data[new_i][new_j]
        if (new_i == len(data) and new_j == len(data[i]) - 1):
            break
    return np.nanmean(data[i])

def fullMissingData(data, i, j):
    if (i == 0 and j == 0):
        return getNextData(data, i, j)
    if (i == len(data) and j == len(data[i]) - 1):
        return getPreData(data, i, j)

    return getNextData(data, i, j)


# 全部資料檢查一次，補齊缺失值
for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        try:
            raw_data[i][j] = float(raw_data[i][j])
        except ValueError:
            raw_data[i][j] = np.nan

for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        if (np.isnan(raw_data[i][j])):
            if (checkAllRowIsNan(raw_data[i])):
                # print(raw_data[i], raw_data[i - FEATURE_NUM])
                raw_data[i] = raw_data[i - FEATURE_NUM]
                # raw_data[i][j] = np.nanmean(raw_data[i - FEATURE_NUM])
            else:
                raw_data[i][j] = fullMissingData(raw_data, i, j)
                # raw_data[i][j] = np.nanmean(raw_data[i])

# (每個月20day24h=480h橫向小時 ), FEATURE_NUM (features) * 480 (hours) 的資料。
# 這樣,每個月的資料在橫向上是連續的時間
month_data = {}
for month in range(12):
    sample = np.empty([FEATURE_NUM, 480])
    for day in range(20):
        startDay = FEATURE_NUM * (20 * month + day)
        endDay = FEATURE_NUM * (20 * month + day + 1)
        data = raw_data[startDay : endDay, :]
        sample[:, day * 24 : (day + 1) * 24] = data
        # 取raw_data的行,對sample的列進行拼接
    month_data[month] = sample

with open('month.csv', mode='w', newline='') as month_file:
    csv_writer = csv.writer(month_file)
    # 逐一寫入每一列
    for row in month_data[0]:
        csv_writer.writerow(row)

# 前9h數據作为train_x, 第10h的pm2.5作为train_y, 製作xy樣本
# 每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * FEATURE_NUM 的 features (一小時 FEATURE_NUM 個 features * 9 小時)。
# 對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5)
x = np.empty([12 * 471, FEATURE_NUM * LAST_HOURS], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour + (9 - LAST_HOURS) : day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

# Z-score standardization
mean_x = np.mean(x, axis = 0) # FEATURE_NUM * 9 
std_x = np.std(x, axis = 0) # FEATURE_NUM * 9 
for i in range(len(x)): # 12 * 471
    for j in range(len(x[0])): # FEATURE_NUM * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# Max-min standardization
# max_x = np.max(x, axis = 0)
# min_x = np.min(x, axis = 0)
# for i in range(len(x)):
#     for j in range(len(x[0])):
#         if max_x[j] != min_x[j]:
#             x[i][j] = (x[i][j] - min_x[j]) / (max_x[j] - min_x[j])

# 訓練
dim = FEATURE_NUM * LAST_HOURS + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)#rmse
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)

testData = pd.read_csv('test_X.csv', header = None, encoding = 'big5')
for item in REMOVE_ITEM:
    testData = testData[testData[1] != item]
test_data_np = testData.iloc[:, 2:].to_numpy()

# test data 將浮點數轉換為整數
for i in range(len(test_data_np)):
    for j in range(len(test_data_np[i])):
        try:
            test_data_np[i][j] = float(test_data_np[i][j])
        except ValueError:
            test_data_np[i][j] = np.nan

for i in range(len(test_data_np)):
    for j in range(len(test_data_np[i])):
        if (np.isnan(test_data_np[i][j])):
            if (checkAllRowIsNan(test_data_np[i])):
                test_data_np[i] = test_data_np[i - FEATURE_NUM]
                # test_data_np[i][j] = np.nanmean(test_data_np[i - FEATURE_NUM])
            else:
                test_data_np[i][j] = fullMissingData(test_data_np, i, j)
                # test_data_np[i][j] = np.nanmean(test_data_np[i])

# 載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 244 個維度為 FEATURE_NUM * 9 + 1 的資料。
test_x = np.empty([244, FEATURE_NUM * LAST_HOURS], dtype = float)
for i in range(244):
    test_x[i, :] = test_data_np[FEATURE_NUM * i : FEATURE_NUM * (i + 1), (9 - LAST_HOURS): 9].reshape(1, -1)

# 標準化 z-score standardization
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

# # 正規化 min-max normalization
# for i in range(len(test_x)):
#     for j in range(len(test_x[0])):
#         if max_x[j] != min_x[j]:
#             test_x[i][j] = (test_x[i][j] - min_x[j]) / (max_x[j] - min_x[j])

test_x = np.concatenate((np.ones([244, 1]), test_x), axis = 1).astype(float)

# 預測
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

# 輸出
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['index', 'answer']
    # print(header)
    csv_writer.writerow(header)
    for i in range(244):
        ans = ans_y[i][0]
        if ans_y[i][0] < 0:
            ans = 0
        row = ['index_' + str(i), ans]
        csv_writer.writerow(row)
        # print(row)
print('done')
