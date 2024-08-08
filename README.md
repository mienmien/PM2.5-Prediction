# PM2.5-Prediction
2023 NYCU Data Mining class HW1  
Implement linear regression using only numpy to predict the value of PM2.5

# Dataset Description  
**train.csv** - Climate data for the first 20 days of each month.  

**test_X.csv** - Sample the continuous 10 hours data from the remaining 10 days of each month,take the data of the previous 9 hours as a feature, and the last hour as the answer, get total 244 unique test datas.  

**sample_submission.csv** - A 245*2 .csv file, first row is for the column name and the last 244 rows for your result, column name must be index and answer.

# Result  
Based on RMSE  

**Public test data** - 3.22300  
ranking - 22/72  

**Private test data** - 4.01510  
ranking - 45/72  

---strong baseline--- 3.47311  
---simple baseline--- 5.04415  
