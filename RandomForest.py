# importing labraries
from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split ,RandomizedSearchCV , learning_curve 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# handling data
data = pd.read_csv(r"C:\Users\lenovo\Downloads\student_performancezip\Student_Performance.csv")
renamed_data = {
    'Extracurricular Activities': 'Activities' , 
    'Sample Question Papers Practiced' : 'SQPP',
    'Performance Index': 'performace'
}
data = data.rename(columns=renamed_data)
col = data.shape[1]
x = data.iloc[: , 0: col-1]
y = data.iloc[ : , col-1 : col ]

# converting Activities from strings to numirical values 
le = LabelEncoder()
fill = x['Activities'] 
fill = le.fit_transform(fill)
x['Activities'] = fill

# testing if data has outliers using IQR method
Q1 = x.quantile(0.25)
Q3 = x.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered = (x < lower_bound) | (x > upper_bound)
outliers = filtered.sum()
print(outliers)
# i found that we got 0 outliers , let's move to next (:

# scaling  
scaler = StandardScaler()
scaled = x['Previous Scores'].values.reshape(-1 , 1)
scaled = scaler.fit_transform(scaled)
x['Previous Scores'] =  scaled

# splitting data
x_train , x_test , y_train , y_test = train_test_split(x  , y , test_size=0.3 , random_state=23)

# testing the best randomforest params
model = RandomForestRegressor()
params = {
    'n_estimators' : [200 , 300 , 400 , 500 , 800] , 
    'max_depth' : [3 ,6 ,9 , 10 ,12 , 'auto'] , 
    'min_samples_split' : [10 , 15 , 20 , 30 , 50] , 
    'min_samples_leaf' : [2 , 4 , 6 ,10] , 
    'max_features' : [0.2 , 0.3 , 0.6 , 'auto']
}
# results
values = RandomizedSearchCV(model , params , cv=8 , random_state=23)
values.fit(x_train , y_train)
bst_est = values.best_estimator_
bst_params = values.best_params_
bst_score = values.best_score_

# best parameters 
print("bets estimators : \n" , bst_est) # RandomForestRegressor(max_depth=9, max_features=0.6, min_samples_leaf=6, min_samples_split=20, n_estimators=800)
print("bets params : \n" , bst_params) # {'n_estimators': 800, 'min_samples_split': 20, 'min_samples_leaf': 6, 'max_features': 0.6, 'max_depth': 9}
print(f"bets score {bst_score*100:.2f}%") # bets score 98.63%

# training our model 
model = RandomForestRegressor(max_depth=9, max_features=0.6, min_samples_leaf=6, min_samples_split=20, n_estimators=800 , random_state=23)
model.fit(x_train , y_train)
predicitons = model.predict(x_test)

# results
trainscore = model.score(x_train , y_train)
testscore = model.score(x_test , y_test)
mse = mean_squared_error(y_test, predicitons)
mae = mean_absolute_error(y_test, predicitons)

# printing results 
print(f"train score : {trainscore*100:.2f}%")
print(f"test score : {testscore*100:.2f}%")
print(f"mse : {mse}")
print(f"mae : {mae}")

# evaluation 
train_sizes , train_score , test_score = learning_curve(model ,x_train , y_train  , cv=5 , n_jobs=-1 ) # increase cv (cross validation) for better plot visualization
train_mean = np.mean(train_score , axis=1)
test_mean = np.mean(test_score , axis=1)

# additional tip
results = pd.DataFrame({
    'Actual': y_test.values.flatten(),
    'Predicted': predicitons.flatten()
})
print(results.head(10))

# plotting results 
plt.style.use("fivethirtyeight")
plt.plot(train_sizes , train_mean , label = 'train score' , c='orange')
plt.plot(train_sizes , test_mean , label = 'test score' , c='cyan')
plt.xlabel("train sizes")
plt.ylabel("variance")
plt.title("learning curves")
plt.legend()
plt.show()

# the difference between real data and prediction
residuals = y_test.values.flatten() - predicitons.flatten()
plt.scatter(predicitons, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()
