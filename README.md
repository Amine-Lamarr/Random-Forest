<h1>Student Performance Prediction using Random Forest</h1>
<h2>Project Overview</h2>

The goal of this project is to predict the performance of students based on their background and activities. By analyzing data, we aim to understand the factors that affect student performance and use this insight to predict future outcomes.

<h2>Data</h2>

This project uses a dataset containing information about students' extracurricular activities, practice of sample question papers, previous scores, and their final performance index. The dataset is processed and cleaned before being fed into the model.

<h2>Steps Taken</h2>

<h3>1. Data Preprocessing:</h3>
   - Renaming columns for better readability.
   - Encoding categorical data (e.g., 'Activities' column).
   - Scaling the 'Previous Scores' feature.
   - Identifying and handling outliers using the IQR method.

<h3>2. Model Training: </h3>
   - Splitting the data into training and testing sets.
   - Hyperparameter tuning using Randomized Search.
   - Training the Random Forest Regressor on the dataset.

<h3>3. Model Evaluation:</h3>
   - Evaluating the model's performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
   - Plotting learning curves and residual plots to assess the model's learning over time.

<h2>Results</h2>

The trained model achieves a high level of accuracy in predicting student performance. The results are presented with:

- **Training score**
- **Testing score**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

The model also visualizes the learning process and residuals to understand how well it fits the data.

## <h2>Technologies Used</h2>

- Python
- Scikit-learn (for Random Forest and model evaluation)
- Pandas (for data manipulation)
- Matplotlib (for visualizations)







