# SpaceX Launch Data Analysis and Prediction

## Project Overview

This project focuses on analyzing historical SpaceX launch data and building predictive models to forecast rocket landing outcomes and estimate launch costs. Using Python and various machine learning techniques, we process, clean, and visualize the data to gain insights into the factors influencing successful rocket landings and overall mission costs.

The analysis relies on a dataset containing information about rocket launches, including payload mass, launch site, booster versions, and reuse status. The primary goal is to predict whether SpaceX will attempt to land a rocket based on this data.

## Technologies Used

The following libraries and tools have been used in this project:

- **Pandas**: For data manipulation and cleaning, especially for reading CSV files and handling dataframes.
- **Matplotlib** and **Seaborn**: For data visualization, including scatter plots and heatmaps, to visually understand relationships between variables.
- **Scikit-learn**: For machine learning model development, including:
  - Random Forest Classifier for predicting landing outcomes.
  - Linear Regression for estimating launch costs.
  - Train-test split for splitting data into training and testing sets.
  - Evaluation metrics like accuracy, classification report, and mean squared error.
  
## Steps in the Analysis

1. **Data Loading**: Load the SpaceX launch data from a CSV file.
2. **Data Exploration**: Explore the dataset to understand its structure and check for missing or incomplete values.
3. **Data Preprocessing**:
    - Handle missing values by removing rows with null data or replacing missing values where appropriate.
    - Encode categorical variables like launch site and booster version to make them usable by machine learning models.
4. **Data Visualization**: Use scatter plots to visualize relationships between features like payload mass and landing outcome, and generate correlation heatmaps to identify patterns in the data.
5. **Model Building**:
    - Train a **Random Forest Classifier** to predict whether a rocket will land successfully based on factors such as payload mass and booster version.
    - Train a **Linear Regression** model to predict launch costs, estimating relationships between various launch parameters and costs.
6. **Model Evaluation**: Assess model performance using accuracy, classification reports, and mean squared error (MSE) to fine-tune and validate the models.
