# Forest Microclimate Predictor

A machine learning project exploring how forest conditions influence temperature buffering and whether those patterns can be used to make predictions.

## Project Question
Forests can create cooler and more stable temperatures under their canopy, but the amount of temperature buffering can change depending on the forest conditions. In this project, I wanted to explore whether factors such as canopy cover, sensor height, elevation, and time of year could be used to predict temperature buffering. I also wanted to compare different machine learning models and see how well they could learn these patterns from real environmental data.

## Dataset 
This project uses the Processed_microclimate_data.csv dataset from the Dryad dataset "Microclimatic buffering in forests of the future: the role of local water balance." The data was collected from environmental sensors in forests of the northwestern United States and includes daily measurements of temperature, relative humidity, and vapor pressure deficit. It also includes information about the conditions surrounding each sensor, such as canopy cover and sensor height.

For this project, I focused on predicting temperature buffering using four features:
- Canopy cover
- Sensor height
- Elevation 
- Month

## Data Source
DOI: https://datadryad.org/dataset/doi:10.5061/dryad.406ms4g 

## Data Exploration
Before building the models. I explored and cleaned the dataset using Python and pandas. I checked the size of the dataset, looked for missing values, examined the different variables, and created visualizations to better understand the environmental patterns in the data.Some of the relationships I explored included how temperature buffering changes with canopy cover and sensor height.

## Machine Learning
I first trained a Linear Regression model as a simpler baseline. I then trained a Random Forest model to see whether a model that can learn more complicated relationships between the features would perform better.
I evaluated the models using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R^2
After comparing the models and experimenting with different features, I selected a Random Forest using canopy cover, sensor height, elevation, and month as my final model.

## Results
Using a normal random 80/20 train-test split, the final Random Forest achieved approximately:
- MAE: 1.40 Celsius
- RMSE: 1.99 Celsius
- R^2: 0.72
I also experimented with adding the research location as another feature. This slightly lowered the MAE to about 1.35 Celsius. However, I decided not to use location in the final model because I wanted the predictions to depend on environmental features instead of the identity of one of the locations already represented in the dataset.

## Testing on Unseen Sensors
I wanted to test whether the good results from the random split meant that the model could also generalize to sensors it had never seen before.
Instead of randomly separating individual rows, I used grouped splitting so that all measurements from a sensor entirely in either the training set or the testing set. This prevented the model from training on other measurements from a sensor that later appeared in the test data.
Across five grouped tests, the model achieved approximately:
- Average MAE: 2.55 Celcius
- Average R^2: 0.11
This was much worse than the normal random split. It showed that although the model learned useful patterns in the dataset, those patterns did not generalize nearly as well to completely unseen sensors.

## Limitations
The model only uses four features, while real forest microclimates are affected by many different environmental conditions. The dataset also contains repeated measurements from a limited number of sensors and locations. Because of this, the web application should be viewed as an experimental demostration of the machine learning model rather than a tool that can accurately predict temperature buffering in every forest. A future version of the project could use data from more locations, include additional environmental variables, and investigate ways to improve performance on completely unseen sensors.

## Web Application
I built an interactive web application using StreamLit. A user can select:
- Canopy cover
- Sensor height
- Elevation
- Month
The application sends these inputs to the trained Random Forest model and displays its predicted temperature buffering. The model was trained separately and saved using Joblib so tha the Streamlit application can load the trained model without retraining it every time the application runs.

## Tools and Technologies
- Python
- pandas
- scikit-learn
- matplotlib
- Streamlit
- Joblib
- Git and GitHub

## What I learned 
This project helped me understand the complete process of building a small machine learning project using real environmental data. I practiced cleaning and exploring data, selecting features, training and comparing models, evaluating predictions, and building an interactive web-app around a trained model.

One of the most important things I learned was that a good evaluation score does not always mean that a model will work equally well on new data. The difference between the random split and the grouped sensor tests showed me why the way a machine learning model is tested matters just as much as the final score.