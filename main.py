import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Load the data
# Replace 'your_data.csv' with the actual path to your dataset
data = pd.read_csv('jobfair_2023/jobfair_train.csv')

# Features (X) and target variable (y)
features = [
    'cohort_season',
    'avg_age_top_11_players',
    'avg_stars_top_11_players',
    'avg_stars_top_14_players',
    'avg_training_factor_top_11_players',
    'days_active_last_28_days',
    'league_match_watched_count_last_28_days',
    'session_count_last_28_days',
    'playtime_last_28_days',
    'league_match_won_count_last_28_days',
    'training_count_last_28_days',
    'global_competition_level',
    'tokens_spent_last_28_days',
    'tokens_stash',
    'rests_stash',
    'morale_boosters_stash'
]

X = data[features]
y = data['league_rank']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean absolute error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Now you can use this model to predict the league rank for new data
# Replace 'new_data.csv' with the actual path to your new data
new_data = pd.read_csv('jobfair_2023/jobfair_test.csv')
new_data_imputed = pd.DataFrame(imputer.transform(new_data[features]), columns=features)
new_predictions = model.predict(new_data_imputed)

# Round the predictions
rounded_predictions = new_predictions.round()

# Create a DataFrame for the submission file
submission_df = pd.DataFrame({
    'club_id': new_data['club_id'],
    'predicted_league_rank': rounded_predictions
})

# Save the results to a CSV file
submission_df.to_csv('league_rank_predictions.csv', index=False)

# Print the predicted league ranks for new data
print('Predicted League Ranks for New Data:')
print(submission_df)