import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv('data.csv')

# Define thresholds
mouse_movement_threshold = 150
screen_width_threshold = 1536
screen_height_threshold = 864
time_on_page_threshold = 12
click_threshold_min = 6
click_threshold_max = 10

# Convert 'timeOnPage' to numeric values
data['timeOnPage'] = pd.to_numeric(data['timeOnPage'], errors='coerce')
data = data.dropna(subset=['timeOnPage'])  # Drop rows with non-numeric values if needed

# Create new binary features based on thresholds
data['timeOnPage_above_threshold'] = data['timeOnPage'].apply(lambda x: 1 if x >= time_on_page_threshold else 0)
data['mouseMovements_above_threshold'] = data['mouseMovements'].apply(lambda x: 1 if x > mouse_movement_threshold else 0)
data['screenWidth_above_threshold'] = data['screenWidth'].apply(lambda x: 1 if x >= screen_width_threshold else 0)
data['screenHeight_above_threshold'] = data['screenHeight'].apply(lambda x: 1 if x >= screen_height_threshold else 0)
data['clicks_within_range'] = data['clicks'].apply(lambda x: 1 if click_threshold_min <= x <= click_threshold_max else 0)
data['language_is_en_gb'] = data['language'].apply(lambda x: 1 if x == 'en-GB' else 0)

# Drop the original columns if not needed
data = data.drop(['mouseMovements', 'screenWidth', 'screenHeight', 'clicks', 'language'], axis=1)

# Encode categorical variables
data['browserName'] = data['browserName'].astype('category').cat.codes
data['userAgent'] = data['userAgent'].astype('category').cat.codes
data['referrer'] = data['referrer'].astype('category').cat.codes

# Encode the 'label' column
data['label_encoded'] = data['label'].astype('category').cat.codes

# Ensure balanced class distribution
class_distribution_initial = data['label'].value_counts()
print("\nInitial Class Distribution in 'label' column:")
print(class_distribution_initial)

if class_distribution_initial.min() < 2:
    print("Class distribution is unbalanced. Resampling the data to ensure at least 2 samples in each class.")
    data_resampled = data.groupby('label', group_keys=False).apply(lambda x: x.sample(class_distribution_initial.max(), replace=True))
    data = data_resampled.reset_index(drop=True)

# Verify the updated class distribution
class_distribution_updated = data['label'].value_counts()
print("Updated Class Distribution in 'label' column:")
print(class_distribution_updated)

# Define features (X) and label (y)
X = data[['mouseMovements_above_threshold', 'screenWidth_above_threshold',
          'screenHeight_above_threshold', 'browserName', 'userAgent',
          'language_is_en_gb', 'timeOnPage_above_threshold', 'timeOnPage',
          'clicks_within_range', 'keyPresses', 'referrer']]
y = data['label']

unique_labels = data['label'].unique()

if len(unique_labels) < 2:
    print('Error: The dataset contains only one class. Unable to train the model.')
else:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'trained_model_with_thresholds.pkl')

    print('Model trained and saved successfully.')
