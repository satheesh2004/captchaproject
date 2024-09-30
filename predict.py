from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load('trained_model_with_thresholds.pkl')
except Exception as e:
    print("Error loading model:", e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure 'mouseMovements' is a list
        if not isinstance(data.get('mouseMovements'), list):
            raise ValueError("Expected 'mouseMovements' to be a list")

        # Convert 'mouseMovements' to list of integers
        try:
            mousemovement_data = [int(i.get("timestamp", 0)) for i in data['mouseMovements']]
        except ValueError:
            raise ValueError("All elements in 'mouseMovements' should be convertible to integers")

        # Calculate average mouse movement
        if len(mousemovement_data) == 0:
            raise ValueError("The list 'mouseMovements' is empty")

        mouse_movement_sum = sum(mousemovement_data)
        average_mouse_movement = mouse_movement_sum / len(mousemovement_data)

        # Define thresholds
        mouse_movement_threshold = 150
        screen_width_threshold = 1536
        screen_height_threshold = 864
        time_on_page_threshold = 12
        click_threshold_min = 6
        click_threshold_max = 10
        time_on_page = data.get('timeOnPage', 0)
        clicks = data.get('clicks', 0)
        key_presses = data.get('keyPresses', 0)

        # Convert input data to DataFrame with original feature names and in the specified order
        input_data = pd.DataFrame([{
            'mouseMovements_above_threshold': 1 if average_mouse_movement > mouse_movement_threshold else 0,
            'screenWidth_above_threshold': 1 if data.get('screenWidth', 0) >= screen_width_threshold else 0,
            'screenHeight_above_threshold': 1 if data.get('screenHeight', 0) >= screen_height_threshold else 0,
            'browserName': data.get('browserName', ''),
            'userAgent': data.get('userAgent', ''),
            'language_is_en_gb': 1 if data.get('language') == 'en-GB' else 0,
            'timeOnPage_above_threshold': 1 if time_on_page > time_on_page_threshold else 0,
            'timeOnPage': time_on_page,
            'clicks_within_range': 1 if click_threshold_min <= clicks <= click_threshold_max else 0,
            'keyPresses': key_presses,
            'referrer': data.get('referrer', '')
        }])

        # Preprocess the categorical features (ensure same encoding as training)
        input_data['browserName'] = input_data['browserName'].astype('category').cat.codes
        input_data['userAgent'] = input_data['userAgent'].astype('category').cat.codes
        input_data['language_is_en_gb'] = input_data['language_is_en_gb'].astype('category').cat.codes
        input_data['referrer'] = input_data['referrer'].astype('category').cat.codes

        # Reorder columns to match model's expected input order
        input_data = input_data[['mouseMovements_above_threshold', 'screenWidth_above_threshold',
                                 'screenHeight_above_threshold', 'browserName', 'userAgent',
                                 'language_is_en_gb', 'timeOnPage_above_threshold', 'timeOnPage',
                                 'clicks_within_range', 'keyPresses', 'referrer']]

        # Make a prediction
        try:
            prediction = model.predict(input_data)
            return jsonify({'prediction': int(prediction[0])})
        except Exception as e:
            print(e)
            return jsonify({'error': f"Prediction error: {str(e)}"}), 500
    except ValueError as e:
        print(f"ValueError: {e}")
        return jsonify({"error": f"ValueError: {str(e)}"}), 400
    except TypeError as e:
        print(f"TypeError: {e}")
        return jsonify({"error": f"TypeError: {str(e)}"}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)
