from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from flask_cors import CORS
import os
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import joblib
import re
import requests
import folium
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure Gemini AI with API key
API_KEY = "AIzaSyDqgRQe8nu1lJry7NI0MgF21WSdRSOLEmw"
genai.configure(api_key=API_KEY)
#AIzaSyAnOuinmsV8FH4SpPX1NmPH2p5GiLK_OMU

# Multi-language support dictionary for farm solution generator
LANGUAGES = {
    "English": {
        "land_types": ["Clay Soil", "Sandy Soil", "Loamy Soil", "Silt Soil", "Black Soil"],
        "seasons": ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"],
        "crops": ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Vegetables", "Fruits", "Oil Seeds"]
    },
    "Hindi": {
        "land_types": ["मिट्टी की मिट्टी", "रेतीली मिट्टी", "दोमट मिट्टी", "गाद मिट्टी", "काली मिट्टी"],
        "seasons": ["खरीफ (मानसून)", "रबी (सर्दी)", "जायद (गर्मी)"],
        "crops": ["चावल", "गेहूं", "कपास", "गन्ना", "दालें", "सब्जियां", "फल", "तिलहन"]
    },
    "Telugu": {
        "land_types": ["బంక మట్టి", "ఇసుక నేల", "లోమి నేల", "బురద నేల", "నల్ల నేల"],
        "seasons": ["ఖరీఫ్ (వర్షాకాలం)", "రబీ (శీతాకాలం)", "జైద్ (వేసవి)"],
        "crops": ["వరి", "గోధుమ", "పత్తి", "చెరకు", "పప్పు ధాన్యాలు", "కూరగాయలు", "పండ్లు", "నూనె గింజలు"]
    }
}

# ================= Helper Functions for Crop Recommendation & Production Prediction =================
def clean_string(value):
    """
    Clean and standardize string values
    """
    if isinstance(value, str):
        # Remove extra whitespace and convert to lowercase
        return re.sub(r'\s+', ' ', value).strip().lower()
    return value

def get_gemini_fertilizer_recommendation(crop, soil_type, location):
    """
    Get fertilizer recommendations using Gemini API
    """
    try:
        # Use the correct model name
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Prompt for fertilizer recommendation
        prompt = f"""
        Provide a detailed fertilizer recommendation for {crop} grown in {soil_type} soil type in {location}.
        Include:
        1. Recommended NPK fertilizer ratio
        2. Quantity of fertilizer per hectare
        3. Application method
        4. Timing of fertilizer application
        5. Additional soil health recommendations

        Ensure the response is practical, scientifically accurate, and formatted in clear, readable markdown.
        """

        # Generate the content
        response = model.generate_content(prompt)
        
        # Return the text of the response
        return response.text
    except Exception as e:
        # More detailed error handling
        return f"""
        ### Fertilizer Recommendation Unavailable

        Unable to fetch fertilizer recommendation due to an API error. 
        Possible reasons:
        - API key may be invalid
        - Service might be temporarily unavailable
        - Network connectivity issues

        General Fertilizer Recommendation Tips:
        1. Consult local agricultural experts
        2. Get a soil test from a local laboratory
        3. Consider crop-specific nutrient requirements
        4. Use balanced NPK fertilizers
        5. Follow recommended application rates
        """

def get_weather_data(location):
    """
    Fetch weather data using OpenWeatherMap API
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': os.environ.get('OPENWEATHER_API_KEY'),
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'latitude': data['coord']['lat'],
                'longitude': data['coord']['lon']
            }
        else:
            return None
    except Exception as e:
        return None

def create_location_map(latitude, longitude):
    """
    Create a Folium map for location visualization
    """
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.Marker(
        [latitude, longitude], 
        popup='Your Location'
    ).add_to(m)
    
    # Save map to HTML string
    map_html = BytesIO()
    m.save(map_html, close_file=False)
    return map_html.getvalue().decode()

def load_and_preprocess_recommendation_data():
    """
    Load and preprocess crop recommendation dataset
    """
    try:
        crop = pd.read_csv("dataset/Crop_recommendation.csv")
        crop.drop(crop[crop.label == 'muskmelon'].index, inplace=True)
        data = crop.copy().drop_duplicates()
        encod = LabelEncoder()
        data['Encoded_label'] = encod.fit_transform(data.label)
        
        classes = pd.DataFrame({'label': pd.unique(data.label), 'encoded': pd.unique(data.Encoded_label)})
        classes = classes.sort_values('encoded').set_index('label')
        return data, classes, encod
    except FileNotFoundError:
        return None, None, None

def train_crop_recommendation_model(data):
    """
    Train Random Forest Classifier for crop recommendation
    """
    x = data.iloc[:,:-2]
    y = data.Encoded_label
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    param = {
        'n_estimators': [10, 50, 100], 
        'criterion': ['gini', 'entropy'], 
        'max_depth': [5, 10, 15, None]
    }
    rand = RandomForestClassifier(random_state=42)
    grid_rand = GridSearchCV(rand, param, cv=5, n_jobs=-1, verbose=1)
    grid_rand.fit(x_train_scaled, y_train)
    
    pred_rand = grid_rand.predict(x_test_scaled)
    print('Classification Report:\n', classification_report(y_test, pred_rand))
    return grid_rand, x.columns, scaler

def load_and_preprocess_production_data():
    """
    Load and preprocess crop production dataset with improved error handling
    """
    try:
        # Read the CSV file
        crop_data = pd.read_csv("dataset/crop_production.csv")
        
        # Apply string cleaning to all object columns
        for col in crop_data.select_dtypes(include=['object']).columns:
            crop_data[col] = crop_data[col].apply(clean_string)
        
        # Clean and standardize crop names
        crop_data['Crop'] = crop_data['Crop'].replace({
            'moth': 'mothbeans', 
            'peas  (vegetable)': 'pigeonpeas', 
            'bean': 'kidneybeans',
            'moong(green gram)': 'mungbean', 
            'pome granet': 'pomegranate',
            'water melon': 'watermelon', 
            'cotton(lint)': 'cotton', 
            'gram': 'chickpea'
        })
        
        # Define valid crops
        valid_crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
            'banana', 'mango', 'grapes', 'watermelon', 'apple', 'orange', 
            'papaya', 'coconut', 'cotton', 'jute', 'coffee'
        ]
        
        # Filter and clean data
        crop_data = crop_data[crop_data['Crop'].isin(valid_crops)]
        
        # Remove columns that might cause issues
        columns_to_drop = [col for col in ['State_Name', 'District_Name'] if col in crop_data.columns]
        if columns_to_drop:
            crop_data = crop_data.drop(columns_to_drop, axis=1)
        
        # Convert numeric columns, handling potential string values
        numeric_columns = ['Crop_Year', 'Area', 'Production']
        for col in numeric_columns:
            # First clean any non-numeric characters
            crop_data[col] = crop_data[col].apply(lambda x: re.sub(r'[^\d.]', '', str(x)) if isinstance(x, str) else x)
            
            # Attempt to convert to numeric, replacing errors with NaN
            crop_data[col] = pd.to_numeric(crop_data[col], errors='coerce')
        
        # Additional check to remove rows with non-numeric values in critical columns
        crop_data = crop_data.dropna(subset=numeric_columns)
        
        # Ensure crop year is integer
        crop_data['Crop_Year'] = crop_data['Crop_Year'].astype(int)
        
        return crop_data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None
def get_weather_data_by_coords(latitude, longitude):
    """
    Fetch weather data using coordinates
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': latitude,
        'lon': longitude,
        'appid': os.environ.get('OPENWEATHER_API_KEY'),
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'latitude': latitude,
                'longitude': longitude
            }
        else:
            return None
    except Exception as e:
        return None
def get_location_info(latitude, longitude):
    """
    Get location name from coordinates using reverse geocoding
    """
    base_url = "https://api.openweathermap.org/geo/1.0/reverse"
    params = {
        'lat': latitude,
        'lon': longitude,
        'limit': 1,
        'appid': os.environ.get('OPENWEATHER_API_KEY')
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200 and data:
            # Extract location name from response
            location_name = data[0].get('name', '')
            country = data[0].get('country', '')
            
            if location_name:
                return f"{location_name}, {country}"
            return "Unknown Location"
        else:
            return "Unknown Location"
    except Exception as e:
        return "Unknown Location"
def train_crop_production_model(data):
    """
    Train Random Forest Regressor for crop production prediction
    """
    # Ensure data is not None
    if data is None:
        return None, None, None, None, None
    
    # Use median crop year as default
    default_crop_year = int(data['Crop_Year'].median())
    
    # Perform one-hot encoding
    dummy = pd.get_dummies(data, columns=['Crop'])
    
    # Verify data types before training
    numeric_columns = dummy.select_dtypes(include=[np.number]).columns
    dummy = dummy[numeric_columns]
    
    x = dummy.drop(["Production"], axis=1)
    y = dummy["Production"]
    
    # Scale the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Check if we have enough data
    if len(x) == 0 or len(y) == 0:
        return None, None, None, None, None
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=5)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    rf_predict = model.predict(x_test)
    
    # Calculate R-squared and other metrics
    r1 = r2_score(y_test, rf_predict)
    mse = mean_squared_error(y_test, rf_predict)
    rmse = np.sqrt(mse)
    
    print(f"R2 score: {r1}")
    print(f"Root Mean Squared Error: {rmse}")
    
    return model, x, x.columns.tolist(), scaler, default_crop_year

def preprocess_input_data(input_data, feature_columns, recommendation_scaler):
    """
    Preprocess input data for crop recommendation
    """
    # Convert season to numeric encoding
    season_mapping = {
        'kharif': 0, 
        'rabi': 1, 
        'zaid': 2, 
        'whole year': 2, 
        'summer': 2
    }
    # Clean and map the season
    season = clean_string(input_data[0])
    input_data[0] = season_mapping.get(season, input_data[0])
    
    # Create a dictionary with feature values
    features_dict = {
        'N': input_data[1],
        'P': input_data[2],
        'K': input_data[3],
        'temperature': input_data[5]['temperature'],  # Using weather data
        'humidity': input_data[5]['humidity'],        # Using weather data
        'ph': 6.5,          # Default average pH
        'rainfall': 1500    # Default average rainfall
    }
    
    # Get feature values in the original order
    features = [features_dict[col] for col in feature_columns]
    
    # Scale the features
    features_scaled = recommendation_scaler.transform([features])
    
    return features_scaled[0]

def predict_crop_and_production(recommendation_model, production_model, x_train, production_columns, classes, feature_columns, recommendation_scaler, production_scaler, default_crop_year, input_data):
    """
    Predict recommended crop and estimated production
    """
    # Preprocess input data for recommendation
    preprocessed_input = preprocess_input_data(input_data, feature_columns, recommendation_scaler)
    
    crop_probabilities = recommendation_model.predict_proba([preprocessed_input])
    crop_prob_df = pd.DataFrame(data=np.round(crop_probabilities.T*100, 2), index=classes.index, columns=['predicted_values'])
    recommended_crop = crop_prob_df.predicted_values.idxmax()
    
    # Prepare test row with one-hot encoded crop
    test_row = x_train.head(1).copy()
    test_row.iloc[0] = 0  # Reset all values to 0
    test_row['Crop_Year'] = default_crop_year
    test_row['Area'] = input_data[4]
    
    # Set the crop column for the recommended crop
    crop_column = f'Crop_{recommended_crop}'
    if crop_column in test_row.columns:
        test_row[crop_column] = 1
    
    # Scale the test row features
    test_row_scaled = production_scaler.transform(test_row)
    
    # Predict production
    production = production_model.predict(test_row_scaled)[0]
    yield_per_area = production / test_row['Area'].values[0]
    
    return recommended_crop, production, yield_per_area, crop_prob_df.sort_values('predicted_values', ascending=False)

def train_and_save_models():
    """
    Train and save models
    """
    # Training recommendation model
    recommendation_data, classes, encoder = load_and_preprocess_recommendation_data()
    if recommendation_data is not None:
        recommendation_model, feature_columns, recommendation_scaler = train_crop_recommendation_model(recommendation_data)
        
        # Save components as a dictionary
        recommendation_model_dict = {
            'model': recommendation_model,
            'classes': classes,
            'encoder': encoder,
            'scaler': recommendation_scaler,
            'feature_columns': feature_columns
        }
        
        joblib.dump(recommendation_model_dict, 'crop_recommendation_model.pkl')
        result = "Crop Recommendation Model Trained and Saved!"
    else:
        result = "Error: Crop recommendation dataset not found!"
        return jsonify({"status": "error", "message": result})
    
    # Training production model
    production_data = load_and_preprocess_production_data()
    if production_data is not None:
        production_model, x_train, production_columns, production_scaler, default_crop_year = train_crop_production_model(production_data)
        if production_model is not None:
            # Save all components together in a dictionary
            production_model_dict = {
                'model': production_model, 
                'x_train': x_train,
                'production_columns': production_columns,
                'scaler': production_scaler, 
                'default_crop_year': default_crop_year
            }
            joblib.dump(production_model_dict, 'crop_production_model.pkl')
            result += " Crop Production Model Trained and Saved!"
            return jsonify({"status": "success", "message": result})
        else:
            return jsonify({"status": "error", "message": "Error training production model!"})
    else:
        return jsonify({"status": "error", "message": "Error: Crop production dataset not found!"})

def load_saved_models():
    """
    Load pre-trained models and scalers with robust error handling
    """
    try:
        # Check if model files exist
        if not (os.path.exists('crop_recommendation_model.pkl') and 
                os.path.exists('crop_production_model.pkl')):
            return None

        # Load recommendation model components
        recommendation_model_dict = joblib.load('crop_recommendation_model.pkl')
        
        # Verify all required keys are present
        recommendation_keys = ['model', 'classes', 'encoder', 'scaler', 'feature_columns']
        if not all(key in recommendation_model_dict for key in recommendation_keys):
            return None

        # Load production model components
        production_model_dict = joblib.load('crop_production_model.pkl')
        
        # Verify all required keys are present
        production_keys = ['model', 'x_train', 'production_columns', 'scaler', 'default_crop_year']
        if not all(key in production_model_dict for key in production_keys):
            return None

        # Unpack components
        recommendation_model = recommendation_model_dict['model']
        classes = recommendation_model_dict['classes']
        encoder = recommendation_model_dict['encoder']
        recommendation_scaler = recommendation_model_dict['scaler']
        feature_columns = recommendation_model_dict['feature_columns']

        production_model = production_model_dict['model']
        x_train = production_model_dict['x_train']
        production_columns = production_model_dict['production_columns']
        production_scaler = production_model_dict['scaler']
        default_crop_year = production_model_dict['default_crop_year']
        
        return (recommendation_model, classes, encoder, recommendation_scaler, 
                feature_columns, production_model, x_train, production_columns, 
                production_scaler, default_crop_year)
    
    except Exception as e:
        return None

# ================= Route Handlers for Both Applications =================

# Shared route for index
@app.route('/')
def home():
    return render_template('landing_page.html')
@app.route('/index.html')
def home1():
    return render_template('index.html')
@app.route('/landing_page.html')
def home2():
    return render_template('landing_page.html')
# Farm solution routes
@app.route('/api/get_options/<language>')
def get_options(language):
    if language in LANGUAGES:
        return jsonify(LANGUAGES[language])
    return jsonify({"error": "Language not supported"}), 400

@app.route('/api/list_models')
def list_models():
    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        return jsonify({"models": model_names})
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error listing models: {str(e)}")
        print(error_details)
        return jsonify({"error": str(e), "traceback": error_details}), 500

@app.route('/api/generate_solution', methods=['POST'])
def generate_solution():
    try:
        # Print received data for debugging
        print("Received data:", request.data)
        
        # Check if request has JSON data
        if not request.is_json:
            print("Error: Request does not contain JSON data")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        print("Parsed JSON data:", data)
        
        # Validate required fields
        required_fields = ['land_type', 'season', 'crop_type', 'acres', 'problem']
        for field in required_fields:
            if field not in data:
                print(f"Error: Missing required field '{field}'")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        language = data.get('language', 'English')
        
        # Set up model and generate content
        print("Setting up Gemini model")
        
        # Use a specific model instead of auto-selecting the first one
        model_name = 'models/gemini-1.5-flash'
        print(f"Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        As an agricultural expert, provide a detailed solution in {language} for the following farming situation:
        
        Land Type: {data['land_type']}
        Season: {data['season']}
        Crop Type: {data['crop_type']}
        Land Area: {data['acres']} acres
        Problem Description: {data['problem']}
        
        Please provide:
        1. Problem analysis
        2. Recommended solutions
        3. Preventive measures for the future
        4. Additional tips specific to the land type, crop, and season
        """
        
        print("Sending prompt to Gemini API")
        response = model.generate_content(prompt)
        print("Received response from Gemini API")
        
        # Create solutions directory if it doesn't exist
        os.makedirs('solutions', exist_ok=True)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solutions/farm_solution_{timestamp}.txt"
        
        print(f"Writing solution to file: {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("FARM PROBLEM DETAILS\n")
            f.write("-------------------\n\n")
            for key, value in data.items():
                f.write(f"{key.title()}: {value}\n")
            f.write("\nRECOMMENDED SOLUTION\n")
            f.write("-------------------\n\n")
            f.write(response.text)
        
        print("Successfully generated solution")
        return jsonify({
            "solution": response.text,
            "filename": filename
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in generate_solution: {str(e)}")
        print(error_details)
        return jsonify({"error": str(e), "traceback": error_details}), 500

# Crop recommendation routes
@app.route('/index1.html')
def index():
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Silt', 'Chalky', 'Peaty']
    return render_template('index1.html', soil_types=soil_types)

@app.route('/train', methods=['GET'])
def train_models():
    return render_template('train.html')

@app.route('/train_models', methods=['POST'])
def train_models_api():
    return train_and_save_models()
@app.route('/get_map', methods=['GET'])
def get_map():
    """
    Return a map centered at a default location for initial display
    """
    # Default coordinates (can be set to a common agricultural region)
    default_lat = 20.5937
    default_lng = 78.9629  # Center of India
    
    # Create a map centered at the default location
    map_html = create_location_map(default_lat, default_lng)
    
    return jsonify({
        "status": "success",
        "map_html": map_html
    })
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form
        season = data.get('season')
        nitrogen = float(data.get('nitrogen'))
        phosphorus = float(data.get('phosphorus'))
        potassium = float(data.get('potassium'))
        area = float(data.get('area'))
        soil_type = data.get('soil_type')
        
        # Get location data from coordinates
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        
        # Get location name using reverse geocoding
        location_name = get_location_info(latitude, longitude)
        
        # Load models
        model_components = load_saved_models()
        
        if model_components is None:
            return jsonify({
                "status": "error", 
                "message": "Models not found or incomplete. Please train the models first."
            })
        
        # Unpack model components
        (recommendation_model, classes, encoder, recommendation_scaler, 
         feature_columns, production_model, x_train, production_columns, 
         production_scaler, default_crop_year) = model_components
        
        # Get weather data using coordinates
        weather_data = get_weather_data_by_coords(latitude, longitude)
        
        if not weather_data:
            return jsonify({
                "status": "error", 
                "message": "Weather data could not be retrieved for the given location."
            })
        
        # Create input data array with weather data
        input_data = [season, nitrogen, phosphorus, potassium, area, weather_data]
        
        # Predict crop and production
        recommended_crop, production, yield_per_area, crop_prob_df = predict_crop_and_production(
            recommendation_model, production_model, x_train, production_columns, 
            classes, feature_columns, recommendation_scaler, 
            production_scaler, default_crop_year, input_data
        )
        
        # Get fertilizer recommendation
        fertilizer_recommendation = get_gemini_fertilizer_recommendation(
            recommended_crop, soil_type, location_name
        )
        
        # Create map
        map_html = create_location_map(
            weather_data['latitude'], 
            weather_data['longitude']
        )
        
        # Prepare crop probabilities for JSON response
        crop_probs = crop_prob_df.head(5).reset_index()
        crop_probs_list = []
        for _, row in crop_probs.iterrows():
            crop_probs_list.append({
                "crop": row['label'].capitalize(),
                "probability": float(row['predicted_values'])
            })
        
        # Build response
        response = {
            "status": "success",
            "recommended_crop": recommended_crop.capitalize(),
            "production": round(float(production), 2),
            "yield_per_hectare": round(float(yield_per_area), 2),
            "location": location_name,
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            },
            "weather": {
                "temperature": weather_data['temperature'],
                "humidity": weather_data['humidity'],
                "description": weather_data['description'].capitalize(),
                "wind_speed": weather_data['wind_speed']
            },
            "crop_probabilities": crop_probs_list,
            "fertilizer_recommendation": fertilizer_recommendation,
            "map_html": map_html
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred during prediction: {str(e)}"
        })
if __name__ == '__main__':
    app.run(debug=True)