def process_web_location_query(input_string):
    # Make sure this function returns a dictionary 
    # containing at least the key 'location'
    demo_response = {
        'location': input_string,
        'aqi_score': 42
    }
    return demo_response
    
import pandas as pd
import numpy as np
import datetime
import hashlib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# BLOCK 1: CORE AI ENGINE & MODEL TRAINING (Runs once when server starts)
# =====================================================================
print("[SYSTEM] Step 1: Initialising environmental training baseline...")

np.random.seed(42)
known_neighborhoods = [
    'Downtown Hub St', 'Greenlake Park Area', 'Westside Industrial Road', 
    'Sunnyvale Senior Community', 'Riverside Boulevard', 'Oakwood Meadow Zone'
]
num_rows = 2500

data = {
    'Timestamp': [datetime.datetime(2026, 10, 1) + datetime.timedelta(hours=i%500) for i in range(num_rows)],
    'Neighborhood': np.random.choice(known_neighborhoods, size=num_rows),
    'Temperature_C': np.random.uniform(15, 35, size=num_rows),
    'Humidity_Percent': np.random.uniform(30, 90, size=num_rows),
    'Wind_Speed_kmh': np.random.uniform(2, 25, size=num_rows),
    'Rainfall_mm': np.random.choice([0.0, 0.0, 0.0, 1.2, 4.5], size=num_rows), 
    'PM10_ug_m3': np.random.uniform(10, 120, size=num_rows),
    'NO2_ppm': np.random.uniform(0.01, 0.08, size=num_rows)
}

df = pd.DataFrame(data)

df['Hour'] = df['Timestamp'].dt.hour
df['Neighborhood'] = df['Neighborhood'].astype(str)
df['Neighborhood_Code'] = df['Neighborhood'].astype('category').cat.codes

# Physics laws mapping urban topology features to target pollutant load thresholds
df['PM2_5_ug_m3'] = (
    (df['PM10_ug_m3'] * 0.55) + 
    (35 / (df['Wind_Speed_kmh'] + 0.5)) + 
    np.where(df['Neighborhood'].str.contains('Industrial|Road|St'), 35, 0) - 
    np.where(df['Neighborhood'].str.contains('Park|Green|Meadow'), 22, 0) + 
    np.random.normal(0, 3, size=num_rows)
)
df['PM2_5_ug_m3'] = df['PM2_5_ug_m3'].clip(lower=1)

features = ['Temperature_C', 'Humidity_Percent', 'Wind_Speed_kmh', 'Rainfall_mm', 'PM10_ug_m3', 'NO2_ppm', 'Hour', 'Neighborhood_Code']
X = df[features]
y = df['PM2_5_ug_m3']

code_mapping = dict(zip(df['Neighborhood'], df['Neighborhood_Code']))

# Train the primary prediction model
model = RandomForestRegressor(n_estimators=40, random_state=42)
model.fit(X.values, y.values)
print("[ENGINE] AI Model Engine successfully trained and cached in memory.")
print("-" * 60)


# =====================================================================
# BLOCK 2: REQUEST CONTROLLER & COMPREHENSIVE MEDICAL IMPACT ENGINE
# =====================================================================
def calculate_aqi(pm25):
    """
    Standard EPA mathematical formula to convert raw PM2.5 mass concentrations 
    into a traditional scalar Air Quality Index value (0 - 500 scale).
    """
    if pm25 <= 12.0:
        return round(((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    else:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)

def get_aqi_status(aqi):
    if aqi <= 50:
        return '[GOOD] Safe for travel'
    elif aqi <= 100:
        return '[MODERATE] Acceptable quality'
    elif aqi <= 150:
        return '[WARNING] Unhealthy for Vulnerable Groups'
    else:
        return '[ALERT] Unhealthy / Hazardous Hotspot'

def get_medical_breakdown(status, aqi):
    if "GOOD" in status:
        return (
            "🩺 HEALTH IMPACT REPORT: MINIMAL RISK [AQI RANGE: 0 - 50]\n"
            "At this concentration level, the air quality is highly favorable and poses negligible risk. "
            "Particulate tracking models show zero significant cellular barrier irritation. It is entirely "
            "safe for all demographic cohorts, including senior citizens, infants, and asthmatics, to engage "
            "in vigorous outdoor exercise, open-air transit commutes, and unchecked prolonged exposure.\n\n"
            "Physiologically, blood oxygenation remains optimal, and lungs face zero exogenous oxidative "
            "stress from vehicle combustion emissions. Travel routing options across this zone do not require "
            "any containment or respiration barriers."
        )
    elif "MODERATE" in status:
        return (
            "🩺 HEALTH IMPACT REPORT: MODERATE SENSITIVITY [AQI RANGE: 51 - 100]\n"
            "The localized atmosphere is structurally acceptable; however, a micro-concentration of respirable "
            "particles is present. Individuals possessing hyper-acute lung hypersensitivity or conditions like "
            "Chronic Obstructive Pulmonary Disease (COPD) may begin to experience subtle airway resistance or "
            "mild throat tickling over prolonged outdoor stays.\n\n"
            "While healthy individuals will cross this zone completely unimpeded, highly sensitive people are "
            "advised to monitor their respiratory cycles carefully during long journeys, as minor alveolar "
            "inflammation can manifest if physical exertion levels are elevated."
        )
    elif "WARNING" in status:
        return (
            "🩺 HEALTH IMPACT REPORT: ELEVATED RISKS FOR VULNERABLE POPULATIONS [AQI RANGE: 101 - 150]\n"
            "WARNING: Fine particulate matter (PM2.5) concentrations have reached tiers capable of penetrating "
            "deeply into lung tissue and bypassing primary throat defense membranes. For vulnerable demographics "
            "— including senior citizens, pregnant women, young children, and individuals diagnosed with asthma "
            "or cardiovascular conditions — this level causes immediate adverse biological side effects.\n\n"
            "Common symptoms manifest quickly as localized wheezing, deep chest tightness, persistent dry coughing, "
            "and accelerated shortness of breath. Extended exposure accelerates vascular irritation, increasing the "
            "risk of acute respiratory flare-ups, chest pain, and reduced endurance. Vulnerable groups must actively "
            "avoid outdoor excursions and re-route travel paths entirely away from this sector."
        )
    else:
        return (
            "🚨 HEALTH IMPACT REPORT: SEVERE SYSTEMIC TOXICITY WARNING [AQI RANGE: 151+]\n"
            "CRITICAL HAZARD: The local environment is saturated with dense emissions, triggering widespread toxic "
            "irritation across the entire population. PM2.5 particulates are small enough to pass from alveoli directly "
            "into the human bloodstream, generating acute oxidative stress, systemic vascular inflammation, and severe "
            "endothelial stress.\n\n"
            "Healthy individuals will experience immediate airway irritation, vocal hoarseness, dizziness, and reduced "
            "lung function during travel. For senior citizens and individuals with pre-existing heart or lung disorders, "
            "prolonged exposure inside this hotspot exponentially spikes the incidence rate of severe medical emergencies, "
            "including asthma attacks, arrhythmia, and cardiac fatigue. Everyone should strictly avoid outdoor activity, "
            "seal all residential vents, and avoid transit through this zone unless protected by industrial filtration masks."
        )

def process_web_location_query(user_location):
    if not user_location.strip():
        user_location = "General Urban Sector"
        
    nanosecond_seed = int(time.time() * 1000000) % 1234567
    np.random.seed(nanosecond_seed)
    
    loc_lower = user_location.lower()
    
    green_keywords = ['park', 'garden', 'woods', 'lake', 'green', 'lawn', 'meadow', 'forest']
    traffic_keywords = ['road', 'street', 'highway', 'main', 'avenue', 'lane', 'drive', 'blvd', 'way', 'hub', 'st', 'rd']
    
    if any(kw in loc_lower for kw in green_keywords):
        base_pm10 = np.random.uniform(5.0, 13.0)      
        base_wind = np.random.uniform(14.0, 24.0)     
        base_no2 = np.random.uniform(0.003, 0.010)
    elif any(kw in loc_lower for kw in traffic_keywords) or loc_lower.endswith((' rd', ' st', ' ave', ' lane', ' blvd')):
        base_pm10 = np.random.uniform(94.0, 150.0)    
        base_wind = np.random.uniform(0.6, 2.8)       
        base_no2 = np.random.uniform(0.058, 0.095)
    else:
        base_pm10 = np.random.uniform(32.0, 66.0)     
        base_wind = np.random.uniform(5.5, 13.0)
        base_no2 = np.random.uniform(0.018, 0.040)
        
    hash_val = int(hashlib.md5(user_location.encode('utf-8')).hexdigest(), 16)
    assigned_code = code_mapping.get(user_location, (hash_val % 10) + len(code_mapping))
    
    current_hour = datetime.datetime.now().hour
    current_temp = np.random.uniform(19.0, 27.0)
    current_humidity = np.random.uniform(45.0, 75.0)
    
    # Passing dynamic metrics matrix as plain numeric array values to keep feature dimensions uncompromised
    input_payload = np.array([[
        current_temp, current_humidity, base_wind, 0.0, base_pm10, base_no2, current_hour, assigned_code
    ]])
    
    raw_pred = model.predict(input_payload)
    final_pm25 = max(0.5, float(raw_pred[0]) + np.random.uniform(-1.2, 1.2))
    
    computed_aqi = calculate_aqi(final_pm25)
    status = get_aqi_status(computed_aqi)
    medical_text = get_medical_breakdown(status, computed_aqi)
    
    return {
        "location": user_location.upper(),
        "time": f"{current_hour}:00",
        "wind_speed": round(base_wind, 2),
        "pm10": round(base_pm10, 2),
        "pm25": round(final_pm25, 2),
        "aqi_score": computed_aqi,
        "safety_status": status,
        "medical_profile": medical_text
    }
    
print("🔍 Search for a location on the map:")
input_string = input("📍 Enter location: ")

api_response = process_web_location_query(input_string)

print("\n" + "="*55)
print(f"🖥️ WEBSITE FRONTEND RENDER FOR: {api_response['location']}")
print("="*55)
print(f"🌬️ Wind Speed: {api_response['wind_speed']} km/h | 📋 PM10: {api_response['pm10']} ug/m3")
print(f"😷 Predicted PM2.5: {api_response['pm25']} ug/m3")
print(f"📊 AQI Score: {api_response['aqi_score']} ({api_response['safety_status']})")
print(f"⏰ Timestamp Reference: {api_response['time']}:00 Hours")
print("="*55)
print(api_response['medical_profile'])
print("="*55)


# =====================================================================