#!/usr/bin/env python3
"""
AgriMind Edge - Main Application
Ultra-low-power AI Agricultural Advisory System

This is the core application that runs on resource-constrained devices
to provide offline agricultural intelligence to farmers across Africa.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict

# Configure logging for resource-constrained environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agrimind.log', mode='a', maxBytes=1024*100),  # 100KB max
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Lightweight sensor data structure"""
    timestamp: str
    temperature: float
    humidity: float
    soil_moisture: float
    light_intensity: float
    battery_level: float

@dataclass
class CropAnalysis:
    """Crop disease analysis result"""
    timestamp: str
    disease_detected: bool
    disease_type: str
    confidence: float
    treatment_advice: str
    urgency_level: int  # 1-5 scale

@dataclass
class WeatherPrediction:
    """Local weather prediction"""
    timestamp: str
    next_24h_temp: Tuple[float, float]  # min, max
    rain_probability: float
    humidity_forecast: float
    farming_advice: str

class TinyMLModel:
    """Ultra-compressed ML model for edge inference"""
    
    def __init__(self, model_path: str = None):
        self.model_loaded = False
        self.model_size_kb = 0
        self.inference_time_ms = 0
        
        # Simulated model weights (in real implementation, load quantized model)
        self.crop_disease_weights = self._load_compressed_model()
        self.weather_prediction_weights = self._load_weather_model()
        
        logger.info(f"TinyML models loaded - Size: {self.model_size_kb}KB")
    
    def _load_compressed_model(self) -> Dict:
        """Load ultra-compressed crop disease detection model"""
        # Simulated quantized model weights (8-bit quantization)
        # Real implementation would load TensorFlow Lite Micro model
        return {
            'conv_layers': np.random.randint(-128, 127, (64, 32), dtype=np.int8),
            'dense_weights': np.random.randint(-128, 127, (128, 10), dtype=np.int8),
            'disease_classes': [
                'healthy', 'leaf_spot', 'rust', 'blight', 'mosaic_virus',
                'bacterial_wilt', 'powdery_mildew', 'anthracnose'
            ]
        }
    
    def _load_weather_model(self) -> Dict:
        """Load compressed weather prediction model"""
        return {
            'lstm_weights': np.random.randint(-128, 127, (32, 16), dtype=np.int8),
            'output_weights': np.random.randint(-128, 127, (16, 4), dtype=np.int8)
        }
    
    def predict_crop_disease(self, image_features: np.ndarray) -> CropAnalysis:
        """Predict crop disease from image features"""
        start_time = time.time()
        
        # Simulated inference on quantized model
        # Real implementation would use TensorFlow Lite Micro
        prediction_scores = np.random.random(8)  # 8 disease classes
        predicted_class = np.argmax(prediction_scores)
        confidence = float(prediction_scores[predicted_class])
        
        disease_type = self.crop_disease_weights['disease_classes'][predicted_class]
        disease_detected = disease_type != 'healthy'
        
        # Generate treatment advice based on disease type
        treatment_advice = self._get_treatment_advice(disease_type)
        urgency_level = self._calculate_urgency(disease_type, confidence)
        
        self.inference_time_ms = (time.time() - start_time) * 1000
        
        return CropAnalysis(
            timestamp=datetime.now().isoformat(),
            disease_detected=disease_detected,
            disease_type=disease_type,
            confidence=confidence,
            treatment_advice=treatment_advice,
            urgency_level=urgency_level
        )
    
    def predict_weather(self, sensor_history: List[SensorReading]) -> WeatherPrediction:
        """Predict local weather using sensor data history"""
        if len(sensor_history) < 24:  # Need at least 24 hours of data
            return self._default_weather_prediction()
        
        # Extract features from sensor history
        temps = [reading.temperature for reading in sensor_history[-24:]]
        humidity = [reading.humidity for reading in sensor_history[-24:]]
        
        # Simple trend analysis (real model would be more sophisticated)
        temp_trend = np.mean(temps[-6:]) - np.mean(temps[-12:-6])
        humidity_trend = np.mean(humidity[-6:]) - np.mean(humidity[-12:-6])
        
        # Predict next 24h temperature range
        current_temp = temps[-1]
        next_min_temp = current_temp + temp_trend - 2
        next_max_temp = current_temp + temp_trend + 3
        
        # Rain probability based on humidity trend
        rain_prob = min(0.9, max(0.1, (humidity[-1] + humidity_trend) / 100))
        
        # Generate farming advice
        farming_advice = self._generate_farming_advice(
            (next_min_temp, next_max_temp), rain_prob
        )
        
        return WeatherPrediction(
            timestamp=datetime.now().isoformat(),
            next_24h_temp=(next_min_temp, next_max_temp),
            rain_probability=rain_prob,
            humidity_forecast=humidity[-1] + humidity_trend,
            farming_advice=farming_advice
        )
    
    def _get_treatment_advice(self, disease_type: str) -> str:
        """Get treatment advice for detected disease"""
        treatments = {
            'healthy': 'Continue current care practices. Monitor regularly.',
            'leaf_spot': 'Remove affected leaves. Apply neem oil spray. Improve air circulation.',
            'rust': 'Apply copper-based fungicide. Remove infected plant debris.',
            'blight': 'URGENT: Apply fungicide immediately. Remove infected plants.',
            'mosaic_virus': 'Remove infected plants. Control aphid vectors. Use resistant varieties.',
            'bacterial_wilt': 'Improve drainage. Apply copper bactericide. Rotate crops.',
            'powdery_mildew': 'Increase air circulation. Apply sulfur-based fungicide.',
            'anthracnose': 'Remove infected fruits. Apply copper fungicide preventively.'
        }
        return treatments.get(disease_type, 'Consult agricultural extension officer.')
    
    def _calculate_urgency(self, disease_type: str, confidence: float) -> int:
        """Calculate urgency level (1-5 scale)"""
        urgency_map = {
            'healthy': 1,
            'leaf_spot': 2,
            'rust': 3,
            'blight': 5,
            'mosaic_virus': 4,
            'bacterial_wilt': 4,
            'powdery_mildew': 2,
            'anthracnose': 3
        }
        base_urgency = urgency_map.get(disease_type, 3)
        # Adjust based on confidence
        if confidence > 0.8:
            return min(5, base_urgency + 1)
        elif confidence < 0.5:
            return max(1, base_urgency - 1)
        return base_urgency
    
    def _default_weather_prediction(self) -> WeatherPrediction:
        """Default weather prediction when insufficient data"""
        return WeatherPrediction(
            timestamp=datetime.now().isoformat(),
            next_24h_temp=(20.0, 30.0),
            rain_probability=0.3,
            humidity_forecast=60.0,
            farming_advice="Insufficient data for accurate prediction. Monitor weather closely."
        )
    
    def _generate_farming_advice(self, temp_range: Tuple[float, float], rain_prob: float) -> str:
        """Generate contextual farming advice"""
        min_temp, max_temp = temp_range
        
        if rain_prob > 0.7:
            return "High rain probability. Delay spraying. Ensure good drainage."
        elif rain_prob < 0.2 and max_temp > 35:
            return "Hot and dry conditions. Increase irrigation. Provide shade if possible."
        elif min_temp < 10:
            return "Cold weather expected. Protect sensitive crops. Delay planting."
        elif 20 <= max_temp <= 30 and 0.3 <= rain_prob <= 0.6:
            return "Optimal growing conditions. Good time for planting and field work."
        else:
            return "Monitor crops closely. Adjust irrigation based on soil moisture."

class PowerManager:
    """Ultra-low power management system"""
    
    def __init__(self):
        self.battery_level = 100.0
        self.solar_charging = False
        self.power_mode = 'normal'  # normal, eco, sleep
        self.daily_power_budget = 1000  # mWh per day
        self.power_consumed_today = 0
        
    def get_battery_status(self) -> Dict:
        """Get current battery and power status"""
        return {
            'battery_level': self.battery_level,
            'solar_charging': self.solar_charging,
            'power_mode': self.power_mode,
            'estimated_runtime_hours': self._estimate_runtime()
        }
    
    def _estimate_runtime(self) -> float:
        """Estimate remaining runtime in hours"""
        if self.power_mode == 'normal':
            consumption_mw = 50  # 50mW average
        elif self.power_mode == 'eco':
            consumption_mw = 25  # 25mW in eco mode
        else:  # sleep mode
            consumption_mw = 5   # 5mW in sleep mode
        
        battery_mwh = self.battery_level * 10  # Assume 1000mWh battery
        return battery_mwh / consumption_mw
    
    def optimize_power_mode(self, current_hour: int) -> str:
        """Optimize power mode based on time and battery level"""
        if self.battery_level < 20:
            return 'sleep'
        elif self.battery_level < 50 or (18 <= current_hour <= 6):
            return 'eco'
        else:
            return 'normal'
    
    def simulate_power_consumption(self, operation: str) -> float:
        """Simulate power consumption for different operations"""
        power_costs = {
            'sensor_reading': 2,    # 2mW for 1 second
            'ai_inference': 15,     # 15mW for 100ms
            'mesh_transmit': 100,   # 100mW for 10ms
            'camera_capture': 200,  # 200mW for 50ms
            'data_storage': 1       # 1mW for 10ms
        }
        
        cost = power_costs.get(operation, 5)
        self.power_consumed_today += cost
        self.battery_level -= cost / 1000  # Convert to percentage
        
        return cost

if __name__ == "__main__":
    logger.info("Starting AgriMind Edge System...")
    
    # Initialize components
    ml_model = TinyMLModel()
    power_manager = PowerManager()
    
    logger.info("AgriMind Edge initialized successfully")
    logger.info(f"System ready - Battery: {power_manager.battery_level}%")
    logger.info(f"Model size: {ml_model.model_size_kb}KB")