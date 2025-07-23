#!/usr/bin/env python3
"""
AgriMind Edge - Sensor Interface Module
Handles all sensor data collection and processing for agricultural monitoring

This module interfaces with various sensors to collect environmental data
optimized for ultra-low power consumption and resource-constrained devices.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SensorCalibration:
    """Sensor calibration parameters"""
    sensor_type: str
    offset: float = 0.0
    scale: float = 1.0
    last_calibrated: str = ""
    calibration_valid: bool = True

class SensorInterface:
    """Base class for all sensor interfaces"""
    
    def __init__(self, sensor_id: str, power_consumption_mw: float):
        self.sensor_id = sensor_id
        self.power_consumption_mw = power_consumption_mw
        self.last_reading_time = 0
        self.reading_interval = 60  # Default 1 minute
        self.calibration = SensorCalibration(sensor_type=sensor_id)
        self.error_count = 0
        self.max_errors = 5
        
    def read_raw(self) -> Optional[float]:
        """Read raw sensor value - to be implemented by subclasses"""
        raise NotImplementedError
    
    def read_calibrated(self) -> Optional[float]:
        """Read calibrated sensor value"""
        raw_value = self.read_raw()
        if raw_value is None:
            return None
        
        return (raw_value + self.calibration.offset) * self.calibration.scale
    
    def is_ready_for_reading(self) -> bool:
        """Check if sensor is ready for next reading"""
        current_time = time.time()
        return (current_time - self.last_reading_time) >= self.reading_interval
    
    def set_reading_interval(self, interval_seconds: int):
        """Set reading interval for power optimization"""
        self.reading_interval = max(10, interval_seconds)  # Minimum 10 seconds
        logger.info(f"Sensor {self.sensor_id} interval set to {interval_seconds}s")

class TemperatureSensor(SensorInterface):
    """Temperature sensor interface (DHT22/SHT30 compatible)"""
    
    def __init__(self, sensor_id: str = "temp_sensor"):
        super().__init__(sensor_id, power_consumption_mw=2.5)
        self.temperature_range = (-40, 80)  # Celsius
        
    def read_raw(self) -> Optional[float]:
        """Simulate temperature reading"""
        try:
            # Simulate realistic temperature with some noise
            base_temp = 25.0 + 10 * np.sin(time.time() / 3600)  # Daily variation
            noise = np.random.normal(0, 0.5)  # ±0.5°C noise
            temperature = base_temp + noise
            
            # Clamp to sensor range
            temperature = max(self.temperature_range[0], 
                            min(self.temperature_range[1], temperature))
            
            self.last_reading_time = time.time()
            self.error_count = 0
            return temperature
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Temperature sensor error: {e}")
            return None

class HumiditySensor(SensorInterface):
    """Humidity sensor interface"""
    
    def __init__(self, sensor_id: str = "humidity_sensor"):
        super().__init__(sensor_id, power_consumption_mw=2.5)
        self.humidity_range = (0, 100)  # Percentage
        
    def read_raw(self) -> Optional[float]:
        """Simulate humidity reading"""
        try:
            # Simulate realistic humidity with daily variation
            base_humidity = 60.0 + 20 * np.sin(time.time() / 3600 + np.pi)
            noise = np.random.normal(0, 2.0)  # ±2% noise
            humidity = base_humidity + noise
            
            # Clamp to valid range
            humidity = max(0, min(100, humidity))
            
            self.last_reading_time = time.time()
            self.error_count = 0
            return humidity
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Humidity sensor error: {e}")
            return None

class SoilMoistureSensor(SensorInterface):
    """Soil moisture sensor interface (capacitive)"""
    
    def __init__(self, sensor_id: str = "soil_moisture"):
        super().__init__(sensor_id, power_consumption_mw=5.0)
        self.moisture_range = (0, 100)  # Percentage
        self.soil_type = "loam"  # affects calibration
        
    def read_raw(self) -> Optional[float]:
        """Simulate soil moisture reading"""
        try:
            # Simulate soil moisture with slower changes
            base_moisture = 40.0 + 15 * np.sin(time.time() / 7200)  # 2-hour cycle
            noise = np.random.normal(0, 1.0)  # ±1% noise
            moisture = base_moisture + noise
            
            # Clamp to valid range
            moisture = max(0, min(100, moisture))
            
            self.last_reading_time = time.time()
            self.error_count = 0
            return moisture
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Soil moisture sensor error: {e}")
            return None

class LightSensor(SensorInterface):
    """Light intensity sensor interface (BH1750/TSL2561)"""
    
    def __init__(self, sensor_id: str = "light_sensor"):
        super().__init__(sensor_id, power_consumption_mw=1.0)
        self.light_range = (0, 65535)  # Lux
        
    def read_raw(self) -> Optional[float]:
        """Simulate light intensity reading"""
        try:
            # Simulate day/night cycle
            hour_of_day = (time.time() % 86400) / 3600  # 0-24 hours
            
            if 6 <= hour_of_day <= 18:  # Daytime
                base_light = 20000 * np.sin(np.pi * (hour_of_day - 6) / 12)
            else:  # Nighttime
                base_light = 0.1
            
            noise = np.random.normal(0, base_light * 0.05)  # 5% noise
            light_intensity = max(0, base_light + noise)
            
            self.last_reading_time = time.time()
            self.error_count = 0
            return light_intensity
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Light sensor error: {e}")
            return None

class BatterySensor(SensorInterface):
    """Battery voltage/level sensor"""
    
    def __init__(self, sensor_id: str = "battery_sensor"):
        super().__init__(sensor_id, power_consumption_mw=0.1)
        self.voltage_range = (3.0, 4.2)  # Li-ion battery range
        self.initial_voltage = 4.0
        self.discharge_rate = 0.001  # V per hour
        
    def read_raw(self) -> Optional[float]:
        """Simulate battery voltage reading"""
        try:
            # Simulate battery discharge over time
            hours_elapsed = time.time() / 3600
            current_voltage = self.initial_voltage - (hours_elapsed * self.discharge_rate)
            
            # Add small noise
            noise = np.random.normal(0, 0.01)
            voltage = current_voltage + noise
            
            # Clamp to valid range
            voltage = max(self.voltage_range[0], 
                         min(self.voltage_range[1], voltage))
            
            self.last_reading_time = time.time()
            self.error_count = 0
            return voltage
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Battery sensor error: {e}")
            return None
    
    def voltage_to_percentage(self, voltage: float) -> float:
        """Convert battery voltage to percentage"""
        min_v, max_v = self.voltage_range
        percentage = ((voltage - min_v) / (max_v - min_v)) * 100
        return max(0, min(100, percentage))

class CameraSensor:
    """Camera interface for crop image capture"""
    
    def __init__(self, camera_id: str = "crop_camera"):
        self.camera_id = camera_id
        self.power_consumption_mw = 200  # High power consumption
        self.resolution = (640, 480)  # VGA resolution for efficiency
        self.last_capture_time = 0
        self.capture_interval = 3600  # Default 1 hour
        self.error_count = 0
        
    def capture_image(self) -> Optional[Dict]:
        """Simulate image capture and feature extraction"""
        try:
            current_time = time.time()
            if (current_time - self.last_capture_time) < self.capture_interval:
                return None  # Too soon for next capture
            
            # Simulate image capture and basic feature extraction
            # In real implementation, this would interface with camera hardware
            image_features = {
                "timestamp": datetime.now().isoformat(),
                "resolution": self.resolution,
                "brightness": np.random.uniform(50, 200),
                "contrast": np.random.uniform(0.5, 2.0),
                "green_ratio": np.random.uniform(0.3, 0.8),  # Vegetation indicator
                "texture_features": np.random.random(64).tolist(),  # Simplified features
                "color_histogram": np.random.random(32).tolist(),
                "edge_density": np.random.uniform(0.1, 0.9)
            }
            
            self.last_capture_time = current_time
            self.error_count = 0
            
            logger.info("Image captured and features extracted")
            return image_features
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Camera capture error: {e}")
            return None
    
    def set_capture_interval(self, interval_seconds: int):
        """Set image capture interval for power optimization"""
        self.capture_interval = max(300, interval_seconds)  # Minimum 5 minutes
        logger.info(f"Camera capture interval set to {interval_seconds}s")

class SensorManager:
    """Manages all sensors and coordinates data collection"""
    
    def __init__(self):
        self.sensors = {}
        self.camera = CameraSensor()
        self.data_history = []
        self.max_history_size = 1000  # Keep last 1000 readings
        self.power_budget_mw = 50  # Total power budget
        self.adaptive_sampling = True
        
        # Initialize sensors
        self._initialize_sensors()
        
    def _initialize_sensors(self):
        """Initialize all sensor interfaces"""
        self.sensors = {
            "temperature": TemperatureSensor(),
            "humidity": HumiditySensor(),
            "soil_moisture": SoilMoistureSensor(),
            "light": LightSensor(),
            "battery": BatterySensor()
        }
        
        logger.info(f"Initialized {len(self.sensors)} sensors")
    
    def collect_sensor_data(self, force_reading: bool = False) -> Optional[Dict]:
        """Collect data from all sensors"""
        sensor_data = {
            "timestamp": datetime.now().isoformat(),
            "node_id": "agrimind_node_001",  # Would be unique per device
            "readings": {},
            "power_consumption_mw": 0
        }
        
        readings_taken = 0
        total_power = 0
        
        for sensor_name, sensor in self.sensors.items():
            if force_reading or sensor.is_ready_for_reading():
                try:
                    value = sensor.read_calibrated()
                    if value is not None:
                        sensor_data["readings"][sensor_name] = {
                            "value": value,
                            "unit": self._get_sensor_unit(sensor_name),
                            "quality": self._assess_reading_quality(sensor, value)
                        }
                        total_power += sensor.power_consumption_mw
                        readings_taken += 1
                        
                except Exception as e:
                    logger.error(f"Error reading {sensor_name}: {e}")
                    sensor_data["readings"][sensor_name] = {
                        "value": None,
                        "error": str(e)
                    }
        
        if readings_taken == 0:
            return None
        
        sensor_data["power_consumption_mw"] = total_power
        sensor_data["readings_count"] = readings_taken
        
        # Add to history
        self.data_history.append(sensor_data)
        if len(self.data_history) > self.max_history_size:
            self.data_history.pop(0)
        
        # Optimize sampling intervals based on power budget
        if self.adaptive_sampling:
            self._optimize_sampling_intervals()
        
        logger.info(f"Collected {readings_taken} sensor readings, power: {total_power:.1f}mW")
        return sensor_data
    
    def capture_crop_image(self) -> Optional[Dict]:
        """Capture and process crop image"""
        return self.camera.capture_image()
    
    def get_sensor_status(self) -> Dict:
        """Get status of all sensors"""
        status = {
            "total_sensors": len(self.sensors),
            "active_sensors": 0,
            "error_sensors": 0,
            "power_consumption_mw": 0,
            "sensor_details": {}
        }
        
        for sensor_name, sensor in self.sensors.items():
            is_active = sensor.error_count < sensor.max_errors
            if is_active:
                status["active_sensors"] += 1
                status["power_consumption_mw"] += sensor.power_consumption_mw
            else:
                status["error_sensors"] += 1
            
            status["sensor_details"][sensor_name] = {
                "active": is_active,
                "error_count": sensor.error_count,
                "last_reading": sensor.last_reading_time,
                "interval": sensor.reading_interval,
                "power_mw": sensor.power_consumption_mw
            }
        
        # Add camera status
        status["camera"] = {
            "active": self.camera.error_count < 5,
            "last_capture": self.camera.last_capture_time,
            "interval": self.camera.capture_interval,
            "power_mw": self.camera.power_consumption_mw
        }
        
        return status
    
    def get_recent_data(self, hours: int = 24) -> List[Dict]:
        """Get sensor data from recent hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_data = []
        for data in self.data_history:
            data_time = datetime.fromisoformat(data["timestamp"])
            if data_time >= cutoff_time:
                recent_data.append(data)
        
        return recent_data
    
    def _get_sensor_unit(self, sensor_name: str) -> str:
        """Get unit for sensor reading"""
        units = {
            "temperature": "°C",
            "humidity": "%",
            "soil_moisture": "%",
            "light": "lux",
            "battery": "V"
        }
        return units.get(sensor_name, "")
    
    def _assess_reading_quality(self, sensor: SensorInterface, value: float) -> str:
        """Assess quality of sensor reading"""
        if sensor.error_count > 0:
            return "poor"
        elif hasattr(sensor, 'calibration') and not sensor.calibration.calibration_valid:
            return "uncalibrated"
        else:
            return "good"
    
    def _optimize_sampling_intervals(self):
        """Optimize sensor sampling intervals based on power budget"""
        total_power = sum(sensor.power_consumption_mw for sensor in self.sensors.values())
        
        if total_power > self.power_budget_mw:
            # Increase intervals to reduce power consumption
            scale_factor = total_power / self.power_budget_mw
            for sensor in self.sensors.values():
                new_interval = int(sensor.reading_interval * scale_factor)
                sensor.set_reading_interval(new_interval)
            
            logger.info(f"Increased sampling intervals due to power constraints")
        
        elif total_power < self.power_budget_mw * 0.7:
            # Decrease intervals for more frequent sampling
            scale_factor = 0.8
            for sensor in self.sensors.values():
                new_interval = max(60, int(sensor.reading_interval * scale_factor))
                sensor.set_reading_interval(new_interval)
            
            logger.info(f"Decreased sampling intervals - power budget available")

if __name__ == "__main__":
    # Test sensor functionality
    sensor_manager = SensorManager()
    
    # Collect sensor data
    data = sensor_manager.collect_sensor_data(force_reading=True)
    if data:
        print("Sensor Data:")
        print(json.dumps(data, indent=2))
    
    # Get sensor status
    status = sensor_manager.get_sensor_status()
    print("\nSensor Status:")
    print(json.dumps(status, indent=2))
    
    # Test camera
    image_data = sensor_manager.capture_crop_image()
    if image_data:
        print("\nImage captured successfully")
        print(f"Features extracted: {len(image_data)} parameters")