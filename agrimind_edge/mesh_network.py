#!/usr/bin/env python3
"""
AgriMind Edge - Mesh Network Module
Implements LoRa-based mesh networking for offline device communication

This module enables devices to share agricultural data and AI insights
without requiring internet connectivity, creating resilient farming networks.
"""

import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in the mesh network"""
    SENSOR_DATA = "sensor_data"
    CROP_ANALYSIS = "crop_analysis"
    WEATHER_PREDICTION = "weather_prediction"
    ALERT = "alert"
    MODEL_UPDATE = "model_update"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"

@dataclass
class MeshMessage:
    """Standard message format for mesh network"""
    message_id: str
    sender_id: str
    message_type: MessageType
    payload: Dict
    timestamp: str
    ttl: int = 5  # Time to live (hops)
    route_path: List[str] = None
    
    def __post_init__(self):
        if self.route_path is None:
            self.route_path = [self.sender_id]

@dataclass
class NodeInfo:
    """Information about a mesh network node"""
    node_id: str
    last_seen: str
    signal_strength: int  # RSSI
    battery_level: float
    location: Optional[Dict] = None  # GPS coordinates if available
    capabilities: List[str] = None  # What this node can do
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["sensor", "ai_inference"]

class MeshNetworkManager:
    """Manages mesh network communication and routing"""
    
    def __init__(self, node_id: str, max_neighbors: int = 10):
        self.node_id = node_id
        self.max_neighbors = max_neighbors
        self.neighbors: Dict[str, NodeInfo] = {}
        self.message_cache: Dict[str, MeshMessage] = {}
        self.routing_table: Dict[str, str] = {}  # destination -> next_hop
        self.message_queue: List[MeshMessage] = []
        
        # Network statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_forwarded = 0
        self.network_discovery_interval = 300  # 5 minutes
        self.last_discovery = 0
        
        logger.info(f"Mesh network initialized for node {self.node_id}")
    
    def discover_neighbors(self) -> List[NodeInfo]:
        """Discover nearby nodes in the mesh network"""
        current_time = time.time()
        
        if current_time - self.last_discovery < self.network_discovery_interval:
            return list(self.neighbors.values())
        
        # Simulate neighbor discovery (in real implementation, use LoRa radio)
        discovery_message = MeshMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            message_type=MessageType.DISCOVERY,
            payload={
                "node_capabilities": ["sensor", "ai_inference", "weather_prediction"],
                "battery_level": 85.0,
                "discovery_timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Simulate discovering 2-5 neighbors
        import random
        num_neighbors = random.randint(2, min(5, self.max_neighbors))
        
        for i in range(num_neighbors):
            neighbor_id = f"agri_node_{random.randint(1000, 9999)}"
            if neighbor_id not in self.neighbors:
                self.neighbors[neighbor_id] = NodeInfo(
                    node_id=neighbor_id,
                    last_seen=datetime.now().isoformat(),
                    signal_strength=random.randint(-80, -40),  # RSSI in dBm
                    battery_level=random.uniform(30, 100),
                    capabilities=["sensor", "ai_inference"]
                )
        
        self.last_discovery = current_time
        logger.info(f"Discovered {len(self.neighbors)} neighbors")
        return list(self.neighbors.values())
    
    def send_message(self, message: MeshMessage, target_node: str = None) -> bool:
        """Send message through mesh network"""
        try:
            # Add to message cache to prevent loops
            self.message_cache[message.message_id] = message
            
            if target_node:
                # Unicast to specific node
                next_hop = self._find_route_to_node(target_node)
                if next_hop:
                    success = self._transmit_to_neighbor(message, next_hop)
                else:
                    # No route found, broadcast for route discovery
                    success = self._broadcast_message(message)
            else:
                # Broadcast to all neighbors
                success = self._broadcast_message(message)
            
            if success:
                self.messages_sent += 1
                logger.info(f"Message {message.message_id} sent successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def receive_message(self, raw_message: str) -> Optional[MeshMessage]:
        """Process received message from mesh network"""
        try:
            message_data = json.loads(raw_message)
            message = MeshMessage(**message_data)
            
            # Check if we've seen this message before (loop prevention)
            if message.message_id in self.message_cache:
                return None
            
            # Add to cache
            self.message_cache[message.message_id] = message
            self.messages_received += 1
            
            # Update neighbor information
            if message.sender_id in self.neighbors:
                self.neighbors[message.sender_id].last_seen = datetime.now().isoformat()
            
            # Check if message is for us
            if self._is_message_for_us(message):
                logger.info(f"Received message {message.message_id} from {message.sender_id}")
                return message
            
            # Forward message if TTL allows
            if message.ttl > 0:
                self._forward_message(message)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to process received message: {e}")
            return None
    
    def broadcast_sensor_data(self, sensor_data: Dict) -> bool:
        """Broadcast sensor data to mesh network"""
        message = MeshMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            message_type=MessageType.SENSOR_DATA,
            payload=sensor_data,
            timestamp=datetime.now().isoformat()
        )
        
        return self.send_message(message)
    
    def broadcast_crop_analysis(self, analysis_data: Dict) -> bool:
        """Broadcast crop analysis results"""
        message = MeshMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            message_type=MessageType.CROP_ANALYSIS,
            payload=analysis_data,
            timestamp=datetime.now().isoformat()
        )
        
        return self.send_message(message)
    
    def broadcast_alert(self, alert_data: Dict) -> bool:
        """Broadcast urgent agricultural alerts"""
        message = MeshMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            message_type=MessageType.ALERT,
            payload=alert_data,
            timestamp=datetime.now().isoformat(),
            ttl=10  # Alerts get higher TTL for wider propagation
        )
        
        return self.send_message(message)
    
    def get_network_status(self) -> Dict:
        """Get current mesh network status"""
        active_neighbors = [
            neighbor for neighbor in self.neighbors.values()
            if self._is_neighbor_active(neighbor)
        ]
        
        return {
            "node_id": self.node_id,
            "active_neighbors": len(active_neighbors),
            "total_neighbors": len(self.neighbors),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_forwarded": self.messages_forwarded,
            "network_health": self._calculate_network_health(),
            "last_discovery": self.last_discovery
        }
    
    def cleanup_old_messages(self, max_age_hours: int = 24):
        """Clean up old messages from cache"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        messages_to_remove = []
        for msg_id, message in self.message_cache.items():
            msg_time = datetime.fromisoformat(message.timestamp)
            if msg_time < cutoff_time:
                messages_to_remove.append(msg_id)
        
        for msg_id in messages_to_remove:
            del self.message_cache[msg_id]
        
        logger.info(f"Cleaned up {len(messages_to_remove)} old messages")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = str(time.time())
        content = f"{self.node_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _find_route_to_node(self, target_node: str) -> Optional[str]:
        """Find next hop to reach target node"""
        if target_node in self.neighbors:
            return target_node  # Direct neighbor
        
        # Check routing table
        if target_node in self.routing_table:
            return self.routing_table[target_node]
        
        # No known route
        return None
    
    def _transmit_to_neighbor(self, message: MeshMessage, neighbor_id: str) -> bool:
        """Simulate transmission to specific neighbor"""
        if neighbor_id not in self.neighbors:
            return False
        
        # Simulate transmission success/failure based on signal strength
        neighbor = self.neighbors[neighbor_id]
        success_probability = min(0.95, max(0.1, (neighbor.signal_strength + 100) / 60))
        
        import random
        success = random.random() < success_probability
        
        if success:
            # Update route path
            message.route_path.append(neighbor_id)
            logger.debug(f"Transmitted message to {neighbor_id}")
        
        return success
    
    def _broadcast_message(self, message: MeshMessage) -> bool:
        """Broadcast message to all active neighbors"""
        active_neighbors = [
            neighbor_id for neighbor_id, neighbor in self.neighbors.items()
            if self._is_neighbor_active(neighbor)
        ]
        
        if not active_neighbors:
            logger.warning("No active neighbors for broadcast")
            return False
        
        success_count = 0
        for neighbor_id in active_neighbors:
            if self._transmit_to_neighbor(message, neighbor_id):
                success_count += 1
        
        # Consider broadcast successful if at least 50% of neighbors received it
        return success_count >= len(active_neighbors) * 0.5
    
    def _forward_message(self, message: MeshMessage):
        """Forward message to other nodes"""
        if message.ttl <= 0:
            return
        
        # Decrease TTL and forward
        message.ttl -= 1
        message.route_path.append(self.node_id)
        
        # Don't forward back to sender or nodes already in route
        exclude_nodes = set(message.route_path)
        
        forward_candidates = [
            neighbor_id for neighbor_id in self.neighbors.keys()
            if neighbor_id not in exclude_nodes and 
               self._is_neighbor_active(self.neighbors[neighbor_id])
        ]
        
        if forward_candidates:
            # Forward to best neighbor (highest signal strength)
            best_neighbor = max(
                forward_candidates,
                key=lambda n: self.neighbors[n].signal_strength
            )
            
            if self._transmit_to_neighbor(message, best_neighbor):
                self.messages_forwarded += 1
                logger.debug(f"Forwarded message {message.message_id} to {best_neighbor}")
    
    def _is_message_for_us(self, message: MeshMessage) -> bool:
        """Check if message is intended for this node"""
        # For now, all messages are processed (broadcast nature)
        # In real implementation, could have targeted messages
        return True
    
    def _is_neighbor_active(self, neighbor: NodeInfo) -> bool:
        """Check if neighbor is considered active"""
        try:
            last_seen = datetime.fromisoformat(neighbor.last_seen)
            time_since_seen = datetime.now() - last_seen
            return time_since_seen.total_seconds() < 1800  # 30 minutes
        except:
            return False
    
    def _calculate_network_health(self) -> float:
        """Calculate overall network health score (0-1)"""
        if not self.neighbors:
            return 0.0
        
        active_neighbors = sum(1 for n in self.neighbors.values() if self._is_neighbor_active(n))
        neighbor_health = active_neighbors / len(self.neighbors)
        
        # Factor in message success rate
        total_messages = self.messages_sent + self.messages_received
        if total_messages > 0:
            message_health = (self.messages_sent + self.messages_received) / (total_messages + 1)
        else:
            message_health = 1.0
        
        return (neighbor_health + message_health) / 2

class FederatedLearningManager:
    """Manages federated learning across mesh network"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_model_updates = []
        self.received_updates = {}
        self.aggregation_threshold = 3  # Minimum updates needed for aggregation
        
    def add_local_update(self, model_update: Dict):
        """Add local model update"""
        update = {
            "node_id": self.node_id,
            "timestamp": datetime.now().isoformat(),
            "update_data": model_update,
            "sample_count": model_update.get("sample_count", 1)
        }
        self.local_model_updates.append(update)
        logger.info("Added local model update")
    
    def receive_model_update(self, sender_id: str, update_data: Dict):
        """Receive model update from another node"""
        self.received_updates[sender_id] = {
            "timestamp": datetime.now().isoformat(),
            "update_data": update_data,
            "sample_count": update_data.get("sample_count", 1)
        }
        logger.info(f"Received model update from {sender_id}")
    
    def should_aggregate(self) -> bool:
        """Check if we have enough updates to perform aggregation"""
        total_updates = len(self.local_model_updates) + len(self.received_updates)
        return total_updates >= self.aggregation_threshold
    
    def aggregate_updates(self) -> Optional[Dict]:
        """Aggregate model updates using federated averaging"""
        if not self.should_aggregate():
            return None
        
        # Simple federated averaging (weighted by sample count)
        total_samples = 0
        aggregated_weights = {}
        
        # Include local updates
        for update in self.local_model_updates:
            sample_count = update["sample_count"]
            total_samples += sample_count
            
            for param_name, param_value in update["update_data"].items():
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = 0
                aggregated_weights[param_name] += param_value * sample_count
        
        # Include received updates
        for update in self.received_updates.values():
            sample_count = update["sample_count"]
            total_samples += sample_count
            
            for param_name, param_value in update["update_data"].items():
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = 0
                aggregated_weights[param_name] += param_value * sample_count
        
        # Average the weights
        for param_name in aggregated_weights:
            aggregated_weights[param_name] /= total_samples
        
        # Clear processed updates
        self.local_model_updates.clear()
        self.received_updates.clear()
        
        logger.info(f"Aggregated model updates from {total_samples} samples")
        
        return {
            "aggregated_weights": aggregated_weights,
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat(),
            "aggregator_node": self.node_id
        }

if __name__ == "__main__":
    # Test mesh network functionality
    mesh = MeshNetworkManager("test_node_001")
    
    # Discover neighbors
    neighbors = mesh.discover_neighbors()
    print(f"Discovered {len(neighbors)} neighbors")
    
    # Test message broadcasting
    test_data = {
        "temperature": 25.5,
        "humidity": 65.0,
        "soil_moisture": 45.0
    }
    
    success = mesh.broadcast_sensor_data(test_data)
    print(f"Broadcast success: {success}")
    
    # Show network status
    status = mesh.get_network_status()
    print(f"Network status: {status}")