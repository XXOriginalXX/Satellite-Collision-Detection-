import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime, timedelta
import time
import 
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import queue
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
API_KEY = "Use Your Own API buddy3543BHD47"
BASE_URL = "https://api.n2yo.com/rest/v1/satellite"

@dataclass
class SatelliteData:
    satid: int
    name: str
    latitude: float
    longitude: float
    altitude: float
    azimuth: float
    elevation: float
    timestamp: int
    category: str = "Unknown"

@dataclass
class CollisionRisk:
    sat1_id: int
    sat2_id: int
    distance: float
    risk_score: float
    time_to_closest_approach: float
    predicted_miss_distance: float

class SatelliteAPI:
    """Enhanced N2YO API wrapper with rate limiting and error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str) -> Dict:
        """Make API request with rate limiting"""
        current_time = time.time()
        if current_time - self.last_request_time < 1:  # Rate limiting
            time.sleep(1 - (current_time - self.last_request_time))
        
        try:
            url = f"{self.base_url}/{endpoint}&apiKey={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self.request_count += 1
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            return {}
    
    def get_satellite_positions(self, sat_id: int, observer_lat: float, 
                              observer_lng: float, observer_alt: float, 
                              seconds: int = 300) -> List[SatelliteData]:
        """Get satellite positions for tracking"""
        endpoint = f"positions/{sat_id}/{observer_lat}/{observer_lng}/{observer_alt}/{seconds}"
        data = self._make_request(endpoint)
        
        positions = []
        if 'positions' in data:
            for pos in data['positions']:
                positions.append(SatelliteData(
                    satid=data['info']['satid'],
                    name=data['info']['satname'],
                    latitude=pos['satlatitude'],
                    longitude=pos['satlongitude'],
                    altitude=pos.get('sataltitude', 0),
                    azimuth=pos['azimuth'],
                    elevation=pos['elevation'],
                    timestamp=pos['timestamp']
                ))
        return positions
    
    def get_satellites_above(self, observer_lat: float, observer_lng: float,
                           observer_alt: float, search_radius: int = 90,
                           category_id: int = 0) -> List[SatelliteData]:
        """Get all satellites above observer location"""
        endpoint = f"above/{observer_lat}/{observer_lng}/{observer_alt}/{search_radius}/{category_id}"
        data = self._make_request(endpoint)
        
        satellites = []
        if 'above' in data:
            for sat in data['above']:
                satellites.append(SatelliteData(
                    satid=sat['satid'],
                    name=sat['satname'],
                    latitude=sat['satlat'],
                    longitude=sat['satlng'],
                    altitude=sat['satalt'],
                    azimuth=0,  # Not provided in 'above' endpoint
                    elevation=0,
                    timestamp=int(time.time()),
                    category=data['info'].get('category', 'Unknown')
                ))
        return satellites

class OrbitPredictor:
    """AI-powered orbital prediction system"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['timestamp', 'altitude', 'latitude', 'longitude', 
                               'azimuth', 'elevation']
    
    def prepare_features(self, positions: List[SatelliteData]) -> np.ndarray:
        """Extract features from satellite positions"""
        features = []
        for pos in positions:
            features.append([
                pos.timestamp,
                pos.altitude,
                pos.latitude,
                pos.longitude,
                pos.azimuth,
                pos.elevation
            ])
        return np.array(features)
    
    def train(self, historical_positions: List[List[SatelliteData]]):
        """Train the orbital prediction model"""
        if len(historical_positions) < 2:
            return
        
        X, y = [], []
        for sat_positions in historical_positions:
            if len(sat_positions) < 10:
                continue
            
            features = self.prepare_features(sat_positions[:-5])
            targets = self.prepare_features(sat_positions[5:])
            
            if features.shape[0] == targets.shape[0]:
                X.extend(features)
                y.extend(targets)
        
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            print(f"Orbital predictor trained on {len(X)} samples")
    
    def predict_future_positions(self, current_positions: List[SatelliteData], 
                               steps_ahead: int = 10) -> List[SatelliteData]:
        """Predict future satellite positions"""
        if not self.is_trained or len(current_positions) < 5:
            return []
        
        features = self.prepare_features(current_positions[-5:])
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        last_features = features_scaled[-1].reshape(1, -1)
        
        for step in range(steps_ahead):
            pred = self.model.predict(last_features)[0]
            
            predicted_pos = SatelliteData(
                satid=current_positions[0].satid,
                name=current_positions[0].name + "_PREDICTED",
                latitude=pred[2],
                longitude=pred[3],
                altitude=pred[1],
                azimuth=pred[4],
                elevation=pred[5],
                timestamp=int(pred[0]),
                category="PREDICTED"
            )
            predictions.append(predicted_pos)
            last_features = pred.reshape(1, -1)
        
        return predictions

class CollisionDetector:
    """AI-powered collision detection and risk assessment"""
    
    def __init__(self):
        self.risk_threshold = 100  # km
        self.critical_threshold = 10  # km
    
    def calculate_distance(self, sat1: SatelliteData, sat2: SatelliteData) -> float:
        """Calculate 3D distance between satellites"""
        # Convert to cartesian coordinates
        R = 6371  # Earth radius in km
        
        lat1, lon1, alt1 = np.radians(sat1.latitude), np.radians(sat1.longitude), sat1.altitude
        lat2, lon2, alt2 = np.radians(sat2.latitude), np.radians(sat2.longitude), sat2.altitude
        
        r1 = R + alt1
        r2 = R + alt2
        
        x1 = r1 * np.cos(lat1) * np.cos(lon1)
        y1 = r1 * np.cos(lat1) * np.sin(lon1)
        z1 = r1 * np.sin(lat1)
        
        x2 = r2 * np.cos(lat2) * np.cos(lon2)
        y2 = r2 * np.cos(lat2) * np.sin(lon2)
        z2 = r2 * np.sin(lat2)
        
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def assess_collision_risk(self, satellites: List[SatelliteData]) -> List[CollisionRisk]:
        """Assess collision risks between all satellite pairs"""
        risks = []
        
        for i in range(len(satellites)):
            for j in range(i+1, len(satellites)):
                sat1, sat2 = satellites[i], satellites[j]
                distance = self.calculate_distance(sat1, sat2)
                
                if distance < self.risk_threshold:
                    # Calculate risk score (higher = more dangerous)
                    risk_score = max(0, (self.risk_threshold - distance) / self.risk_threshold)
                    
                    risks.append(CollisionRisk(
                        sat1_id=sat1.satid,
                        sat2_id=sat2.satid,
                        distance=distance,
                        risk_score=risk_score,
                        time_to_closest_approach=abs(sat1.timestamp - sat2.timestamp),
                        predicted_miss_distance=distance  # Simplified
                    ))
        
        return sorted(risks, key=lambda x: x.risk_score, reverse=True)

class AnomalyDetector:
    """Detect anomalous satellite behavior using isolation forest"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, satellites: List[SatelliteData]) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = []
        for sat in satellites:
            features.append([
                sat.latitude,
                sat.longitude,
                sat.altitude,
                sat.azimuth,
                sat.elevation
            ])
        return np.array(features)
    
    def train(self, normal_data: List[List[SatelliteData]]):
        """Train anomaly detector on normal satellite behavior"""
        all_features = []
        for sat_list in normal_data:
            features = self.prepare_features(sat_list)
            all_features.extend(features)
        
        if len(all_features) > 10:
            X = np.array(all_features)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.is_trained = True
            print(f"Anomaly detector trained on {len(all_features)} samples")
    
    def detect_anomalies(self, satellites: List[SatelliteData]) -> List[Tuple[SatelliteData, float]]:
        """Detect anomalous satellites"""
        if not self.is_trained:
            return []
        
        features = self.prepare_features(satellites)
        features_scaled = self.scaler.transform(features)
        
        anomaly_scores = self.model.decision_function(features_scaled)
        predictions = self.model.predict(features_scaled)
        
        anomalies = []
        for i, (sat, pred, score) in enumerate(zip(satellites, predictions, anomaly_scores)):
            if pred == -1:  # Anomaly detected
                anomalies.append((sat, score))
        
        return sorted(anomalies, key=lambda x: x[1])

class ConstellationOptimizer:
    """AI-powered constellation optimization"""
    
    def __init__(self):
        self.coverage_threshold = 0.8
    
    def calculate_coverage(self, satellites: List[SatelliteData], 
                         ground_points: List[Tuple[float, float]]) -> float:
        """Calculate ground coverage percentage"""
        covered_points = 0
        
        for lat, lon in ground_points:
            for sat in satellites:
                # Simple coverage model - satellite covers ground within ~1000km
                distance = self._ground_distance(lat, lon, sat.latitude, sat.longitude)
                if distance < 1000:  # km
                    covered_points += 1
                    break
        
        return covered_points / len(ground_points) if ground_points else 0
    
    def _ground_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate ground distance between two points"""
        R = 6371  # Earth radius
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
             np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def optimize_constellation(self, satellites: List[SatelliteData]) -> Dict:
        """Analyze and suggest constellation improvements"""
        # Generate test ground points
        ground_points = [(lat, lon) for lat in range(-60, 61, 10) 
                        for lon in range(-180, 180, 20)]
        
        current_coverage = self.calculate_coverage(satellites, ground_points)
        
        # Cluster analysis
        if len(satellites) > 3:
            coords = np.array([[sat.latitude, sat.longitude] for sat in satellites])
            clustering = DBSCAN(eps=30, min_samples=2).fit(coords)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            n_clusters = 1
        
        return {
            'current_coverage': current_coverage,
            'cluster_count': n_clusters,
            'recommendations': self._generate_recommendations(current_coverage, n_clusters, len(satellites)),
            'satellite_count': len(satellites)
        }
    
    def _generate_recommendations(self, coverage: float, clusters: int, satellite_count: int) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if coverage < 0.5:
            recommendations.append("Low coverage detected - consider adding more satellites")
        if clusters > satellite_count * 0.3:
            recommendations.append("High clustering detected - redistribute satellites for better coverage")
        if coverage > 0.9:
            recommendations.append("Excellent coverage - constellation well optimized")
        
        return recommendations

class SatelliteIntelligenceSystem:
    """Main AI-powered satellite intelligence system"""
    
    def __init__(self, api_key: str, observer_location: Tuple[float, float, float]):
        self.api = SatelliteAPI(api_key)
        self.observer_lat, self.observer_lng, self.observer_alt = observer_location
        
        # AI Components
        self.orbit_predictor = OrbitPredictor()
        self.collision_detector = CollisionDetector()
        self.anomaly_detector = AnomalyDetector()
        self.constellation_optimizer = ConstellationOptimizer()
        
        # Data storage
        self.satellite_history = {}
        self.current_satellites = []
        self.collision_risks = []
        self.anomalies = []
        
        # Threading
        self.data_queue = queue.Queue()
        self.is_running = False
    
    def start_monitoring(self):
        """Start real-time satellite monitoring"""
        self.is_running = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        print("Satellite monitoring started...")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get current satellites
                satellites = self.api.get_satellites_above(
                    self.observer_lat, self.observer_lng, self.observer_alt
                )
                
                if satellites:
                    self.current_satellites = satellites
                    self._update_history(satellites)
                    self._analyze_data()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _update_history(self, satellites: List[SatelliteData]):
        """Update satellite position history"""
        for sat in satellites:
            if sat.satid not in self.satellite_history:
                self.satellite_history[sat.satid] = []
            
            self.satellite_history[sat.satid].append(sat)
            
            # Keep only recent history (last 100 positions)
            if len(self.satellite_history[sat.satid]) > 100:
                self.satellite_history[sat.satid] = self.satellite_history[sat.satid][-100:]
    
    def _analyze_data(self):
        """Perform AI analysis on current data"""
        if not self.current_satellites:
            return
        
        # Train models if enough data
        if len(self.satellite_history) > 5:
            history_data = list(self.satellite_history.values())
            self.orbit_predictor.train(history_data)
            self.anomaly_detector.train(history_data)
        
        # Collision detection
        self.collision_risks = self.collision_detector.assess_collision_risk(self.current_satellites)
        
        # Anomaly detection
        if self.anomaly_detector.is_trained:
            self.anomalies = self.anomaly_detector.detect_anomalies(self.current_satellites)
        
        # Print alerts
        self._print_alerts()
    
    def _print_alerts(self):
        """Print important alerts"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Critical collision risks
        critical_risks = [r for r in self.collision_risks if r.distance < 50]
        if critical_risks:
            print(f"\n🚨 CRITICAL COLLISION ALERT [{current_time}]")
            for risk in critical_risks[:3]:
                print(f"  Satellites {risk.sat1_id} & {risk.sat2_id}: {risk.distance:.1f}km apart")
        
        # Anomalies
        if self.anomalies:
            print(f"\n⚠️  ANOMALY DETECTED [{current_time}]")
            for sat, score in self.anomalies[:3]:
                print(f"  {sat.name} (ID: {sat.satid}): Anomaly score {score:.3f}")
    
    def generate_intelligence_report(self) -> Dict:
        """Generate comprehensive intelligence report"""
        if not self.current_satellites:
            return {"error": "No current satellite data available"}
        
        # Constellation analysis
        constellation_analysis = self.constellation_optimizer.optimize_constellation(
            self.current_satellites
        )
        
        # Predict future positions
        predictions = {}
        for sat_id, history in self.satellite_history.items():
            if len(history) >= 10:
                future_pos = self.orbit_predictor.predict_future_positions(history, 5)
                if future_pos:
                    predictions[sat_id] = future_pos
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'observer_location': {
                'latitude': self.observer_lat,
                'longitude': self.observer_lng,
                'altitude': self.observer_alt
            },
            'current_satellites': {
                'count': len(self.current_satellites),
                'categories': list(set(sat.category for sat in self.current_satellites))
            },
            'collision_risks': {
                'total_risks': len(self.collision_risks),
                'critical_risks': len([r for r in self.collision_risks if r.distance < 50]),
                'highest_risk': self.collision_risks[0].__dict__ if self.collision_risks else None
            },
            'anomalies': {
                'count': len(self.anomalies),
                'satellites': [{'id': sat.satid, 'name': sat.name, 'score': score} 
                              for sat, score in self.anomalies[:5]]
            },
            'constellation_analysis': constellation_analysis,
            'predictions': {
                'satellites_predicted': len(predictions),
                'models_trained': {
                    'orbit_predictor': self.orbit_predictor.is_trained,
                    'anomaly_detector': self.anomaly_detector.is_trained
                }
            },
            'api_stats': {
                'requests_made': self.api.request_count
            }
        }
        
        return report
    
    def visualize_satellite_network(self):
        """Create visualization of satellite network"""
        if not self.current_satellites:
            print("No satellite data to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI-Powered Satellite Intelligence Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Global satellite positions
        lats = [sat.latitude for sat in self.current_satellites]
        lons = [sat.longitude for sat in self.current_satellites]
        alts = [sat.altitude for sat in self.current_satellites]
        
        scatter = ax1.scatter(lons, lats, c=alts, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Current Satellite Positions')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Altitude (km)')
        
        # 2. Altitude distribution
        ax2.hist(alts, bins=20, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Altitude (km)')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_title('Altitude Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Collision risk heatmap
        if self.collision_risks:
            risk_matrix = np.zeros((min(10, len(self.current_satellites)), 
                                  min(10, len(self.current_satellites))))
            for risk in self.collision_risks[:50]:  # Top 50 risks
                if risk.sat1_id < 10 and risk.sat2_id < 10:
                    risk_matrix[risk.sat1_id % 10][risk.sat2_id % 10] = risk.risk_score
            
            sns.heatmap(risk_matrix, ax=ax3, cmap='Reds', annot=True, fmt='.2f')
            ax3.set_title('Collision Risk Matrix (Top Satellites)')
        else:
            ax3.text(0.5, 0.5, 'No collision risks detected', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Collision Risk Matrix')
        
        # 4. Category distribution
        categories = [sat.category for sat in self.current_satellites]
        category_counts = pd.Series(categories).value_counts()
        
        if len(category_counts) > 0:
            ax4.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            ax4.set_title('Satellite Categories')
        else:
            ax4.text(0.5, 0.5, 'No category data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def export_data(self, filename: str):
        """Export data to JSON file"""
        report = self.generate_intelligence_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Intelligence report exported to {filename}")

def main():
    """Main execution function"""
    print("🛰️  AI-Powered Satellite Constellation Intelligence System")
    print("=" * 60)
    
    # Configuration - Default location (adjust as needed)
    observer_location = (37.7749, -122.4194, 0)  # San Francisco
    
    # Initialize system
    system = SatelliteIntelligenceSystem(API_KEY, observer_location)
    
    try:
        # Start monitoring
        system.start_monitoring()
        
        # Wait for initial data
        print("Collecting initial satellite data...")
        time.sleep(45)
        
        # Generate and display report
        print("\n📊 Generating Intelligence Report...")
        report = system.generate_intelligence_report()
        
        print(f"\n🛰️  SATELLITE INTELLIGENCE SUMMARY")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"Current Satellites Tracked: {report['current_satellites']['count']}")
        print(f"Collision Risks Identified: {report['collision_risks']['total_risks']}")
        print(f"Critical Risks: {report['collision_risks']['critical_risks']}")
        print(f"Anomalies Detected: {report['anomalies']['count']}")
        print(f"Coverage Analysis: {report['constellation_analysis']['current_coverage']:.1%}")
        
        if report['collision_risks']['highest_risk']:
            hr = report['collision_risks']['highest_risk']
            print(f"\n🚨 HIGHEST RISK:")
            print(f"   Satellites: {hr['sat1_id']} & {hr['sat2_id']}")
            print(f"   Distance: {hr['distance']:.1f} km")
            print(f"   Risk Score: {hr['risk_score']:.3f}")
        
        # Show recommendations
        recommendations = report['constellation_analysis']['recommendations']
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Visualizations
        print("\n📈 Generating visualizations...")
        system.visualize_satellite_network()
        
        # Export data
        filename = f"satellite_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        system.export_data(filename)
        
        print(f"\n✅ Analysis complete! Check {filename} for detailed data.")
        print("\nThis system demonstrates:")
        print("• Real-time satellite tracking and monitoring")
        print("• AI-powered collision detection and risk assessment")
        print("• Machine learning orbital prediction")
        print("• Anomaly detection using isolation forests")
        print("• Constellation optimization analysis")
        print("• Space situational awareness capabilities")
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down system...")
    finally:
        system.stop_monitoring()

if __name__ == "__main__":
    main()
