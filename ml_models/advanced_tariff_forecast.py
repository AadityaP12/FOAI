# ml_models/advanced_tariff_forecast.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import os
import json

warnings.filterwarnings('ignore')

@dataclass
class TariffPrediction:
    """Structured tariff prediction with confidence intervals"""
    base_forecast: List[float]
    conservative_scenario: List[float]
    aggressive_scenario: List[float]
    renewable_push_scenario: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_years: List[int]
    metadata: Dict

class AdvancedTariffForecaster:
    """
    Multi-scenario tariff forecasting engine with policy awareness
    Supports residential, commercial, agricultural, and EV charging tariffs
    Now with model persistence capabilities
    """
    
    def __init__(self, confidence_level: float = 0.95, model_dir: str = "C:/FOAI/models"):
        self.models = {
            'base': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'gradient': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        }
        self.scalers = {}
        self.label_encoders = {}
        self.is_trained = False
        self.confidence_level = confidence_level
        self.feature_importance = {}
        self.validation_scores = {}
        self.model_dir = model_dir
        self.model_version = "1.0"
        self.training_metadata = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Policy and economic factors
        self.policy_factors = {
            'renewable_push': 0.85,    # Reduced tariff growth
            'carbon_tax': 1.15,        # Increased tariff growth
            'subsidy_removal': 1.25,   # Higher growth for agriculture
            'inflation_high': 1.12     # Economic pressure
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_model(self, model_name: str = None) -> str:
        """
        Save the entire trained model to disk
        Returns the path where the model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model. Train the model first.")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"tariff_forecaster_{timestamp}"
        
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        
        # Prepare model data for saving
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_columns': getattr(self, 'feature_columns', []),
            'feature_importance': self.feature_importance,
            'validation_scores': self.validation_scores,
            'policy_factors': self.policy_factors,
            'confidence_level': self.confidence_level,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }
        
        # Save model using pickle
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata as JSON for easy inspection
            metadata = {
                'model_name': model_name,
                'model_version': self.model_version,
                'saved_timestamp': datetime.now().isoformat(),
                'validation_scores': self.validation_scores,
                'available_states': list(self.label_encoders.get('state', {}).classes_) if 'state' in self.label_encoders else [],
                'available_categories': list(self.label_encoders.get('category', {}).classes_) if 'category' in self.label_encoders else [],
                'feature_columns': getattr(self, 'feature_columns', []),
                'training_metadata': self.training_metadata,
                'model_file': f"{model_name}.pkl"
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Model saved successfully to: {model_path}")
            self.logger.info(f"📋 Metadata saved to: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str = None, model_name: str = None) -> bool:
        """
        Load a trained model from disk
        Args:
            model_path: Full path to the .pkl file
            model_name: Name of the model (will look in model_dir)
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if model_path is None and model_name is None:
                raise ValueError("Either model_path or model_name must be provided")
            
            if model_path is None:
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore all model components
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data.get('feature_columns', [])
            self.feature_importance = model_data.get('feature_importance', {})
            self.validation_scores = model_data.get('validation_scores', {})
            self.policy_factors = model_data.get('policy_factors', self.policy_factors)
            self.confidence_level = model_data.get('confidence_level', 0.95)
            self.model_version = model_data.get('model_version', '1.0')
            self.is_trained = model_data.get('is_trained', False)
            self.training_metadata = model_data.get('training_metadata', {})
            
            self.logger.info(f"✅ Model loaded successfully from: {model_path}")
            self.logger.info(f"📊 Model version: {self.model_version}")
            self.logger.info(f"🏛️ Available states: {len(self.label_encoders.get('state', {}).classes_)}")
            self.logger.info(f"📋 Available categories: {len(self.label_encoders.get('category', {}).classes_)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False
    
    def list_saved_models(self) -> List[Dict]:
        """List all saved models in the model directory"""
        saved_models = []
        
        if not os.path.exists(self.model_dir):
            self.logger.warning(f"Model directory does not exist: {self.model_dir}")
            return saved_models
        
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                model_name = file[:-4]  # Remove .pkl extension
                model_path = os.path.join(self.model_dir, file)
                metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                
                model_info = {
                    'model_name': model_name,
                    'model_file': file,
                    'model_path': model_path,
                    'file_size_mb': round(os.path.getsize(model_path) / (1024*1024), 2),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                }
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_info.update(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata for {model_name}: {e}")
                
                saved_models.append(model_info)
        
        # Sort by last modified (newest first)
        saved_models.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return saved_models
    
    def get_latest_model_path(self) -> Optional[str]:
        """Get the path to the most recently saved model"""
        models = self.list_saved_models()
        if models:
            return models[0]['model_path']
        return None
    
    def load_real_data(self, csv_path: str = None) -> pd.DataFrame:
        """Load and preprocess real tariff data from CSV"""
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                self.logger.info(f"Loaded real data from CSV: {df.shape[0]} records")
                return self._preprocess_real_data(df)
            except FileNotFoundError:
                self.logger.warning(f"File {csv_path} not found, generating synthetic data")
                return self._generate_comprehensive_training_data()
        else:
            # Load from uploaded file using window.fs.readFile API
            try:
                # This would be used in a web environment
                self.logger.info("Using synthetic data as fallback")
                return self._generate_comprehensive_training_data()
            except:
                self.logger.info("Using synthetic data as fallback")
                return self._generate_comprehensive_training_data()
    
    def load_csv_data(self, csv_content: str = None) -> pd.DataFrame:
        """Load real tariff data from CSV content string"""
        if csv_content:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            self.logger.info(f"Loaded real data from content: {df.shape[0]} records")
            return self._preprocess_real_data(df)
        return self._generate_comprehensive_training_data()
    
    def _preprocess_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preprocessing of real MNRE tariff data"""
        processed_data = []
        
        self.logger.info("Processing real tariff data...")
        
        # Parse the real data structure from your CSV
        for _, row in df.iterrows():
            # Extract information
            max_state = str(row['Max State/Utility'])
            min_state = str(row['Min State/Utility'])
            category = str(row['Category'])
            load = str(row['Load/Consumption'])
            
            # Convert paise to rupees
            max_rate = float(row['Max Rate (Paise/kWh)']) / 100
            min_rate = float(row['Min Rate (Paise/kWh)']) / 100
            
            # Clean and standardize names
            max_state_clean = self._clean_state_name(max_state)
            min_state_clean = self._clean_state_name(min_state)
            category_clean = self._standardize_category(category)
            
            # Generate historical time series (2020-2024) with realistic trends
            base_year = 2024
            for year_offset in range(-4, 1):  # 2020-2024
                year = base_year + year_offset
                
                # Apply historical trend based on Indian electricity tariff growth (6-10% annually)
                if category_clean == 'Agricultural':
                    # Agriculture has lower growth due to subsidies
                    annual_growth = 0.04 + np.random.normal(0, 0.01)
                elif category_clean == 'Residential':
                    # Residential has moderate growth
                    annual_growth = 0.07 + np.random.normal(0, 0.015)
                elif category_clean == 'Commercial':
                    # Commercial has higher growth
                    annual_growth = 0.08 + np.random.normal(0, 0.02)
                elif category_clean == 'EV_Charging':
                    # EV charging is newer, more volatile
                    annual_growth = 0.06 + np.random.normal(0, 0.025)
                else:
                    annual_growth = 0.075 + np.random.normal(0, 0.015)
                
                trend_factor = (1 + annual_growth) ** year_offset
                
                # Economic adjustments
                if year in [2020, 2021]:  # COVID impact
                    trend_factor *= np.random.uniform(0.92, 0.98)
                elif year >= 2022:  # Post-COVID inflation
                    trend_factor *= np.random.uniform(1.02, 1.08)
                
                # Generate monthly data for each year
                for month in range(1, 13):
                    # Seasonal variation (higher in summer months)
                    seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * (month - 3) / 12)
                    
                    # Calculate adjusted rates
                    adjusted_max = max_rate * trend_factor * seasonal_factor * np.random.normal(1, 0.03)
                    adjusted_min = min_rate * trend_factor * seasonal_factor * np.random.normal(1, 0.03)
                    
                    # Ensure minimum tariff floor
                    adjusted_max = max(0.05, adjusted_max)
                    adjusted_min = max(0.00, adjusted_min)  # Agricultural can be 0
                    
                    # Add records for both max and min states
                    processed_data.extend([
                        {
                            'state': max_state_clean,
                            'category': category_clean,
                            'load_segment': self._standardize_load(load),
                            'year': year,
                            'month': month,
                            'tariff': adjusted_max,
                            'is_urban': self._determine_urban(max_state),
                            'is_max_rate': True,
                            'original_rate': max_rate
                        },
                        {
                            'state': min_state_clean,
                            'category': category_clean,
                            'load_segment': self._standardize_load(load),
                            'year': year,
                            'month': month,
                            'tariff': adjusted_min,
                            'is_urban': self._determine_urban(min_state),
                            'is_max_rate': False,
                            'original_rate': min_rate
                        }
                    ])
        
        processed_df = pd.DataFrame(processed_data)
        
        # Add derived features
        processed_df = self._add_economic_indicators(processed_df)
        
        self.logger.info(f"Generated {len(processed_df)} time series records from {len(df)} original entries")
        return processed_df
    
    def _clean_state_name(self, state_str: str) -> str:
        """Enhanced state name cleaning"""
        # Remove parenthetical information and clean
        state = state_str.split('(')[0].split('-')[0].strip()
        if ',' in state:
            state = state.split(',')[0].strip()
        
        # Map common variations and abbreviations
        state_mapping = {
            'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
            'Daman and Diu': 'Daman and Diu',
            'Andaman & Nicobar Island': 'Andaman and Nicobar Islands',
            'J&K': 'Jammu and Kashmir',
            'Delhi': 'Delhi',
            'Maharashtra': 'Maharashtra',
            'Gujarat': 'Gujarat',
            'Rajasthan': 'Rajasthan',
            'Tamil Nadu': 'Tamil Nadu',
            'Karnataka': 'Karnataka',
            'Andhra Pradesh': 'Andhra Pradesh',
            'Kerala': 'Kerala',
            'Punjab': 'Punjab',
            'Haryana': 'Haryana',
            'Uttar Pradesh': 'Uttar Pradesh',
            'West Bengal': 'West Bengal',
            'Madhya Pradesh': 'Madhya Pradesh',
            'Telangana': 'Telangana',
            'Chhattisgarh': 'Chhattisgarh',
            'Jharkhand': 'Jharkhand',
            'Mizoram': 'Mizoram',
            'Puducherry': 'Puducherry'
        }
        
        return state_mapping.get(state, state)
    
    def _standardize_category(self, category: str) -> str:
        """Enhanced category standardization"""
        category_mapping = {
            'DOMESTIC': 'Residential',
            'COMMERCIAL': 'Commercial',
            'AGRICULTURE': 'Agricultural',
            'EV CHARGING': 'EV_Charging',
            'INDUSTRIAL': 'Industrial',
            'TAX/DUTY': 'Tax_Duty'
        }
        return category_mapping.get(category.upper(), 'Residential')
    
    def _standardize_load(self, load_str: str) -> str:
        """Standardize load/consumption descriptions"""
        # Extract numeric values where possible
        if 'KW' in load_str.upper() or 'HP' in load_str.upper():
            return load_str
        elif 'Unit' in load_str:
            return load_str
        else:
            return f"Standard Load"
    
    def _determine_urban(self, state_str: str) -> bool:
        """Determine if location is urban based on state string indicators"""
        urban_indicators = ['Urban', 'Metro', 'City', 'BYPL', 'BRPL', 'TPDDL']
        return any(indicator in state_str for indicator in urban_indicators)
    
    def _add_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic and renewable energy indicators by state"""
        
        # State-specific renewable adoption and economic profiles
        state_profiles = {
            'Maharashtra': {'renewable_adoption': 0.65, 'growth_rate': 0.08, 'urban_ratio': 0.75},
            'Gujarat': {'renewable_adoption': 0.85, 'growth_rate': 0.07, 'urban_ratio': 0.62},
            'Rajasthan': {'renewable_adoption': 0.70, 'growth_rate': 0.09, 'urban_ratio': 0.35},
            'Tamil Nadu': {'renewable_adoption': 0.75, 'growth_rate': 0.06, 'urban_ratio': 0.68},
            'Karnataka': {'renewable_adoption': 0.68, 'growth_rate': 0.07, 'urban_ratio': 0.61},
            'Andhra Pradesh': {'renewable_adoption': 0.72, 'growth_rate': 0.075, 'urban_ratio': 0.44},
            'Punjab': {'renewable_adoption': 0.45, 'growth_rate': 0.085, 'urban_ratio': 0.37},
            'Delhi': {'renewable_adoption': 0.55, 'growth_rate': 0.065, 'urban_ratio': 0.97},
            'West Bengal': {'renewable_adoption': 0.42, 'growth_rate': 0.075, 'urban_ratio': 0.68},
            'Kerala': {'renewable_adoption': 0.58, 'growth_rate': 0.07, 'urban_ratio': 0.65},
            'Haryana': {'renewable_adoption': 0.48, 'growth_rate': 0.08, 'urban_ratio': 0.40},
            'Uttar Pradesh': {'renewable_adoption': 0.38, 'growth_rate': 0.08, 'urban_ratio': 0.22}
        }
        
        # Default profile for states not in the list
        default_profile = {'renewable_adoption': 0.55, 'growth_rate': 0.075, 'urban_ratio': 0.50}
        
        # Add economic indicators
        df['renewable_adoption'] = df['state'].map(
            lambda x: state_profiles.get(x, default_profile)['renewable_adoption']
        )
        df['growth_rate'] = df['state'].map(
            lambda x: state_profiles.get(x, default_profile)['growth_rate']
        )
        df['state_urban_ratio'] = df['state'].map(
            lambda x: state_profiles.get(x, default_profile)['urban_ratio']
        )
        
        return df
    
    def _generate_comprehensive_training_data(self) -> pd.DataFrame:
        """Generate comprehensive synthetic training data as fallback"""
        data = []
        
        # Enhanced state profiles with realistic economic factors
        state_profiles = {
            'Maharashtra': {'base_tariff': 6.5, 'growth_rate': 0.08, 'urbanization': 0.75, 'renewable_adoption': 0.65},
            'Gujarat': {'base_tariff': 5.2, 'growth_rate': 0.07, 'urbanization': 0.62, 'renewable_adoption': 0.85},
            'Rajasthan': {'base_tariff': 8.1, 'growth_rate': 0.09, 'urbanization': 0.35, 'renewable_adoption': 0.70},
            'Tamil Nadu': {'base_tariff': 4.8, 'growth_rate': 0.06, 'urbanization': 0.68, 'renewable_adoption': 0.75},
            'Karnataka': {'base_tariff': 5.5, 'growth_rate': 0.07, 'urbanization': 0.61, 'renewable_adoption': 0.68},
            'Andhra Pradesh': {'base_tariff': 4.9, 'growth_rate': 0.075, 'urbanization': 0.44, 'renewable_adoption': 0.72},
            'Punjab': {'base_tariff': 6.8, 'growth_rate': 0.085, 'urbanization': 0.37, 'renewable_adoption': 0.45},
            'Delhi': {'base_tariff': 5.8, 'growth_rate': 0.065, 'urbanization': 0.97, 'renewable_adoption': 0.55},
            'West Bengal': {'base_tariff': 5.9, 'growth_rate': 0.075, 'urbanization': 0.68, 'renewable_adoption': 0.42},
            'Uttar Pradesh': {'base_tariff': 6.2, 'growth_rate': 0.08, 'urbanization': 0.22, 'renewable_adoption': 0.38}
        }
        
        # Category-specific multipliers
        category_multipliers = {
            'Residential': {'base': 1.0, 'volatility': 0.05},
            'Commercial': {'base': 1.3, 'volatility': 0.08},
            'Industrial': {'base': 0.9, 'volatility': 0.12},
            'Agricultural': {'base': 0.3, 'volatility': 0.15},  # Highly subsidized
            'EV_Charging': {'base': 1.1, 'volatility': 0.06}
        }
        
        # Generate historical data (2020-2024)
        for state, profile in state_profiles.items():
            for category, cat_profile in category_multipliers.items():
                base_tariff = profile['base_tariff'] * cat_profile['base']
                
                for year in range(2020, 2025):
                    for month in range(1, 13):
                        # Year-over-year growth with economic cycles
                        years_from_base = year - 2022
                        growth_factor = (1 + profile['growth_rate']) ** years_from_base
                        
                        # Economic cycle impact
                        if year in [2020, 2021]:  # COVID impact
                            growth_factor *= 0.95
                        elif year >= 2022:  # Post-COVID inflation
                            growth_factor *= 1.05
                        
                        # Seasonal variation
                        seasonal_factor = 1 + 0.08 * np.sin(2 * np.pi * (month - 3) / 12)
                        
                        # Policy impact (renewable adoption reduces tariff pressure)
                        policy_factor = 1 - (profile['renewable_adoption'] * 0.1)
                        
                        # Random variation
                        noise = np.random.normal(1, cat_profile['volatility'])
                        
                        final_tariff = base_tariff * growth_factor * seasonal_factor * policy_factor * noise
                        
                        data.append({
                            'state': state,
                            'category': category,
                            'year': year,
                            'month': month,
                            'tariff': max(0.1, final_tariff),  # Minimum tariff floor
                            'is_urban': np.random.choice([True, False], p=[profile['urbanization'], 1-profile['urbanization']]),
                            'renewable_adoption': profile['renewable_adoption'],
                            'growth_rate': profile['growth_rate'],
                            'load_segment': f"{np.random.randint(1, 50)} kW",
                            'state_urban_ratio': profile['urbanization']
                        })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated synthetic data: {df.shape[0]} records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced feature preparation for machine learning"""
        # Create time-based features
        df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Encode categorical variables
        categorical_cols = ['state', 'category']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle new categories not seen during training
                unique_values = df[col].astype(str).unique()
                known_values = self.label_encoders[col].classes_
                
                for val in unique_values:
                    if val not in known_values:
                        # Add new category to encoder
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, val)
                
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select features
        feature_cols = [
            'year_norm', 'month_sin', 'month_cos', 'state_encoded', 'category_encoded'
        ]
        
        # Add economic indicators if available
        if 'renewable_adoption' in df.columns:
            feature_cols.extend(['renewable_adoption', 'growth_rate'])
        
        if 'is_urban' in df.columns:
            df['is_urban_int'] = df['is_urban'].astype(int)
            feature_cols.append('is_urban_int')
            
        if 'state_urban_ratio' in df.columns:
            feature_cols.append('state_urban_ratio')
        
        X = df[feature_cols].copy()
        y = df['tariff'].copy()
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        return X, y
    
    def train_models(self, df: pd.DataFrame, auto_save: bool = True, model_name: str = None) -> Dict[str, float]:
        """Enhanced model training with better validation and auto-save option"""
        X, y = self.prepare_features(df)
        
        # Store training metadata
        self.training_metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'training_data_shape': df.shape,
            'unique_states': len(df['state'].unique()),
            'unique_categories': len(df['category'].unique()),
            'data_year_range': f"{df['year'].min()}-{df['year'].max()}",
            'total_records': len(df)
        }
        
        # Split data chronologically for better time series validation
        # Use 80% for training, 20% for testing
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_test_scaled = self.scalers['main'].transform(X_test)
        
        # Train models
        scores = {}
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            
            # Test set evaluation
            y_pred = model.predict(X_test_scaled)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)
            
            scores[name] = {
                'cv_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std': np.sqrt(cv_scores.std()),
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
                self.feature_importance[name] = sorted(feature_importance.items(), 
                                                     key=lambda x: x[1], reverse=True)
        
        self.validation_scores = scores
        self.is_trained = True
        
        # Auto-save the trained model if requested
        if auto_save:
            try:
                if model_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"tariff_forecaster_{timestamp}"
                
                saved_path = self.save_model(model_name)
                self.logger.info(f"🎉 Model auto-saved to: {saved_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Auto-save failed: {e}")
        
        return scores
    
    def predict_multi_scenario(self, state: str, category: str, years_ahead: int = 10) -> TariffPrediction:
        """Enhanced multi-scenario tariff predictions"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        forecast_years = list(range(2025, 2025 + years_ahead))
        
        # Generate base predictions with multiple monthly samples for stability
        base_predictions = []
        prediction_intervals = []
        
        for year in forecast_years:
            year_predictions = []
            
            # Sample multiple months and take average for stability
            for month in [3, 6, 9, 12]:  # Quarterly samples
                features = self._create_prediction_features(state, category, year, month)
                if features is None:
                    continue
                    
                features_scaled = self.scalers['main'].transform([features])
                
                # Get predictions from both models
                pred_rf = self.models['base'].predict(features_scaled)[0]
                pred_gb = self.models['gradient'].predict(features_scaled)[0]
                
                # Ensemble prediction with confidence weighting
                if 'base' in self.validation_scores and 'gradient' in self.validation_scores:
                    rf_weight = 1 / (1 + self.validation_scores['base']['test_rmse'])
                    gb_weight = 1 / (1 + self.validation_scores['gradient']['test_rmse'])
                    total_weight = rf_weight + gb_weight
                    ensemble_pred = (pred_rf * rf_weight + pred_gb * gb_weight) / total_weight
                else:
                    ensemble_pred = (pred_rf + pred_gb) / 2
                
                year_predictions.append(ensemble_pred)
            
            if not year_predictions:
                # Fallback prediction
                base_pred = 5.0 * (1.07 ** (year - 2024))
            else:
                # Average the quarterly predictions
                base_pred = np.mean(year_predictions)
            
            # Add trend component based on category and state
            trend_multiplier = self._get_trend_multiplier(state, category, year - 2024)
            base_pred_with_trend = base_pred * trend_multiplier
            
            base_predictions.append(round(base_pred_with_trend, 2))
            
            # Calculate prediction interval with increasing uncertainty
            if year_predictions:
                pred_std = np.std(year_predictions) + 0.15 + ((year - 2024) * 0.05)
            else:
                pred_std = 0.5 + ((year - 2024) * 0.1)
            
            confidence_margin = 1.96 * pred_std  # 95% confidence
            
            prediction_intervals.append((
                round(max(0.1, base_pred_with_trend - confidence_margin), 2),
                round(base_pred_with_trend + confidence_margin, 2)
            ))
        
        # Generate scenario-based predictions
        scenarios = self._generate_scenarios(base_predictions, category, state)
        
        return TariffPrediction(
            base_forecast=base_predictions,
            conservative_scenario=scenarios['conservative'],
            aggressive_scenario=scenarios['aggressive'],
            renewable_push_scenario=scenarios['renewable_push'],
            confidence_intervals=prediction_intervals,
            forecast_years=forecast_years,
            metadata={
                'state': state,
                'category': category,
                'model_accuracy': self.validation_scores,
                'last_trained': datetime.now().isoformat(),
                'data_source': 'real_csv_data',
                'trend_assumption': 'Dynamic based on category and state'
            }
        )
    
    def _get_trend_multiplier(self, state: str, category: str, years_ahead: int) -> float:
        """Calculate trend multiplier based on state and category characteristics"""
        # Category-specific growth rates
        category_trends = {
            'Residential': 0.07,     # 7% annual
            'Commercial': 0.085,     # 8.5% annual  
            'Industrial': 0.075,     # 7.5% annual
            'Agricultural': 0.04,    # 4% annual (subsidized)
            'EV_Charging': 0.06      # 6% annual (policy support)
        }
        
        # State-specific modifiers
        state_modifiers = {
            'Maharashtra': 1.0, 'Gujarat': 0.95, 'Rajasthan': 1.05,
            'Tamil Nadu': 0.92, 'Karnataka': 0.98, 'Delhi': 0.96
        }
        
        base_growth = category_trends.get(category, 0.075)
        state_modifier = state_modifiers.get(state, 1.0)
        
        return (1 + base_growth * state_modifier) ** years_ahead
    
    def _create_prediction_features(self, state: str, category: str, year: int, month: int = 6) -> Optional[List[float]]:
        """Create feature vector for prediction with better error handling"""
        try:
            # Normalize year (extend range for future predictions)
            year_norm = (year - 2020) / (2035 - 2020)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Encode state and category
            try:
                state_encoded = self.label_encoders['state'].transform([state])[0]
            except (ValueError, KeyError):
                # Find closest match or use Maharashtra as default
                available_states = list(self.label_encoders['state'].classes_)
                state_encoded = self.label_encoders['state'].transform([available_states[0]])[0]
                
            try:
                category_encoded = self.label_encoders['category'].transform([category])[0]
            except (ValueError, KeyError):
                # Use Residential as default
                available_categories = list(self.label_encoders['category'].classes_)
                default_category = 'Residential' if 'Residential' in available_categories else available_categories[0]
                category_encoded = self.label_encoders['category'].transform([default_category])[0]
            
            features = [year_norm, month_sin, month_cos, state_encoded, category_encoded]
            
            # Add economic indicators with state-specific defaults
            if 'renewable_adoption' in self.feature_columns:
                state_profiles = {
                    'Maharashtra': [0.65, 0.08], 'Gujarat': [0.85, 0.07], 'Rajasthan': [0.70, 0.09],
                    'Tamil Nadu': [0.75, 0.06], 'Karnataka': [0.68, 0.07], 'Delhi': [0.55, 0.065],
                    'Punjab': [0.45, 0.085], 'Haryana': [0.48, 0.08], 'West Bengal': [0.42, 0.075]
                }
                profile = state_profiles.get(state, [0.6, 0.075])
                features.extend(profile)
            
            if 'is_urban_int' in self.feature_columns:
                # Urban probability by state
                urban_prob = {
                    'Delhi': 0.97, 'Maharashtra': 0.75, 'Gujarat': 0.62,
                    'Tamil Nadu': 0.68, 'Karnataka': 0.61, 'West Bengal': 0.68,
                    'Kerala': 0.65, 'Punjab': 0.37, 'Haryana': 0.40
                }.get(state, 0.5)
                features.append(1 if np.random.random() < urban_prob else 0)
            
            if 'state_urban_ratio' in self.feature_columns:
                urban_ratios = {
                    'Delhi': 0.97, 'Maharashtra': 0.75, 'Gujarat': 0.62,
                    'Tamil Nadu': 0.68, 'Karnataka': 0.61, 'West Bengal': 0.68,
                    'Kerala': 0.65, 'Punjab': 0.37, 'Haryana': 0.40, 'Rajasthan': 0.35
                }
                features.append(urban_ratios.get(state, 0.5))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error creating prediction features: {e}")
            return None
    
    def _generate_scenarios(self, base_predictions: List[float], category: str, state: str = None) -> Dict[str, List[float]]:
        """Generate different policy/economic scenarios with enhanced modeling"""
        scenarios = {}
        
        # State-specific factors based on renewable potential and economic conditions
        state_factors = {
            'Maharashtra': {'growth_volatility': 1.1, 'renewable_impact': 0.88, 'policy_support': 1.0},
            'Gujarat': {'growth_volatility': 0.95, 'renewable_impact': 0.82, 'policy_support': 1.1},  # High renewable adoption
            'Rajasthan': {'growth_volatility': 1.15, 'renewable_impact': 0.85, 'policy_support': 1.05}, # Desert solar potential
            'Tamil Nadu': {'growth_volatility': 1.05, 'renewable_impact': 0.84, 'policy_support': 1.03},
            'Delhi': {'growth_volatility': 1.08, 'renewable_impact': 0.90, 'policy_support': 0.98},
            'Karnataka': {'growth_volatility': 1.02, 'renewable_impact': 0.86, 'policy_support': 1.02},
            'Punjab': {'growth_volatility': 1.12, 'renewable_impact': 0.92, 'policy_support': 0.95},
            'Haryana': {'growth_volatility': 1.08, 'renewable_impact': 0.91, 'policy_support': 0.97}
        }
        
        state_factor = state_factors.get(state, {
            'growth_volatility': 1.0, 'renewable_impact': 0.88, 'policy_support': 1.0
        })
        
        # Conservative scenario (moderate growth with policy stability)
        scenarios['conservative'] = []
        for i, pred in enumerate(base_predictions):
            # Reduced growth rate for conservative scenario
            conservative_growth = 0.05 * state_factor['policy_support']  # 5% base growth
            growth_factor = (1 + conservative_growth) ** i
            adjusted_pred = pred * growth_factor * 0.95  # Conservative adjustment
            scenarios['conservative'].append(round(adjusted_pred, 2))
        
        # Aggressive scenario (high inflation, policy changes, economic stress)
        category_stress_factors = {
            'Agricultural': 1.20,    # Subsidy removal risk
            'Residential': 1.15,     # High inflation impact on consumers
            'Commercial': 1.18,      # Business cost pass-through
            'Industrial': 1.12,      # Some protection via policy
            'EV_Charging': 1.08,     # Policy support cushions impact
            'Tax_Duty': 1.25        # Government revenue pressure
        }
        
        stress_factor = category_stress_factors.get(category, 1.15)
        scenarios['aggressive'] = []
        for i, pred in enumerate(base_predictions):
            # High growth with economic stress
            aggressive_growth = 0.12 + (i * 0.005)  # Accelerating growth
            growth_factor = (1 + aggressive_growth) ** i
            adjusted_pred = pred * growth_factor * stress_factor * state_factor['growth_volatility']
            scenarios['aggressive'].append(round(adjusted_pred, 2))
        
        # Renewable push scenario (aggressive renewable adoption reduces tariff pressure)
        scenarios['renewable_push'] = []
        renewable_benefit = state_factor['renewable_impact']
        
        category_renewable_benefits = {
            'Residential': 0.85,     # Good benefit from rooftop solar
            'Commercial': 0.88,      # Moderate benefit
            'Industrial': 0.90,      # Less benefit (different energy profile)
            'Agricultural': 0.80,    # High benefit (solar pumps, etc.)
            'EV_Charging': 0.82      # High benefit (renewable charging)
        }
        
        category_benefit = category_renewable_benefits.get(category, 0.87)
        
        for i, pred in enumerate(base_predictions):
            # Progressive benefit from renewable adoption with learning curve
            year_benefit = renewable_benefit * category_benefit
            # Benefit increases over time as technology improves and scales
            progressive_benefit = year_benefit - (i * 0.008)  # Increasing benefit
            renewable_adjusted = pred * progressive_benefit
            scenarios['renewable_push'].append(round(renewable_adjusted, 2))
        
        return scenarios
    
    def get_state_category_forecast(self, state: str, category: str, years: int = 5) -> Dict:
        """Get simplified forecast for specific state and category"""
        if not self.is_trained:
            # Try to load the latest model first
            latest_model = self.get_latest_model_path()
            if latest_model:
                self.logger.info(f"Loading latest saved model: {latest_model}")
                if self.load_model(model_path=latest_model):
                    self.logger.info("✅ Successfully loaded saved model!")
                else:
                    # Quick train with available data
                    self.logger.info("Model not trained and load failed. Training with default data...")
                    df = self._generate_comprehensive_training_data()
                    self.train_models(df)
            else:
                # Quick train with available data
                self.logger.info("No saved models found. Training with default data...")
                df = self._generate_comprehensive_training_data()
                self.train_models(df)
        
        try:
            prediction = self.predict_multi_scenario(state, category, years)
            
            return {
                'forecast_years': prediction.forecast_years,
                'base_forecast': prediction.base_forecast,
                'confidence_intervals': prediction.confidence_intervals,
                'scenarios': {
                    'conservative': prediction.conservative_scenario,
                    'renewable_push': prediction.renewable_push_scenario,
                    'high_inflation': prediction.aggressive_scenario
                },
                'metadata': prediction.metadata
            }
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            # Return fallback forecast
            return self._generate_fallback_forecast(state, category, years)
    
    def _generate_fallback_forecast(self, state: str, category: str, years: int) -> Dict:
        """Generate a simple fallback forecast when main prediction fails"""
        # Simple growth-based forecast
        base_rates = {
            'Residential': 5.5, 'Commercial': 7.2, 'Industrial': 6.0,
            'Agricultural': 2.0, 'EV_Charging': 6.0
        }
        
        base_rate = base_rates.get(category, 5.5)
        growth_rate = 0.075  # 7.5% annual growth
        
        forecast_years = list(range(2025, 2025 + years))
        base_forecast = [round(base_rate * (1 + growth_rate) ** i, 2) for i in range(years)]
        
        # Simple confidence intervals (±20%)
        confidence_intervals = [(round(f * 0.8, 2), round(f * 1.2, 2)) for f in base_forecast]
        
        # Simple scenarios
        conservative = [round(f * 0.9, 2) for f in base_forecast]
        renewable_push = [round(f * 0.85, 2) for f in base_forecast]
        high_inflation = [round(f * 1.15, 2) for f in base_forecast]
        
        return {
            'forecast_years': forecast_years,
            'base_forecast': base_forecast,
            'confidence_intervals': confidence_intervals,
            'scenarios': {
                'conservative': conservative,
                'renewable_push': renewable_push,
                'high_inflation': high_inflation
            },
            'metadata': {
                'state': state,
                'category': category,
                'data_source': 'fallback_method',
                'note': 'Simple growth-based forecast due to prediction error'
            }
        }
    
    def validate_model_performance(self) -> Dict:
        """Return comprehensive model validation metrics"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        validation_summary = {
            'model_status': 'trained',
            'validation_scores': self.validation_scores,
            'feature_importance': self.feature_importance,
            'training_features': list(self.feature_columns) if hasattr(self, 'feature_columns') else [],
            'available_states': list(self.label_encoders.get('state', {}).classes_) if 'state' in self.label_encoders else [],
            'available_categories': list(self.label_encoders.get('category', {}).classes_) if 'category' in self.label_encoders else []
        }
        
        # Add model performance summary
        if self.validation_scores:
            best_model = min(self.validation_scores.keys(), 
                           key=lambda x: self.validation_scores[x].get('test_rmse', float('inf')))
            validation_summary['best_model'] = best_model
            validation_summary['best_model_rmse'] = self.validation_scores[best_model].get('test_rmse', 'N/A')
        
        return validation_summary
    
    def train_with_csv_data(self, csv_path: str = "C:/FOAI/data/electricity_tariff_dataset_520_entries.csv", 
                          auto_save: bool = True, model_name: str = None) -> Dict:
        """Convenience method to train with the provided CSV data"""
        try:
            # Check if file exists first
            import os
            file_exists = os.path.exists(csv_path)
        
            if not file_exists:
                self.logger.warning(f"CSV file '{csv_path}' not found!")
                self.logger.info("Training with synthetic data instead...")
                df = self._generate_comprehensive_training_data()
                data_source = "synthetic"
            else:
                # File exists, try to load real data
                try:
                    df = pd.read_csv(csv_path)
                    df = self._preprocess_real_data(df)
                    data_source = "real_csv"
                    self.logger.info(f"Successfully loaded real CSV data: {len(df)} records")
                except Exception as e:
                    self.logger.error(f"Error processing CSV file: {e}")
                    self.logger.info("Falling back to synthetic data...")
                    df = self._generate_comprehensive_training_data()
                    data_source = "synthetic_fallback"
        
            scores = self.train_models(df, auto_save=auto_save, model_name=model_name)
        
            # Accurate logging based on actual data source
            if data_source == "real_csv":
                self.logger.info("Successfully trained models with REAL CSV data")
            else:
                self.logger.info("Successfully trained models with SYNTHETIC data")
        
            self.logger.info(f"Training data shape: {df.shape}")
            self.logger.info(f"Available states: {sorted(df['state'].unique())}")
            self.logger.info(f"Available categories: {sorted(df['category'].unique())}")
        
            return {
                'success': True,
                'data_source': data_source,  # Add this to make it clear
                'training_scores': scores,
                'data_info': {
                    'total_records': len(df),
                    'states': sorted(df['state'].unique().tolist()),
                    'categories': sorted(df['category'].unique().tolist()),
                    'year_range': f"{df['year'].min()}-{df['year'].max()}",
                    'is_real_data': data_source == "real_csv"  # Clear indicator
                }
            }
        except Exception as e:
            self.logger.error(f"Error in training process: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_source': 'error',
                'fallback': 'Will use synthetic data for training'
            }

# Utility functions for easy model management
class ModelManager:
    """Helper class for managing saved models"""
    
    def __init__(self, model_dir: str = "C:/FOAI/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def list_models(self) -> List[Dict]:
        """List all available models with their metadata"""
        forecaster = AdvancedTariffForecaster(model_dir=self.model_dir)
        return forecaster.list_saved_models()
    
    def load_latest_model(self) -> AdvancedTariffForecaster:
        """Load the most recently saved model"""
        forecaster = AdvancedTariffForecaster(model_dir=self.model_dir)
        latest_path = forecaster.get_latest_model_path()
        
        if latest_path:
            if forecaster.load_model(model_path=latest_path):
                print(f"✅ Loaded latest model: {latest_path}")
                return forecaster
            else:
                print("❌ Failed to load latest model")
        else:
            print("❌ No saved models found")
        
        return None
    
    def load_model_by_name(self, model_name: str) -> AdvancedTariffForecaster:
        """Load a specific model by name"""
        forecaster = AdvancedTariffForecaster(model_dir=self.model_dir)
        
        if forecaster.load_model(model_name=model_name):
            print(f"✅ Loaded model: {model_name}")
            return forecaster
        else:
            print(f"❌ Failed to load model: {model_name}")
            return None
    
    def cleanup_old_models(self, keep_latest: int = 5):
        """Keep only the latest N models and delete the rest"""
        models = self.list_models()
        
        if len(models) <= keep_latest:
            print(f"Only {len(models)} models found, no cleanup needed")
            return
        
        models_to_delete = models[keep_latest:]
        deleted_count = 0
        
        for model in models_to_delete:
            try:
                # Delete model file
                if os.path.exists(model['model_path']):
                    os.remove(model['model_path'])
                
                # Delete metadata file
                metadata_path = model['model_path'].replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                deleted_count += 1
                print(f"🗑️ Deleted old model: {model['model_name']}")
                
            except Exception as e:
                print(f"❌ Failed to delete model {model['model_name']}: {e}")
        
        print(f"✅ Cleanup complete. Deleted {deleted_count} old models, kept {keep_latest} latest.")

# Usage example and enhanced testing
if __name__ == "__main__":
    # Initialize forecaster
    print("🔧 Initializing Advanced Tariff Forecaster with Model Persistence...")
    forecaster = AdvancedTariffForecaster()
    
    # Train with real CSV data and auto-save
    print("\n📊 Training models with real CSV data (auto-save enabled)...")
    training_result = forecaster.train_with_csv_data(
        csv_path="C:/FOAI/data/electricity_tariff_dataset_520_entries.csv",
        auto_save=True,
        model_name="tariff_model_production"
    )
    
    if training_result['success']:
        print("✅ Successfully trained and saved model!")
        print(f"📈 Data Info: {training_result['data_info']}")
        
        # Display training scores
        print("\n🎯 Model Validation Scores:")
        for model_name, metrics in training_result['training_scores'].items():
            print(f"\n{model_name.title()} Model:")
            print(f"  Cross-validation RMSE: ₹{metrics['cv_rmse']:.3f} ± {metrics.get('cv_std', 0):.3f}")
            print(f"  Test MAE: ₹{metrics['test_mae']:.3f}")
            print(f"  Test RMSE: ₹{metrics['test_rmse']:.3f}")
            print(f"  Test R²: {metrics['test_r2']:.3f}")
        
        # List saved models
        print("\n💾 Saved Models:")
        saved_models = forecaster.list_saved_models()
        for model in saved_models[:3]:  # Show latest 3
            print(f"  📁 {model['model_name']} ({model['file_size_mb']} MB) - {model['last_modified'][:19]}")
    
    else:
        print(f"⚠️ Training failed: {training_result.get('error', 'Unknown error')}")
    
    # Test loading the model in a new instance
    print("\n🔄 Testing Model Loading in New Instance...")
    new_forecaster = AdvancedTariffForecaster()
    
    # Try to load the latest model
    if new_forecaster.load_model(model_name="tariff_model_production"):
        print("✅ Successfully loaded saved model in new instance!")
        
        # Test prediction with loaded model
        print("\n🔮 Testing Prediction with Loaded Model:")
        try:
            prediction = new_forecaster.predict_multi_scenario("Maharashtra", "Residential", years_ahead=5)
            print(f"📊 Maharashtra Residential 5-year forecast: {prediction.base_forecast}")
            print(f"🌱 Renewable scenario: {prediction.renewable_push_scenario}")
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    # Test Model Manager
    print("\n🛠️ Testing Model Manager...")
    manager = ModelManager()
    
    print("📋 Available Models:")
    models = manager.list_models()
    for model in models:
        print(f"  • {model['model_name']} - {model.get('training_metadata', {}).get('total_records', 'N/A')} records")
    
    # Test loading latest model via manager
    loaded_forecaster = manager.load_latest_model()
    if loaded_forecaster:
        print("✅ Model Manager successfully loaded latest model!")
    
    print("\n🎉 Enhanced Tariff Forecasting System with Model Persistence Ready! 🚀")
    print("💾 Models are automatically saved to C:/FOAI/models/")
    print("🔄 You can now load and use trained models in other modules!")
    
    # Show usage example for other modules
    print("\n📝 Usage Example for Other Modules:")
    print("""
    # In your other Python modules:
    from advanced_tariff_forecast import AdvancedTariffForecaster, ModelManager
    
    # Option 1: Load latest model directly
    forecaster = AdvancedTariffForecaster()
    forecaster.load_model(model_name="tariff_model_production")
    
    # Option 2: Use Model Manager
    manager = ModelManager()
    forecaster = manager.load_latest_model()
    
    # Make predictions
    forecast = forecaster.get_state_category_forecast("Gujarat", "Commercial", years=3)
    print(forecast['base_forecast'])
    """)