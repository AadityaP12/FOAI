# integration_manager.py - FIXED VERSION

from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np
import os
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import pickle

# -----------------------------------------------------------------------------
from ml_models.weather_intelligence import EnhancedWeatherIntelligence
from ml_models.advanced_tariff_forecast import AdvancedTariffForecaster
from ml_models.technology_trend_analysis import RealDataSolarTrendAnalyzer
from ml_models.user_clustering import FixedUserClusteringV4

from engines.system_sizing_and_cost import EnhancedSolarSystemSizer
from engines.advanced_roi_calculator import (
    AdvancedROICalculator, SolarSystemSpec, LocationData,
    FinancialParameters, RiskParameters, TimeHorizon
)
from engines.comprehensive_risk_analysis import EnhancedRiskAnalyzer
from engines.intelligent_company_comparison import EnhancedCompanyComparator
from engines.mistake_prevention_engine import EnhancedMistakePreventionEngine
from engines._heuristic_search import run_simplified_heuristic_search
from engines.rooftop_feasibility_analyzer import AdvancedRooftopFeasibilityAnalyzer, RooftopGeometry
# Add these imports near the top
from engines.system_sizing_and_cost import (
    EnhancedSolarSystemSizer, LocationParameters, RegionalZone,
    RoofType, MountingMaterial
)

# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("FOAI.Integration")

# -----------------------------------------------------------------------------
# Data Classes (unchanged)
# -----------------------------------------------------------------------------

@dataclass
class UserRequest:
    location: str
    state: str
    category: str
    monthly_consumption_kwh: float
    monthly_bill: float = 2500.0
    roof_area_m2: Optional[float] = None
    budget_inr: Optional[float] = None
    house_type: str = "independent"
    income_bracket: str = "Medium"
    risk_tolerance: str = "moderate"
    timeline_preference: str = "flexible"
    priority: str = "cost"
    goals: Optional[List[str]] = field(default_factory=list)

@dataclass
class WeatherOut:
    monthly_generation_factors: Dict[str, float] = field(default_factory=dict)
    annual_generation_per_kw: Optional[float] = None
    confidence_bands: Optional[Dict[str, Tuple[float, float]]] = None
    notes: Optional[str] = None

@dataclass
class TariffOut:
    base_forecast: Dict[int, float] = field(default_factory=dict)
    scenarios: Dict[str, Dict[int, float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechTrendOut:
    efficiency_now_pct: Optional[float] = None
    efficiency_12mo_pct: Optional[float] = None
    cost_now_inr_per_w: Optional[float] = None
    cost_12mo_inr_per_w: Optional[float] = None
    subsidy_now: Optional[float] = None
    subsidy_12mo: Optional[float] = None
    policy_summary: Optional[str] = None
    model_summary: Optional[str] = None

@dataclass
class SizingOut:
    recommended_panels: Optional[int] = None
    system_capacity_kw: Optional[float] = None
    monthly_generation_kwh: Optional[float] = None
    selected_panel: Optional[str] = None
    selected_inverter: Optional[str] = None
    cost_breakdown_inr: Dict[str, float] = field(default_factory=dict)
    cost_range_inr: Optional[Tuple[float, float]] = None
    payback_years: Optional[float] = None
    confidence_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    feasibility_data: Optional[Dict[str, Any]] = None
    constraint_violated: bool = False  # ADD THIS LINE
    constraint_violation_reason: Optional[str] = None  # ADD THIS LINE
@dataclass
class ROIOut:
    npv_15y_inr: Optional[float] = None
    payback_years: Optional[float] = None
    annual_savings_inr: Optional[float] = None
    scenario_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class RiskOut:
    overall_risk: Optional[str] = None
    components: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserClusteringOut:
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    business_value: Optional[Dict[str, Any]] = None
    readiness_scores: Optional[Dict[str, float]] = None
    confidence: Optional[str] = None
    prediction_method: Optional[str] = None

@dataclass
class HeuristicSearchOut:
    optimal_scenario_type: Optional[str] = None
    roi: Optional[float] = None
    risk_score: Optional[float] = None
    payback_period: Optional[float] = None
    confidence: Optional[float] = None
    cost: Optional[float] = None
    f_score: Optional[float] = None  # FIXED: Make it a regular field, not a property
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    action_plan: List[Dict[str, Any]] = field(default_factory=list)   # NEW: structured multi-step plan
    
    def calculate_f_score(self) -> Optional[float]:
        """Calculate f_score from cost and other factors - now as a method"""
        if self.cost is None:
            return None
        base_cost = self.cost
        
        # Add heuristic component based on roi and risk
        if self.roi is not None and self.risk_score is not None:
            heuristic = (20 - self.roi) * 1000 + self.risk_score * 500
        else:
            heuristic = 5000  # Default heuristic
            
        calculated_f_score = base_cost + heuristic
        
        # Update the field with calculated value
        self.f_score = calculated_f_score
        return calculated_f_score


@dataclass
class VendorOut:
    ranked_vendors: List[Dict[str, Any]] = field(default_factory=list)
    method: Optional[str] = None

@dataclass
class SafetyGateOut:
    issues: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ok: bool = True

@dataclass
class PipelineResult:
    user: UserRequest
    weather: WeatherOut
    tariff: TariffOut
    tech: TechTrendOut
    sizing: SizingOut
    roi: ROIOut
    risk: RiskOut
    user_clustering: UserClusteringOut
    heuristic_search: HeuristicSearchOut
    vendors: VendorOut
    safety: SafetyGateOut

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)

# -----------------------------------------------------------------------------
# Helper Functions - FIXED
# -----------------------------------------------------------------------------

def safe_division(numerator, denominator, default=0.0):
    """Safely perform division with proper type checking"""
    try:
        num = safe_float_conversion(numerator)
        den = safe_float_conversion(denominator)
        if den == 0:
            return default
        return num / den
    except:
        return default

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def extract_cost_range(cost_range_data):
    """Safely extract cost range"""
    if cost_range_data is None:
        return 300000.0, 400000.0
    
    if isinstance(cost_range_data, (tuple, list)) and len(cost_range_data) >= 2:
        return safe_float_conversion(cost_range_data[0], 300000), safe_float_conversion(cost_range_data[1], 400000)
    elif isinstance(cost_range_data, (int, float)):
        base = safe_float_conversion(cost_range_data, 300000)
        return base * 0.9, base * 1.1
    else:
        return 300000.0, 400000.0







# -----------------------------------------------------------------------------
# Pipeline implementations - FIXED VERSIONS
# -----------------------------------------------------------------------------

# UPDATED WEATHER FUNCTION - NOW USES PERSISTENT MODELS
# UPDATED WEATHER FUNCTION - FIXED VERSION with proper historical data handling
def run_weather(user: UserRequest) -> WeatherOut:
    """
    FIXED: Weather analysis with persistent model support - proper historical data handling
    """
    try:
        region = user.location.lower().split(',')[0].strip()
        log.info(f"Starting OPTIMIZED weather analysis for region: {region}")
        
        # STEP 1: Try to load existing trained model first
        try:
            inst = EnhancedWeatherIntelligence(region=region)
            
            # Try to load saved model
            model_loaded = inst.load_trained_model()
            
            if model_loaded:
                log.info(f"✅ Successfully loaded saved weather model for {region}")
                log.info(f"   Model type: {inst.model_type}")
                log.info(f"   Data source: {inst.data_source}")
                
                # CRITICAL FIX: Load historical data even when model is loaded
                # The model needs historical data to generate proper impact analysis
                if inst.historical_data is None:
                    log.info("Loading historical data for saved model...")
                    data_loaded = inst.load_historical_data(years=3, use_cache=True)
                    if not data_loaded:
                        log.warning("Failed to load historical data, will retrain model")
                        model_loaded = False
                
                # Use the loaded model with historical data for impact analysis
                if model_loaded and inst.historical_data is not None:
                    impact_result = inst.get_weather_impact_analysis(system_capacity_kw=5.0, use_saved_model=True)
                    
                    if 'error' not in impact_result:
                        log.info("✅ Weather impact analysis completed using saved model")
                        
                        annual_gen = safe_division(impact_result.get('annual_generation_kwh', 5200), 5.0, 1040)
                        
                        # Extract monthly factors using the same strategies as before
                        monthly_factors = {}
                        
                        # Try seasonal performance data first
                        seasonal_perf = impact_result.get('seasonal_performance', {})
                        if isinstance(seasonal_perf, dict):
                            for i in range(1, 13):
                                month_name = pd.Timestamp(2024, i, 1).strftime('%B')
                                factor_value = seasonal_perf.get(f'month_{i}_performance')
                                if factor_value is not None:
                                    factor = safe_float_conversion(factor_value, 1.0)
                                    factor = max(0.3, min(2.0, factor))  # Bounded
                                    monthly_factors[month_name] = factor
                            
                            if len(monthly_factors) >= 12:
                                log.info("✅ Successfully extracted monthly factors from saved model")
                                return WeatherOut(
                                    monthly_generation_factors=monthly_factors,
                                    annual_generation_per_kw=annual_gen,
                                    confidence_bands={"annual": (annual_gen * 0.9, annual_gen * 1.1)},
                                    notes=f"Using saved {inst.model_type} model with {inst.data_source} data"
                                )
                    
                    log.warning("Saved model loaded but impact analysis failed, will retrain")
                else:
                    log.warning("Saved model exists but historical data unavailable, will retrain")
            else:
                log.info("No saved model found, will train new model")
        
        except Exception as model_load_error:
            log.warning(f"Model loading failed: {model_load_error}, will train new model")
        
        # STEP 2: If no saved model or it failed, train new model (only if needed)
        try:
            if 'inst' not in locals() or inst.historical_data is None:
                inst = EnhancedWeatherIntelligence(region=region)
            
            # Load data and train (with auto-save enabled)
            if inst.historical_data is None:
                data_loaded = inst.load_historical_data(years=3)
                if not data_loaded:
                    log.warning("Failed to load real data, using synthetic")
            
            training_result = inst.train_sarima_model(auto_save=True)  # This will save the model
            
            if 'error' not in training_result:
                log.info(f"✅ New model trained and saved for future use")
                log.info(f"   Model type: {training_result.get('model_type', 'SARIMA')}")
                log.info(f"   Data source: {training_result['data_source']}")
                log.info(f"   Model saved: {training_result.get('model_saved', False)}")
                
                # Generate impact analysis with new model
                impact_result = inst.get_weather_impact_analysis(system_capacity_kw=5.0)
                
                if 'error' not in impact_result:
                    annual_gen = safe_division(impact_result.get('annual_generation_kwh', 5200), 5.0, 1040)
                    
                    # Extract monthly factors
                    monthly_factors = {}
                    seasonal_perf = impact_result.get('seasonal_performance', {})
                    
                    if isinstance(seasonal_perf, dict):
                        for i in range(1, 13):
                            month_name = pd.Timestamp(2024, i, 1).strftime('%B')
                            factor_value = seasonal_perf.get(f'month_{i}_performance')
                            if factor_value is not None:
                                factor = safe_float_conversion(factor_value, 1.0)
                                factor = max(0.3, min(2.0, factor))
                                monthly_factors[month_name] = factor
                    
                    if len(monthly_factors) >= 12:
                        log.info("✅ Successfully extracted monthly factors from new model")
                        return WeatherOut(
                            monthly_generation_factors=monthly_factors,
                            annual_generation_per_kw=annual_gen,
                            confidence_bands={"annual": (annual_gen * 0.9, annual_gen * 1.1)},
                            notes=f"Newly trained {training_result.get('model_type', 'SARIMA')} model with {training_result['data_source']} data"
                        )
            
            log.warning(f"ML model training failed: {training_result.get('error', 'Unknown error')}")
        
        except Exception as ml_error:
            log.warning(f"ML weather analysis failed completely: {ml_error}")
    
    except Exception as weather_error:
        log.warning(f"Weather module initialization failed: {weather_error}")
    
    # STEP 3: Fallback to realistic seasonal patterns
    log.info("Using guaranteed realistic seasonal patterns based on Indian solar data")
    
    realistic_factors = get_location_based_seasonal_factors(user.location)
    realistic_annual_gen = get_location_based_annual_generation(user.location)
    
    return WeatherOut(
        monthly_generation_factors=realistic_factors,
        annual_generation_per_kw=realistic_annual_gen,
        confidence_bands={"annual": (realistic_annual_gen * 0.9, realistic_annual_gen * 1.1)},
        notes="Realistic Indian solar patterns - location-optimized seasonal model (ML model unavailable)"
    )


def get_location_based_seasonal_factors(location: str) -> Dict[str, float]:
    """
    GUARANTEED: Realistic seasonal factors based on actual Indian solar irradiance data
    """
    # Base Indian solar generation pattern (Source: NREL/MNRE data)
    # Normalized so annual average = 1.0
    base_indian_pattern = {
        'January': 0.75,    # Winter - low sun angle, clear skies
        'February': 0.85,   # Late winter - improving sun angle
        'March': 1.10,      # Pre-summer - excellent conditions
        'April': 1.25,      # Peak summer - maximum solar irradiance
        'May': 1.35,        # Peak summer - highest generation
        'June': 1.15,       # Early monsoon - still good before clouds
        'July': 0.65,       # Monsoon - heavy cloud cover
        'August': 0.55,     # Peak monsoon - lowest generation
        'September': 0.75,  # Late monsoon - gradual improvement
        'October': 1.05,    # Post-monsoon - excellent conditions return
        'November': 1.15,   # Post-monsoon - peak clear weather
        'December': 0.85    # Early winter - good but declining
    }
    
    # Apply location-specific adjustments based on real climate data
    if location:
        location_lower = location.lower()
        
        # Western Coast (heavy monsoon impact)
        if any(city in location_lower for city in ['mumbai', 'pune', 'goa']):
            monsoon_heavy = {
                'June': 0.90, 'July': 0.45, 'August': 0.40, 'September': 0.65,
                'January': 0.80, 'December': 0.90  # Better winter
            }
            for month, factor in monsoon_heavy.items():
                base_indian_pattern[month] = factor
                
        # Northern Plains (extreme seasonal variation)
        elif any(city in location_lower for city in ['delhi', 'jaipur', 'lucknow', 'chandigarh']):
            north_extreme = {
                'December': 0.60, 'January': 0.55, 'February': 0.75,  # Harsh winter
                'April': 1.40, 'May': 1.45, 'June': 1.35,  # Extreme summer
                'July': 0.70, 'August': 0.65  # Moderate monsoon
            }
            for month, factor in north_extreme.items():
                base_indian_pattern[month] = factor
                
        # Eastern India (longer monsoon, moderate winter)
        elif any(city in location_lower for city in ['kolkata', 'bhubaneswar']):
            east_monsoon = {
                'June': 0.85, 'July': 0.50, 'August': 0.45, 'September': 0.60, 'October': 0.90,
                'January': 0.70, 'December': 0.75  # Moderate winter
            }
            for month, factor in east_monsoon.items():
                base_indian_pattern[month] = factor
                
        # Southern India (more consistent, less extreme)
        elif any(city in location_lower for city in ['bangalore', 'hyderabad', 'chennai', 'coimbatore']):
            south_consistent = {
                'January': 0.85, 'December': 0.90,  # Mild winter
                'May': 1.25, 'April': 1.20,  # Moderate summer
                'July': 0.70, 'August': 0.65,  # Moderate monsoon
                'October': 1.10, 'November': 1.20  # Excellent post-monsoon
            }
            for month, factor in south_consistent.items():
                base_indian_pattern[month] = factor
                
        # Western Arid (excellent winter, extreme summer, minimal monsoon)
        elif any(city in location_lower for city in ['ahmedabad', 'jodhpur', 'udaipur']):
            arid_western = {
                'January': 0.85, 'February': 0.95, 'December': 0.90,  # Great winter
                'April': 1.45, 'May': 1.50, 'June': 1.30,  # Extreme summer
                'July': 0.85, 'August': 0.80  # Minimal monsoon impact
            }
            for month, factor in arid_western.items():
                base_indian_pattern[month] = factor
                
        # Kerala/Coastal South (unique monsoon pattern)
        elif any(city in location_lower for city in ['kochi', 'trivandrum']):
            kerala_pattern = {
                'June': 0.70, 'July': 0.40, 'August': 0.35, 'September': 0.50,  # Heavy monsoon
                'October': 0.85, 'November': 1.00,  # Post-monsoon recovery
                'January': 0.90, 'February': 1.00, 'March': 1.15,  # Excellent winter/spring
                'December': 0.95  # Good winter
            }
            for month, factor in kerala_pattern.items():
                base_indian_pattern[month] = factor
    
    # Validate and normalize the pattern
    annual_sum = sum(base_indian_pattern.values())
    target_average = 1.0
    normalization_factor = (target_average * 12) / annual_sum
    
    normalized_pattern = {}
    for month, factor in base_indian_pattern.items():
        normalized_factor = factor * normalization_factor
        # Apply realistic bounds
        normalized_factor = max(0.4, min(1.6, normalized_factor))
        normalized_pattern[month] = round(normalized_factor, 3)
    
    # Final validation
    final_average = sum(normalized_pattern.values()) / 12
    log.info(f"REALISTIC SEASONAL FACTORS APPLIED FOR {location.upper()}:")
    
    # Show seasonal insights
    best_month = max(normalized_pattern.items(), key=lambda x: x[1])
    worst_month = min(normalized_pattern.items(), key=lambda x: x[1])
    
    log.info(f"  Best month: {best_month[0]} ({best_month[1]:.3f})")
    log.info(f"  Worst month: {worst_month[0]} ({worst_month[1]:.3f})")
    log.info(f"  Seasonal variation: {(best_month[1]/worst_month[1]):.1f}x")
    log.info(f"  Annual average: {final_average:.3f}")
    
    for month, factor in normalized_pattern.items():
        log.info(f"     {month}: {factor:.3f}")
    
    return normalized_pattern


def get_location_based_annual_generation(location: str) -> float:
    """
    Get realistic annual generation per kW based on location
    """
    # Actual solar irradiance data for Indian cities (kWh/kW/year)
    location_generation = {
        'mumbai': 1420,     # Coastal, moderate monsoon impact
        'delhi': 1580,      # Northern plains, good winter/summer
        'bangalore': 1350,  # Southern plateau, consistent
        'chennai': 1480,    # Coastal south, good irradiance
        'pune': 1450,       # Western ghats, balanced
        'hyderabad': 1520,  # Deccan plateau, excellent
        'ahmedabad': 1680,  # Arid west, highest in India
        'kolkata': 1290,    # Eastern, monsoon impact
        'jaipur': 1620,     # Desert edge, excellent
        'kochi': 1380,      # Kerala coast, monsoon limited
        'surat': 1550,      # Gujarat coast, very good
        'lucknow': 1480,    # UP plains, good
        'nagpur': 1510,     # Central India, excellent
        'bhopal': 1540,     # Central plateau, very good
        'indore': 1560      # MP plateau, excellent
    }
    
    if location:
        location_lower = location.lower()
        for city, generation in location_generation.items():
            if city in location_lower:
                log.info(f"ðŸ“ Using location-specific generation: {generation} kWh/kW/year for {city}")
                return generation
    
    # Regional fallbacks
    if location:
        location_lower = location.lower()
        # State-based fallbacks
        if any(state in location_lower for state in ['maharashtra']):
            return 1450
        elif any(state in location_lower for state in ['gujarat', 'rajasthan']):
            return 1600
        elif any(state in location_lower for state in ['karnataka', 'andhra', 'telangana']):
            return 1500
        elif any(state in location_lower for state in ['tamil nadu']):
            return 1480
        elif any(state in location_lower for state in ['kerala']):
            return 1380
        elif any(state in location_lower for state in ['west bengal']):
            return 1300
        elif any(state in location_lower for state in ['delhi', 'haryana', 'punjab', 'uttar pradesh']):
            return 1550
    
    log.info("ðŸ“ Using India average: 1500 kWh/kW/year")
    return 1500.0  # India average

def run_tariff(user: UserRequest) -> TariffOut:
    """FIXED: Use pre-trained model first, train only if needed - with proper model validation"""
    try:
        log.info(f"Starting OPTIMIZED tariff forecasting for {user.state}, {user.category}")
        inst = AdvancedTariffForecaster()
        
        # FIXED: Look specifically for tariff forecaster models, not any .pkl file
        model_loaded = False
        latest_model_path = None
        
        try:
            # Check for tariff-specific models only
            saved_models = inst.list_saved_models()
            tariff_models = [m for m in saved_models if 'tariff' in m['model_name'].lower()]
            
            if tariff_models:
                latest_model_path = tariff_models[0]['model_path']  # Most recent tariff model
                log.info(f"Found tariff model: {latest_model_path}")
                
                # FIXED: Validate that this is actually a tariff forecaster model
                model_loaded = inst.load_model(model_path=latest_model_path)
                
                if model_loaded:
                    # FIXED: Additional validation to ensure it's the right type of model
                    if (hasattr(inst, 'label_encoders') and 
                        isinstance(inst.label_encoders, dict) and
                        hasattr(inst, 'models') and 
                        isinstance(inst.models, dict)):
                        
                        log.info("Successfully loaded and validated pre-trained tariff model!")
                        
                        # Safe access to label encoders
                        available_states = []
                        available_categories = []
                        
                        try:
                            if 'state' in inst.label_encoders and hasattr(inst.label_encoders['state'], 'classes_'):
                                available_states = list(inst.label_encoders['state'].classes_)
                            if 'category' in inst.label_encoders and hasattr(inst.label_encoders['category'], 'classes_'):
                                available_categories = list(inst.label_encoders['category'].classes_)
                        except Exception as attr_error:
                            log.warning(f"Could not access label encoder classes: {attr_error}")
                            available_states = []
                            available_categories = []
                        
                        user_state_supported = user.state in available_states if available_states else False
                        user_category_supported = user.category in available_categories if available_categories else False
                        
                        if user_state_supported and user_category_supported:
                            log.info(f"Pre-trained model supports {user.state}/{user.category}")
                        else:
                            log.warning(f"Pre-trained model missing {user.state}/{user.category}, will use fallback")
                            model_loaded = False
                    else:
                        log.warning("Loaded model is not a valid tariff forecaster, will train new model")
                        model_loaded = False
                else:
                    log.warning("Failed to load tariff model")
            else:
                log.info("No tariff-specific models found")
        
        except Exception as model_check_error:
            log.warning(f"Error checking for existing models: {model_check_error}")
            model_loaded = False
        
        if not model_loaded:
            log.info("No suitable pre-trained tariff model found, training new model...")
            training_result = inst.train_with_csv_data(auto_save=True, model_name=f"tariff_model_pipeline_{datetime.now().strftime('%Y%m%d')}")
            
            if not training_result.get('success', False):
                log.warning(f"Training failed: {training_result.get('error', 'Unknown error')}")
                return _get_economics_driven_forecast(user)
            
            log.info(f"New model trained and saved successfully")
        
        # Generate forecast using loaded/trained model
        try:
            forecast_result = inst.get_state_category_forecast(
                state=user.state, 
                category=user.category, 
                years=6
            )
            
            base_forecast = {}
            forecast_years = forecast_result.get('forecast_years', list(range(2025, 2031)))
            base_predictions = forecast_result.get('base_forecast', [])
            
            for i, year in enumerate(forecast_years):
                if i < len(base_predictions):
                    base_forecast[year] = safe_float_conversion(base_predictions[i], 7.0)
            
            if _validate_ml_forecast(base_forecast, user.state, user.category):
                scenarios = forecast_result.get('scenarios', {})
                
                meta = {
                    "model_trained": not model_loaded,
                    "model_loaded_from_disk": model_loaded,
                    "model_path": latest_model_path if model_loaded else "newly_trained",
                    "data_source": "pre_trained_ml" if model_loaded else "trained_ml",
                    "forecast_method": "ML_cached" if model_loaded else "ML_fresh",
                    "state_analyzed": user.state,
                    "category_analyzed": user.category,
                    "validation_passed": True
                }
                
                log.info(f"ML forecast completed using {'pre-trained' if model_loaded else 'fresh'} model")
                return TariffOut(base_forecast=base_forecast, scenarios=scenarios, meta=meta)
            
        except Exception as forecast_error:
            log.warning(f"ML forecast failed: {forecast_error}")
        
        # Fallback to economics-driven approach
        log.info("Using economics-driven fallback...")
        return _get_economics_driven_forecast(user)
        
    except Exception as e:
        log.error(f"All tariff forecasting methods failed: {e}")
        return _create_minimal_emergency_fallback(user)
    

def _get_economics_driven_forecast(user: UserRequest) -> TariffOut:
    """Economics-driven forecast with proper validation"""
    log.info("Using economics-driven tariff forecasting...")
    
    # Get realistic current tariff and growth rate
    current_tariff = get_realistic_current_tariff(user.state, user.category)
    growth_rate = calculate_realistic_growth_rate(user.state, user.category)
    
    # Generate base forecast with economic modeling
    base_forecast = {}
    for year in range(2025, 2031):
        years_ahead = year - 2024
        
        # Compound growth with economic cycle adjustments
        base_rate = current_tariff * ((1 + growth_rate) ** years_ahead)
        
        # Economic cycle adjustments
        if year <= 2026:  # Near-term inflation pressure
            base_rate *= 1.05
        elif year >= 2030:  # Long-term renewable impact
            base_rate *= 0.96
        
        base_forecast[year] = round(base_rate, 2)
    
    # Generate realistic scenarios with economics logic
    scenarios = generate_realistic_scenarios(base_forecast, user.state, user.category)
    
    # Calculate and log economic implications
    annual_growth_actual = ((base_forecast[2030] / base_forecast[2025]) ** (1/5)) - 1

    _demonstrate_comprehensive_payback_impact(
        current_tariff, base_forecast, scenarios, user, annual_growth_actual
    )
    
    meta = {
        "model_trained": False,
        "data_source": "economics_driven",
        "forecast_method": "economic_projection_enhanced",
        "state_analyzed": user.state,
        "category_analyzed": user.category,
        "current_tariff": current_tariff,
        "annual_growth_rate": annual_growth_actual,
        "payback_impact": "beneficial" if annual_growth_actual > 0.05 else "neutral",
        "economic_adjustments_applied": True
    }
    
    log.info(f"✓ Economics-driven forecast completed:")
    log.info(f"   Current tariff: ₹{current_tariff:.2f}/kWh")
    log.info(f"   2025 forecast: ₹{base_forecast[2025]}/kWh")
    log.info(f"   2030 forecast: ₹{base_forecast[2030]}/kWh")
    log.info(f"   Annual growth: {annual_growth_actual:.1%}")
    log.info(f"   Solar payback impact: {meta['payback_impact']}")
    
    return TariffOut(
        base_forecast=base_forecast,
        scenarios=scenarios,
        meta=meta
    )

def _has_sufficient_tariff_data(state: str, category: str) -> bool:
    """Check if we have sufficient data for reliable ML forecasting"""
    # Known states with good data coverage
    well_covered_states = ['maharashtra', 'gujarat', 'tamil nadu', 'karnataka', 
                          'rajasthan', 'delhi', 'uttar pradesh']
    
    # Known categories with good data coverage
    well_covered_categories = ['residential', 'commercial', 'industrial']
    
    state_lower = state.lower() if state else ''
    category_lower = category.lower() if category else ''
    
    # Check if both state and category have sufficient data
    has_sufficient_data = (state_lower in well_covered_states and 
                          category_lower in well_covered_categories)
    
    if not has_sufficient_data:
        log.warning(f"Insufficient data for ML forecasting: {state}/{category}")
    
    return has_sufficient_data


def _demonstrate_comprehensive_payback_impact(current_tariff: float, base_forecast: Dict[int, float], 
                                            scenarios: Dict[str, Dict[int, float]], user: UserRequest, 
                                            annual_growth_rate: float):
    """
    NEW: Comprehensive present vs future payback comparison analysis
    Shows exactly how tariff increases reduce payback periods over time
    """
    try:
        log.info(f"")
        log.info(f"ðŸŽ¯ PAYBACK IMPACT ANALYSIS - How Rising Tariffs Help Solar Economics")
        log.info(f"=" * 80)
        
        # Estimate system parameters for realistic payback calculation
        monthly_consumption = safe_float_conversion(user.monthly_consumption_kwh, 350)
        monthly_bill = safe_float_conversion(user.monthly_bill, 2500)
        
        # If consumption not provided, estimate from bill and current tariff
        if monthly_consumption == 350:  # Default value indicates no user input
            monthly_consumption = monthly_bill / current_tariff
            log.info(f"ðŸ“‹ Estimated consumption from bill: {monthly_consumption:.0f} kWh/month")
        
        # Realistic system sizing
        annual_consumption = monthly_consumption * 12
        target_coverage = 0.90
        
        # Location-based generation factors
        generation_factors = {
            'mumbai': 1450, 'delhi': 1520, 'bangalore': 1380, 'chennai': 1500,
            'pune': 1480, 'hyderabad': 1550, 'ahmedabad': 1620, 'kolkata': 1320
        }
        
        annual_gen_per_kw = 1500  # Default
        if user.location:
            for city, factor in generation_factors.items():
                if city in user.location.lower():
                    annual_gen_per_kw = factor
                    break
        
        # Calculate system specifications
        system_efficiency = 0.85
        effective_gen_per_kw = annual_gen_per_kw * system_efficiency
        system_capacity_kw = (annual_consumption * target_coverage) / effective_gen_per_kw
        system_capacity_kw = max(1.5, min(20.0, system_capacity_kw))
        
        annual_generation_kwh = system_capacity_kw * effective_gen_per_kw
        
        # Cost calculation with subsidies
        cost_per_kw = 60000 if system_capacity_kw <= 5 else 55000
        total_system_cost = system_capacity_kw * cost_per_kw
        
        # Apply subsidies (PM Surya Ghar scheme)
        subsidy_amount = min(78000, total_system_cost * 0.30)
        net_system_cost = total_system_cost - subsidy_amount
        
        log.info(f"ðŸ“‹ SYSTEM SPECIFICATIONS:")
        log.info(f"   System capacity: {system_capacity_kw:.1f} kW")
        log.info(f"   Annual generation: {annual_generation_kwh:,.0f} kWh")
        log.info(f"   Net cost: â‚¹{net_system_cost:,.0f}")
        log.info(f"")
        
        # PRESENT PAYBACK CALCULATION
        annual_savings_present = annual_generation_kwh * current_tariff
        payback_present = net_system_cost / annual_savings_present if annual_savings_present > 0 else 15
        
        log.info(f"ðŸ• PRESENT SCENARIO (Current Tariff):")
        log.info(f"   Current rate: â‚¹{current_tariff:.2f}/kWh")
        log.info(f"   Annual savings: â‚¹{annual_savings_present:,.0f}")
        log.info(f"   Simple payback: {payback_present:.1f} years")
        log.info(f"")
        
        # FUTURE PAYBACK CALCULATIONS
        log.info(f"ðŸ”® FUTURE SCENARIOS - Payback Improvement:")
        
        # 2030 base forecast
        if 2030 in base_forecast:
            tariff_2030 = base_forecast[2030]
            annual_savings_2030 = annual_generation_kwh * tariff_2030
            payback_2030 = net_system_cost / annual_savings_2030 if annual_savings_2030 > 0 else 15
            improvement = payback_present - payback_2030
            
            log.info(f"ðŸ“… 2030 BASE FORECAST:")
            log.info(f"   Future rate: â‚¹{tariff_2030:.2f}/kWh (+{((tariff_2030/current_tariff)-1)*100:.0f}%)")
            log.info(f"   Future payback: {payback_2030:.1f} years")
            log.info(f"   ðŸŽ‰ IMPROVEMENT: {improvement:.1f} years shorter!")
            log.info(f"")
        
        # Scenario analysis
        for scenario_name in ['conservative', 'aggressive']:
            if scenario_name in scenarios and 2030 in scenarios[scenario_name]:
                scenario_tariff = scenarios[scenario_name][2030]
                scenario_savings = annual_generation_kwh * scenario_tariff
                scenario_payback = net_system_cost / scenario_savings if scenario_savings > 0 else 15
                scenario_improvement = payback_present - scenario_payback
                
                symbol = "ðŸš€" if scenario_improvement > 2 else "ðŸ“ˆ" if scenario_improvement > 0 else "ðŸ“‰"
                
                log.info(f"   {scenario_name.title():12s}: {scenario_payback:.1f} years ({scenario_improvement:+.1f}) {symbol}")
        
        log.info(f"")
        log.info(f"ðŸ’¡ KEY INSIGHT: Higher tariff growth = Faster payback = Better solar ROI")
        log.info(f"=" * 80)
        
    except Exception as e:
        log.warning(f"Could not calculate payback demonstration: {e}")


def _validate_ml_forecast(forecast: Dict[int, float], state: str, category: str) -> bool:
    """STRICTER validation of ML forecast against economic reality"""
    if len(forecast) < 2:
        return False
    
    years = sorted(forecast.keys())
    initial = forecast[years[0]]
    final = forecast[years[-1]]
    
    # Get expected current tariff for this state/category
    expected_tariff = get_realistic_current_tariff(state, category)
    
    # STRICTER: Check for reasonable bounds
    if initial < expected_tariff * 0.5 or initial > expected_tariff * 2.0:
        log.warning(f"ML forecast outside reasonable bounds: ₹{initial:.2f} vs expected ₹{expected_tariff:.2f}")
        return False
    
    if final < expected_tariff * 0.7 or final > expected_tariff * 3.0:
        log.warning(f"ML forecast final value unrealistic: ₹{final:.2f} vs expected ₹{expected_tariff:.2f}")
        return False
    
    # STRICTER: Check for reasonable growth rate (-5% to 15%)
    annual_growth = (final / initial) ** (1 / (len(years) - 1)) - 1
    if annual_growth < -0.05 or annual_growth > 0.15:
        log.warning(f"ML forecast unrealistic growth rate: {annual_growth:.1%}")
        return False
    
    # STRICTER: Check category-specific reasonableness (within 40%)
    if abs(initial - expected_tariff) > expected_tariff * 0.4:
        log.warning(f"ML forecast differs significantly from expected for {state}/{category}")
        log.warning(f"ML: ₹{initial:.2f}, Expected: ₹{expected_tariff:.2f}, Diff: {((initial/expected_tariff-1)*100):+.0f}%")
        return False
    
    log.info(f"✓ ML forecast validation passed: {annual_growth:.1%} annual growth")
    return True


def _extract_ml_scenarios(forecast_result: dict, forecast_years: list) -> Dict[str, Dict[int, float]]:
    """Extract and format ML scenarios"""
    scenarios = forecast_result.get('scenarios', {})
    formatted_scenarios = {}
    
    for scenario_name, predictions in scenarios.items():
        formatted_scenarios[scenario_name] = {}
        for i, year in enumerate(forecast_years):
            if i < len(predictions):
                formatted_scenarios[scenario_name][year] = safe_float_conversion(predictions[i], 7.0)
    
    # Add confidence intervals if available
    confidence_intervals = forecast_result.get('confidence_intervals', [])
    if confidence_intervals:
        formatted_scenarios['confidence_lower'] = {}
        formatted_scenarios['confidence_upper'] = {}
        for i, year in enumerate(forecast_years):
            if i < len(confidence_intervals):
                lower, upper = confidence_intervals[i]
                formatted_scenarios['confidence_lower'][year] = safe_float_conversion(lower, 5.0)
                formatted_scenarios['confidence_upper'][year] = safe_float_conversion(upper, 10.0)
    
    return formatted_scenarios


def _create_minimal_emergency_fallback(user: UserRequest) -> TariffOut:
    """Absolute minimal fallback when everything fails"""
    log.error("Using emergency minimal fallback")
    
    # Basic India-wide averages
    category_base = {
        'residential': 7.0,
        'commercial': 9.0,
        'industrial': 6.5,
        'agricultural': 3.0,
        'ev_charging': 8.0
    }
    
    base_tariff = category_base.get(user.category.lower() if user.category else 'residential', 7.0)
    
    # Simple 7% annual growth
    base_forecast = {}
    for year in range(2025, 2031):
        base_forecast[year] = round(base_tariff * (1.07 ** (year - 2024)), 2)
    
    scenarios = {
        'conservative': {year: rate * 0.85 for year, rate in base_forecast.items()},
        'aggressive': {year: rate * 1.25 for year, rate in base_forecast.items()}
    }
    
    return TariffOut(
        base_forecast=base_forecast,
        scenarios=scenarios,
        meta={
            "model_trained": False,
            "data_source": "emergency_fallback",
            "forecast_method": "minimal_projection"
        }
    )


# Helper functions from version 1 (economics-driven logic)

def get_realistic_current_tariff(state: str, category: str) -> float:
    """Get realistic current tariff rates based on state and category"""
    
    # State-wise residential tariff averages (â‚¹/kWh) - 2024 data
    state_tariffs = {
        'maharashtra': {'residential': 8.5, 'commercial': 11.2, 'industrial': 8.8, 'agricultural': 2.5},
        'gujarat': {'residential': 6.8, 'commercial': 9.8, 'industrial': 7.2, 'agricultural': 1.8},
        'rajasthan': {'residential': 8.0, 'commercial': 10.5, 'industrial': 8.0, 'agricultural': 3.2},
        'tamil nadu': {'residential': 7.2, 'commercial': 9.5, 'industrial': 7.8, 'agricultural': 2.0},
        'karnataka': {'residential': 7.8, 'commercial': 10.0, 'industrial': 8.2, 'agricultural': 2.8},
        'delhi': {'residential': 9.5, 'commercial': 12.0, 'industrial': 9.8, 'agricultural': 4.0},
        'west bengal': {'residential': 7.5, 'commercial': 9.8, 'industrial': 7.5, 'agricultural': 2.2},
        'kerala': {'residential': 6.5, 'commercial': 9.2, 'industrial': 7.0, 'agricultural': 1.5},
        'punjab': {'residential': 6.9, 'commercial': 9.5, 'industrial': 7.8, 'agricultural': 1.0},
        'haryana': {'residential': 7.2, 'commercial': 9.8, 'industrial': 8.0, 'agricultural': 1.2},
        'uttar pradesh': {'residential': 7.8, 'commercial': 10.2, 'industrial': 8.5, 'agricultural': 2.5},
        'andhra pradesh': {'residential': 4.9, 'commercial': 8.5, 'industrial': 6.8, 'agricultural': 1.8},
        'telangana': {'residential': 8.2, 'commercial': 10.8, 'industrial': 8.5, 'agricultural': 2.0}
    }
    
    # Category mapping
    category_map = {
        'residential': 'residential',
        'commercial': 'commercial', 
        'industrial': 'industrial',
        'agriculture': 'agricultural',
        'agricultural': 'agricultural',
        'ev_charging': 'commercial',
        'tax_duty': 'commercial'
    }
    
    state_key = state.lower() if state else 'maharashtra'
    category_key = category_map.get(category.lower() if category else 'residential', 'residential')
    
    tariffs = state_tariffs.get(state_key, state_tariffs['maharashtra'])
    current_tariff = tariffs.get(category_key, tariffs['residential'])
    
    return current_tariff


def calculate_realistic_growth_rate(state: str, category: str) -> float:
    """Calculate realistic annual growth rate based on state and category"""
    
    # Base growth rates by category
    category_growth = {
        'residential': 0.075,    # 7.5% - moderate growth due to cross-subsidization
        'commercial': 0.085,     # 8.5% - higher growth, less subsidized
        'industrial': 0.070,     # 7% - some policy protection
        'agricultural': 0.040,   # 4% - heavily subsidized
        'ev_charging': 0.060     # 6% - policy support
    }
    
    # State-specific modifiers based on renewable adoption
    state_modifiers = {
        'gujarat': 0.90,         # High renewable adoption
        'rajasthan': 0.95,       # Growing renewable capacity
        'tamil nadu': 0.92,      # Strong renewable program
        'karnataka': 0.95,       # Balanced energy mix
        'maharashtra': 1.00,     # Baseline
        'delhi': 0.98,           # Policy support
        'west bengal': 1.05,     # Higher coal dependency
        'kerala': 0.93,          # Good hydro resources
        'punjab': 1.08,          # Agricultural subsidy pressure
        'haryana': 1.03,         # Agricultural subsidy pressure
        'uttar pradesh': 1.02,   # Infrastructure stress
        'andhra pradesh': 0.88,  # Low current rates, higher growth potential
        'telangana': 0.96        # Reasonable renewable adoption
    }
    
    category_key = category.lower() if category else 'residential'
    if category_key not in category_growth:
        category_key = 'residential'
    
    state_key = state.lower() if state else 'maharashtra'
    
    base_growth = category_growth[category_key]
    state_modifier = state_modifiers.get(state_key, 1.0)
    
    final_growth_rate = base_growth * state_modifier
    final_growth_rate = max(0.02, min(0.12, final_growth_rate))  # 2% to 12% bounds
    
    return final_growth_rate


def generate_realistic_scenarios(base_forecast: Dict[int, float], state: str, category: str) -> Dict[str, Dict[int, float]]:
    """Generate realistic tariff scenarios with economic logic"""
    
    scenarios = {}
    
    # Conservative scenario (renewable success)
    scenarios['conservative'] = {}
    for year, base_rate in base_forecast.items():
        conservative_factor = 0.82 + (0.03 * np.sin(2 * np.pi * (year - 2024) / 8))
        scenarios['conservative'][year] = round(base_rate * conservative_factor, 2)
    
    # Aggressive scenario (economic stress + policy failures)
    scenarios['aggressive'] = {}
    for year, base_rate in base_forecast.items():
        years_ahead = year - 2024
        aggressive_factor = 1.25 + (0.10 * years_ahead * 0.1)
        
        # Category-specific stress factors
        category_stress = {
            'residential': 1.15, 'commercial': 1.20, 'industrial': 1.10,
            'agricultural': 1.30, 'ev_charging': 1.08
        }
        
        stress_multiplier = category_stress.get(category.lower(), 1.15)
        scenarios['aggressive'][year] = round(base_rate * aggressive_factor * stress_multiplier, 2)
    
    # Renewable push scenario (moderate renewable benefit)
    scenarios['renewable_push'] = {}
    for year, base_rate in base_forecast.items():
        years_ahead = year - 2024
        renewable_benefit = 0.95 - (years_ahead * 0.02)  # Increasing benefit
        renewable_benefit = max(0.85, renewable_benefit)  # Minimum 15% benefit
        scenarios['renewable_push'][year] = round(base_rate * renewable_benefit, 2)
    
    # ML confidence bounds (if ML was attempted)
    scenarios['ml_lower_bound'] = {year: rate * 0.88 for year, rate in base_forecast.items()}
    scenarios['ml_upper_bound'] = {year: rate * 1.18 for year, rate in base_forecast.items()}
    
    # Log scenario implications
    log.info(f"TARIFF SCENARIOS FOR SOLAR PAYBACK:")
    log.info(f"  Conservative 2030: â‚¹{scenarios['conservative'][2030]}/kWh â†’ Longer payback")
    log.info(f"  Base 2030:         â‚¹{base_forecast[2030]}/kWh â†’ Baseline payback")
    log.info(f"  Aggressive 2030:   â‚¹{scenarios['aggressive'][2030]}/kWh â†’ Shorter payback âœ…")
    
    return scenarios



def run_rooftop_feasibility(user: UserRequest, sizing: SizingOut) -> Dict[str, Any]:
    """
    Step 5.5: Rooftop feasibility analysis
    Validates if the sized system can actually be installed on the given roof
    """
    try:
        log.info("Starting rooftop feasibility analysis...")
        
        # Initialize analyzer
        analyzer = AdvancedRooftopFeasibilityAnalyzer(debug_mode=False)
        
        # Extract roof data from user input
        roof_area = safe_float_conversion(user.roof_area_m2, None)
        if roof_area is None:
            # Estimate based on system size
            system_capacity = safe_float_conversion(sizing.system_capacity_kw, 5.0)
            roof_area = max(50, system_capacity * 8)  # 8 sq.m per kW conservative
            log.info(f"Estimated roof area: {roof_area:.0f} sq.m")
        
        # Create rooftop geometry
        rooftop_geometry = RooftopGeometry(
            total_area_sqm=roof_area,
            usable_area_sqm=roof_area * 0.85,  # 85% usable
            length_m=math.sqrt(roof_area * 1.2),  # Assume rectangular with 1.2:1 ratio
            width_m=math.sqrt(roof_area / 1.2),
            shape="rectangular",
            roof_orientation=180,  # South facing default
            tilt_angle=5,  # Nearly flat
            height_above_ground_m=3.5 if user.house_type.lower() == 'independent' else 12,
            perimeter_setback_m=1.0
        )
        
        # Building details
        building_details = {
            'roof_type': 'CONCRETE_FLAT',
            'building_type': 'RESIDENTIAL_INDEPENDENT' if user.house_type.lower() == 'independent' else 'RESIDENTIAL_APARTMENT',
            'age_years': 10,
            'structural_condition': 'good',
            'seismic_zone': 'III',
            'wind_zone': 'III',
            'rooftop_obstacles': [
                {'type': 'WATER_TANK', 'x': 2, 'y': 2, 'dimensions': (2, 1.5, 2), 'permanent': True, 'relocatable': False}
            ],
            'surrounding_buildings': []
        }
        
        # Location details
        location_details = {
            'city': user.location.split(',')[0].strip() if user.location else 'Mumbai',
            'state': user.state,
            'latitude': 19.0760,
            'longitude': 72.8777,
            'climate_zone': 'tropical'
        }
        
        # System requirements from sizing result
        system_requirements = {
            'target_capacity_kw': safe_float_conversion(sizing.system_capacity_kw, 5.0),
            'panel_preference': 'residential',
            'panel_wattage': 540,
            'mounting_preference': 'penetrating'
        }
        
        # Perform feasibility analysis
        feasibility_result = analyzer.analyze_rooftop_feasibility(
            rooftop_geometry=rooftop_geometry,
            building_details=building_details,
            location_details=location_details,
            system_requirements=system_requirements,
            regulatory_context={'building_approval': 'approved'}
        )
        
        # Extract key metrics
        feasibility_data = {
            'is_feasible': feasibility_result.is_technically_feasible,
            'feasibility_score': feasibility_result.overall_feasibility_score,
            'max_capacity_kw': feasibility_result.maximum_installable_capacity_kw,
            'cost_multiplier': feasibility_result.cost_implications.get('base_cost_multiplier', 1.0),
            'additional_costs': feasibility_result.cost_implications.get('additional_fixed_costs', 0),
            'shading_loss_percent': feasibility_result.shading_analysis.annual_shading_loss_percent,
            'warnings': feasibility_result.warnings[:3],
            'recommendations': feasibility_result.recommendations[:3]
        }
        
        log.info(f"Feasibility: Score {feasibility_data['feasibility_score']:.1f}/100, Max {feasibility_data['max_capacity_kw']:.1f}kW")
        return feasibility_data
        
    except Exception as e:
        log.error(f"Rooftop feasibility analysis failed: {e}")
        return {
            'is_feasible': True, 'feasibility_score': 75.0,
            'max_capacity_kw': safe_float_conversion(sizing.system_capacity_kw, 5.0),
            'cost_multiplier': 1.0, 'additional_costs': 0,
            'shading_loss_percent': 5.0, 'warnings': ['Feasibility analysis unavailable'],
            'recommendations': ['Manual site assessment recommended']
        }
def run_tech_trends(user: UserRequest, tariff: TariffOut) -> TechTrendOut:
    """
    FIXED: Use pre-trained models for tech trend analysis
    """
    try:
        # STEP 1: Try to use pre-trained analyzer
        log.info("Loading technology trend analyzer...")
        
        # Check if we have a saved analyzer state
        analyzer_cache_path = "C:/FOAI/models/tech_analyzer_cache.pkl"
        analyzer = None
        
        if os.path.exists(analyzer_cache_path):
            try:
                with open(analyzer_cache_path, 'rb') as f:
                    analyzer = pickle.load(f)
                log.info("✅ Loaded cached tech trend analyzer")
            except Exception as load_error:
                log.warning(f"Failed to load cached analyzer: {load_error}")
        
        # STEP 2: Initialize new analyzer if cache load failed
        if analyzer is None:
            log.info("🔄 Initializing new tech trend analyzer...")
            analyzer = RealDataSolarTrendAnalyzer(data_source="real_market")
            
            # Save for future use
            try:
                os.makedirs(os.path.dirname(analyzer_cache_path), exist_ok=True)
                with open(analyzer_cache_path, 'wb') as f:
                    pickle.dump(analyzer, f)
                log.info("💾 Cached tech trend analyzer for future use")
            except Exception as save_error:
                log.warning(f"Failed to cache analyzer: {save_error}")
        
        # STEP 3: Generate forecasts using cached/loaded analyzer
        monocrystalline_forecast = None
        cost_forecast = None
        
        try:
            # Use the pre-loaded analyzer methods
            if hasattr(analyzer, 'forecast_efficiency_with_real_trends'):
                mono_forecast = analyzer.forecast_efficiency_with_real_trends('Monocrystalline', 12)
                monocrystalline_forecast = mono_forecast
                log.info("✅ Efficiency forecast generated using cached model")
        except Exception as eff_error:
            log.warning(f"Efficiency forecast failed: {eff_error}")
        
        try:
            if hasattr(analyzer, 'forecast_cost_with_real_market_trends'):
                cost_forecast_result = analyzer.forecast_cost_with_real_market_trends('Monocrystalline', 12)
                cost_forecast = cost_forecast_result
                log.info("✅ Cost forecast generated using cached model")
        except Exception as cost_error:
            log.warning(f"Cost forecast failed: {cost_error}")
        
        # Process results
        current_eff = safe_float_conversion(
            monocrystalline_forecast.get('current_efficiency', 20.5) if monocrystalline_forecast else 20.5, 20.5
        )
        future_eff = safe_float_conversion(
            monocrystalline_forecast.get('forecasted_efficiency', 20.7) if monocrystalline_forecast else 20.7, 20.7
        )
        current_cost = safe_float_conversion(
            cost_forecast.get('current_cost_per_watt_inr', 30.0) if cost_forecast else 30.0, 30.0
        )
        future_cost = safe_float_conversion(
            cost_forecast.get('forecasted_cost_per_watt_inr', 28.5) if cost_forecast else 28.5, 28.5
        )
        
        model_summary = "cached_real_data_analyzer" if os.path.exists(analyzer_cache_path) else "fresh_real_data_analyzer"
        
        return TechTrendOut(
            efficiency_now_pct=current_eff,
            efficiency_12mo_pct=future_eff,
            cost_now_inr_per_w=current_cost,
            cost_12mo_inr_per_w=future_cost,
            subsidy_now=0.3,
            subsidy_12mo=0.25,
            policy_summary="PM Surya Ghar scheme active with reducing subsidies",
            model_summary=model_summary
        )
        
    except Exception as e:
        log.error(f"Technology trend analysis failed: {e}")
        return TechTrendOut(
            efficiency_now_pct=20.5,
            efficiency_12mo_pct=20.8,
            cost_now_inr_per_w=30.0,
            cost_12mo_inr_per_w=28.5,
            subsidy_now=0.3,
            subsidy_12mo=0.25,
            policy_summary="Standard market trends",
            model_summary="Fallback analysis"
        )

# Fixed run_sizing function for integration_manager.py

# Fixed run_sizing function for integration_manager.py

def run_sizing(user: UserRequest, weather: WeatherOut, tech: TechTrendOut) -> SizingOut:
    """CORRECTED sizing function with realistic capacity calculation"""
    try:
        from engines.system_sizing_and_cost import (
            EnhancedSolarSystemSizer, LocationParameters, RegionalZone,
            RoofType, MountingMaterial
        )
        
        # Get actual monthly consumption
        monthly_consumption_kwh = safe_float_conversion(user.monthly_consumption_kwh, 300)
        monthly_bill = safe_float_conversion(user.monthly_bill, 2500)
        
        # FIXED: Better consumption estimation from bill
        if monthly_consumption_kwh == 300:  # default value indicates no input
            # Realistic tariff estimation
            base_tariff = 8.5  # Average residential tariff
            if user.state:
                state_tariffs = {
                    'maharashtra': 8.5, 'delhi': 9.5, 'karnataka': 7.8, 'tamil nadu': 7.2,
                    'gujarat': 6.8, 'telangana': 8.2, 'west bengal': 7.5, 'rajasthan': 8.0
                }
                base_tariff = state_tariffs.get(user.state.lower(), 8.5)
            
            monthly_consumption_kwh = monthly_bill / base_tariff
            log.info(f"Estimated consumption from bill: {monthly_consumption_kwh:.0f} kWh/month using tariff â‚¹{base_tariff}/unit")
        
        # CORRECTED SYSTEM SIZING CALCULATION
        # Get realistic annual generation potential
        annual_gen_per_kw = safe_float_conversion(weather.annual_generation_per_kw, 1500)
        
        # Validate and correct unrealistic weather data
        if annual_gen_per_kw < 1200 or annual_gen_per_kw > 2000:
            # Use location-based defaults
            location_defaults = {
                'mumbai': 1450, 'delhi': 1520, 'bangalore': 1380, 'chennai': 1500,
                'pune': 1480, 'hyderabad': 1550, 'ahmedabad': 1620, 'kolkata': 1320
            }
            city = 'mumbai'  # default
            if user.location:
                for known_city in location_defaults:
                    if known_city in user.location.lower():
                        city = known_city
                        break
            annual_gen_per_kw = location_defaults[city]
            log.warning(f"Using location default generation: {annual_gen_per_kw} kWh/kW/year")
        
        # Calculate monthly generation per kW
        monthly_gen_per_kw = annual_gen_per_kw / 12
        
        # System efficiency (realistic)
        system_efficiency = 0.85  # 15% total losses
        effective_monthly_gen_per_kw = monthly_gen_per_kw * system_efficiency
        
        # Calculate required capacity
        required_capacity_kw = monthly_consumption_kwh / effective_monthly_gen_per_kw
        
        # Add modest buffer
        recommended_capacity_kw = required_capacity_kw * 1.1  # 10% buffer
        
        # Apply realistic bounds
        recommended_capacity_kw = max(1.0, min(20.0, recommended_capacity_kw))
        
        log.info(f"CORRECTED SIZING CALCULATION:")
        log.info(f"  Monthly consumption: {monthly_consumption_kwh:.0f} kWh")
        log.info(f"  Annual generation per kW: {annual_gen_per_kw:.0f} kWh/kW/year")
        log.info(f"  Monthly generation per kW: {monthly_gen_per_kw:.0f} kWh/kW")
        log.info(f"  System efficiency: {system_efficiency:.0%}")
        log.info(f"  Required capacity: {required_capacity_kw:.2f} kW")
        log.info(f"  Recommended capacity: {recommended_capacity_kw:.2f} kW")


        # ✅ PATCH: Align sizing for heuristic search with demand-based requirement
        adjusted_capacity_kw = required_capacity_kw  # from earlier calculation (~6.3 kW)
        log.info(f"⚡ Adjusted capacity for heuristic search: {adjusted_capacity_kw:.2f} kW")

        
        
        # Set up budget
        budget = safe_float_conversion(user.budget_inr, None)
        if budget is None or budget <= 0:
            # Realistic cost estimation
            cost_per_kw = 55000 if recommended_capacity_kw > 5 else 60000
            estimated_budget = recommended_capacity_kw * cost_per_kw
            budget = max(150000, estimated_budget)
            log.info(f"Estimated budget: â‚¹{budget:,.0f}")
        
        budget_constraints = {
            'min': budget * 0.7,
            'max': budget * 1.3
        }
        
        # Enhanced roof area calculation
        roof_area = safe_float_conversion(user.roof_area_m2, None)
        if roof_area is None:
            area_per_kw = 7  # sq.m per kW
            estimated_roof_area = recommended_capacity_kw * area_per_kw * 1.3
            roof_area = max(50, estimated_roof_area)
            log.info(f"Estimated roof area: {roof_area:.0f} sq.m")
        
        roof_specifications = {
            'area_sqm': roof_area,
            'type': 'CONCRETE_FLAT',
            'height_floors': 1 if user.house_type.lower() == 'independent' else 3,
            'shading_factor': 0.05 if user.house_type.lower() == 'independent' else 0.10,
            'access_difficulty': 'easy' if user.house_type.lower() == 'independent' else 'moderate'
        }
        
        preferences = {
            'quality_tier': 'any',
            'priority': user.priority if user.priority else 'cost',
            'preferred_brands': [],
            'max_price_per_watt': 60,
            'min_efficiency': 18.0,
            'financing_required': False,
            'electricity_tariff': 8.5,
            'tariff_escalation_rate': 6.0,
            'distance_from_dealer': 30,
            'shading_issues': False,
            'complex_roof': user.house_type.lower() == 'apartment',
            'budget_category': 'flexible'
        }
        
        # Get location parameters
        location_params = create_location_parameters_enhanced(user, weather)
        
        # Initialize sizer
        sizer = EnhancedSolarSystemSizer(debug_mode=True)
        
        try:
            result = sizer.size_optimal_system(
                energy_requirement_kwh_month=monthly_consumption_kwh,
                location=location_params,
                roof_specifications=roof_specifications,
                budget_constraints=budget_constraints,
                preferences=preferences,
                optimization_objectives=['cost', 'performance']
            )


            # UPDATED: Check for constraint violations
            if hasattr(result, 'constraint_violations') and result.constraint_violations:
                log.warning(f"System sizing constraints violated: {result.constraint_violations}")
                return SizingOut(
                    recommended_panels=0,
                    system_capacity_kw=0.0,
                    monthly_generation_kwh=0,
                    selected_panel="No viable system",
                    selected_inverter="No viable system",
                    cost_breakdown_inr={},
                    cost_range_inr=(0, 0),
                    payback_years=float('inf'),
                    confidence_score=0.0,
                    warnings=["System not viable due to constraint violations"] + list(result.constraint_violations),
                    constraint_violated=True,  # UPDATED
                    constraint_violation_reason="Budget and/or technical constraints not met"  # UPDATED
                )
            
            # Process results
            config = result.system_config
            performance = result.performance_prediction
            cost_breakdown = result.cost_breakdown
            
            # Calculate monthly generation
            annual_generation = performance.annual_generation_kwh[0] if performance.annual_generation_kwh else 0
            monthly_generation = annual_generation / 12
            
            # Validate coverage
            coverage_ratio = monthly_generation / monthly_consumption_kwh if monthly_consumption_kwh > 0 else 0
            warnings = list(result.warnings) if result.warnings else []
            
            if coverage_ratio < 0.8:
                warnings.append(f"System undersized: {coverage_ratio:.0%} coverage")
            elif coverage_ratio > 1.3:
                warnings.append(f"System oversized: {coverage_ratio:.0%} coverage")
            
            cost_range = (cost_breakdown.total_after_incentives * 0.95, 
                         cost_breakdown.total_after_incentives * 1.05)
            
            cost_dict = {
                'panels': cost_breakdown.panels,
                'inverter': cost_breakdown.inverter,
                'mounting_structure': cost_breakdown.mounting_structure,
                'electrical_components': cost_breakdown.electrical_components,
                'labor': cost_breakdown.labor,
                'transport': cost_breakdown.transport,
                'permits_approvals': cost_breakdown.permits_approvals,
                'miscellaneous': cost_breakdown.miscellaneous,
                'contingency': cost_breakdown.contingency,
                'total': cost_breakdown.total_after_incentives
            }
            
            log.info(f"CORRECTED RESULTS:")
            log.info(f"  System capacity: {config.total_capacity_kw:.2f} kW")
            log.info(f"  Monthly generation: {monthly_generation:.0f} kWh")
            log.info(f"  Coverage ratio: {coverage_ratio:.0%}")
            
            return SizingOut(
                recommended_panels=config.num_panels,
                system_capacity_kw=config.total_capacity_kw,
                monthly_generation_kwh=monthly_generation,
                selected_panel=f"{config.panels.brand} {config.panels.model}",
                selected_inverter=f"{config.inverter.brand} {config.inverter.model}",
                cost_breakdown_inr=cost_dict,
                cost_range_inr=cost_range,
                payback_years=result.payback_analysis.get('simple_payback_years') if result.payback_analysis else None,
                confidence_score=result.confidence_metrics.get('overall_confidence') if result.confidence_metrics else None,
                warnings=warnings
            )
            
        except Exception as sizing_error:
            log.error(f"Enhanced system sizing failed: {sizing_error}")
            # UPDATED: Also check if it's a constraint violation
            if "budget" in str(sizing_error).lower() or "constraint" in str(sizing_error).lower():
                return SizingOut(
                    recommended_panels=0,
                    system_capacity_kw=0.0,
                    monthly_generation_kwh=0,
                    selected_panel="No viable system",
                    selected_inverter="No viable system", 
                    cost_breakdown_inr={},
                    cost_range_inr=(0, 0),
                    payback_years=float('inf'),
                    confidence_score=0.0,
                    warnings=[f"System sizing failed: {str(sizing_error)}"],
                    constraint_violated=True,  # UPDATED
                    constraint_violation_reason=str(sizing_error)  # UPDATED
                )
            
            else:
                return create_corrected_fallback_sizing(user, monthly_consumption_kwh, recommended_capacity_kw)
            
    except Exception as e:
        log.error(f"System sizing function failed: {e}")
        monthly_consumption = safe_float_conversion(user.monthly_consumption_kwh, 300)
        # FIXED: Use realistic capacity calculation in fallback
        annual_gen = 1500  # Default for India
        monthly_gen_per_kw = (annual_gen / 12) * 0.85  # With efficiency
        required_capacity = monthly_consumption / monthly_gen_per_kw
        return create_corrected_fallback_sizing(user, monthly_consumption, required_capacity)

def create_corrected_fallback_sizing(user: UserRequest, monthly_consumption_kwh: float, 
                                   required_capacity_kw: float) -> SizingOut:
    """CORRECTED fallback sizing with realistic calculations"""
    
    # Ensure realistic system size
    capacity_kw = max(1.0, min(15.0, required_capacity_kw))
    
    # Calculate realistic panel count
    panel_wattage = 540
    num_panels = max(2, int(capacity_kw * 1000 / panel_wattage))
    actual_capacity = num_panels * panel_wattage / 1000
    
    # Realistic cost calculation based on market rates
    if actual_capacity <= 3:
        cost_per_kw = 65000
    elif actual_capacity <= 10:
        cost_per_kw = 55000
    else:
        cost_per_kw = 50000
    
    # Location adjustment
    location_multiplier = 1.0
    if user.location:
        location_lower = user.location.lower()
        if any(metro in location_lower for metro in ['mumbai', 'delhi', 'bangalore', 'chennai']):
            location_multiplier = 1.15
    
    total_cost = actual_capacity * cost_per_kw * location_multiplier
    cost_range = (total_cost * 0.9, total_cost * 1.1)
    
    # CORRECTED generation calculation
    # Get realistic annual generation for location
    location_annual_gen = {
        'mumbai': 1450, 'delhi': 1520, 'bangalore': 1380, 'chennai': 1500,
        'pune': 1480, 'hyderabad': 1550, 'ahmedabad': 1620, 'kolkata': 1320
    }
    
    city = 'mumbai'  # default
    if user.location:
        for known_city in location_annual_gen:
            if known_city in user.location.lower():
                city = known_city
                break
    
    annual_gen_per_kw = location_annual_gen[city]
    system_efficiency = 0.85
    monthly_generation = actual_capacity * (annual_gen_per_kw / 12) * system_efficiency
    
    # Calculate payback
    tariff = 8.5  # Default tariff
    if user.state:
        state_tariffs = {
            'maharashtra': 8.5, 'delhi': 9.5, 'karnataka': 7.8, 'tamil nadu': 7.2
        }
        tariff = state_tariffs.get(user.state.lower(), 8.5)
    
    annual_savings = monthly_generation * 12 * tariff
    payback_years = total_cost / annual_savings if annual_savings > 0 else 8.0
    
    # Coverage validation
    coverage_ratio = monthly_generation / monthly_consumption_kwh if monthly_consumption_kwh > 0 else 1.0
    warnings = []
    
    if coverage_ratio < 0.8:
        warnings.append(f"System undersized: {coverage_ratio:.0%} coverage")
    elif coverage_ratio > 1.3:
        warnings.append(f"System oversized: {coverage_ratio:.0%} coverage")
    
    if payback_years > 10:
        warnings.append(f"Long payback period: {payback_years:.1f} years")
    
    warnings.append("Fallback sizing used - manual verification recommended")
    
    cost_dict = {
        'panels': total_cost * 0.40,
        'inverter': total_cost * 0.15,
        'mounting_structure': total_cost * 0.12,
        'electrical_components': total_cost * 0.08,
        'labor': total_cost * 0.15,
        'transport': total_cost * 0.03,
        'permits_approvals': total_cost * 0.04,
        'miscellaneous': total_cost * 0.02,
        'contingency': total_cost * 0.01,
        'total': total_cost
    }
    
    log.info(f"CORRECTED FALLBACK SIZING:")
    log.info(f"  Capacity: {actual_capacity:.2f} kW ({num_panels} panels)")
    log.info(f"  Monthly generation: {monthly_generation:.0f} kWh")
    log.info(f"  Coverage: {coverage_ratio:.0%}")
    log.info(f"  Payback: {payback_years:.1f} years")

    # UPDATED: Check if this is actually a constraint violation
    if actual_capacity < 1.0 or payback_years > 15:
        return SizingOut(
            recommended_panels=0,
            system_capacity_kw=0.0,
            monthly_generation_kwh=0,
            selected_panel="No viable system",
            selected_inverter="No viable system",
            cost_breakdown_inr={},
            cost_range_inr=(0, 0),
            payback_years=float('inf'),
            confidence_score=0.0,
            warnings=["System not economically viable with current parameters"],
            constraint_violated=True,  # UPDATED
            constraint_violation_reason="Poor economics - system too small or payback too long"  # UPDATED
        )
    
    return SizingOut(
        recommended_panels=num_panels,
        system_capacity_kw=actual_capacity,
        monthly_generation_kwh=monthly_generation,
        selected_panel=f"Standard 540W Mono-PERC Panel",
        selected_inverter=f"{actual_capacity:.1f}kW String Inverter",
        cost_breakdown_inr=cost_dict,
        cost_range_inr=cost_range,
        payback_years=payback_years,
        confidence_score=0.75,
        warnings=warnings,
        constraint_violated=False,  # UPDATED
        constraint_violation_reason=None  # UPDATED
    )


def create_location_parameters_enhanced(user: UserRequest, weather: WeatherOut) -> LocationParameters:
    """CORRECTED location parameters with realistic peak sun hours"""
    
    # CORRECTED city data with realistic peak sun hours for India
    city_data = {
        'mumbai': {'lat': 19.0760, 'lng': 72.8777, 'zone': RegionalZone.TIER_1_METROS, 'temp': 28, 'peak_sun_hours': 4.5},
        'delhi': {'lat': 28.7041, 'lng': 77.1025, 'zone': RegionalZone.TIER_1_METROS, 'temp': 25, 'peak_sun_hours': 5.0},
        'bangalore': {'lat': 12.9716, 'lng': 77.5946, 'zone': RegionalZone.TIER_1_METROS, 'temp': 24, 'peak_sun_hours': 4.2},
        'chennai': {'lat': 13.0827, 'lng': 80.2707, 'zone': RegionalZone.TIER_1_METROS, 'temp': 32, 'peak_sun_hours': 4.8},
        'pune': {'lat': 18.5204, 'lng': 73.8567, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 26, 'peak_sun_hours': 4.6},
        'hyderabad': {'lat': 17.3850, 'lng': 78.4867, 'zone': RegionalZone.TIER_1_METROS, 'temp': 30, 'peak_sun_hours': 4.7},
        'ahmedabad': {'lat': 23.0225, 'lng': 72.5714, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 30, 'peak_sun_hours': 5.2},
        'kolkata': {'lat': 22.5726, 'lng': 88.3639, 'zone': RegionalZone.TIER_1_METROS, 'temp': 30, 'peak_sun_hours': 3.8},
        'surat': {'lat': 21.1702, 'lng': 72.8311, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 29, 'peak_sun_hours': 4.9},
        'jaipur': {'lat': 26.9124, 'lng': 75.7873, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 28, 'peak_sun_hours': 5.1},
        'kochi': {'lat': 9.9312, 'lng': 76.2673, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 29, 'peak_sun_hours': 3.5},
        'lucknow': {'lat': 26.8467, 'lng': 80.9462, 'zone': RegionalZone.TIER_2_CITIES, 'temp': 26, 'peak_sun_hours': 4.3}
    }
    
    # Extract city
    city_name = 'mumbai'  # default
    if user.location:
        location_lower = user.location.lower()
        for city in city_data.keys():
            if city in location_lower:
                city_name = city
                break
    
    city_info = city_data[city_name]
    
    # Use realistic annual generation
    annual_gen = safe_float_conversion(weather.annual_generation_per_kw, 1500)
    if annual_gen < 1200 or annual_gen > 2000:
        # Use location-based realistic values
        location_gen_map = {
            'mumbai': 1450, 'delhi': 1520, 'bangalore': 1380, 'chennai': 1500,
            'pune': 1480, 'hyderabad': 1550, 'ahmedabad': 1620, 'kolkata': 1320
        }
        annual_gen = location_gen_map.get(city_name, 1500)
    
    return LocationParameters(
        city=city_name.title(),
        state=user.state,
        latitude=city_info['lat'],
        longitude=city_info['lng'],
        annual_irradiance=annual_gen,
        irradiance_std=120,
        peak_sun_hours=city_info['peak_sun_hours'],  # CORRECTED realistic values
        weather_reliability=0.88,
        regional_zone=city_info['zone'],
        grid_stability_score=0.85,
        average_temperature=city_info['temp'],
        pollution_factor=0.88
    )
def improved_budget_estimation(user: UserRequest, recommended_capacity_kw: float) -> float:
    """CORRECTED budget estimation with realistic market rates"""
    
    # Current market rates (2024-25) with economies of scale
    if recommended_capacity_kw <= 3:
        cost_per_kw = 65000  # Small systems
    elif recommended_capacity_kw <= 5:
        cost_per_kw = 60000  # Standard residential
    elif recommended_capacity_kw <= 10:
        cost_per_kw = 55000  # Large residential  
    else:
        cost_per_kw = 50000  # Very large systems
    
    base_budget = recommended_capacity_kw * cost_per_kw
    
    # Location adjustment (realistic regional variations)
    location_multiplier = 1.0
    if user.location:
        location_lower = user.location.lower()
        if any(metro in location_lower for metro in ['mumbai', 'delhi', 'bangalore', 'chennai']):
            location_multiplier = 1.15  # Metro premium
        elif any(tier2 in location_lower for tier2 in ['pune', 'hyderabad', 'ahmedabad']):
            location_multiplier = 1.05  # Tier-2 premium
    
    # House type adjustment
    house_multiplier = 1.0
    if user.house_type.lower() == 'apartment':
        house_multiplier = 1.1  # More complex installation
    
    # Income-based adjustment
    income_multiplier = 1.0
    if user.income_bracket:
        income_lower = user.income_bracket.lower()
        if income_lower in ['low']:
            income_multiplier = 0.85  # Budget systems
        elif income_lower in ['high', 'premium']:
            income_multiplier = 1.2   # Premium systems
    
    estimated_budget = base_budget * location_multiplier * house_multiplier * income_multiplier
    
    # Use user's budget if reasonable
    user_budget = safe_float_conversion(user.budget_inr, 0)
    if user_budget > 0:
        # Check if user budget is in reasonable range
        ratio = user_budget / estimated_budget
        if 0.7 <= ratio <= 2.0:  # Within reasonable range
            return user_budget
        else:
            log.warning(f"User budget â‚¹{user_budget:,} unrealistic for {recommended_capacity_kw:.1f}kW, using estimated â‚¹{estimated_budget:,.0f}")
    
    # Apply bounds
    min_budget = 150000   # Minimum â‚¹1.5 lakh
    max_budget = 1000000  # Maximum â‚¹10 lakh for residential
    
    final_budget = max(min_budget, min(max_budget, estimated_budget))
    
    log.info(f"CORRECTED BUDGET ESTIMATION:")
    log.info(f"  System capacity: {recommended_capacity_kw:.1f} kW")
    log.info(f"  Cost per kW: â‚¹{cost_per_kw:,}")
    log.info(f"  Location multiplier: {location_multiplier:.2f}")
    log.info(f"  Final budget: â‚¹{final_budget:,.0f}")
    
    return final_budget

def run_roi(user: UserRequest, sizing: SizingOut, tariff: TariffOut, tech: TechTrendOut, weather: WeatherOut) -> ROIOut:
    """
    FIXED: ROI calculation with proper dynamic tariff integration
    Shows how rising tariffs improve solar ROI over time
    """
    try:
        # Get system specifications
        cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
        total_cost = (cost_min + cost_max) / 2
        system_capacity = safe_float_conversion(sizing.system_capacity_kw, 5.0)
        monthly_generation = safe_float_conversion(sizing.monthly_generation_kwh, 400)
        annual_generation = monthly_generation * 12
        
        # Get tariff information
        current_tariff = 7.0
        tariff_growth_rate = 0.08  # Default 8% growth
        
        if tariff.base_forecast:
            # Use first available year as current tariff
            sorted_years = sorted(tariff.base_forecast.keys())
            current_tariff = safe_float_conversion(tariff.base_forecast[sorted_years[0]], 7.0)
            
            # Calculate actual growth rate from forecast
            if len(sorted_years) >= 2:
                initial_tariff = tariff.base_forecast[sorted_years[0]]
                final_tariff = tariff.base_forecast[sorted_years[-1]]
                years_span = sorted_years[-1] - sorted_years[0]
                if years_span > 0 and initial_tariff > 0:
                    tariff_growth_rate = ((final_tariff / initial_tariff) ** (1/years_span)) - 1
                    tariff_growth_rate = max(0.02, min(0.15, tariff_growth_rate))  # Reasonable bounds
        
        log.info(f"ROI CALCULATION WITH DYNAMIC TARIFFS:")
        log.info(f"  Current tariff: â‚¹{current_tariff:.2f}/kWh")
        log.info(f"  Annual tariff growth: {tariff_growth_rate:.1%}")
        log.info(f"  System capacity: {system_capacity:.1f} kW")
        log.info(f"  Annual generation: {annual_generation:,.0f} kWh")
        
        # Create comprehensive system specifications
        system_spec = SolarSystemSpec(
            capacity_kw=system_capacity,
            panel_efficiency=safe_float_conversion(tech.efficiency_now_pct, 20.5),
            panel_degradation_rate=0.5,
            inverter_efficiency=0.96,
            system_losses=0.14
        )
        
        # Location data with proper annual irradiance
        annual_irradiance = safe_float_conversion(weather.annual_generation_per_kw, 1500)
        location_data = LocationData(
            latitude=19.0760,  # Default to Mumbai
            longitude=72.8777,
            annual_irradiance=annual_irradiance,
            irradiance_std=150,
            weather_reliability=0.92
        )
        
        # FIXED: Financial parameters with dynamic tariff escalation
        financial_params = FinancialParameters(
            system_cost_per_kw=safe_division(total_cost, system_capacity, 60000),
            installation_cost=15000,
            maintenance_cost_annual=3000,
            insurance_cost_annual=1500,
            electricity_tariff=current_tariff,
            tariff_escalation_rate=tariff_growth_rate,  # Use calculated growth rate
            subsidy_amount=total_cost * safe_float_conversion(tech.subsidy_now, 0.3),
            discount_rate=0.08,  # Realistic discount rate
            inflation_rate=0.06   # Current inflation
        )
        
        risk_params = RiskParameters()
        
        # Use the advanced ROI calculator
        calculator = AdvancedROICalculator(monte_carlo_iterations=500)  # Reduced for speed
        results = calculator.calculate_comprehensive_roi(
            system_spec, location_data, financial_params, risk_params, TimeHorizon.LONG_TERM
        )
        
        # FIXED: Extract proper annual savings (first year from dynamic calculation)
        first_year_savings = safe_float_conversion(results.annual_cashflows[0] if results.annual_cashflows else 0, 0)
        
        # ENHANCED: Calculate scenario-based ROI analysis using tariff scenarios
        scenario_table = {}
        
        if tariff.scenarios:
            log.info(f"SCENARIO-BASED ROI ANALYSIS:")
            
            for scenario_name, scenario_tariffs in tariff.scenarios.items():
                if scenario_name in ['conservative', 'aggressive', 'renewable_push']:
                    try:
                        # Create financial params for this scenario
                        scenario_financial = FinancialParameters(**financial_params.__dict__)
                        
                        # Calculate growth rate for this scenario
                        if len(scenario_tariffs) >= 2:
                            scenario_years = sorted(scenario_tariffs.keys())
                            initial = scenario_tariffs[scenario_years[0]]
                            final = scenario_tariffs[scenario_years[-1]]
                            years_span = scenario_years[-1] - scenario_years[0]
                            if years_span > 0 and initial > 0:
                                scenario_growth = ((final / initial) ** (1/years_span)) - 1
                                scenario_financial.tariff_escalation_rate = max(0.02, min(0.15, scenario_growth))
                        
                        # Calculate ROI for this scenario
                        scenario_results = calculator.calculate_comprehensive_roi(
                            system_spec, location_data, scenario_financial, risk_params, TimeHorizon.LONG_TERM
                        )
                        
                        scenario_payback = safe_float_conversion(scenario_results.payback_period, 10)
                        scenario_npv = safe_float_conversion(scenario_results.npv, 0)
                        scenario_roi = safe_float_conversion(scenario_results.simple_roi, 0)
                        
                        scenario_table[scenario_name] = {
                            'roi': scenario_roi,
                            'npv': scenario_npv,
                            'payback': scenario_payback,
                            'tariff_growth': scenario_financial.tariff_escalation_rate * 100,
                            'annual_savings_y1': scenario_results.annual_cashflows[0] if scenario_results.annual_cashflows else 0
                        }
                        
                        log.info(f"  {scenario_name:15s}: ROI {scenario_roi:5.1f}%, Payback {scenario_payback:4.1f}y, NPV â‚¹{scenario_npv:,.0f}")
                        
                    except Exception as scenario_error:
                        log.warning(f"Scenario {scenario_name} calculation failed: {scenario_error}")
        
        # ENHANCED: Show dynamic vs static tariff comparison
        _demonstrate_roi_tariff_impact(
            current_tariff, tariff_growth_rate, annual_generation, total_cost, 
            safe_float_conversion(tech.subsidy_now, 0.3)
        )
        
        return ROIOut(
            npv_15y_inr=safe_float_conversion(results.npv),
            payback_years=safe_float_conversion(results.payback_period),
            annual_savings_inr=first_year_savings,
            scenario_table=scenario_table
        )
        
    except Exception as e:
        log.error(f"Advanced ROI calculation failed: {e}")
        
        # ENHANCED FALLBACK: Use dynamic tariff calculation even in fallback
        return _calculate_enhanced_fallback_roi(user, sizing, tariff, tech)


def _demonstrate_roi_tariff_impact(current_tariff: float, growth_rate: float, 
                                 annual_generation: float, system_cost: float, subsidy_rate: float):
    """
    NEW: Demonstrate how dynamic tariffs affect ROI compared to static assumptions
    """
    try:
        log.info("")
        log.info("ðŸ“Š ROI IMPACT: Dynamic vs Static Tariff Assumptions")
        log.info("=" * 60)
        
        net_cost = system_cost * (1 - subsidy_rate)
        
        # STATIC TARIFF SCENARIO (traditional calculation)
        static_annual_savings = annual_generation * current_tariff
        static_payback = net_cost / static_annual_savings if static_annual_savings > 0 else 15
        static_15y_savings = static_annual_savings * 15
        static_npv = static_15y_savings - net_cost
        
        # DYNAMIC TARIFF SCENARIO (with escalation)
        dynamic_total_savings = 0
        dynamic_discounted_savings = 0
        
        for year in range(1, 16):  # 15-year analysis
            # Tariff grows each year
            year_tariff = current_tariff * ((1 + growth_rate) ** (year - 1))
            
            # Generation decreases due to degradation (0.5% per year)
            year_generation = annual_generation * ((1 - 0.005) ** (year - 1))
            
            # Annual savings for this year
            year_savings = year_generation * year_tariff
            dynamic_total_savings += year_savings
            
            # Discounted savings (8% discount rate)
            discounted_savings = year_savings / ((1.08) ** year)
            dynamic_discounted_savings += discounted_savings
        
        dynamic_npv = dynamic_discounted_savings - net_cost
        
        # Calculate when investment breaks even (dynamic payback)
        cumulative_savings = 0
        dynamic_payback = 15  # Default if never breaks even
        
        for year in range(1, 16):
            year_tariff = current_tariff * ((1 + growth_rate) ** (year - 1))
            year_generation = annual_generation * ((1 - 0.005) ** (year - 1))
            year_savings = year_generation * year_tariff
            cumulative_savings += year_savings
            
            if cumulative_savings >= net_cost:
                # Interpolate for fractional payback
                prev_cumulative = cumulative_savings - year_savings
                if year_savings > 0:
                    fraction = (net_cost - prev_cumulative) / year_savings
                    dynamic_payback = year - 1 + fraction
                break
        
        # Calculate the benefit of dynamic tariffs
        additional_npv = dynamic_npv - static_npv
        payback_improvement = static_payback - dynamic_payback
        
        log.info(f"ðŸ  SYSTEM INVESTMENT:")
        log.info(f"   System cost: â‚¹{system_cost:,.0f}")
        log.info(f"   Net cost (after subsidy): â‚¹{net_cost:,.0f}")
        log.info(f"   Annual generation: {annual_generation:,.0f} kWh")
        log.info("")
        
        log.info(f"ðŸ“‰ STATIC TARIFF SCENARIO (Traditional Method):")
        log.info(f"   Fixed tariff: â‚¹{current_tariff:.2f}/kWh")
        log.info(f"   Annual savings: â‚¹{static_annual_savings:,.0f}")
        log.info(f"   15-year savings: â‚¹{static_15y_savings:,.0f}")
        log.info(f"   Simple payback: {static_payback:.1f} years")
        log.info(f"   Static NPV: â‚¹{static_npv:,.0f}")
        log.info("")
        
        log.info(f"ðŸ“ˆ DYNAMIC TARIFF SCENARIO (Rising Prices):")
        log.info(f"   Starting tariff: â‚¹{current_tariff:.2f}/kWh")
        log.info(f"   Annual growth: {growth_rate:.1%}")
        log.info(f"   Year 15 tariff: â‚¹{current_tariff * ((1 + growth_rate) ** 14):.2f}/kWh")
        log.info(f"   15-year total savings: â‚¹{dynamic_total_savings:,.0f}")
        log.info(f"   Dynamic payback: {dynamic_payback:.1f} years")
        log.info(f"   Dynamic NPV: â‚¹{dynamic_npv:,.0f}")
        log.info("")
        
        log.info(f"ðŸŽ¯ TARIFF ESCALATION BENEFIT:")
        log.info(f"   Additional NPV: â‚¹{additional_npv:,.0f}")
        log.info(f"   Payback improvement: {payback_improvement:.1f} years faster")
        log.info(f"   ROI enhancement: {(additional_npv/net_cost)*100:.1f} percentage points")
        log.info("")
        
        # Year-by-year demonstration (first 5 years)
        log.info(f"ðŸ“… YEAR-BY-YEAR SAVINGS GROWTH:")
        for year in range(1, 6):
            year_tariff = current_tariff * ((1 + growth_rate) ** (year - 1))
            year_generation = annual_generation * ((1 - 0.005) ** (year - 1))
            year_savings = year_generation * year_tariff
            
            log.info(f"   Year {year}: â‚¹{year_tariff:.2f}/kWh â†’ â‚¹{year_savings:,.0f} savings")
        
        log.info(f"   ... (escalation continues)")
        log.info("")
        log.info(f"ðŸ’¡ KEY INSIGHT: Rising tariffs make solar MORE valuable over time")
        log.info(f"ðŸ›¡ï¸ Solar protects against ALL future electricity price increases")
        log.info("=" * 60)
        
    except Exception as e:
        log.warning(f"ROI tariff impact demonstration failed: {e}")


def _calculate_enhanced_fallback_roi(user: UserRequest, sizing: SizingOut, tariff: TariffOut, tech: TechTrendOut) -> ROIOut:
    """
    ENHANCED FALLBACK: ROI calculation with dynamic tariff integration even in fallback mode
    """
    try:
        # Basic system parameters
        cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
        total_cost = (cost_min + cost_max) / 2
        monthly_gen = safe_float_conversion(sizing.monthly_generation_kwh, 400)
        annual_generation = monthly_gen * 12
        
        # Get dynamic tariff parameters
        current_tariff = 7.0
        tariff_growth_rate = 0.08
        
        if tariff.base_forecast:
            sorted_years = sorted(tariff.base_forecast.keys())
            current_tariff = safe_float_conversion(tariff.base_forecast[sorted_years[0]], 7.0)
            
            # Calculate growth rate from forecast
            if len(sorted_years) >= 2:
                initial = tariff.base_forecast[sorted_years[0]]
                final = tariff.base_forecast[sorted_years[-1]]
                years_span = sorted_years[-1] - sorted_years[0]
                if years_span > 0 and initial > 0:
                    tariff_growth_rate = ((final / initial) ** (1/years_span)) - 1
        
        # Apply subsidy
        subsidy_rate = safe_float_conversion(tech.subsidy_now, 0.3)
        net_cost = total_cost * (1 - subsidy_rate)
        
        log.info(f"ENHANCED FALLBACK ROI (Dynamic Tariff Integration):")
        log.info(f"  Net system cost: â‚¹{net_cost:,.0f}")
        log.info(f"  Current tariff: â‚¹{current_tariff:.2f}/kWh")
        log.info(f"  Tariff growth: {tariff_growth_rate:.1%}/year")
        
        # DYNAMIC CALCULATION: Year-by-year cash flows with escalating tariffs
        annual_cashflows = []
        cumulative_savings = 0
        total_savings_15y = 0
        payback_year = None
        
        for year in range(1, 16):  # 15-year analysis
            # Tariff for this year (escalating)
            year_tariff = current_tariff * ((1 + tariff_growth_rate) ** (year - 1))
            
            # Generation for this year (with degradation)
            year_generation = annual_generation * ((1 - 0.005) ** (year - 1))
            
            # Savings for this year
            year_savings = year_generation * year_tariff
            annual_cashflows.append(year_savings)
            
            # Maintenance costs (escalating with inflation)
            year_maintenance = 3000 * ((1.05) ** (year - 1))
            
            # Net cash flow
            net_cashflow = year_savings - year_maintenance
            cumulative_savings += net_cashflow
            total_savings_15y += year_savings
            
            # Track payback
            if payback_year is None and cumulative_savings >= net_cost:
                payback_year = year
                if year_savings > 0:
                    prev_cumulative = cumulative_savings - net_cashflow
                    fraction = (net_cost - prev_cumulative) / net_cashflow
                    payback_year = year - 1 + fraction
        
        # Calculate NPV with dynamic cash flows
        npv_15y = 0
        for year, cashflow in enumerate(annual_cashflows, 1):
            maintenance = 3000 * ((1.05) ** (year - 1))
            net_annual = cashflow - maintenance
            discounted_value = net_annual / ((1.08) ** year)
            npv_15y += discounted_value
        
        npv_15y -= net_cost  # Subtract initial investment
        
        # Calculate payback with fallback
        if payback_year is None:
            # Simple payback if dynamic didn't work
            first_year_savings = annual_generation * current_tariff
            payback_year = net_cost / first_year_savings if first_year_savings > 0 else 15
        
        # ENHANCED: Create scenario table with tariff scenarios
        scenario_table = {}
        
        if tariff.scenarios:
            for scenario_name in ['conservative', 'aggressive']:
                if scenario_name in tariff.scenarios:
                    scenario_data = tariff.scenarios[scenario_name]
                    if scenario_data:
                        # Get final year tariff for this scenario
                        final_year = max(scenario_data.keys())
                        final_tariff = scenario_data[final_year]
                        scenario_growth = ((final_tariff / current_tariff) ** (1/(final_year - 2024))) - 1
                        
                        # Estimate NPV for this scenario
                        scenario_total = 0
                        for year in range(1, 16):
                            year_tariff = current_tariff * ((1 + scenario_growth) ** (year - 1))
                            year_gen = annual_generation * ((1 - 0.005) ** (year - 1))
                            scenario_total += (year_tariff * year_gen) / ((1.08) ** year)
                        
                        scenario_npv = scenario_total - net_cost
                        scenario_payback = net_cost / (annual_generation * current_tariff * 1.5) if scenario_name == 'aggressive' else payback_year * 1.1
                        
                        scenario_table[scenario_name] = {
                            'roi': (scenario_npv / net_cost) * 100 / 15,  # Annualized
                            'npv': scenario_npv,
                            'payback': scenario_payback,
                            'tariff_growth': scenario_growth * 100
                        }
        
        # Enhanced logging
        log.info(f"DYNAMIC ROI RESULTS:")
        log.info(f"  Dynamic payback: {payback_year:.1f} years")
        log.info(f"  15-year NPV: â‚¹{npv_15y:,.0f}")
        log.info(f"  First year savings: â‚¹{annual_cashflows[0]:,.0f}")
        log.info(f"  Total 15-year savings: â‚¹{total_savings_15y:,.0f}")
        
        return ROIOut(
            npv_15y_inr=npv_15y,
            payback_years=payback_year,
            annual_savings_inr=annual_cashflows[0] if annual_cashflows else 0,
            scenario_table=scenario_table
        )
        
    except Exception as e:
        log.error(f"Enhanced fallback ROI calculation failed: {e}")
        
        # FINAL FALLBACK: Basic calculation with some dynamic elements
        cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
        total_cost = (cost_min + cost_max) / 2
        monthly_gen = safe_float_conversion(sizing.monthly_generation_kwh, 400)
        
        # Try to get current tariff from forecast
        current_tariff = 7.0
        if tariff.base_forecast:
            current_tariff = safe_float_conversion(next(iter(tariff.base_forecast.values())), 7.0)
        
        # Basic calculation with some growth assumption
        annual_savings = monthly_gen * 12 * current_tariff
        
        # Assume 8% tariff growth for basic NPV
        total_savings_with_growth = 0
        for year in range(1, 16):
            year_tariff = current_tariff * ((1.08) ** (year - 1))
            year_savings = monthly_gen * 12 * year_tariff
            discounted = year_savings / ((1.08) ** year)
            total_savings_with_growth += discounted
        
        subsidy_rate = safe_float_conversion(tech.subsidy_now, 0.3)
        net_cost = total_cost * (1 - subsidy_rate)
        
        payback = net_cost / annual_savings if annual_savings > 0 else 10
        npv_15y = total_savings_with_growth - net_cost
        
        log.info(f"FINAL FALLBACK ROI (with growth estimation):")
        log.info(f"  Assumed tariff growth: 8%/year")
        log.info(f"  Dynamic NPV: â‚¹{npv_15y:,.0f}")
        log.info(f"  Basic payback: {payback:.1f} years")
        
        return ROIOut(
            npv_15y_inr=npv_15y,
            payback_years=payback,
            annual_savings_inr=annual_savings,
            scenario_table={"base": {"roi": annual_savings, "payback": payback, "npv": npv_15y}}
        )

# Replace the run_risk function in integration_manager.py with this updated version

def run_risk(user: UserRequest, sizing: SizingOut, roi: ROIOut, tech: TechTrendOut, weather: WeatherOut) -> RiskOut:
    try:
        payload = {
            "user_request": asdict(user),
            "sizing": asdict(sizing),
            "roi": asdict(roi),
            "tech": asdict(tech),
            "weather": asdict(weather)
        }
        analyzer = EnhancedRiskAnalyzer()
        
        # Call the new integration method
        risk_results = analyzer.analyze_from_pipeline(payload)
        
        # Handle both success and error cases
        if "error" in risk_results:
            log.warning(f"Risk analysis completed with warnings: {risk_results['error']}")
        else:
            log.info("Risk analysis completed successfully")
        
        overall_risk = risk_results.get("overall_risk", "Moderate")
        components = risk_results.get("risk_components", {})
        
        # Add additional insights to components
        components.update({
            "overall_risk_score": risk_results.get("overall_risk_score", 0.4),
            "success_probability": risk_results.get("success_probability", 0.7),
            "investment_recommendation": risk_results.get("investment_recommendation", "PROCEED WITH CAUTION"),
            "key_risk_drivers": risk_results.get("key_risk_drivers", []),
            "confidence": risk_results.get("confidence", 0.75),
            "ownership_feasibility": risk_results.get("ownership_feasibility", {}),
            "monte_carlo_summary": risk_results.get("monte_carlo_summary", {}),
            "vendor_insights": risk_results.get("vendor_insights", {})
        })
        
        return RiskOut(overall_risk=overall_risk, components=components)
        
    except Exception as e:
        log.error(f"Risk analysis failed: {e}")
        return RiskOut(
            overall_risk="Moderate", 
            components={
                "financial_risk": 0.4,
                "technical_risk": 0.3,
                "policy_risk": 0.2,
                "environmental_risk": 0.25,
                "error": f"Analysis failed: {str(e)}"
            }
        )



def run_user_clustering_FIXED(user: UserRequest, roi: ROIOut, sizing: SizingOut) -> UserClusteringOut:
    """
    FIXED: Enhanced user clustering with proper NaN handling and error recovery
    """
    try:
        log.info("Starting FIXED user clustering analysis...")
        
        # Initialize the FIXED V4 clustering system
        cluster_system = FixedUserClusteringV4(
            algorithm='agglomerative', 
            scaler_type='robust', 
            use_pca=False
        )
        
        # Train on synthetic data (this should work)
        log.info("Training clustering model...")
        df = cluster_system.create_synthetic_data_V4(n_samples=1000)
        train_result = cluster_system.train_FIXED_clustering_V4(df)
        
        if 'error' in train_result:
            log.error(f"User clustering training failed: {train_result['error']}")
            return UserClusteringOut()
        
        log.info(f"Training successful - Silhouette: {train_result['validation_metrics']['silhouette_score']:.3f}")
        
        # Clean user profile with enhanced validation
        user_profile = clean_user_data_for_clustering_FIXED(user)
        
        # Additional validation before prediction
        log.info("Validating user profile for clustering...")
        
        # Check for any remaining issues
        profile_issues = []
        for key, value in user_profile.items():
            if value is None:
                profile_issues.append(f"Null value for {key}")
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                profile_issues.append(f"Invalid float value for {key}: {value}")
            elif isinstance(value, str) and value.strip() == "":
                profile_issues.append(f"Empty string for {key}")
        
        if profile_issues:
            log.warning(f"Profile validation issues found: {profile_issues}")
            # Apply final cleanup
            for key, value in user_profile.items():
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    defaults = {
                        "location": "Mumbai",
                        "risk_tolerance": "medium",
                        "timeline_preference": "flexible",
                        "priority": "cost", 
                        "income_bracket": "Medium",
                        "house_type": "independent",
                        "monthly_bill": 2500.0,
                        "budget_max": 300000.0,
                        "roof_area": 100.0
                    }
                    user_profile[key] = defaults.get(key, "unknown")
        
        # Final validation - ensure no NaN values exist
        log.info("Performing final NaN check...")
        for key, value in user_profile.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                log.error(f"NaN still present for {key}: {value}")
                return UserClusteringOut()
        
        log.info(f"Clean user profile: {user_profile}")
        
        # Make prediction with enhanced error handling
        try:
            prediction = cluster_system.predict_user_cluster_V4(user_profile)
            
            if 'error' in prediction:
                log.error(f"Clustering prediction failed: {prediction['error']}")
                return UserClusteringOut()
            
            log.info(f"Clustering prediction successful: {prediction.get('cluster_name', 'Unknown')}")
            
            return UserClusteringOut(
                cluster_id=prediction.get('cluster_id'),
                cluster_name=prediction.get('cluster_name'),
                strategy=prediction.get('strategy'),
                business_value=prediction.get('business_value'),
                readiness_scores=prediction.get('user_scores'),
                confidence=prediction.get('confidence'),
                prediction_method=prediction.get('prediction_method')
            )
            
        except Exception as prediction_error:
            log.error(f"Prediction execution failed: {str(prediction_error)}")
            # Check if it's a NaN-related error
            if "NaN" in str(prediction_error) or "missing values" in str(prediction_error):
                log.error("NaN values detected in prediction pipeline - this indicates a data preparation issue")
                
                # Debug: Print the user profile to see what's causing NaN
                log.error(f"Problematic user profile: {user_profile}")
                
                # Try to identify which field has NaN
                for key, value in user_profile.items():
                    if isinstance(value, float) and np.isnan(value):
                        log.error(f"Found NaN in field: {key} = {value}")
                
            return UserClusteringOut()
            
    except Exception as e:
        log.error(f"User clustering system failed: {str(e)}")
        return UserClusteringOut()


# Updated integration function to use the fixed version
def run_user_clustering(user: UserRequest, roi: ROIOut, sizing: SizingOut) -> UserClusteringOut:
    """Updated to use the FIXED version"""
    return run_user_clustering_FIXED(user, roi, sizing)

def clean_user_data_for_clustering_FIXED(user: UserRequest) -> Dict[str, Any]:
    """
    FIXED: Clean and validate user data for clustering with proper null handling and NaN prevention
    """
    
    # Enhanced mapping with fallback handling
    risk_tolerance_map = {
        'conservative': 'low', 'low': 'low',
        'moderate': 'medium', 'medium': 'medium', 
        'aggressive': 'high', 'high': 'high',
        'dynamic': 'high'
    }
    
    timeline_map = {
        'immediate': 'immediate', 'asap': 'immediate',
        'flexible': 'flexible', 'moderate': 'flexible',
        'patient': 'wait', 'long_term': 'wait', 'wait': 'wait'
    }
    
    priority_map = {
        'cost': 'cost', 'price': 'cost', 'budget': 'cost',
        'quality': 'quality', 'performance': 'quality',
        'sustainability': 'sustainability', 'green': 'sustainability', 'environmental': 'sustainability'
    }
    
    house_type_map = {
        'independent': 'independent', 'villa': 'villa', 'bungalow': 'independent',
        'apartment': 'apartment', 'flat': 'apartment', 'condo': 'apartment'
    }
    
    income_bracket_map = {
        'low': 'Low', 'medium': 'Medium', 'high': 'High', 'premium': 'High'
    }
    
    # Clean and validate each field with proper fallbacks
    location = "Mumbai"  # Default fallback
    if user.location:
        location_parts = user.location.split(',')
        if location_parts:
            location_clean = location_parts[0].strip()
            # Map to known cities
            known_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad', 'Ahmedabad', 'Kolkata', 'Kochi', 'Jaipur']
            if location_clean in known_cities:
                location = location_clean
            else:
                # Find closest match or use default
                location_lower = location_clean.lower()
                for city in known_cities:
                    if city.lower() in location_lower or location_lower in city.lower():
                        location = city
                        break
    
    # Ensure all categorical values are properly mapped
    risk_tolerance = risk_tolerance_map.get(user.risk_tolerance.lower() if user.risk_tolerance else 'medium', 'medium')
    timeline_preference = timeline_map.get(user.timeline_preference.lower() if user.timeline_preference else 'flexible', 'flexible')
    priority = priority_map.get(user.priority.lower() if user.priority else 'cost', 'cost')
    house_type = house_type_map.get(user.house_type.lower() if user.house_type else 'independent', 'independent')
    income_bracket = income_bracket_map.get(user.income_bracket.lower() if user.income_bracket else 'medium', 'Medium')
    
    # Ensure all numerical values are clean
    monthly_bill = safe_float_conversion(user.monthly_bill, 2500)
    budget_max = safe_float_conversion(user.budget_inr, 300000)
    roof_area = safe_float_conversion(user.roof_area_m2, 100)
    
    # Final validation - ensure no None or NaN values
    cleaned_data = {
        "location": str(location),
        "risk_tolerance": str(risk_tolerance),
        "timeline_preference": str(timeline_preference),
        "priority": str(priority),
        "income_bracket": str(income_bracket),
        "house_type": str(house_type),
        "monthly_bill": float(monthly_bill),
        "budget_max": float(budget_max),
        "roof_area": float(roof_area)
    }
    
    # Double-check for any remaining issues
    for key, value in cleaned_data.items():
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            log.warning(f"Found invalid value for {key}: {value}, applying fallback")
            if key in ["monthly_bill"]:
                cleaned_data[key] = 2500.0
            elif key in ["budget_max"]:
                cleaned_data[key] = 300000.0
            elif key in ["roof_area"]:
                cleaned_data[key] = 100.0
            else:
                # For categorical fields, use safe defaults
                defaults = {
                    "location": "Mumbai",
                    "risk_tolerance": "medium",
                    "timeline_preference": "flexible", 
                    "priority": "cost",
                    "income_bracket": "Medium",
                    "house_type": "independent"
                }
                cleaned_data[key] = defaults.get(key, "unknown")
    
    return cleaned_data



# integration_manager.py

def run_heuristic_search(user: UserRequest, roi: ROIOut, sizing: SizingOut,
                        tariff: TariffOut, tech: TechTrendOut, risk: RiskOut) -> HeuristicSearchOut:
    """
    Reworked: Use A* time-only search (TimeAStarSolarSearch) that evaluates 'when to install'
    using AdvancedROICalculator for authentic payback & coverage. Final ROI run uses higher
    Monte Carlo iterations for precision.
    """
    try:
        log.info("Running enhanced heuristic search with DYNAMIC A* time-search...")

        # UPDATED: Check if sizing already determined non-viability
        if (hasattr(sizing, 'constraint_violated') and sizing.constraint_violated) or \
           (sizing.system_capacity_kw == 0.0 and 
            any("not viable" in warning.lower() for warning in (sizing.warnings or []))):
            
            log.info("Heuristic search: System already determined non-viable by sizing engine")
            rejection_reason = getattr(sizing, 'constraint_violation_reason', 'System constraints not met')
            
            return HeuristicSearchOut(
                optimal_scenario_type="system_not_viable",  # UPDATED: Instead of "install_now"
                roi=0.0,
                risk_score=10.0,  # High risk since system is not viable
                payback_period=float('inf'),
                confidence=0.0,
                cost=0,
                search_metadata={
                    'algorithm': 'Constraint Violation Detected',
                    'rejection_reason': rejection_reason,
                    'sizing_constraint_violated': True
                },
                action_plan=[{
                    'action': 'system_rejected',
                    'year': 2025,
                    'description': f'Solar system not viable: {rejection_reason}',
                    'cost': 0,
                    'trigger': 'constraint_violation'
                }]
            )

        # UPDATED: Additional safety check for zero capacity systems
        if sizing.system_capacity_kw <= 0:
            log.warning("Heuristic search: Zero capacity system detected, treating as non-viable")
            return HeuristicSearchOut(
                optimal_scenario_type="system_not_viable",
                roi=0.0,
                risk_score=10.0,
                payback_period=float('inf'),
                confidence=0.0,
                cost=0,
                search_metadata={
                    'algorithm': 'Zero Capacity Rejection',
                    'rejection_reason': 'System capacity is zero or negative'
                },
                action_plan=[]
            )

        # --- 1) dynamic rates (keep your logic) ---
        tariff_growth_rate = 0.08
        if tariff.base_forecast and len(tariff.base_forecast) >= 2:
            sorted_years = sorted(tariff.base_forecast.keys())
            initial_tariff = safe_float_conversion(tariff.base_forecast[sorted_years[0]])
            final_tariff = safe_float_conversion(tariff.base_forecast[sorted_years[-1]])
            years_span = sorted_years[-1] - sorted_years[0]
            if years_span > 0 and initial_tariff > 0:
                calculated_rate = ((final_tariff / initial_tariff) ** (1 / years_span)) - 1
                tariff_growth_rate = max(0.01, min(0.15, calculated_rate))
        log.info(f"  Using dynamic tariff growth rate: {tariff_growth_rate:.2%}")

        tech_decline_rate = 0.05
        cost_now = safe_float_conversion(tech.cost_now_inr_per_w)
        cost_12mo = safe_float_conversion(tech.cost_12mo_inr_per_w)
        if cost_now > 0 and cost_12mo > 0:
            calculated_decline = 1.0 - (cost_12mo / cost_now)
            tech_decline_rate = max(0.01, min(0.15, calculated_decline))
        log.info(f"  Using dynamic tech cost decline rate: {tech_decline_rate:.2%}")

        # --- 2) core inputs from pipeline ---
        user_budget = safe_float_conversion(user.budget_inr, 400000.0)
        monthly_consumption = safe_float_conversion(user.monthly_consumption_kwh, 350.0)
        monthly_bill = safe_float_conversion(user.monthly_bill, 2500.0)

        # Robust current tariff: pick the first value from base_forecast if present; else fallback
        if tariff.base_forecast:
            current_tariff = safe_float_conversion(next(iter(tariff.base_forecast.values()), 8.5))
        else:
            current_tariff = 8.5

        # required capacity from sizing (target capacity)
        required_capacity_kw = safe_float_conversion(sizing.system_capacity_kw, 5.0)

        # tech.cost_now_inr_per_w is per W in your pipeline; convert to per kW
        current_tech_cost_per_kw = safe_float_conversion(tech.cost_now_inr_per_w, 27.35) * 1000.0

        log.info(f"Search inputs: budget=₹{user_budget:,.0f}, consumption={monthly_consumption:.0f} kWh/month, tariff={current_tariff:.2f} INR/kWh, cost_per_kW=₹{current_tech_cost_per_kw:,.0f}")

        # --- 3) prepare templates passed to A* (will be cloned/modified inside search) ---
        # Reuse dataclasses you already defined in pipeline:
        system_spec_template = SolarSystemSpec(
            capacity_kw=required_capacity_kw,
            panel_efficiency=safe_float_conversion(getattr(tech, 'efficiency_percent', None), 19.42),
            panel_degradation_rate=safe_float_conversion(getattr(tech, 'degradation_percent', None), 0.45),
            inverter_efficiency=0.97,
            system_losses=0.10
        )

        financial_params_template = FinancialParameters(
            system_cost_per_kw=current_tech_cost_per_kw,
            installation_cost=safe_float_conversion(getattr(sizing, 'installation_cost_est_inr', None), 10000),
            maintenance_cost_annual=safe_float_conversion(getattr(tech, 'maintenance_est_inr', None), 2000),
            insurance_cost_annual=1000.0,
            electricity_tariff=current_tariff,
            discount_rate=0.06,
            inflation_rate=0.05,
            tariff_escalation_rate=tariff_growth_rate,
            net_metering_rate=1.0,
            subsidy_amount=safe_float_conversion(getattr(sizing, 'subsidy_inr', None), 0.0),
            loan_amount=safe_float_conversion(getattr(user, 'loan_amount_inr', None), 0.0),
            loan_interest_rate=0.09,
            loan_tenure_years=10,
            tax_benefits=0.0
        )

        # risk params from pipeline (pass-through)
        risk_params = RiskParameters(
            technology_risk=safe_float_conversion(getattr(risk, 'technology_risk', None), 0.05),
            weather_risk=safe_float_conversion(getattr(risk, 'weather_risk', None), 0.10),
            policy_risk=safe_float_conversion(getattr(risk, 'policy_risk', None), 0.15),
            market_risk=safe_float_conversion(getattr(risk, 'market_risk', None), 0.08),
            execution_risk=safe_float_conversion(getattr(risk, 'execution_risk', None), 0.05),
            correlation_matrix=getattr(risk, 'correlation_matrix', None)
        )

        # location object (from pipeline). If not present create a minimal one from sizing/weather outputs
        if hasattr(sizing, 'location_data') and sizing.location_data:
            location_data = sizing.location_data
        else:
            # fall back to constructing LocationData from known pipeline values
            location_data = LocationData(
                latitude=safe_float_conversion(getattr(sizing, 'latitude', None), 18.5),
                longitude=safe_float_conversion(getattr(sizing, 'longitude', None), 73.8),
                annual_irradiance=safe_float_conversion(getattr(tariff, 'annual_irradiance_kwh_per_kw', None), safe_float_conversion(getattr(sizing, 'annual_generation_kwh_per_kw', None), 1300.0)),
                irradiance_std=0.1,
                weather_reliability=safe_float_conversion(getattr(tech, 'weather_reliability', None), 0.9),
                grid_connection_cost=8000,
                permitting_cost=3000,
                installation_complexity=1.0,
                local_labor_cost_per_day=1200
            )

        # --- 4) instantiate ROI calculator & A* search engine ---
        # Use a lower MC count during the search (speed), and we'll run a final precise MC after the candidate is selected.
        fast_mc_iters = 300
        final_mc_iters = 1200

        roi_calc_fast = AdvancedROICalculator(monte_carlo_iterations=fast_mc_iters)
        from engines._heuristic_search import TimeAStarSolarSearch  # ensure module available per earlier code
        a_star = TimeAStarSolarSearch(roi_calculator=roi_calc_fast, max_months=36)

        # Run the A* time-search
        a_star_result = a_star.search(
            user_budget=user_budget,
            current_tariff=current_tariff,
            required_capacity_kw=required_capacity_kw,
            current_tech_cost_per_kw=current_tech_cost_per_kw,
            monthly_consumption_kwh=monthly_consumption,
            tariff_growth_rate=tariff_growth_rate,
            tech_decline_rate=tech_decline_rate,
            system_spec_template=system_spec_template,
            location_data=location_data,
            financial_params_template=financial_params_template,
            risk_params=risk_params
        )

        log.info(f"Time-A* result: {a_star_result}")

        # --- 5) Final high-precision verification: run full ROI calc at chosen timestamp ---
        chosen_months = int(a_star_result.get('months_to_wait', 0))
        # project tariff & tech cost to chosen_months
        years_proj = chosen_months / 12.0
        proj_tariff = current_tariff * ((1 + tariff_growth_rate) ** years_proj)
        proj_tech_cost_per_kw = current_tech_cost_per_kw * ((1 - tech_decline_rate) ** years_proj)

        # clone templates and override
        system_spec_final = SolarSystemSpec(**system_spec_template.__dict__)
        system_spec_final.capacity_kw = required_capacity_kw

        fin_final = FinancialParameters(**financial_params_template.__dict__)
        fin_final.system_cost_per_kw = proj_tech_cost_per_kw
        fin_final.electricity_tariff = proj_tariff
        fin_final.tariff_escalation_rate = tariff_growth_rate

        roi_calc_final = AdvancedROICalculator(monte_carlo_iterations=final_mc_iters)
        final_roi = roi_calc_final.calculate_comprehensive_roi(
            system_spec=system_spec_final,
            location_data=location_data,
            financial_params=fin_final,
            risk_params=risk_params,
            time_horizon=TimeHorizon.LONG_TERM
        )

        # Build a clean base_result structure compatible with rest of code
        base_result = {
            'optimal_scenario_type': a_star_result.get('optimal_scenario_type', 'install_now'),
            'months_to_wait': chosen_months,
            'payback_period': final_roi.payback_period,
            'roi': final_roi.simple_roi,
            'cost': round(final_roi.initial_cost, 2) if getattr(final_roi, 'initial_cost', None) is not None else round(required_capacity_kw * proj_tech_cost_per_kw, 2),
            'coverage_pct': (
        (final_roi.energy_production_forecast[0] / (monthly_consumption * 12) * 100)
        if final_roi.energy_production_forecast else
        (required_capacity_kw * 1400) / (monthly_consumption * 12) * 100
    ),
            'confidence': safe_float_conversion(final_roi.confidence_level, 0.7),
            'risk_score': safe_float_conversion(final_roi.uncertainty_score, 5.0),
            'search_metadata': {
                'algorithm': 'Time-A* Search (time-only)',
                'a_star_iterations': a_star_result.get('iterations', 0),
                'chosen_months': chosen_months,
                'fast_mc_iters': fast_mc_iters,
                'final_mc_iters': final_mc_iters
            },
            'search_successful': a_star_result.get('search_successful', False)
        }

        # Compute coverage safely (sizing.monthly_generation_kwh may exist)
        coverage = safe_division(sizing.monthly_generation_kwh, user.monthly_consumption_kwh, 1.0) * 100

        # Build scenarios (trade-offs) - keep original shape but update fields to use final_roi / base_result
        scenarios = []
        scenarios.append({
            "action": "install_now_optimal",
            "capacity_kw": sizing.system_capacity_kw,
            "coverage": coverage,
            "cost": (extract_cost_range(sizing.cost_range_inr)[0] + extract_cost_range(sizing.cost_range_inr)[1]) / 2 if sizing.cost_range_inr else base_result['cost'],
            "payback": roi.payback_years if hasattr(roi, 'payback_years') else base_result['payback_period'],
            "roi": (roi.npv_15y_inr / sizing.cost_range_inr[0] * 100) if sizing.cost_range_inr else base_result['roi']
        })

        scenarios.append({
            "action": base_result['optimal_scenario_type'],
            "capacity_kw": sizing.system_capacity_kw,
            "coverage": round(base_result.get('coverage_pct', coverage), 1),
            "cost": base_result['cost'],
            "payback": base_result['payback_period'],
            "roi": base_result['roi']
        })

        # Return HeuristicSearchOut (use same fields as before)
        return HeuristicSearchOut(
            optimal_scenario_type=base_result.get('optimal_scenario_type', "install_now"),
            roi=safe_float_conversion(base_result.get('roi')),
            risk_score=safe_float_conversion(base_result.get('risk_score')),
            payback_period=safe_float_conversion(base_result.get('payback_period')),
            confidence=safe_float_conversion(base_result.get('confidence')),
            cost=safe_float_conversion(base_result.get('cost')),
            search_metadata=base_result.get('search_metadata', {'algorithm': 'Dynamic A*'}),
            action_plan=scenarios
        )

    except Exception as e:
        log.error(f"Enhanced heuristic search failed: {e}", exc_info=True)
        
        # UPDATED: Check if failure is due to non-viable system
        if sizing.system_capacity_kw == 0.0:
            return HeuristicSearchOut(
                optimal_scenario_type="system_not_viable",
                roi=0.0,
                risk_score=10.0,
                payback_period=float('inf'),
                confidence=0.0,
                cost=0,
                search_metadata={
                    'algorithm': 'Error Fallback - Non-viable System',
                    'error': str(e)
                },
                action_plan=[]
            )
        else:
            # Existing fallback for viable systems with errors
            return HeuristicSearchOut(
                optimal_scenario_type="install_now",
                roi=15.0,
                risk_score=5.0,
                payback_period=8.0,
                confidence=0.6,
                cost=250000,
                search_metadata={'algorithm': 'Error Fallback'},
                action_plan=[]
            )


   

def create_simplified_heuristic_result(user: UserRequest, sizing: SizingOut, roi: ROIOut, 
                                     tariff: TariffOut, tech: TechTrendOut) -> HeuristicSearchOut:
    """
    ENHANCED: Create simplified heuristic result with better logic
    """
    try:
        log.info("Creating enhanced simplified heuristic decision result...")
        
        # Get basic parameters
        user_budget = safe_float_conversion(user.budget_inr, 300000)
        cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
        system_cost = (cost_min + cost_max) / 2
        
        payback_years = safe_float_conversion(roi.payback_years, 8)
        roi_value = safe_float_conversion(roi.npv_15y_inr, 200000) / system_cost * 100 if system_cost > 0 else 15
        
        # Enhanced decision logic
        risk_tolerance = user.risk_tolerance.lower()
        timeline_pref = user.timeline_preference.lower()
        priority = user.priority.lower()
        
        # Budget analysis
        budget_ratio = system_cost / user_budget if user_budget > 0 else 2.0
        budget_stress = "high" if budget_ratio > 1.2 else "medium" if budget_ratio > 0.9 else "low"
        
        log.info(f"Decision factors: budget_ratio={budget_ratio:.2f}, payback={payback_years:.1f}y, roi={roi_value:.1f}%")
        
        # Enhanced decision matrix
        scenario = "install_now"  # default
        confidence = 0.7
        
        # Budget-first decision logic
        if budget_stress == "high":
            if timeline_pref == 'patient':
                scenario = "wait_12_months"
                confidence = 0.65
            else:
                scenario = "wait_6_months" 
                confidence = 0.60
        elif budget_stress == "medium":
            if payback_years > 8 or roi_value < 12:
                scenario = "wait_3_months"
                confidence = 0.70
            else:
                scenario = "install_now"
                confidence = 0.75
        else:  # budget_stress == "low"
            if timeline_pref == 'immediate' or risk_tolerance in ['aggressive', 'high']:
                scenario = "install_now"
                confidence = 0.85
            elif payback_years <= 6:
                scenario = "install_now"
                confidence = 0.80
            else:
                scenario = "wait_3_months"
                confidence = 0.75
        
        # Priority adjustments
        if priority == 'sustainability' and scenario in ['wait_6_months', 'wait_12_months']:
            # Environmental priority favors immediate action
            scenario = "wait_3_months" if scenario == "wait_6_months" else "wait_6_months"
            confidence += 0.05
        
        # High tariff growth adjustment
        if tariff.base_forecast:
            sorted_years = sorted(tariff.base_forecast.keys())
            if len(sorted_years) >= 2:
                initial_tariff = tariff.base_forecast[sorted_years[0]]
                final_tariff = tariff.base_forecast[sorted_years[-1]]
                years_span = sorted_years[-1] - sorted_years[0]
                if years_span > 0 and initial_tariff > 0:
                    growth_rate = ((final_tariff / initial_tariff) ** (1/years_span)) - 1
                    
                    # High tariff growth (>10%) strongly favors immediate installation
                    if growth_rate > 0.10:
                        if scenario in ['wait_6_months', 'wait_12_months'] and budget_ratio <= 1.3:
                            scenario = "install_now" if budget_ratio <= 1.1 else "wait_3_months"
                            confidence = min(confidence + 0.15, 0.9)
                            log.info(f"High tariff growth ({growth_rate:.1%}) adjusted to: {scenario}")
        
        # Risk tolerance final adjustment
        if risk_tolerance == 'conservative':
            if scenario == "install_now" and budget_ratio > 0.8:
                scenario = "wait_3_months"
            confidence *= 0.95
        elif risk_tolerance in ['aggressive', 'high']:
            if scenario in ['wait_6_months', 'wait_12_months']:
                scenario = "wait_3_months" if scenario == 'wait_6_months' else "wait_6_months"
            confidence *= 1.05
        
        # Ensure confidence bounds
        confidence = max(0.5, min(0.9, confidence))
        
        log.info(f"Enhanced simplified result: {scenario} (confidence: {confidence:.1%})")
        log.info(f"  Budget stress: {budget_stress}, ROI: {roi_value:.1f}%, Payback: {payback_years:.1f}y")
        
        return HeuristicSearchOut(
    optimal_scenario_type=scenario,
    roi=roi_value,
    risk_score=5.0,
    payback_period=payback_years,
    confidence=confidence,
    cost=system_cost,
    search_metadata={
        'algorithm_type': 'Enhanced Decision Logic',
        'decision_factors': {
            'budget_ratio': budget_ratio,
            'budget_stress': budget_stress,
            'payback_years': payback_years,
            'roi_estimate': roi_value,
            'risk_tolerance': risk_tolerance,
            'timeline_preference': timeline_pref,
            'priority': priority
        },
        'confidence_level': 'enhanced',
        'adjustments_applied': ['budget_analysis', 'tariff_growth', 'risk_tolerance'],
        'calculated_f_score': system_cost + (15 - roi_value) * 5000
    },
    action_plan=[]
)

        
    except Exception as e:
        log.error(f"Enhanced simplified heuristic creation failed: {e}")
        
        # Ultimate fallback
        return HeuristicSearchOut(
    optimal_scenario_type="install_now",
    roi=15.0,
    risk_score=5.0,
    payback_period=8.0,
    confidence=0.6,
    cost=300000,
    search_metadata={
        'algorithm_type': 'Emergency Fallback',
        'fallback_reason': 'All heuristic methods failed',
        'calculated_f_score': 350000
    },
    action_plan=[]
)


# Replace the run_vendors function in your integration_manager.py with this updated version

def run_vendors(user: UserRequest, sizing: SizingOut, risk: RiskOut, heuristic: HeuristicSearchOut) -> VendorOut:
    try:
        payload = {
            "user_request": asdict(user),
            "sizing": asdict(sizing),
            "risk": asdict(risk),
            "heuristic": asdict(heuristic)
        }
        comparator = EnhancedCompanyComparator()
        
        # Call the new integration method
        result = comparator.compare_vendors_for_pipeline(payload)
        
        # Handle both success and error cases
        if "error" in result:
            log.warning(f"Company comparison failed: {result['error']}")
            ranked_vendors = result.get("companies", [])
            method = result.get("method", "fallback")
        else:
            ranked_vendors = result.get("companies", []) or result.get("ranked_vendors", [])
            method = result.get("method", "Risk-Adjusted MCDA")
            
            # Log additional intelligence if available
            if "negotiation_intelligence" in result:
                log.info(f"Negotiation intelligence generated for {len(result['negotiation_intelligence'])} vendors")
            
            if "recommendations" in result:
                recommendations = result["recommendations"]
                log.info(f"Best overall: {recommendations.get('best_overall')}")
                log.info(f"Best value: {recommendations.get('best_value')}")
                log.info(f"Lowest risk: {recommendations.get('lowest_risk')}")
        
        return VendorOut(
            ranked_vendors=ranked_vendors, 
            method=method
        )
        
    except Exception as e:
        log.error(f"Vendor comparison failed: {e}")
        # Enhanced fallback with more realistic data
        cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
        system_capacity = safe_float_conversion(sizing.system_capacity_kw, 5.0)
        
        fallback_vendors = [
            {
                "name": "Tata Power Solar",
                "rank": 1,
                "rating": 4.5,
                "overall_score": 85.2,
                "tier": "Premium",
                "risk_category": "Low Risk",
                "estimated_cost": cost_min,
                "warranty_years": 25,
                "installation_time": "13 days",
                "negotiation_potential": "Moderate",
                "value_proposition": "Premium service quality with strong market presence",
                "strengths": ["Excellent service quality", "Strong warranty support", "Low vendor risk"],
                "weaknesses": ["Higher pricing"],
                "service_metrics": {
                    "installation_time": "13 days",
                    "warranty_claim_success": "92%",
                    "response_time": "24 hours",
                    "service_coverage": "85%"
                },
                "financial_indicators": {
                    "cost_per_kw": f"â‚¹{int(cost_min/system_capacity):,}",
                    "financing_available": True
                }
            },
            {
                "name": "Waaree Energies",
                "rank": 2,
                "rating": 4.2,
                "overall_score": 82.1,
                "tier": "Premium",
                "risk_category": "Low Risk",
                "estimated_cost": cost_max,
                "warranty_years": 25,
                "installation_time": "11 days",
                "negotiation_potential": "High",
                "value_proposition": "Superior installation quality with competitive pricing",
                "strengths": ["Excellent installation standards", "Fast delivery", "In-house manufacturing"],
                "weaknesses": ["Slightly lower service quality"],
                "service_metrics": {
                    "installation_time": "11 days",
                    "warranty_claim_success": "89%",
                    "response_time": "48 hours",
                    "service_coverage": "78%"
                },
                "financial_indicators": {
                    "cost_per_kw": f"â‚¹{int(cost_max/system_capacity):,}",
                    "financing_available": True
                }
            },
            {
                "name": "Adani Solar",
                "rank": 3,
                "rating": 4.1,
                "overall_score": 80.3,
                "tier": "Premium",
                "risk_category": "Moderate Risk",
                "estimated_cost": (cost_min + cost_max) / 2,
                "warranty_years": 25,
                "installation_time": "12 days",
                "negotiation_potential": "Moderate",
                "value_proposition": "Balanced quality and service with strong after-sales support",
                "strengths": ["Good after-sales service", "Strong financial backing", "Reliable performance"],
                "weaknesses": ["Limited financing options", "Slower complaint resolution"],
                "service_metrics": {
                    "installation_time": "12 days",
                    "warranty_claim_success": "87%",
                    "response_time": "36 hours",
                    "service_coverage": "82%"
                },
                "financial_indicators": {
                    "cost_per_kw": f"â‚¹{int((cost_min + cost_max) / 2 / system_capacity):,}",
                    "financing_available": False
                }
            }
        ]
        
        return VendorOut(ranked_vendors=fallback_vendors, method="enhanced_fallback")

def run_safety(user: UserRequest, sizing: SizingOut, roi: ROIOut, risk: RiskOut,
               heuristic_search: HeuristicSearchOut) -> SafetyGateOut:
    """
    FIXED: Enhanced safety gate using the mistake prevention engine with proper result handling
    """
    try:
        payload = {
            "user_request": asdict(user),
            "sizing": asdict(sizing),
            "roi": asdict(roi),
            "risk": asdict(risk),
            "heuristic_search": asdict(heuristic_search)
        }
        
        engine = EnhancedMistakePreventionEngine()
        
        # Try the new integration methods with proper result handling
        result = None
        
        if hasattr(engine, 'prevent_mistakes_from_pipeline'):
            try:
                result = engine.prevent_mistakes_from_pipeline(payload)
            except Exception as e:
                log.warning(f"prevent_mistakes_from_pipeline failed: {e}")
        
        if not result and hasattr(engine, 'validate_pipeline_data'):
            try:
                result = engine.validate_pipeline_data(payload)
            except Exception as e:
                log.warning(f"validate_pipeline_data failed: {e}")
        
        if not result and hasattr(engine, 'check_mistakes_in_pipeline'):
            try:
                result = engine.check_mistakes_in_pipeline(payload)
            except Exception as e:
                log.warning(f"check_mistakes_in_pipeline failed: {e}")
        
        if not result:
            log.warning("No integration method found for EnhancedMistakePreventionEngine, using enhanced fallback")
            return enhanced_fallback_safety_check(user, sizing, roi, risk, heuristic_search)
        
        # FIXED: Properly handle the result based on its type
        if hasattr(result, 'data'):
            # It's a wrapped result object - access the data
            result_data = result.data
        elif isinstance(result, dict):
            # It's already a dictionary
            result_data = result
        else:
            # Try to convert to dict or fall back
            try:
                result_data = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            except:
                log.warning("Could not extract data from result, using fallback")
                return enhanced_fallback_safety_check(user, sizing, roi, risk, heuristic_search)
        
        # Extract results from the data
        issues = result_data.get("warnings", []) or result_data.get("issues", []) or []
        confidence_intervals = result_data.get("confidence_intervals", {})
        
        # Additional safety metrics
        overall_confidence = result_data.get("overall_confidence", 0.7)
        success_probability = result_data.get("success_probability", 0.65)
        
        # Determine if system passes safety gate
        ok = len(issues) == 0 or (overall_confidence > 0.6 and success_probability > 0.6)
        
        # Log comprehensive results
        if result_data.get("status") == "success":
            log.info(f"Mistake prevention analysis completed: {len(issues)} issues found")
            log.info(f"Overall confidence: {overall_confidence:.0%}, Success probability: {success_probability:.0%}")
        else:
            log.warning(f"Mistake prevention analysis had issues: {result_data.get('error', 'Unknown error')}")
        
        return SafetyGateOut(
            issues=issues, 
            confidence_intervals=confidence_intervals, 
            ok=ok
        )
        
    except Exception as e:
        log.error(f"Enhanced mistake prevention engine failed: {e}")
        return enhanced_fallback_safety_check(user, sizing, roi, risk, heuristic_search)

def enhanced_fallback_safety_check(user: UserRequest, sizing: SizingOut, roi: ROIOut, 
                                 risk: RiskOut, heuristic_search: HeuristicSearchOut) -> SafetyGateOut:
    """
    Enhanced fallback safety checks when the main engine fails
    """
    issues = []
    confidence_intervals = {}
    
    try:
        # Enhanced payback period check
        payback = safe_float_conversion(roi.payback_years)
        if payback > 0:
            if payback < 3:
                issues.append(f"Payback period suspiciously short ({payback:.1f} years) - Verify calculations and assumptions")
            elif payback > 15:
                issues.append(f"Payback period too long ({payback:.1f} years) - Consider smaller system or alternative solutions")
            elif payback > 10:
                issues.append(f"Long payback period ({payback:.1f} years) - Review financial assumptions")
        
        # Enhanced cost reality check
        cost_range = extract_cost_range(sizing.cost_range_inr)
        budget = safe_float_conversion(user.budget_inr, 0)
        if budget > 0 and cost_range[0] > budget * 1.2:
            issues.append(f"System cost (â‚¹{cost_range[0]:,.0f}) significantly exceeds budget (â‚¹{budget:,.0f})")
        
        # System size vs consumption check
        monthly_consumption = safe_float_conversion(user.monthly_consumption_kwh, 0)
        monthly_generation = safe_float_conversion(sizing.monthly_generation_kwh, 0)
        if monthly_consumption > 0 and monthly_generation > 0:
            coverage_ratio = monthly_generation / monthly_consumption
            if coverage_ratio < 0.5:
                issues.append(f"System severely undersized: Only {coverage_ratio:.0%} coverage of consumption")
            elif coverage_ratio > 1.5:
                issues.append(f"System oversized: {coverage_ratio:.0%} coverage - Check grid export policies and economics")
        
        # Risk level assessment
        if risk.overall_risk:
            risk_level = risk.overall_risk.lower()
            if 'very high' in risk_level or 'extreme' in risk_level:
                issues.append(f"Very high risk level detected: {risk.overall_risk} - Comprehensive risk mitigation required")
            elif 'high' in risk_level:
                issues.append(f"High risk level: {risk.overall_risk} - Consider risk mitigation strategies")
        
        # Confidence score check
        confidence = safe_float_conversion(sizing.confidence_score, 1.0)
        if confidence < 0.6:
            issues.append(f"Low system sizing confidence ({confidence:.0%}) - Additional validation recommended")
        
        # ROI sanity check
        annual_savings = safe_float_conversion(roi.annual_savings_inr, 0)
        monthly_bill = safe_float_conversion(user.monthly_bill, 2500)
        if annual_savings > monthly_bill * 12 * 1.2:  # More than 120% of current bill
            issues.append(f"Unrealistic savings projection: â‚¹{annual_savings:,.0f} annual vs â‚¹{monthly_bill * 12:,.0f} current annual bill")
        
        # Location and house type compatibility
        location = user.location.lower() if user.location else ""
        house_type = user.house_type.lower() if user.house_type else ""
        
        if house_type == 'apartment':
            issues.append("Apartment installation - Verify society permissions and shared infrastructure compatibility")
        
        # Roof area feasibility
        roof_area = safe_float_conversion(user.roof_area_m2, 0)
        system_capacity = safe_float_conversion(sizing.system_capacity_kw, 0)
        if roof_area > 0 and system_capacity > 0:
            required_area = system_capacity * 6  # 6 sq.m per kW rough estimate
            if required_area > roof_area * 0.9:  # 90% utilization threshold
                issues.append(f"Roof space constraint: Need ~{required_area:.0f} sq.m for {system_capacity}kW system on {roof_area:.0f} sq.m roof")
        
        # Financial stress check
        if budget > 0 and monthly_bill > 0:
            estimated_emi = budget * 0.008  # Rough EMI at 9.5% for 10 years
            if estimated_emi > monthly_bill * 1.5:
                issues.append(f"Potential EMI stress: Est. EMI â‚¹{estimated_emi:,.0f} vs current bill â‚¹{monthly_bill:,.0f}")
        
        # Heuristic search validation
        if heuristic_search.confidence and heuristic_search.confidence < 0.5:
            issues.append(f"Low heuristic search confidence ({heuristic_search.confidence:.0%}) - Multiple scenarios show suboptimal results")
        
        # Calculate basic confidence intervals
        confidence_intervals = {
            "cost_accuracy": (0.7, 0.9),
            "generation_accuracy": (0.6, 0.8),
            "roi_accuracy": (0.5, 0.8),
            "timeline_accuracy": (0.6, 0.9)
        }
        
        # Adjust intervals based on issues found
        if len(issues) > 3:
            confidence_intervals = {k: (v[0] * 0.8, v[1] * 0.9) for k, v in confidence_intervals.items()}
        
    except Exception as e:
        log.error(f"Enhanced fallback safety check failed: {e}")
        issues.append(f"Safety analysis incomplete due to error: {str(e)}")
    
    # Determine overall safety status
    critical_issues = len([issue for issue in issues if any(word in issue.lower() 
                          for word in ['severe', 'critical', 'unrealistic', 'suspiciously', 'significantly exceeds'])])
    ok = critical_issues == 0 and len(issues) <= 2
    
    log.info(f"Enhanced fallback safety check completed: {len(issues)} issues, {critical_issues} critical, Status: {'PASS' if ok else 'REVIEW REQUIRED'}")
    
    return SafetyGateOut(
        issues=issues, 
        confidence_intervals=confidence_intervals, 
        ok=ok
    )

def save_pipeline_result_to_json(result: PipelineResult, filepath: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        log.info(f"Pipeline result saved to {filepath}")
    except Exception as e:
        log.error(f"Failed to save pipeline result: {e}")

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def run_pipeline(user: UserRequest) -> PipelineResult:
    log.info(f"Running FOAI pipeline for {user.location} / {user.state} / {user.category}")

    weather = run_weather(user)
    log.info(f"Weather done: {weather.annual_generation_per_kw} kWh/kW/year")

    tariff = run_tariff(user)
    log.info(f"Tariff done: {len(tariff.base_forecast)} years forecasted")

    tech = run_tech_trends(user, tariff)
    log.info(f"Tech done: eff={tech.efficiency_now_pct}% cost=â‚¹{tech.cost_now_inr_per_w}/W")

    sizing = run_sizing(user, weather, tech)
    log.info(f"Sizing done: {sizing.system_capacity_kw} kW, {sizing.recommended_panels} panels")

    # NEW STEP 5.5: Rooftop Feasibility Analysis
    feasibility = run_rooftop_feasibility(user, sizing)
    log.info(f"Feasibility done: Score {feasibility['feasibility_score']:.1f}/100, Feasible: {feasibility['is_feasible']}")

    # FIXED: Store feasibility data in sizing object for frontend access
    sizing.feasibility_data = feasibility  # Add this line
    
    # Apply feasibility constraints to sizing
    if not feasibility['is_feasible'] or feasibility['max_capacity_kw'] < sizing.system_capacity_kw * 0.8:
        log.warning("Applying feasibility constraints to system sizing...")

        # Adjust sizing based on roof constraints
        max_feasible = feasibility['max_capacity_kw']
        if sizing.system_capacity_kw > max_feasible:
            # Scale down proportionally
            scale_factor = max_feasible / sizing.system_capacity_kw
            sizing.system_capacity_kw = max_feasible
            sizing.recommended_panels = int(sizing.recommended_panels * scale_factor)
            sizing.monthly_generation_kwh *= scale_factor

            # Adjust costs
            cost_min, cost_max = extract_cost_range(sizing.cost_range_inr)
            adjusted_cost_min = cost_min * scale_factor + feasibility['additional_costs']
            adjusted_cost_max = cost_max * scale_factor + feasibility['additional_costs']
            sizing.cost_range_inr = (adjusted_cost_min, adjusted_cost_max)

            log.info(f"System adjusted to {sizing.system_capacity_kw:.1f} kW due to roof constraints")

        # Add feasibility warnings
        sizing.warnings.extend(feasibility.get('warnings', []))

    # Continue with existing pipeline...
    roi = run_roi(user, sizing, tariff, tech, weather)
    log.info(f"ROI done: payback={roi.payback_years} years, NPV={roi.npv_15y_inr}")

    risk = run_risk(user, sizing, roi, tech, weather)
    log.info(f"Risk done: {risk.overall_risk}")

    user_clustering = run_user_clustering(user, roi, sizing)
    log.info(f"User Clustering done: {user_clustering.cluster_name} (ID: {user_clustering.cluster_id})")

    heuristic_search = run_heuristic_search(user, roi, sizing, tariff, tech, risk)
    log.info(f"Heuristic Search done: {heuristic_search.optimal_scenario_type}")

    vendors = run_vendors(user, sizing, risk, heuristic_search)
    log.info(f"Vendors done: {len(vendors.ranked_vendors)} vendors ranked")

    safety = run_safety(user, sizing, roi, risk, heuristic_search)
    log.info(f"Safety gate: {'PASS' if safety.ok else 'ISSUES FOUND'}")

    log.info(f"========================================================================================================================================================")

    return PipelineResult(
        user=user,
        weather=weather,
        tariff=tariff,
        tech=tech,
        sizing=sizing,
        roi=roi,
        risk=risk,
        user_clustering=user_clustering,
        heuristic_search=heuristic_search,
        vendors=vendors,
        safety=safety
    )



if __name__ == "__main__":
    example = UserRequest(
        location="Pune, Maharashtra",
        state="Maharashtra",
        category="Residential",
        monthly_consumption_kwh=450.0,
        monthly_bill=2850.0,
        roof_area_m2=85.0,
        budget_inr=400000,
        house_type="independent",
        income_bracket="Medium",
        risk_tolerance="moderate",
        timeline_preference="flexible",
        priority="cost",
        goals=["min_payback", "balanced_risk", "green"]
    )

    print("Running FOAI Pipeline Integration Test...")
    print("=" * 60)
    try:
        result = run_pipeline(example)
        print("\nPipeline completed successfully!")
        print("\n=== FOAI Unified Result Summary ===")
        print(f"Location: {result.user.location}")
        print(f"System Size: {result.sizing.system_capacity_kw}kW")
        print(f"Cost Range: â‚¹{result.sizing.cost_range_inr}")
        print(f"NPV (15y): â‚¹{result.roi.npv_15y_inr:,.0f}" if result.roi.npv_15y_inr else "N/A")
        print(f"Payback: {result.roi.payback_years:.1f} years" if result.roi.payback_years else "N/A")
        print(f"Risk Level: {result.risk.overall_risk}")
        print(f"User Cluster: {result.user_clustering.cluster_name}")
        print(f"Optimal Scenario: {result.heuristic_search.optimal_scenario_type}")
        print(f"Safety Check: {'PASS' if result.safety.ok else 'ISSUES FOUND'}")
        if result.vendors.ranked_vendors:
            print(f"Top Vendor: {result.vendors.ranked_vendors[0]['name']}")
        if result.safety.issues:
            print(f"\nSafety Issues:")
            for issue in result.safety.issues:
                print(f"  â€¢ {issue}")

        # Save to JSON for frontend
        output_path = "C:/FOAI/outputs/pipeline_result.json"
        save_pipeline_result_to_json(result, output_path)
        print(f"\nFull result saved to: {output_path}")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()