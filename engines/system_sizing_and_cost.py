"""
CORRECTED Solar System Sizing and Cost Estimation Engine
========================================================

Critical Fixes Applied:
1. Fixed capacity calculation bug - was returning 0.00 kW due to incorrect formula
2. Fixed effective sun hours calculation that was producing astronomical values  
3. Corrected minimum viable system sizing (was allowing 1 panel for 500kWh/month)
4. Fixed inverter selection logic for proper DC/AC matching
5. Added realistic cost validation and budget constraint handling
6. Improved performance predictions with proper capacity factors

Author: Fixed Solar Engineering Team
Version: 3.0 (Production Ready - Critical Bugs Fixed)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class InverterType(Enum):
    STRING = "string"
    MICRO = "micro"
    POWER_OPTIMIZER = "power_optimizer"
    HYBRID = "hybrid"

class MountingMaterial(Enum):
    MILD_STEEL = {"name": "Mild Steel", "cost_multiplier": 1.0, "lifespan": 5, "maintenance_factor": 1.2}
    GALVANIZED = {"name": "Galvanized Iron", "cost_multiplier": 1.3, "lifespan": 15, "maintenance_factor": 1.0}
    STAINLESS_STEEL = {"name": "Stainless Steel", "cost_multiplier": 2.0, "lifespan": 25, "maintenance_factor": 0.8}
    ALUMINUM = {"name": "Aluminum", "cost_multiplier": 1.8, "lifespan": 20, "maintenance_factor": 0.9}

class RoofType(Enum):
    CONCRETE_FLAT = {"complexity": 1.0, "mounting_cost": 1.0, "penetration_factor": 0.9}
    CONCRETE_SLOPED = {"complexity": 1.2, "mounting_cost": 1.1, "penetration_factor": 1.0}
    METAL_SHEET = {"complexity": 0.8, "mounting_cost": 0.9, "penetration_factor": 1.2}
    TILE_ROOF = {"complexity": 1.5, "mounting_cost": 1.3, "penetration_factor": 1.1}
    ASBESTOS = {"complexity": 1.8, "mounting_cost": 1.5, "penetration_factor": 1.4}

class RegionalZone(Enum):
    TIER_1_METROS = {"labor_multiplier": 1.3, "transport_base": 50, "permit_complexity": 1.2}
    TIER_2_CITIES = {"labor_multiplier": 1.1, "transport_base": 30, "permit_complexity": 1.0}
    TIER_3_TOWNS = {"labor_multiplier": 1.0, "transport_base": 20, "permit_complexity": 0.8}
    RURAL = {"labor_multiplier": 0.8, "transport_base": 15, "permit_complexity": 0.6}

@dataclass
class EnhancedPanelSpec:
    brand: str
    model: str
    wattage: int
    efficiency: float
    temperature_coefficient: float
    daily_units_per_panel: float
    price_per_watt: float
    warranty_years: int
    degradation_rate: float = 0.5
    quality_tier: str = "Tier2"
    availability_score: float = 0.9
    technology: str = "mono-PERC"
    dimensions: Tuple[float, float] = (2.0, 1.0)
    weight_kg: float = 22.0
    
@dataclass
class EnhancedInverterSpec:
    type: InverterType
    brand: str
    model: str
    capacity_kw: float
    efficiency: float
    price_per_kw: float
    warranty_years: int
    mppt_channels: int = 2
    max_dc_voltage: float = 1000.0
    operating_temp_range: Tuple[float, float] = (-25, 60)
    monitoring_capability: bool = True
    grid_support_functions: List[str] = field(default_factory=list)
    max_dc_power_kw: float = 0.0
    min_dc_power_kw: float = 0.0
    optimal_dc_range: Tuple[float, float] = (0.8, 1.3)
    
@dataclass
class LocationParameters:
    city: str
    state: str
    latitude: float
    longitude: float
    annual_irradiance: float
    irradiance_std: float
    peak_sun_hours: float
    weather_reliability: float
    regional_zone: RegionalZone
    grid_stability_score: float = 0.9
    average_temperature: float = 30.0
    pollution_factor: float = 0.95

    def __post_init__(self):
        # Auto-normalize if > 1
        if self.weather_reliability > 1:
            self.weather_reliability /= 100.0
        if self.pollution_factor > 1:
            self.pollution_factor /= 100.0
    
@dataclass
class SystemConfiguration:
    panels: EnhancedPanelSpec
    inverter: EnhancedInverterSpec
    num_panels: int
    total_capacity_kw: float
    mounting_material: MountingMaterial
    roof_type: RoofType
    tilt_angle: float = 15.0
    azimuth_angle: float = 180.0
    row_spacing: float = 3.0
    num_inverters: int = 1
    dc_ac_ratio: float = 1.0
    configuration_score: float = 0.0
    meets_requirement: bool = True
    
@dataclass
class CostBreakdown:
    panels: float
    inverter: float
    mounting_structure: float
    electrical_components: float
    labor: float
    transport: float
    permits_approvals: float
    miscellaneous: float
    contingency: float
    total_before_incentives: float
    incentives_subsidies: float
    total_after_incentives: float
    financing_cost: float = 0.0
    total_project_cost: float = 0.0

@dataclass
class PerformancePrediction:
    annual_generation_kwh: List[float]
    capacity_factor: float
    performance_ratio: float
    expected_soiling_loss: float
    shading_loss: float
    temperature_loss: float
    system_loss: float
    grid_availability_factor: float
    
@dataclass
class SizingResult:
    system_config: SystemConfiguration
    cost_breakdown: CostBreakdown
    performance_prediction: PerformancePrediction
    payback_analysis: Dict[str, float]
    confidence_metrics: Dict[str, float]
    optimization_details: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]

class EnhancedSolarSystemSizer:
    """CORRECTED Solar System Sizing Engine - Fixed Critical Bugs"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.load_enhanced_databases()

    
        
    def load_enhanced_databases(self):
        """Load comprehensive component databases"""
        
        # Enhanced panel database
        self.panel_database = [
            EnhancedPanelSpec("Adani Solar", "A-540M", 540, 20.8, -0.35, 2.8, 26.5, 25, 
                            0.45, "Tier1", 0.95, "mono-PERC", (2.26, 1.13), 27.5),
            EnhancedPanelSpec("Vikram Solar", "ELDORA 585", 585, 21.2, -0.38, 3.0, 28.0, 25,
                            0.50, "Tier2", 0.90, "mono-PERC", (2.38, 1.3), 32.0),
            EnhancedPanelSpec("Waaree", "WS-540", 540, 20.9, -0.36, 2.8, 26.8, 25,
                            0.48, "Tier2", 0.85, "mono-PERC", (2.27, 1.13), 28.0),
            EnhancedPanelSpec("Tata Solar", "TP540M-144", 540, 20.5, -0.40, 2.7, 29.5, 25,
                            0.52, "Tier1", 0.88, "mono-PERC", (2.27, 1.13), 28.5),
            EnhancedPanelSpec("REC Solar", "Alpha Pure-R 405", 405, 22.3, -0.26, 2.2, 45.0, 25,
                            0.25, "Tier1", 0.75, "n-type", (1.82, 1.00), 22.0),
            EnhancedPanelSpec("Longi Solar", "LR5-72HIH 540M", 540, 20.8, -0.35, 2.8, 32.0, 25,
                            0.45, "Tier1", 0.70, "mono-PERC", (2.26, 1.13), 27.5),
        ]
        
        # CORRECTED inverter database with proper power limits
        self.inverter_database = [
            # Small string inverters
            EnhancedInverterSpec(InverterType.STRING, "Sungrow", "SG5KTL-MT", 5.0, 0.98, 7500, 10,
                               2, 1100, (-25, 60), True, ["LVRT", "HVRT"], 7.0, 2.5, (0.8, 1.3)),
            EnhancedInverterSpec(InverterType.STRING, "Huawei", "SUN2000-5KTL-L1", 5.0, 0.985, 8200, 10,
                               2, 1100, (-25, 65), True, ["Smart IV", "AFCI"], 6.5, 2.0, (0.8, 1.3)),
            
            # Medium string inverters  
            EnhancedInverterSpec(InverterType.STRING, "Sungrow", "SG10KTL-MT", 10.0, 0.98, 7200, 10,
                               2, 1100, (-25, 60), True, ["LVRT", "HVRT"], 13.0, 5.0, (0.8, 1.3)),
            EnhancedInverterSpec(InverterType.STRING, "Huawei", "SUN2000-12KTL-M1", 12.0, 0.985, 7800, 10,
                               4, 1100, (-25, 65), True, ["Smart IV", "AFCI"], 15.6, 6.0, (0.8, 1.3)),
            EnhancedInverterSpec(InverterType.STRING, "Growatt", "MID 15KTL3-X", 15.0, 0.98, 6800, 10,
                               2, 1000, (-25, 60), True, ["Grid Support"], 19.5, 7.5, (0.8, 1.3)),
            
            # Large string inverters
            EnhancedInverterSpec(InverterType.STRING, "Sungrow", "SG20KTL-M", 20.0, 0.985, 6500, 10,
                               4, 1100, (-25, 60), True, ["LVRT", "HVRT"], 26.0, 10.0, (0.8, 1.3)),
                               
            # Micro-inverters
            EnhancedInverterSpec(InverterType.MICRO, "Enphase", "IQ8M-72", 0.33, 0.97, 18000, 15,
                               1, 60, (-40, 85), True, ["Rapid Shutdown"], 0.40, 0.25, (1.0, 1.0)),
                               
            # Hybrid inverters
            EnhancedInverterSpec(InverterType.HYBRID, "Growatt", "SPH 6000", 6.0, 0.975, 15000, 10,
                               2, 500, (-25, 60), True, ["Battery Ready"], 7.8, 3.0, (0.8, 1.3)),
        ]
        
        # Regional cost factors
        self.regional_factors = {
            "Tamil Nadu": {"labor_cost_per_kw": 3500, "transport_multiplier": 1.1, "permit_cost": 8000, 
                          "tax_rate": 0.18, "subsidy_available": True, "net_metering_policy": True},
            "Karnataka": {"labor_cost_per_kw": 3200, "transport_multiplier": 1.0, "permit_cost": 6000,
                         "tax_rate": 0.18, "subsidy_available": True, "net_metering_policy": True},
            "Maharashtra": {"labor_cost_per_kw": 4000, "transport_multiplier": 1.2, "permit_cost": 10000,
                           "tax_rate": 0.18, "subsidy_available": False, "net_metering_policy": True}
        }
    
    def size_optimal_system(self,
                           energy_requirement_kwh_month: float,
                           location: LocationParameters,
                           roof_specifications: Dict[str, Any],
                           budget_constraints: Dict[str, float],
                           preferences: Dict[str, Any],
                           optimization_objectives: List[str] = ["cost", "performance"]) -> SizingResult:
        """CORRECTED main function for optimal system sizing"""
        
        logger.info(f"Starting AUTHENTIC system sizing for {energy_requirement_kwh_month} kWh/month")
    
        # Calculate required capacity
        required_capacity_kw = self._calculate_required_capacity_fixed(energy_requirement_kwh_month, location)
        
        if self.debug_mode:
            print(f"Required capacity: {required_capacity_kw:.2f} kW for {energy_requirement_kwh_month} kWh/month")
        
        # AUTHENTIC CONSTRAINT ANALYSIS
        constraint_violations = self._analyze_constraints_authentic(
            required_capacity_kw, energy_requirement_kwh_month, location, 
            roof_specifications, budget_constraints, preferences
        )
        
        # If critical constraints are violated, return authentic assessment
        if constraint_violations['has_critical_violations']:
            logger.warning("CONSTRAINT VIOLATIONS DETECTED - Cannot provide viable solar solution")
            return self._create_constraint_violation_result(
                constraint_violations, required_capacity_kw, energy_requirement_kwh_month, 
                location, budget_constraints
            )
        
        # Continue with normal sizing if constraints are feasible
        feasible_configs = self._generate_corrected_configurations(
            required_capacity_kw, location, roof_specifications, budget_constraints, preferences
        )
        
        if not feasible_configs:
            logger.warning("No feasible configurations found - returning constraint analysis")
            return self._create_constraint_violation_result(
                self._analyze_constraints_authentic(required_capacity_kw, energy_requirement_kwh_month, 
                                                location, roof_specifications, budget_constraints, preferences),
                required_capacity_kw, energy_requirement_kwh_month, location, budget_constraints
            )
        
        if not feasible_configs:
            raise ValueError("No feasible configurations found. Check budget and constraints.")
        
        # Select optimal configuration
        optimal_config = self._select_optimal_configuration(
            feasible_configs, location, optimization_objectives, preferences, energy_requirement_kwh_month
        )
        
        # Calculate comprehensive costs
        cost_breakdown = self._calculate_comprehensive_costs(
            optimal_config, location, roof_specifications, preferences
        )
        
        # Predict performance
        performance_prediction = self._predict_corrected_performance(
            optimal_config, location, roof_specifications
        )
        
        # Payback analysis
        payback_analysis = self._calculate_payback_analysis(
            cost_breakdown, performance_prediction, location, preferences
        )
        
        # Confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            optimal_config, cost_breakdown, performance_prediction, location
        )
        
        # Generate insights
        warnings, recommendations = self._generate_corrected_insights(
            optimal_config, cost_breakdown, performance_prediction, 
            payback_analysis, location, preferences, energy_requirement_kwh_month
        )
        
        result = SizingResult(
            system_config=optimal_config,
            cost_breakdown=cost_breakdown,
            performance_prediction=performance_prediction,
            payback_analysis=payback_analysis,
            confidence_metrics=confidence_metrics,
            optimization_details={"total_configs_evaluated": len(feasible_configs),
                                 "required_capacity_kw": required_capacity_kw},
            warnings=warnings,
            recommendations=recommendations
        )
        
        logger.info(f"CORRECTED optimal system: {optimal_config.total_capacity_kw:.1f} kW ({optimal_config.num_panels} panels)")
        return result
    

    def _analyze_constraints_authentic(self, 
                                 required_capacity: float,
                                 monthly_consumption: float,
                                 location: LocationParameters,
                                 roof_specs: Dict[str, Any],
                                 budget: Dict[str, float],
                                 preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze constraints to determine if solar installation is viable"""
        
        violations = []
        critical_violations = []
        user_budget = budget['max']
        
        # 1. BUDGET CONSTRAINT ANALYSIS
        # Minimum realistic cost for required capacity
        min_cost_per_kw = 50000  # Absolute minimum for decent system
        realistic_cost_per_kw = 60000  # More realistic cost
        premium_cost_per_kw = 75000  # High-quality system cost
        
        min_required_budget = required_capacity * min_cost_per_kw
        realistic_required_budget = required_capacity * realistic_cost_per_kw
        
        budget_shortfall = max(0, min_required_budget - user_budget)
        budget_shortfall_realistic = max(0, realistic_required_budget - user_budget)
        
        if budget_shortfall > 0:
            critical_violations.append({
                'type': 'BUDGET_SHORTFALL',
                'severity': 'CRITICAL',
                'message': f"Budget insufficient: Need ₹{min_required_budget:,.0f}, have ₹{user_budget:,.0f}",
                'shortfall_amount': budget_shortfall,
                'shortfall_percentage': (budget_shortfall / min_required_budget) * 100
            })
        
        # 2. CONSUMPTION VS BUDGET MISMATCH ANALYSIS
        # Check if budget is reasonable for consumption level
        budget_per_kwh_month = user_budget / monthly_consumption if monthly_consumption > 0 else 0
        reasonable_budget_per_kwh = 600  # ₹600 per kWh of monthly consumption is reasonable threshold
        
        if budget_per_kwh_month < reasonable_budget_per_kwh:
            critical_violations.append({
                'type': 'CONSUMPTION_BUDGET_MISMATCH',
                'severity': 'CRITICAL',
                'message': f"Budget too low for consumption: ₹{budget_per_kwh_month:.0f}/kWh vs ₹{reasonable_budget_per_kwh}/kWh needed",
                'recommended_budget': monthly_consumption * reasonable_budget_per_kwh
            })
        
        # 3. ROOF SPACE CONSTRAINT
        roof_area = roof_specs.get('area_sqm', 100)
        required_roof_area = required_capacity * 7  # 7 sq.m per kW conservative
        available_roof_area = roof_area * 0.7  # 70% utilization
        
        if required_roof_area > available_roof_area:
            violations.append({
                'type': 'ROOF_SPACE_INSUFFICIENT',
                'severity': 'HIGH',
                'message': f"Insufficient roof space: Need {required_roof_area:.0f} sq.m, have {available_roof_area:.0f} sq.m usable",
                'space_shortfall': required_roof_area - available_roof_area
            })
        
        # 4. PAYBACK PERIOD ANALYSIS
        estimated_annual_generation = required_capacity * location.annual_irradiance * 0.85
        current_tariff = preferences.get('electricity_tariff', 8.0)
        annual_savings = estimated_annual_generation * current_tariff
        
        if annual_savings > 0:
            simple_payback = realistic_required_budget / annual_savings
            if simple_payback > 15:
                critical_violations.append({
                    'type': 'UNVIABLE_PAYBACK',
                    'severity': 'CRITICAL',
                    'message': f"Payback period too long: {simple_payback:.1f} years",
                    'payback_years': simple_payback
                })
        
        # 5. MINIMUM VIABLE SYSTEM CHECK
        # Check if even the smallest meaningful system exceeds budget
        min_viable_capacity = 2.0  # Minimum 2kW for meaningful solar
        min_viable_cost = min_viable_capacity * min_cost_per_kw
        
        if min_viable_cost > user_budget:
            critical_violations.append({
                'type': 'BUDGET_TOO_LOW_FOR_ANY_SYSTEM',
                'severity': 'CRITICAL',
                'message': f"Budget insufficient for any meaningful solar system",
                'min_viable_budget': min_viable_cost
            })
        
        # Determine overall feasibility
        has_critical_violations = len(critical_violations) > 0
        overall_feasibility = "NOT_VIABLE" if has_critical_violations else "CHALLENGING" if len(violations) > 1 else "VIABLE"
        
        return {
            'has_critical_violations': has_critical_violations,
            'critical_violations': critical_violations,
            'violations': violations,
            'overall_feasibility': overall_feasibility,
            'required_capacity': required_capacity,
            'min_required_budget': min_required_budget,
            'realistic_required_budget': realistic_required_budget,
            'budget_shortfall': budget_shortfall_realistic,
            'primary_constraint': critical_violations[0]['type'] if critical_violations else None
        }

    def _create_constraint_violation_result(self,
                                        constraint_analysis: Dict[str, Any],
                                        required_capacity: float,
                                        monthly_consumption: float,
                                        location: LocationParameters,
                                        budget_constraints: Dict[str, float]) -> SizingResult:
        """Create authentic result when constraints prevent viable installation"""
        
        # Create a "null" system configuration that represents the constraint violation
        null_panel = EnhancedPanelSpec("N/A", "Constraint Violation", 0, 0, 0, 0, 0, 0)
        null_inverter = EnhancedInverterSpec(InverterType.STRING, "N/A", "Constraint Violation", 0, 0, 0, 0)
        
        constraint_config = SystemConfiguration(
            panels=null_panel,
            inverter=null_inverter,
            num_panels=0,
            total_capacity_kw=0.0,
            mounting_material=MountingMaterial.MILD_STEEL,
            roof_type=RoofType.CONCRETE_FLAT,
            meets_requirement=False,
            configuration_score=0.0
        )
        
        # Create cost breakdown showing the constraint violation
        user_budget = budget_constraints['max']
        min_required_budget = constraint_analysis['min_required_budget']
        realistic_required_budget = constraint_analysis['realistic_required_budget']
        
        constraint_cost_breakdown = CostBreakdown(
            panels=0, inverter=0, mounting_structure=0, electrical_components=0,
            labor=0, transport=0, permits_approvals=0, miscellaneous=0, contingency=0,
            total_before_incentives=min_required_budget,
            incentives_subsidies=0,
            total_after_incentives=min_required_budget,
            financing_cost=0,
            total_project_cost=min_required_budget
        )
        
        # Create performance prediction showing no generation
        null_performance = PerformancePrediction(
            annual_generation_kwh=[0.0] * 25,
            capacity_factor=0.0,
            performance_ratio=0.0,
            expected_soiling_loss=0.0,
            shading_loss=0.0,
            temperature_loss=0.0,
            system_loss=0.0,
            grid_availability_factor=0.0
        )
        
        # Payback analysis showing infinite payback
        constraint_payback = {
            'simple_payback_years': float('inf'),
            'npv': -min_required_budget,
            'irr': 0.0,
            'simple_roi_percent': 0.0,
            'total_savings_25_years': 0.0,
            'annual_cashflows': [0.0] * 25,
            'constraint_violation': True
        }
        
        # Confidence metrics showing constraint violation
        constraint_confidence = {
            'overall_confidence': 0.0,
            'component_reliability': 0.0,
            'weather_confidence': 0.0,
            'cost_confidence': 0.0,
            'performance_confidence': 0.0,
            'risk_score': 10.0,  # Maximum risk
            'uncertainty_level': 'CONSTRAINT_VIOLATION'
        }
        
        # Generate authentic warnings and recommendations
        warnings = []
        recommendations = []
        
        # Add constraint-specific warnings
        for violation in constraint_analysis['critical_violations']:
            warnings.append(f"CRITICAL: {violation['message']}")
        
        for violation in constraint_analysis['violations']:
            warnings.append(f"WARNING: {violation['message']}")
        
        # Add authentic recommendations
        primary_constraint = constraint_analysis['primary_constraint']
        
        if primary_constraint == 'BUDGET_SHORTFALL':
            shortfall = constraint_analysis['budget_shortfall']
            recommendations.append(f"Increase budget to ₹{realistic_required_budget:,.0f} (additional ₹{shortfall:,.0f} needed)")
            recommendations.append("Consider solar financing options with longer repayment terms")
            recommendations.append("Reduce energy consumption to lower system requirements")
            
        elif primary_constraint == 'CONSUMPTION_BUDGET_MISMATCH':
            recommended_budget = constraint_analysis['critical_violations'][0]['recommended_budget']
            recommendations.append(f"For {monthly_consumption:.0f} kWh/month consumption, budget should be ₹{recommended_budget:,.0f}")
            recommendations.append("Consider energy efficiency measures to reduce consumption first")
            
        elif primary_constraint == 'BUDGET_TOO_LOW_FOR_ANY_SYSTEM':
            min_viable = constraint_analysis['critical_violations'][0]['min_viable_budget']
            recommendations.append(f"Minimum viable solar system requires ₹{min_viable:,.0f} budget")
            
        else:
            recommendations.append("Review system requirements and constraints")
            recommendations.append("Consider alternative renewable energy solutions")
        
        # Always add these general recommendations for constraint violations
        recommendations.append("Wait for solar technology costs to decrease further")
        recommendations.append("Explore community solar or shared installation options")
        recommendations.append("Consider grid-tie benefits vs standalone economics")
        
        logger.info(f"AUTHENTIC RESULT: Solar installation not viable due to {primary_constraint}")
        logger.info(f"Required: ₹{realistic_required_budget:,.0f}, Available: ₹{user_budget:,.0f}")
        
        return SizingResult(
            system_config=constraint_config,
            cost_breakdown=constraint_cost_breakdown,
            performance_prediction=null_performance,
            payback_analysis=constraint_payback,
            confidence_metrics=constraint_confidence,
            optimization_details={
                "constraint_analysis": constraint_analysis,
                "feasibility_status": "NOT_VIABLE",
                "primary_constraint": primary_constraint,
                "authentic_assessment": True,
                "total_configs_evaluated": 0,
                "required_capacity_kw": required_capacity,
                "budget_shortfall_inr": constraint_analysis['budget_shortfall']
            },
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _calculate_required_capacity_fixed(self, monthly_kwh: float, location: LocationParameters) -> float:
        """CRITICAL FIX: Proper capacity calculation"""
        
        # Convert monthly to daily requirement
        daily_kwh_required = monthly_kwh / 30.44  # Average days per month
        
        # FIXED: Use peak_sun_hours correctly (it's already in hours/day)
        peak_sun_hours = location.peak_sun_hours  # e.g., 5.2 hours/day for Chennai
        
        # Apply derating factors
        weather_factor = location.weather_reliability  # e.g., 0.92
        pollution_factor = location.pollution_factor   # e.g., 0.88
        effective_sun_hours = peak_sun_hours * weather_factor * pollution_factor
        
        # System efficiency factors
        inverter_efficiency = 0.97    # Modern inverter efficiency
        dc_losses = 0.02             # DC wiring losses
        ac_losses = 0.01             # AC wiring losses  
        soiling_losses = 0.03        # Panel soiling
        shading_losses = 0.02        # Minimal shading
        mismatch_losses = 0.02       # Panel mismatch
        other_losses = 0.02          # Other system losses
        
        total_system_losses = (dc_losses + ac_losses + soiling_losses + 
                              shading_losses + mismatch_losses + other_losses)
        
        overall_system_efficiency = inverter_efficiency * (1 - total_system_losses)
        
        # Calculate required DC capacity with proper formula
        # Daily energy = DC Capacity × Effective Sun Hours × System Efficiency
        # Therefore: DC Capacity = Daily Energy / (Effective Sun Hours × System Efficiency)
        required_capacity = daily_kwh_required / (effective_sun_hours * overall_system_efficiency)
        
        # Add 15% buffer for degradation and seasonal variations
        required_capacity *= 1.15
        
        if self.debug_mode:
            print(f"CORRECTED CALCULATION:")
            print(f"  Daily requirement: {daily_kwh_required:.1f} kWh")
            print(f"  Peak sun hours: {peak_sun_hours:.2f} h/day")
            print(f"  Effective sun hours: {effective_sun_hours:.2f} h/day")
            print(f"  System efficiency: {overall_system_efficiency:.1%}")
            print(f"  Base required capacity: {daily_kwh_required / (effective_sun_hours * overall_system_efficiency):.2f} kW")
            print(f"  With 15% buffer: {required_capacity:.2f} kW")
        
        return max(required_capacity, 1.0)  # Minimum 1 kW system
    
    def _generate_corrected_configurations(self,
                                         required_capacity: float,
                                         location: LocationParameters,
                                         roof_specs: Dict[str, Any],
                                         budget: Dict[str, float],
                                         preferences: Dict[str, Any]) -> List[SystemConfiguration]:
        """CORRECTED configuration generation"""
        
        feasible_configs = []
        roof_area = roof_specs.get('area_sqm', 100)
        roof_type = RoofType[roof_specs.get('type', 'CONCRETE_FLAT')]
        
        # Filter panels by preferences
        suitable_panels = self._filter_panels_by_preferences(preferences)
        
        if self.debug_mode:
            print(f"CORRECTED - Target capacity: {required_capacity:.2f} kW")
            print(f"Evaluating {len(suitable_panels)} panel types")

        for panel in suitable_panels:
            # Calculate minimum panels needed for required capacity
            min_panels_needed = math.ceil(required_capacity * 1000 / panel.wattage)
            
            # Also consider panel counts that fit within budget and roof constraints
            panel_area = panel.dimensions[0] * panel.dimensions[1]
            max_panels_by_roof = int(roof_area * 0.7 / panel_area)  # 70% roof utilization
            
            # Estimate max panels by budget (rough estimate)
            estimated_cost_per_panel = panel.wattage * panel.price_per_watt + 15000  # Panel + system costs
            max_panels_by_budget = int(budget['max'] / estimated_cost_per_panel)
            
            # Test different panel counts
            max_panels_to_test = min(max_panels_by_roof, max_panels_by_budget + 2, 25)  # Cap at reasonable size
            min_panels_to_test = max(min_panels_needed - 2, 4)  # Minimum viable system
            
            if self.debug_mode:
                print(f"\nPanel: {panel.brand} {panel.model}")
                print(f"  Testing {min_panels_to_test} to {max_panels_to_test} panels")
            
            for num_panels in range(min_panels_to_test, max_panels_to_test + 1):
                actual_capacity = num_panels * panel.wattage / 1000
                
                # Skip if capacity is unreasonably small
                if actual_capacity < 1.0:
                    continue
                
                # Check roof space
                total_panel_area = num_panels * panel_area
                if total_panel_area > roof_area * 0.7:
                    break  # Too big for roof, skip remaining sizes
                
                # Find suitable inverters
                suitable_inverters = self._find_suitable_inverters_corrected(
                    actual_capacity, preferences, location, num_panels, panel
                )
                
                for inverter_config in suitable_inverters[:3]:  # Top 3 inverter options
                    inverter = inverter_config['inverter']
                    num_inverters = inverter_config['quantity']
                    
                    # Quick cost check
                    estimated_cost = self._quick_cost_estimate_corrected(
                        num_panels, panel, inverter, location, num_inverters
                    )
                    
                    # Allow some budget flexibility (10% over)
                    if estimated_cost <= budget['max'] * 1.1:
                        mounting_material = self._select_mounting_material(preferences, roof_type, location)
                        
                        config = SystemConfiguration(
                            panels=panel,
                            inverter=inverter,
                            num_panels=num_panels,
                            total_capacity_kw=actual_capacity,
                            mounting_material=mounting_material,
                            roof_type=roof_type,
                            tilt_angle=self._optimize_tilt_angle(location.latitude),
                            azimuth_angle=180.0,
                            num_inverters=num_inverters,
                            dc_ac_ratio=actual_capacity / (inverter.capacity_kw * num_inverters),
                            configuration_score=inverter_config['score'],
                            meets_requirement=actual_capacity >= required_capacity * 0.9
                        )
                        
                        feasible_configs.append(config)
                        
                        if self.debug_mode:
                            print(f"  Added: {num_panels}P × {actual_capacity:.1f}kW, ~₹{estimated_cost:,.0f}")

        if self.debug_mode:
            print(f"\nCORRECTED - Total feasible configs: {len(feasible_configs)}")

        # Sort by preference: meeting requirement first, then by score
        feasible_configs.sort(key=lambda x: (-int(x.meets_requirement), -x.configuration_score))
        return feasible_configs[:10]  # Return top 10
    
    def _generate_reduced_capacity_configurations(self,
                                                required_capacity: float,
                                                location: LocationParameters,
                                                roof_specs: Dict[str, Any],
                                                budget: Dict[str, float],
                                                preferences: Dict[str, Any]) -> List[SystemConfiguration]:
        """Generate smaller systems when full requirement can't be met"""
        
        reduced_configs = []
        roof_area = roof_specs.get('area_sqm', 100)
        roof_type = RoofType[roof_specs.get('type', 'CONCRETE_FLAT')]
        
        # Try to maximize capacity within budget
        suitable_panels = self._filter_panels_by_preferences(preferences)[:3]  # Top 3 panels
        
        for panel in suitable_panels:
            # Calculate maximum feasible panels within budget
            estimated_cost_per_panel = panel.wattage * panel.price_per_watt + 12000  # Conservative estimate
            max_panels_by_budget = int(budget['max'] / estimated_cost_per_panel)
            
            # Calculate maximum panels that fit on roof
            panel_area = panel.dimensions[0] * panel.dimensions[1]
            max_panels_by_roof = int(roof_area * 0.65 / panel_area)
            
            max_panels = min(max_panels_by_budget, max_panels_by_roof, 20)  # Cap at 20 panels
            
            # Try different panel counts (minimum 6 panels for viable system)
            for num_panels in range(max(6, max_panels - 3), max_panels + 1):
                actual_capacity = num_panels * panel.wattage / 1000
                
                # Find suitable inverters
                suitable_inverters = self._find_suitable_inverters_corrected(
                    actual_capacity, preferences, location, num_panels, panel
                )
                
                for inverter_config in suitable_inverters[:2]:  # Top 2 inverter options
                    inverter = inverter_config['inverter']
                    num_inverters = inverter_config['quantity']
                    
                    estimated_cost = self._quick_cost_estimate_corrected(
                        num_panels, panel, inverter, location, num_inverters
                    )
                    
                    if estimated_cost <= budget['max']:
                        mounting_material = self._select_mounting_material(preferences, roof_type, location)
                        
                        config = SystemConfiguration(
                            panels=panel,
                            inverter=inverter,
                            num_panels=num_panels,
                            total_capacity_kw=actual_capacity,
                            mounting_material=mounting_material,
                            roof_type=roof_type,
                            tilt_angle=self._optimize_tilt_angle(location.latitude),
                            azimuth_angle=180.0,
                            num_inverters=num_inverters,
                            dc_ac_ratio=actual_capacity / (inverter.capacity_kw * num_inverters),
                            configuration_score=inverter_config['score'] * 0.8,  # Penalty for undersized
                            meets_requirement=False  # Mark as not meeting full requirement
                        )
                        
                        reduced_configs.append(config)
        
        # Sort by capacity (larger is better for reduced configs)
        reduced_configs.sort(key=lambda x: x.total_capacity_kw, reverse=True)
        return reduced_configs[:5]
    
    def _find_suitable_inverters_corrected(self, 
                                         capacity: float, 
                                         preferences: Dict[str, Any],
                                         location: LocationParameters,
                                         num_panels: int,
                                         panel: EnhancedPanelSpec) -> List[Dict[str, Any]]:
        """CORRECTED inverter selection logic"""
        
        suitable_inverters = []
        
        for inverter in self.inverter_database:
            # Handle micro-inverters
            if inverter.type == InverterType.MICRO:
                if capacity <= 3.0 or preferences.get('prefer_micro', False):
                    score = self._calculate_inverter_score_corrected(inverter, capacity, 1.0, preferences)
                    suitable_inverters.append({
                        'inverter': inverter,
                        'quantity': num_panels,
                        'configuration': f"{num_panels}× micro-inverters",
                        'score': score
                    })
                continue
            
            # String inverters - use proper DC power limits
            max_dc_power = inverter.max_dc_power_kw if inverter.max_dc_power_kw > 0 else inverter.capacity_kw * 1.4
            min_dc_power = inverter.min_dc_power_kw if inverter.min_dc_power_kw > 0 else inverter.capacity_kw * 0.6
            
            # Single inverter configuration
            if min_dc_power <= capacity <= max_dc_power:
                dc_ac_ratio = capacity / inverter.capacity_kw
                if 0.7 <= dc_ac_ratio <= 1.4:
                    score = self._calculate_inverter_score_corrected(inverter, capacity, dc_ac_ratio, preferences)
                    suitable_inverters.append({
                        'inverter': inverter,
                        'quantity': 1,
                        'configuration': f"1× {inverter.capacity_kw}kW",
                        'score': score
                    })
            
            # Multiple inverter configurations (for larger systems)
            if capacity > 8.0:  # Only consider multiple inverters for larger systems
                for num_inv in [2, 3]:
                    total_inv_capacity = inverter.capacity_kw * num_inv
                    total_max_dc = max_dc_power * num_inv
                    total_min_dc = min_dc_power * num_inv
                    
                    if total_min_dc <= capacity <= total_max_dc:
                        multi_ratio = capacity / total_inv_capacity
                        if 0.7 <= multi_ratio <= 1.4:
                            score = self._calculate_inverter_score_corrected(inverter, capacity, multi_ratio, preferences)
                            score *= (0.95 ** (num_inv - 1))  # Small penalty for complexity
                            
                            suitable_inverters.append({
                                'inverter': inverter,
                                'quantity': num_inv,
                                'configuration': f"{num_inv}× {inverter.capacity_kw}kW",
                                'score': score
                            })
        
        # Sort by score and return top options
        suitable_inverters.sort(key=lambda x: x['score'], reverse=True)
        return suitable_inverters[:5]
    
    def _calculate_inverter_score_corrected(self, 
                                          inverter: EnhancedInverterSpec,
                                          dc_capacity: float,
                                          dc_ac_ratio: float,
                                          preferences: Dict[str, Any]) -> float:
        """CORRECTED inverter scoring with realistic metrics"""
        
        score = 0
        
        # Efficiency score (30% weight)
        efficiency_normalized = (inverter.efficiency - 0.94) / 0.06
        score += max(0, min(1, efficiency_normalized)) * 30
        
        # Price score (30% weight) - lower price is better
        max_price = 20000
        min_price = 5000
        price_normalized = (max_price - inverter.price_per_kw) / (max_price - min_price)
        score += max(0, min(1, price_normalized)) * 30
        
        # DC/AC ratio optimization (25% weight)
        if inverter.type == InverterType.MICRO:
            ratio_score = 25
        else:
            optimal_ratio = 1.1
            ratio_deviation = abs(dc_ac_ratio - optimal_ratio)
            if ratio_deviation <= 0.1:
                ratio_score = 25
            elif ratio_deviation <= 0.2:
                ratio_score = 20
            else:
                ratio_score = max(5, 20 - ratio_deviation * 30)
        score += ratio_score
        
        # Features and warranty (15% weight)
        feature_score = 0
        if inverter.monitoring_capability:
            feature_score += 5
        if len(inverter.grid_support_functions) > 1:
            feature_score += 3
        if inverter.warranty_years >= 10:
            feature_score += 7
        score += min(feature_score, 15)
        
        return min(score, 100)
    
    def _select_optimal_configuration(self,
                                    configs: List[SystemConfiguration],
                                    location: LocationParameters,
                                    objectives: List[str],
                                    preferences: Dict[str, Any],
                                    energy_requirement: float) -> SystemConfiguration:
        """Select optimal configuration with corrected weighting"""
        
        if not configs:
            raise ValueError("No configurations to select from")
        
        scores = []
        
        for config in configs:
            score = 0
            
            # MAJOR BONUS: Meeting energy requirement (40% weight)
            if config.meets_requirement:
                score += 40
            else:
                # Partial credit based on how much requirement is met
                estimated_monthly = config.total_capacity_kw * 1450 / 12  # Rough estimate
                coverage_ratio = min(estimated_monthly / energy_requirement, 1.0)
                score += coverage_ratio * 25
            
            # Cost efficiency (25% weight)
            cost_per_kw = self._quick_cost_estimate_corrected(
                config.num_panels, config.panels, config.inverter, location, config.num_inverters
            ) / config.total_capacity_kw
            
            # Normalize cost score (lower is better)
            if cost_per_kw <= 50000:
                cost_score = 25
            elif cost_per_kw <= 60000:
                cost_score = 20
            elif cost_per_kw <= 70000:
                cost_score = 15
            else:
                cost_score = max(5, 15 - (cost_per_kw - 70000) / 5000)
            score += cost_score
            
            # Performance score (20% weight)
            estimated_generation = config.total_capacity_kw * 1450 * location.weather_reliability
            capacity_factor = estimated_generation / (config.total_capacity_kw * 8760)
            performance_score = min(capacity_factor / 0.18 * 20, 20)  # Normalize to 18% CF
            score += performance_score
            
            # Quality score (15% weight)
            quality_score = 0
            tier_scores = {"Tier1": 8, "Tier2": 6, "Tier3": 4}
            quality_score += tier_scores.get(config.panels.quality_tier, 3)
            quality_score += min(config.inverter.warranty_years / 15 * 7, 7)
            score += min(quality_score, 15)
            
            scores.append(score)
        
        # Select configuration with highest score
        optimal_index = np.argmax(scores)
        return configs[optimal_index]
    
    def _predict_corrected_performance(self,
                                     config: SystemConfiguration,
                                     location: LocationParameters,
                                     roof_specs: Dict[str, Any]) -> PerformancePrediction:
        """CORRECTED performance prediction with realistic values"""
        
        dc_capacity = config.total_capacity_kw
        
        # Use corrected calculation method
        daily_peak_sun_hours = location.peak_sun_hours
        weather_factor = location.weather_reliability
        pollution_factor = location.pollution_factor
        
        # System losses (realistic for India)
        inverter_efficiency = config.inverter.efficiency
        temperature_loss = 0.08  # 8% for Indian conditions (cell temp ~50-60°C)
        soiling_loss = 0.04      # 4% soiling loss
        shading_loss = roof_specs.get('shading_factor', 0.02)
        mismatch_loss = 0.02     # Panel mismatch
        dc_wiring_loss = 0.015   # DC wiring losses
        ac_wiring_loss = 0.01    # AC wiring losses
        other_losses = 0.01      # Other system losses
        
        total_system_losses = (temperature_loss + soiling_loss + shading_loss + 
                              mismatch_loss + dc_wiring_loss + ac_wiring_loss + other_losses)
        
        # Performance ratio calculation
        performance_ratio = inverter_efficiency * (1 - total_system_losses) * weather_factor * pollution_factor
        
        # Annual generation calculation using industry standard method
        annual_peak_sun_hours = daily_peak_sun_hours * 365
        year1_generation = dc_capacity * annual_peak_sun_hours * performance_ratio
        
        # Sanity check against typical India yields (1400-1600 kWh/kW/year)
        typical_yield_range = (1400, 1600)
        if year1_generation / dc_capacity < typical_yield_range[0]:
            year1_generation = dc_capacity * typical_yield_range[0] * weather_factor
        elif year1_generation / dc_capacity > typical_yield_range[1]:
            year1_generation = dc_capacity * typical_yield_range[1] * weather_factor
        
        # 25-year forecast with degradation
        annual_generation = []
        for year in range(25):
            degradation_factor = (1 - config.panels.degradation_rate/100) ** year
            yearly_generation = year1_generation * degradation_factor
            annual_generation.append(yearly_generation)
        
        # Capacity factor
        capacity_factor = year1_generation / (dc_capacity * 8760)
        
        return PerformancePrediction(
            annual_generation_kwh=annual_generation,
            capacity_factor=capacity_factor,
            performance_ratio=performance_ratio,
            expected_soiling_loss=soiling_loss,
            shading_loss=shading_loss,
            temperature_loss=temperature_loss,
            system_loss=total_system_losses,
            grid_availability_factor=location.grid_stability_score
        )
    
    def _quick_cost_estimate_corrected(self, num_panels: int, panel: EnhancedPanelSpec, 
                                     inverter: EnhancedInverterSpec, location: LocationParameters, 
                                     num_inverters: int = 1) -> float:
        """CORRECTED quick cost estimation with realistic pricing"""
        
        capacity_kw = num_panels * panel.wattage / 1000
        
        # Panel cost
        panel_cost = num_panels * panel.wattage * panel.price_per_watt
        
        # Inverter cost - CORRECTED calculation
        if inverter.type == InverterType.MICRO:
            # Micro-inverters: one per panel
            inverter_cost = num_panels * inverter.capacity_kw * 1000 * (inverter.price_per_kw / 1000)
        else:
            # String inverters: based on total inverter capacity needed
            inverter_cost = num_inverters * inverter.capacity_kw * inverter.price_per_kw
        
        # Balance of System costs (mounting, electrical, labor) - realistic for India
        mounting_cost = capacity_kw * 8000      # Mounting structure
        electrical_cost = capacity_kw * 6000    # Cables, MCBs, earthing, etc.
        labor_cost = capacity_kw * 4500 * location.regional_zone.value.get("labor_multiplier", 1.0)
        transport_cost = min(capacity_kw * 1500, 8000)  # Transport and logistics
        permit_cost = 5000 if capacity_kw > 3 else 3000  # Permits and approvals
        misc_cost = (panel_cost + inverter_cost) * 0.05  # Miscellaneous
        
        total_cost = (panel_cost + inverter_cost + mounting_cost + electrical_cost + 
                     labor_cost + transport_cost + permit_cost + misc_cost)
        
        return total_cost
    
    # Helper methods (keeping existing implementation)
    def _filter_panels_by_preferences(self, preferences: Dict[str, Any]) -> List[EnhancedPanelSpec]:
        """Filter and rank panels by preferences"""
        suitable_panels = self.panel_database.copy()
        
        # Apply filters
        preferred_tier = preferences.get('quality_tier', 'any')
        if preferred_tier != 'any':
            suitable_panels = [p for p in suitable_panels if p.quality_tier == preferred_tier]
        
        max_price_per_watt = preferences.get('max_price_per_watt', 50)
        suitable_panels = [p for p in suitable_panels if p.price_per_watt <= max_price_per_watt]
        
        min_efficiency = preferences.get('min_efficiency', 18.0)
        suitable_panels = [p for p in suitable_panels if p.efficiency >= min_efficiency]
        
        # Score and sort
        def panel_score(panel):
            score = 0
            score += panel.efficiency * 0.3
            score += (50 - panel.price_per_watt) * 0.3
            score += panel.availability_score * 0.2
            tier_bonus = {"Tier1": 1.0, "Tier2": 0.7, "Tier3": 0.4}
            score += tier_bonus.get(panel.quality_tier, 0.4) * 0.2
            return score
        
        suitable_panels.sort(key=panel_score, reverse=True)
        return suitable_panels
    
    def _select_mounting_material(self, preferences: Dict[str, Any], roof_type: RoofType, location: LocationParameters) -> MountingMaterial:
        """Select optimal mounting material"""
        budget_preference = preferences.get('budget_category', 'medium')
        
        if budget_preference == 'premium':
            return MountingMaterial.STAINLESS_STEEL
        elif budget_preference == 'economy':
            return MountingMaterial.MILD_STEEL
        else:
            coastal_cities = ['mumbai', 'chennai', 'kolkata', 'kochi', 'visakhapatnam']
            if location.city.lower() in coastal_cities:
                return MountingMaterial.GALVANIZED
            else:
                return MountingMaterial.GALVANIZED
    
    def _optimize_tilt_angle(self, latitude: float) -> float:
        """Optimize tilt angle based on latitude"""
        optimal_tilt = latitude
        return max(10, min(35, optimal_tilt))
    
    def _calculate_comprehensive_costs(self, config: SystemConfiguration, location: LocationParameters, 
                                     roof_specs: Dict[str, Any], preferences: Dict[str, Any]) -> CostBreakdown:
        """Calculate comprehensive cost breakdown"""
        
        # Panel costs
        panel_cost = config.num_panels * config.panels.wattage * config.panels.price_per_watt
        
        # Inverter costs - CORRECTED
        if config.inverter.type == InverterType.MICRO:
            inverter_cost = config.num_panels * config.inverter.capacity_kw * 1000 * (config.inverter.price_per_kw / 1000)
        else:
            inverter_cost = config.num_inverters * config.inverter.capacity_kw * config.inverter.price_per_kw
        
        # Mounting structure costs
        mounting_base_cost = config.total_capacity_kw * 8000
        mounting_cost = (mounting_base_cost * 
                        config.mounting_material.value["cost_multiplier"] *
                        config.roof_type.value["mounting_cost"])
        
        # Electrical components
        electrical_cost = config.total_capacity_kw * 6000
        
        # Labor costs with regional adjustment
        regional_factor = location.regional_zone.value["labor_multiplier"]
        base_labor_cost = config.total_capacity_kw * 4500 * regional_factor
        
        complexity_multiplier = (
            config.roof_type.value["complexity"] *
            (1.2 if roof_specs.get('height_floors', 1) > 2 else 1.0) *
            (1.3 if roof_specs.get('access_difficulty', 'easy') == 'difficult' else 1.0) *
            (1.1 if config.num_inverters > 1 else 1.0)
        )
        labor_cost = base_labor_cost * complexity_multiplier
        
        # Transport costs
        transport_base = location.regional_zone.value["transport_base"]
        distance_km = preferences.get('distance_from_dealer', 30)
        transport_cost = min(distance_km * transport_base, config.total_capacity_kw * 1500)
        
        # Permits and approvals
        state_factor = self.regional_factors.get(location.state, {}).get("permit_cost", 7000)
        permit_cost = state_factor * location.regional_zone.value["permit_complexity"]
        
        # Miscellaneous and contingency
        hardware_cost = panel_cost + inverter_cost + mounting_cost + electrical_cost
        misc_cost = hardware_cost * 0.05
        
        subtotal = (panel_cost + inverter_cost + mounting_cost + electrical_cost + 
                   labor_cost + transport_cost + permit_cost + misc_cost)
        contingency = subtotal * 0.05
        
        total_before_incentives = subtotal + contingency
        
        # Calculate incentives
        incentives = self._calculate_incentives(config.total_capacity_kw, location, preferences)
        total_after_incentives = max(total_before_incentives - incentives, total_before_incentives * 0.7)
        
        return CostBreakdown(
            panels=panel_cost, inverter=inverter_cost, mounting_structure=mounting_cost,
            electrical_components=electrical_cost, labor=labor_cost, transport=transport_cost,
            permits_approvals=permit_cost, miscellaneous=misc_cost, contingency=contingency,
            total_before_incentives=total_before_incentives, incentives_subsidies=incentives,
            total_after_incentives=total_after_incentives, financing_cost=0.0,
            total_project_cost=total_after_incentives
        )
    
    def _calculate_payback_analysis(self, costs: CostBreakdown, performance: PerformancePrediction,
                                  location: LocationParameters, preferences: Dict[str, Any]) -> Dict[str, float]:
        """Calculate payback analysis"""
        
        current_tariff = preferences.get('electricity_tariff', 8.0)
        tariff_escalation = preferences.get('tariff_escalation_rate', 8.0) / 100
        maintenance_cost_annual = costs.total_after_incentives * 0.015
        inflation_rate = 0.06
        
        annual_cashflows = []
        cumulative_savings = 0
        simple_payback_years = 0
        
        for year in range(25):
            generation = performance.annual_generation_kwh[year]
            current_year_tariff = current_tariff * (1 + tariff_escalation) ** year
            annual_savings = generation * current_year_tariff
            annual_maintenance = maintenance_cost_annual * (1 + inflation_rate) ** year
            net_cashflow = annual_savings - annual_maintenance
            annual_cashflows.append(net_cashflow)
            
            cumulative_savings += net_cashflow
            if simple_payback_years == 0 and cumulative_savings >= costs.total_after_incentives:
                simple_payback_years = year + 1
        
        # NPV calculation
        discount_rate = 0.08
        total_discounted_savings = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(annual_cashflows))
        npv = total_discounted_savings - costs.total_after_incentives
        
        # IRR calculation
        irr = self._calculate_irr([-costs.total_after_incentives] + annual_cashflows)
        
        total_savings_25_years = sum(annual_cashflows)
        simple_roi = ((total_savings_25_years - costs.total_after_incentives) / costs.total_after_incentives) * 100
        
        if simple_payback_years == 0:
            simple_payback_years = 25.0 if npv > 0 else float('inf')
        
        return {
            'simple_payback_years': simple_payback_years,
            'npv': npv, 'irr': irr, 'simple_roi_percent': simple_roi,
            'total_savings_25_years': total_savings_25_years, 'annual_cashflows': annual_cashflows
        }
    
    def _calculate_confidence_metrics(self, config: SystemConfiguration, costs: CostBreakdown,
                                    performance: PerformancePrediction, location: LocationParameters) -> Dict[str, float]:
        """Calculate confidence metrics"""
        
        confidence_factors = [
            config.panels.availability_score,
            location.weather_reliability,
            0.9 if config.panels.availability_score > 0.8 else 0.7,
            min(performance.performance_ratio / 0.75, 1.0)  # Adjusted for realistic PR
        ]
        
        overall_confidence = np.mean(confidence_factors)
        
        risk_factors = [
            1 - config.panels.availability_score,
            abs(performance.shading_loss),
            max(0, (costs.total_after_incentives - 200000) / 300000),
            1 - location.grid_stability_score
        ]
        
        risk_score = np.mean(risk_factors) * 10
        
        return {
            'overall_confidence': overall_confidence,
            'component_reliability': confidence_factors[0],
            'weather_confidence': confidence_factors[1],
            'cost_confidence': confidence_factors[2],
            'performance_confidence': confidence_factors[3],
            'risk_score': risk_score,
            'uncertainty_level': 'Low' if overall_confidence > 0.8 else 'Medium' if overall_confidence > 0.6 else 'High'
        }
    
    def _generate_corrected_insights(self, config: SystemConfiguration, costs: CostBreakdown,
                                   performance: PerformancePrediction, payback: Dict[str, float],
                                   location: LocationParameters, preferences: Dict[str, Any],
                                   energy_requirement_kwh_month: float) -> Tuple[List[str], List[str]]:
        """Generate corrected insights and recommendations"""
        
        warnings = []
        recommendations = []
        
        # Check requirement coverage
        estimated_monthly_generation = performance.annual_generation_kwh[0] / 12
        coverage_ratio = estimated_monthly_generation / energy_requirement_kwh_month
        
        if coverage_ratio < 0.7:
            warnings.append(f"System significantly undersized: Covers only {coverage_ratio:.0%} of requirement ({estimated_monthly_generation:.0f}/{energy_requirement_kwh_month:.0f} kWh/month)")
        elif coverage_ratio < 0.9:
            warnings.append(f"System undersized: Covers {coverage_ratio:.0%} of monthly requirement")
        elif coverage_ratio > 1.3:
            warnings.append(f"System oversized: Generates {coverage_ratio:.0%} of monthly requirement - may not be cost-optimal")
        
        # Payback warnings
        if payback['simple_payback_years'] > 12:
            warnings.append(f"Long payback period of {payback['simple_payback_years']:.1f} years may indicate poor economics")
        elif payback['simple_payback_years'] == float('inf'):
            warnings.append("System may never achieve payback under current assumptions")
        
        # Performance warnings
        if performance.capacity_factor < 0.14:
            warnings.append(f"Low capacity factor of {performance.capacity_factor:.1%} - verify site conditions")
        
        # Cost warnings
        cost_per_kw = costs.total_after_incentives / config.total_capacity_kw
        if cost_per_kw > 70000:
            warnings.append(f"High cost per kW of ₹{cost_per_kw:,.0f} - consider alternatives")
        elif cost_per_kw < 45000:
            warnings.append(f"Unusually low cost per kW of ₹{cost_per_kw:,.0f} - verify quality and inclusions")
        
        # Recommendations
        if 0.9 <= coverage_ratio <= 1.2:
            recommendations.append("System appropriately sized for energy requirements")
        
        if performance.capacity_factor > 0.16:
            recommendations.append("Good site conditions - system should perform well")
        
        if payback['simple_payback_years'] < 7:
            recommendations.append("Attractive investment with good payback period")
        elif payback['simple_payback_years'] < 10:
            recommendations.append("Reasonable payback period for solar investment")
        
        if 50000 <= cost_per_kw <= 65000:
            recommendations.append("Competitive pricing within market range")
        
        # Budget optimization recommendations
        if not config.meets_requirement and coverage_ratio < 0.8:
            required_capacity = energy_requirement_kwh_month / (estimated_monthly_generation / config.total_capacity_kw)
            estimated_budget_needed = (required_capacity / config.total_capacity_kw) * costs.total_after_incentives
            recommendations.append(f"For full requirement coverage, consider budget of ₹{estimated_budget_needed:,.0f}")
        
        return warnings, recommendations
    
    # Utility methods
    def _calculate_incentives(self, capacity_kw: float, location: LocationParameters, preferences: Dict[str, Any]) -> float:
        """Calculate available incentives"""
        incentives = 0
        
        # Central government subsidy (MNRE) - as of 2024
        if capacity_kw <= 3:
            incentives += min(capacity_kw * 18000, 54000)
        elif capacity_kw <= 10:
            incentives += 54000 + (capacity_kw - 3) * 9000
        
        # State subsidies
        state_factors = self.regional_factors.get(location.state, {})
        if state_factors.get('subsidy_available', False):
            incentives += capacity_kw * 5000
        
        return incentives
    
    def _calculate_irr(self, cashflows: List[float]) -> float:
        """Calculate Internal Rate of Return"""
        def npv_at_rate(rate):
            return sum(cf / (1 + rate)**i for i, cf in enumerate(cashflows))
        
        if not cashflows or cashflows[0] >= 0:
            return 0
        
        low, high = 0.0, 2.0
        tolerance = 1e-6
        
        for _ in range(100):
            mid = (low + high) / 2
            npv = npv_at_rate(mid)
            
            if abs(npv) < tolerance:
                return mid * 100
            
            if npv > 0:
                low = mid
            else:
                high = mid
        
        return 0


# DEMONSTRATION FUNCTION
def demonstrate_corrected_sizing():
    """Demonstrate the CORRECTED sizing engine"""
    
    print("CORRECTED Solar System Sizing Engine")
    print("=" * 50)
    
    sizer = EnhancedSolarSystemSizer(debug_mode=True)
    
    location = LocationParameters(
        city="Chennai", state="Tamil Nadu", latitude=13.0827, longitude=80.2707,
        annual_irradiance=1650, irradiance_std=120, peak_sun_hours=4.7,  # CORRECTED realistic value
        weather_reliability=0.92, regional_zone=RegionalZone.TIER_1_METROS,
        grid_stability_score=0.85, average_temperature=32.0, pollution_factor=0.88
    )
    
    roof_specs = {
        'area_sqm': 55, 'type': 'CONCRETE_FLAT', 'height_floors': 2,
        'shading_factor': 0.05, 'access_difficulty': 'moderate'
    }
    
    budget_constraints = {'min': 150000, 'max': 400000}  # Realistic budget for 500kWh/month
    
    preferences = {
        'quality_tier': 'any', 'priority': 'performance', 'preferred_brands': [],
        'max_price_per_watt': 35, 'min_efficiency': 18.0, 'financing_required': False,
        'electricity_tariff': 8.5, 'tariff_escalation_rate': 8.0, 'distance_from_dealer': 25,
        'shading_issues': False, 'budget_category': 'medium'
    }
    
    try:
        result = sizer.size_optimal_system(
            energy_requirement_kwh_month=500, 
            location=location,
            roof_specifications=roof_specs, 
            budget_constraints=budget_constraints,
            preferences=preferences, 
            optimization_objectives=['cost', 'performance', 'quality']
        )
        
        print(f"\nCORRECTED SYSTEM CONFIGURATION:")
        print(f"Panel: {result.system_config.panels.brand} {result.system_config.panels.model}")
        print(f"Capacity: {result.system_config.total_capacity_kw:.1f} kW ({result.system_config.num_panels} panels)")
        print(f"Inverter: {result.system_config.num_inverters}× {result.system_config.inverter.brand} {result.system_config.inverter.model}")
        print(f"DC/AC Ratio: {result.system_config.dc_ac_ratio:.2f}")
        print(f"Meets Requirement: {'✓' if result.system_config.meets_requirement else '✗'}")
        
        print(f"\nCOST BREAKDOWN:")
        print(f"Total Cost: ₹{result.cost_breakdown.total_after_incentives:,.0f}")
        print(f"Cost per kW: ₹{result.cost_breakdown.total_after_incentives/result.system_config.total_capacity_kw:,.0f}")
        
        print(f"\nPERFORMANCE:")
        print(f"Year 1 Generation: {result.performance_prediction.annual_generation_kwh[0]:,.0f} kWh")
        print(f"Monthly Generation: {result.performance_prediction.annual_generation_kwh[0]/12:,.0f} kWh")
        print(f"Capacity Factor: {result.performance_prediction.capacity_factor:.1%}")
        coverage = (result.performance_prediction.annual_generation_kwh[0]/12)/500
        print(f"Requirement Coverage: {coverage:.1%}")
        
        print(f"\nFINANCIALS:")
        if result.payback_analysis['simple_payback_years'] == float('inf'):
            print(f"Payback Period: Not achievable")
        else:
            print(f"Payback Period: {result.payback_analysis['simple_payback_years']:.1f} years")
        print(f"NPV (25 years): ₹{result.payback_analysis['npv']:,.0f}")
        print(f"IRR: {result.payback_analysis['irr']:.1f}%")
        
        if result.warnings:
            print(f"\nWARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"  • {rec}")
        
        print(f"\nCORRECTED SIZING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_corrected_sizing()