"""
Advanced Rooftop Feasibility Analyzer for Solar Installations
===========================================================

Comprehensive rooftop assessment including:
- Structural load analysis
- Geometric feasibility with 3D modeling
- Shading analysis with sun-path calculations
- Regulatory compliance checks
- Installation complexity assessment
- Risk evaluation and mitigation strategies

Author: Solar Engineering Team
Version: 2.0 (Production Ready)
Pipeline Integration: Engines Layer Module 5.5
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from enum import Enum
import logging
from scipy import spatial
import math
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class RooftopType(Enum):
    CONCRETE_FLAT = {
        "load_capacity_psf": 150, "complexity_score": 1.0, "penetration_allowed": True,
        "mounting_options": ["ballasted", "penetrating", "hybrid"], "typical_lifespan": 50
    }
    CONCRETE_SLOPED = {
        "load_capacity_psf": 120, "complexity_score": 1.3, "penetration_allowed": True,
        "mounting_options": ["rail_mounted", "penetrating"], "typical_lifespan": 50
    }
    METAL_SHEET = {
        "load_capacity_psf": 80, "complexity_score": 0.9, "penetration_allowed": False,
        "mounting_options": ["standing_seam", "trapezoidal_clamps"], "typical_lifespan": 25
    }
    TILE_ROOF = {
        "load_capacity_psf": 100, "complexity_score": 1.6, "penetration_allowed": True,
        "mounting_options": ["tile_replacement", "tile_hooks"], "typical_lifespan": 30
    }
    ASBESTOS_SHEET = {
        "load_capacity_psf": 60, "complexity_score": 2.2, "penetration_allowed": False,
        "mounting_options": ["replacement_required", "overlay_system"], "typical_lifespan": 20
    }

class BuildingType(Enum):
    RESIDENTIAL_INDEPENDENT = {"structural_margin": 1.5, "regulatory_complexity": 1.0, "access_score": 0.8}
    RESIDENTIAL_APARTMENT = {"structural_margin": 2.0, "regulatory_complexity": 1.8, "access_score": 0.6}
    COMMERCIAL_LOW_RISE = {"structural_margin": 2.5, "regulatory_complexity": 1.4, "access_score": 0.9}
    COMMERCIAL_HIGH_RISE = {"structural_margin": 3.0, "regulatory_complexity": 2.0, "access_score": 0.4}
    INDUSTRIAL_SHED = {"structural_margin": 1.2, "regulatory_complexity": 1.2, "access_score": 0.9}

class ObstacleType(Enum):
    WATER_TANK = {"height_m": 2.5, "shadow_factor": 1.8, "clearance_required_m": 2.0}
    AC_UNIT = {"height_m": 1.2, "shadow_factor": 1.2, "clearance_required_m": 1.5}
    STAIRCASE = {"height_m": 3.0, "shadow_factor": 2.0, "clearance_required_m": 2.5}
    PARAPET_WALL = {"height_m": 1.0, "shadow_factor": 1.0, "clearance_required_m": 1.0}
    LIFT_ROOM = {"height_m": 4.0, "shadow_factor": 2.5, "clearance_required_m": 3.0}
    ANTENNA_TOWER = {"height_m": 8.0, "shadow_factor": 4.0, "clearance_required_m": 5.0}
    CHIMNEY = {"height_m": 6.0, "shadow_factor": 3.0, "clearance_required_m": 4.0}
    TREE_LARGE = {"height_m": 10.0, "shadow_factor": 3.5, "clearance_required_m": 6.0}

@dataclass
class RooftopGeometry:
    """Detailed rooftop geometric specifications"""
    total_area_sqm: float
    usable_area_sqm: float
    length_m: float
    width_m: float
    shape: str  # "rectangular", "l_shaped", "irregular", "complex"
    roof_orientation: float  # degrees from north
    tilt_angle: float  # degrees from horizontal
    height_above_ground_m: float
    perimeter_setback_m: float = 1.0
    access_points: List[Dict[str, Any]] = field(default_factory=list)
    structural_zones: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RooftopObstacle:
    """Individual rooftop obstacle specification"""
    obstacle_type: ObstacleType
    position_x: float  # meters from reference point
    position_y: float  # meters from reference point
    dimensions: Tuple[float, float, float]  # length, width, height in meters
    is_permanent: bool = True
    relocation_possible: bool = False
    shadow_impact_score: float = 0.0

@dataclass
class StructuralAssessment:
    """Structural capacity and safety assessment"""
    roof_type: RooftopType
    building_age_years: int
    structural_condition: str  # "excellent", "good", "fair", "poor"
    dead_load_capacity_psf: float
    live_load_capacity_psf: float
    seismic_zone: str  # "I", "II", "III", "IV", "V"
    wind_zone: str  # "I", "II", "III", "IV", "V", "VI"
    requires_structural_analysis: bool = False
    load_bearing_adequacy: float = 0.0  # 0-1 score
    reinforcement_required: bool = False
    engineer_approval_needed: bool = False

@dataclass
class ShadingAnalysis:
    """Comprehensive shading impact analysis"""
    obstacles: List[RooftopObstacle]
    surrounding_buildings: List[Dict[str, Any]]
    terrain_features: List[Dict[str, Any]]
    seasonal_shading_patterns: Dict[str, List[float]]
    hourly_shading_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_shading_loss_percent: float = 0.0
    critical_shading_hours: List[int] = field(default_factory=list)
    mitigation_options: List[str] = field(default_factory=list)

@dataclass
class RegulatoryCompliance:
    """Regulatory and code compliance assessment"""
    building_approval_status: str  # "approved", "pending", "required"
    electricity_board_clearance: str
    fire_safety_compliance: bool
    structural_safety_certificate: str
    environmental_clearance_needed: bool
    height_restrictions: Dict[str, float]
    setback_requirements: Dict[str, float]
    municipal_permissions_needed: List[str]
    estimated_approval_timeline_days: int = 30

@dataclass
class InstallationComplexity:
    """Installation complexity and logistics assessment"""
    access_difficulty_score: float  # 0-10 scale
    crane_requirement: bool
    scaffolding_requirement: bool
    working_area_constraints: List[str]
    weather_sensitivity: float  # 0-1 scale
    safety_risk_level: str  # "low", "medium", "high", "critical"
    specialized_equipment_needed: List[str]
    estimated_installation_days: int
    labor_skill_requirement: str  # "basic", "intermediate", "advanced", "expert"

@dataclass
class FeasibilityResult:
    """Comprehensive feasibility assessment result"""
    overall_feasibility_score: float  # 0-100 scale
    is_technically_feasible: bool
    is_economically_viable: bool
    is_regulatory_compliant: bool
    maximum_installable_capacity_kw: float
    optimal_panel_layout: Dict[str, Any]
    structural_assessment: StructuralAssessment
    shading_analysis: ShadingAnalysis
    regulatory_compliance: RegulatoryCompliance
    installation_complexity: InstallationComplexity
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[Dict[str, Any]]
    cost_implications: Dict[str, float]
    timeline_estimate: Dict[str, int]
    confidence_level: float
    recommendations: List[str]
    warnings: List[str]

class AdvancedRooftopFeasibilityAnalyzer:
    """Comprehensive rooftop feasibility analyzer for solar installations"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.load_assessment_parameters()
    
    def load_assessment_parameters(self):
        """Load assessment parameters and lookup tables"""
        
        # Panel specifications for layout optimization
        self.standard_panel_dimensions = {
            "residential": (2.0, 1.0),  # 2m x 1m
            "commercial": (2.3, 1.3),  # Larger panels
            "high_efficiency": (1.8, 1.0)  # Compact high-efficiency
        }
        
        # Regional building codes and standards
        self.building_codes = {
            "mumbai": {
                "wind_load_psf": 45, "seismic_factor": 0.24, "height_restrictions": {"residential": 70, "commercial": 150},
                "fire_safety_mandatory": True, "structural_audit_threshold_years": 30
            },
            "delhi": {
                "wind_load_psf": 40, "seismic_factor": 0.36, "height_restrictions": {"residential": 65, "commercial": 120},
                "fire_safety_mandatory": True, "structural_audit_threshold_years": 25
            },
            "chennai": {
                "wind_load_psf": 50, "seismic_factor": 0.16, "height_restrictions": {"residential": 60, "commercial": 180},
                "fire_safety_mandatory": False, "structural_audit_threshold_years": 35
            },
            "bangalore": {
                "wind_load_psf": 35, "seismic_factor": 0.16, "height_restrictions": {"residential": 65, "commercial": 140},
                "fire_safety_mandatory": False, "structural_audit_threshold_years": 30
            }
        }
        
        # Default parameters for unknown locations
        self.default_building_code = {
            "wind_load_psf": 40, "seismic_factor": 0.24, "height_restrictions": {"residential": 60, "commercial": 120},
            "fire_safety_mandatory": False, "structural_audit_threshold_years": 30
        }
        
        # Safety factors and margins
        self.safety_factors = {
            "structural_load": 2.0,
            "wind_uplift": 1.5,
            "seismic": 1.8,
            "thermal_expansion": 1.2
        }
    
    def analyze_rooftop_feasibility(self,
                                  rooftop_geometry: RooftopGeometry,
                                  building_details: Dict[str, Any],
                                  location_details: Dict[str, Any],
                                  system_requirements: Dict[str, Any],
                                  regulatory_context: Dict[str, Any]) -> FeasibilityResult:
        """
        Comprehensive rooftop feasibility analysis
        
        Args:
            rooftop_geometry: Detailed geometric specifications
            building_details: Building type, age, condition, etc.
            location_details: City, climate zone, local regulations
            system_requirements: Required capacity, panel preferences
            regulatory_context: Permit requirements, approvals needed
        """
        
        logger.info("Starting comprehensive rooftop feasibility analysis")
        
        # 1. Structural Assessment
        structural_assessment = self._assess_structural_feasibility(
            rooftop_geometry, building_details, location_details
        )
        
        # 2. Geometric Analysis and Layout Optimization
        layout_analysis = self._optimize_panel_layout(
            rooftop_geometry, system_requirements, structural_assessment
        )
        
        # 3. Shading Analysis
        shading_analysis = self._perform_shading_analysis(
            rooftop_geometry, building_details, location_details
        )
        
        # 4. Regulatory Compliance Check
        regulatory_compliance = self._assess_regulatory_compliance(
            rooftop_geometry, building_details, location_details, regulatory_context
        )
        
        # 5. Installation Complexity Assessment
        installation_complexity = self._assess_installation_complexity(
            rooftop_geometry, building_details, structural_assessment, layout_analysis
        )
        
        # 6. Risk Analysis
        risk_factors, mitigation_strategies = self._analyze_risks(
            structural_assessment, shading_analysis, regulatory_compliance, 
            installation_complexity
        )
        
        # 7. Economic Viability Assessment
        cost_implications = self._assess_cost_implications(
            structural_assessment, installation_complexity, regulatory_compliance
        )
        
        # 8. Overall Feasibility Scoring
        overall_score, is_feasible = self._calculate_overall_feasibility(
            structural_assessment, shading_analysis, regulatory_compliance,
            installation_complexity, layout_analysis
        )
        
        # 9. Generate Recommendations and Warnings
        recommendations, warnings = self._generate_recommendations(
            structural_assessment, shading_analysis, regulatory_compliance,
            installation_complexity, risk_factors, layout_analysis
        )
        
        # 10. Timeline Estimation
        timeline_estimate = self._estimate_project_timeline(
            regulatory_compliance, installation_complexity, structural_assessment
        )
        
        result = FeasibilityResult(
            overall_feasibility_score=overall_score,
            is_technically_feasible=is_feasible,
            is_economically_viable=cost_implications.get('viability_score', 0) > 0.6,
            is_regulatory_compliant=regulatory_compliance.fire_safety_compliance and 
                                   regulatory_compliance.building_approval_status != "required",
            maximum_installable_capacity_kw=layout_analysis.get('max_capacity_kw', 0),
            optimal_panel_layout=layout_analysis,
            structural_assessment=structural_assessment,
            shading_analysis=shading_analysis,
            regulatory_compliance=regulatory_compliance,
            installation_complexity=installation_complexity,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            cost_implications=cost_implications,
            timeline_estimate=timeline_estimate,
            confidence_level=self._calculate_confidence_level(structural_assessment, 
                                                           shading_analysis, regulatory_compliance),
            recommendations=recommendations,
            warnings=warnings
        )
        
        if self.debug_mode:
            self._print_feasibility_summary(result)
        
        logger.info(f"Feasibility analysis complete. Overall score: {overall_score:.1f}/100")
        return result
    
    def _assess_structural_feasibility(self,
                                     geometry: RooftopGeometry,
                                     building: Dict[str, Any],
                                     location: Dict[str, Any]) -> StructuralAssessment:
        """Comprehensive structural feasibility assessment"""
        
        roof_type = RooftopType[building.get('roof_type', 'CONCRETE_FLAT')]
        building_age = building.get('age_years', 10)
        building_type = BuildingType[building.get('building_type', 'RESIDENTIAL_INDEPENDENT')]
        
        # Get local building codes
        city = location.get('city', 'mumbai').lower()
        building_code = self.building_codes.get(city, self.default_building_code)
        
        # Calculate solar system loads
        panel_weight_psf = 3.5  # Typical panel weight per sq ft
        mounting_weight_psf = 2.0  # Mounting structure weight
        maintenance_load_psf = 25  # Live load for maintenance
        wind_load_psf = building_code['wind_load_psf']
        seismic_load_factor = building_code['seismic_factor']
        
        total_dead_load = panel_weight_psf + mounting_weight_psf
        total_live_load = maintenance_load_psf
        total_design_load = (total_dead_load + total_live_load) * self.safety_factors['structural_load']
        
        # Assess roof capacity
        roof_capacity = roof_type.value['load_capacity_psf']
        
        # Age-based capacity reduction
        age_factor = max(0.7, 1 - (building_age - 20) * 0.01) if building_age > 20 else 1.0
        effective_capacity = roof_capacity * age_factor
        
        # Load bearing adequacy
        load_bearing_adequacy = min(effective_capacity / total_design_load, 1.0)
        
        # Determine if structural analysis is required
        requires_analysis = (
            building_age > building_code['structural_audit_threshold_years'] or
            load_bearing_adequacy < 0.8 or
            building_type == BuildingType.COMMERCIAL_HIGH_RISE or
            geometry.height_above_ground_m > 30
        )
        
        # Determine if reinforcement is needed
        reinforcement_required = load_bearing_adequacy < 0.7
        
        # Engineer approval requirements
        engineer_approval_needed = (
            requires_analysis or
            reinforcement_required or
            building.get('structural_condition', 'good') == 'poor'
        )
        
        return StructuralAssessment(
            roof_type=roof_type,
            building_age_years=building_age,
            structural_condition=building.get('structural_condition', 'good'),
            dead_load_capacity_psf=effective_capacity,
            live_load_capacity_psf=total_live_load,
            seismic_zone=building.get('seismic_zone', 'III'),
            wind_zone=building.get('wind_zone', 'III'),
            requires_structural_analysis=requires_analysis,
            load_bearing_adequacy=load_bearing_adequacy,
            reinforcement_required=reinforcement_required,
            engineer_approval_needed=engineer_approval_needed
        )
    
    def _optimize_panel_layout(self,
                             geometry: RooftopGeometry,
                             requirements: Dict[str, Any],
                             structural: StructuralAssessment) -> Dict[str, Any]:
        """Optimize panel layout considering geometric and structural constraints"""
        
        # Panel specifications
        panel_type = requirements.get('panel_preference', 'residential')
        panel_length, panel_width = self.standard_panel_dimensions[panel_type]
        panel_area = panel_length * panel_width
        panel_power_w = requirements.get('panel_wattage', 540)
        
        # Available area calculation
        total_area = geometry.usable_area_sqm
        setback_area = geometry.perimeter_setback_m * (
            2 * geometry.length_m + 2 * geometry.width_m - 4 * geometry.perimeter_setback_m
        )
        
        # Obstacle area reduction
        obstacle_area = sum([
            obs.dimensions[0] * obs.dimensions[1] + 
            obs.obstacle_type.value['clearance_required_m'] ** 2 * 3.14159 / 4
            for obs in geometry.structural_zones if 'obstacle' in str(obs).lower()
        ])
        
        effective_area = total_area - setback_area - obstacle_area
        
        # Layout optimization considering roof shape
        if geometry.shape == "rectangular":
            layout_efficiency = 0.85
        elif geometry.shape == "l_shaped":
            layout_efficiency = 0.75
        elif geometry.shape == "irregular":
            layout_efficiency = 0.65
        else:  # complex
            layout_efficiency = 0.60
        
        installable_area = effective_area * layout_efficiency
        
        # Panel count and capacity calculations
        max_panels_by_area = int(installable_area / panel_area)
        
        # Structural load constraints
        if structural.load_bearing_adequacy < 1.0:
            max_panels_by_load = int(max_panels_by_area * structural.load_bearing_adequacy)
        else:
            max_panels_by_load = max_panels_by_area
        
        max_panels = min(max_panels_by_area, max_panels_by_load)
        max_capacity_kw = max_panels * panel_power_w / 1000
        
        # Generate optimal layout configuration
        if geometry.shape == "rectangular":
            panels_length = int(geometry.length_m / panel_length)
            panels_width = int(geometry.width_m / panel_width)
            layout_config = {
                "array_config": f"{panels_length}×{panels_width}",
                "orientation": "landscape" if panel_length > panel_width else "portrait",
                "rows": panels_width,
                "panels_per_row": panels_length
            }
        else:
            layout_config = {
                "array_config": "optimized_irregular",
                "total_panels": max_panels,
                "layout_type": "adaptive"
            }
        
        # Spacing and access considerations
        row_spacing_m = max(2.5, panel_length * 1.5)  # For maintenance access
        inter_panel_spacing_m = 0.02  # Standard gap between panels
        
        return {
            "max_capacity_kw": max_capacity_kw,
            "max_panels": max_panels,
            "effective_area_sqm": installable_area,
            "layout_efficiency": layout_efficiency,
            "panel_dimensions": (panel_length, panel_width),
            "layout_config": layout_config,
            "row_spacing_m": row_spacing_m,
            "access_pathways": self._design_access_pathways(geometry),
            "structural_compliance": structural.load_bearing_adequacy >= 0.7,
            "area_utilization_percent": (installable_area / total_area) * 100
        }
    
    def _perform_shading_analysis(self,
                                geometry: RooftopGeometry,
                                building: Dict[str, Any],
                                location: Dict[str, Any]) -> ShadingAnalysis:
        """Comprehensive shading analysis using sun-path calculations"""
        
        latitude = location.get('latitude', 19.0760)  # Default Mumbai
        longitude = location.get('longitude', 72.8777)
        
        # Create obstacle list from building details
        obstacles = []
        
        # Add obstacles from building details
        for obstacle_data in building.get('rooftop_obstacles', []):
            obstacle_type = ObstacleType[obstacle_data['type']]
            obstacle = RooftopObstacle(
                obstacle_type=obstacle_type,
                position_x=obstacle_data.get('x', 0),
                position_y=obstacle_data.get('y', 0),
                dimensions=obstacle_data.get('dimensions', obstacle_type.value['height_m']),
                is_permanent=obstacle_data.get('permanent', True),
                relocation_possible=obstacle_data.get('relocatable', False)
            )
            obstacles.append(obstacle)
        
        # Surrounding buildings analysis
        surrounding_buildings = building.get('surrounding_buildings', [])
        
        # Calculate seasonal shading patterns
        seasonal_patterns = {}
        critical_hours = []
        annual_shading_loss = 0.0
        
        for season in ['summer', 'monsoon', 'winter']:
            if season == 'summer':
                sun_elevation_range = (65, 85)  # High sun angles
                daylight_hours = range(6, 19)
            elif season == 'monsoon':
                sun_elevation_range = (45, 75)  # Medium sun angles
                daylight_hours = range(7, 18)
            else:  # winter
                sun_elevation_range = (35, 65)  # Lower sun angles
                daylight_hours = range(7, 17)
            
            hourly_shading = []
            for hour in daylight_hours:
                # Simplified shading calculation
                shading_factor = self._calculate_hourly_shading(
                    obstacles, surrounding_buildings, hour, season, latitude
                )
                hourly_shading.append(shading_factor)
                
                # Identify critical shading hours (peak production hours)
                if 10 <= hour <= 14 and shading_factor > 0.2:
                    critical_hours.append(hour)
            
            seasonal_patterns[season] = hourly_shading
            annual_shading_loss += np.mean(hourly_shading) * (365 / 3)  # Seasonal average
        
        # Mitigation options
        mitigation_options = []
        for obstacle in obstacles:
            if obstacle.relocation_possible:
                mitigation_options.append(f"Relocate {obstacle.obstacle_type.name}")
            elif obstacle.shadow_impact_score > 0.15:
                mitigation_options.append(f"Install tilt system to avoid {obstacle.obstacle_type.name} shadow")
        
        return ShadingAnalysis(
            obstacles=obstacles,
            surrounding_buildings=surrounding_buildings,
            terrain_features=building.get('terrain_features', []),
            seasonal_shading_patterns=seasonal_patterns,
            annual_shading_loss_percent=min(annual_shading_loss / 365 * 100, 100),
            critical_shading_hours=critical_hours,
            mitigation_options=mitigation_options
        )
    
    def _assess_regulatory_compliance(self,
                                   geometry: RooftopGeometry,
                                   building: Dict[str, Any],
                                   location: Dict[str, Any],
                                   regulatory: Dict[str, Any]) -> RegulatoryCompliance:
        """Assess regulatory compliance requirements"""
        
        city = location.get('city', 'mumbai').lower()
        building_code = self.building_codes.get(city, self.default_building_code)
        
        # Building approval status
        building_age = building.get('age_years', 10)
        approval_status = regulatory.get('building_approval', 'approved' if building_age > 5 else 'required')
        
        # Electricity board clearance
        capacity_kw = regulatory.get('system_capacity_kw', 5)
        if capacity_kw <= 10:
            eb_clearance = "not_required"
        elif capacity_kw <= 100:
            eb_clearance = "technical_approval_required"
        else:
            eb_clearance = "detailed_feasibility_required"
        
        # Fire safety compliance
        fire_safety_required = (
            building_code['fire_safety_mandatory'] or
            geometry.height_above_ground_m > 15 or
            building.get('building_type') in ['COMMERCIAL_HIGH_RISE', 'RESIDENTIAL_APARTMENT']
        )
        
        # Height restrictions check
        building_type = building.get('building_type', 'residential').lower().split('_')[0]
        max_height = building_code['height_restrictions'].get(building_type, 60)
        height_compliance = geometry.height_above_ground_m <= max_height
        
        # Setback requirements
        setback_requirements = {
            "front": max(3.0, geometry.height_above_ground_m * 0.1),
            "rear": max(2.0, geometry.height_above_ground_m * 0.08),
            "sides": max(1.5, geometry.height_above_ground_m * 0.06)
        }
        
        # Municipal permissions needed
        permissions_needed = []
        if approval_status == 'required':
            permissions_needed.append("Building plan approval")
        if not fire_safety_required:
            permissions_needed.append("Fire NOC")
        if capacity_kw > 10:
            permissions_needed.append("Electrical inspector approval")
        if geometry.height_above_ground_m > 30:
            permissions_needed.append("Aviation clearance")
        
        # Timeline estimation
        timeline_days = 15  # Base timeline
        timeline_days += len(permissions_needed) * 10
        if eb_clearance != "not_required":
            timeline_days += 30
        if not height_compliance:
            timeline_days += 60  # Additional approvals needed
        
        return RegulatoryCompliance(
            building_approval_status=approval_status,
            electricity_board_clearance=eb_clearance,
            fire_safety_compliance=fire_safety_required,
            structural_safety_certificate="required" if building_age > 25 else "not_required",
            environmental_clearance_needed=capacity_kw > 1000,  # 1 MW threshold
            height_restrictions={"max_allowed": max_height, "current": geometry.height_above_ground_m,
                               "compliance": height_compliance},
            setback_requirements=setback_requirements,
            municipal_permissions_needed=permissions_needed,
            estimated_approval_timeline_days=timeline_days
        )
    
    def _assess_installation_complexity(self,
                                      geometry: RooftopGeometry,
                                      building: Dict[str, Any],
                                      structural: StructuralAssessment,
                                      layout: Dict[str, Any]) -> InstallationComplexity:
        """Assess installation complexity and requirements"""
        
        # Access difficulty scoring
        access_score = 0
        height_factor = min(geometry.height_above_ground_m / 30, 3.0)  # 0-3 scale
        access_score += height_factor
        
        roof_type = structural.roof_type
        if roof_type in [RooftopType.TILE_ROOF, RooftopType.ASBESTOS_SHEET]:
            access_score += 2.0
        elif roof_type == RooftopType.METAL_SHEET:
            access_score += 1.0
        
        if geometry.shape in ["irregular", "complex"]:
            access_score += 1.5
        
        building_type = BuildingType[building.get('building_type', 'RESIDENTIAL_INDEPENDENT')]
        access_score *= building_type.value['access_score']
        
        # Equipment requirements
        crane_required = (
            geometry.height_above_ground_m > 15 or
            layout.get('max_capacity_kw', 0) > 20 or
            access_score > 6
        )
        
        scaffolding_required = (
            roof_type == RooftopType.TILE_ROOF or
            roof_type == RooftopType.ASBESTOS_SHEET or
            geometry.height_above_ground_m > 10 or
            geometry.tilt_angle > 15
        )
        
        # Working area constraints
        working_constraints = []
        if len(geometry.access_points) < 2:
            working_constraints.append("Limited access points")
        if geometry.usable_area_sqm < 30:
            working_constraints.append("Cramped working space")
        if structural.reinforcement_required:
            working_constraints.append("Structural reinforcement required")
        
        # Weather sensitivity
        weather_sensitivity = 0.3  # Base sensitivity
        if roof_type in [RooftopType.METAL_SHEET, RooftopType.ASBESTOS_SHEET]:
            weather_sensitivity += 0.3  # More weather dependent
        if geometry.height_above_ground_m > 20:
            weather_sensitivity += 0.2  # Wind concerns
        weather_sensitivity = min(weather_sensitivity, 1.0)
        
        # Safety risk assessment
        risk_factors = [
            height_factor / 3.0,
            1.0 if roof_type == RooftopType.ASBESTOS_SHEET else 0.0,
            0.5 if structural.structural_condition == 'poor' else 0.0,
            weather_sensitivity
        ]
        risk_level = np.mean(risk_factors)
        
        if risk_level > 0.7:
            safety_risk = "critical"
        elif risk_level > 0.5:
            safety_risk = "high"
        elif risk_level > 0.3:
            safety_risk = "medium"
        else:
            safety_risk = "low"
        
        # Specialized equipment needed
        specialized_equipment = []
        if crane_required:
            specialized_equipment.append("Mobile crane")
        if scaffolding_required:
            specialized_equipment.append("Safety scaffolding")
        if roof_type == RooftopType.TILE_ROOF:
            specialized_equipment.append("Tile cutting tools")
        if geometry.height_above_ground_m > 25:
            specialized_equipment.append("High-altitude safety equipment")
        
        # Installation timeline estimation
        base_days = max(2, layout.get('max_panels', 10) / 8)  # 8 panels per day base rate
        
        complexity_multiplier = 1.0
        if roof_type in [RooftopType.TILE_ROOF, RooftopType.ASBESTOS_SHEET]:
            complexity_multiplier *= 1.5
        if crane_required:
            complexity_multiplier *= 1.3
        if structural.reinforcement_required:
            complexity_multiplier *= 2.0
        if geometry.shape in ["irregular", "complex"]:
            complexity_multiplier *= 1.4
        
        installation_days = int(base_days * complexity_multiplier)
        
        # Labor skill requirement
        if roof_type == RooftopType.ASBESTOS_SHEET or safety_risk == "critical":
            skill_requirement = "expert"
        elif roof_type == RooftopType.TILE_ROOF or structural.reinforcement_required:
            skill_requirement = "advanced"
        elif crane_required or geometry.height_above_ground_m > 15:
            skill_requirement = "intermediate"
        else:
            skill_requirement = "basic"
        
        return InstallationComplexity(
            access_difficulty_score=min(access_score, 10),
            crane_requirement=crane_required,
            scaffolding_requirement=scaffolding_required,
            working_area_constraints=working_constraints,
            weather_sensitivity=weather_sensitivity,
            safety_risk_level=safety_risk,
            specialized_equipment_needed=specialized_equipment,
            estimated_installation_days=installation_days,
            labor_skill_requirement=skill_requirement
        )
    
    def _analyze_risks(self,
                      structural: StructuralAssessment,
                      shading: ShadingAnalysis,
                      regulatory: RegulatoryCompliance,
                      installation: InstallationComplexity) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Comprehensive risk analysis and mitigation strategies"""
        
        risk_factors = []
        mitigation_strategies = []
        
        # Structural risks
        if structural.load_bearing_adequacy < 0.8:
            risk_factors.append({
                "category": "structural",
                "risk": "Inadequate load bearing capacity",
                "severity": "high" if structural.load_bearing_adequacy < 0.6 else "medium",
                "probability": 0.8,
                "impact": "System failure, safety hazard"
            })
            mitigation_strategies.append({
                "risk_category": "structural",
                "strategy": "Structural reinforcement with engineer certification",
                "cost_impact": "15-25% increase",
                "timeline_impact": "2-4 weeks additional"
            })
        
        if structural.building_age_years > 25:
            risk_factors.append({
                "category": "structural",
                "risk": "Aging building structure",
                "severity": "medium",
                "probability": 0.4,
                "impact": "Unexpected maintenance, reduced lifespan"
            })
            mitigation_strategies.append({
                "risk_category": "structural",
                "strategy": "Comprehensive structural audit before installation",
                "cost_impact": "₹25,000-50,000",
                "timeline_impact": "1-2 weeks"
            })
        
        # Shading risks
        if shading.annual_shading_loss_percent > 15:
            risk_factors.append({
                "category": "performance",
                "risk": f"Significant shading loss ({shading.annual_shading_loss_percent:.1f}%)",
                "severity": "high" if shading.annual_shading_loss_percent > 25 else "medium",
                "probability": 0.9,
                "impact": "Reduced energy generation, poor ROI"
            })
            mitigation_strategies.append({
                "risk_category": "performance",
                "strategy": "Implement micro-inverters or power optimizers",
                "cost_impact": "10-20% increase",
                "timeline_impact": "No impact"
            })
        
        # Installation risks
        if installation.safety_risk_level in ["high", "critical"]:
            risk_factors.append({
                "category": "safety",
                "risk": f"High installation safety risk ({installation.safety_risk_level})",
                "severity": installation.safety_risk_level,
                "probability": 0.3,
                "impact": "Worker injury, project delays, liability"
            })
            mitigation_strategies.append({
                "risk_category": "safety",
                "strategy": "Enhanced safety protocols and specialized contractors",
                "cost_impact": "8-15% increase",
                "timeline_impact": "20% longer installation"
            })
        
        # Regulatory risks
        if len(regulatory.municipal_permissions_needed) > 2:
            risk_factors.append({
                "category": "regulatory",
                "risk": "Complex approval process",
                "severity": "medium",
                "probability": 0.6,
                "impact": "Project delays, additional costs"
            })
            mitigation_strategies.append({
                "risk_category": "regulatory",
                "strategy": "Engage regulatory consultant for expedited approvals",
                "cost_impact": "₹15,000-30,000",
                "timeline_impact": "Potential 2-4 week reduction"
            })
        
        # Weather and environmental risks
        if installation.weather_sensitivity > 0.6:
            risk_factors.append({
                "category": "environmental",
                "risk": "High weather sensitivity during installation",
                "severity": "medium",
                "probability": 0.4,
                "impact": "Seasonal installation constraints, delays"
            })
            mitigation_strategies.append({
                "risk_category": "environmental",
                "strategy": "Schedule installation during favorable weather windows",
                "cost_impact": "No additional cost",
                "timeline_impact": "Seasonal timing constraints"
            })
        
        return risk_factors, mitigation_strategies
    
    def _assess_cost_implications(self,
                                structural: StructuralAssessment,
                                installation: InstallationComplexity,
                                regulatory: RegulatoryCompliance) -> Dict[str, float]:
        """Assess cost implications of feasibility factors"""
        
        base_cost_multiplier = 1.0
        additional_costs = 0
        
        # Structural cost impacts
        if structural.reinforcement_required:
            base_cost_multiplier *= 1.25
            additional_costs += 50000  # Reinforcement cost
        
        if structural.engineer_approval_needed:
            additional_costs += 25000  # Structural engineer fees
        
        # Installation complexity cost impacts
        complexity_multipliers = {
            "basic": 1.0,
            "intermediate": 1.15,
            "advanced": 1.35,
            "expert": 1.6
        }
        skill_multiplier = complexity_multipliers.get(installation.labor_skill_requirement, 1.2)
        base_cost_multiplier *= skill_multiplier
        
        if installation.crane_requirement:
            additional_costs += 15000  # Crane rental
        
        if installation.scaffolding_requirement:
            additional_costs += 8000  # Scaffolding setup
        
        # Safety equipment costs
        safety_multipliers = {
            "low": 1.0,
            "medium": 1.05,
            "high": 1.15,
            "critical": 1.25
        }
        safety_multiplier = safety_multipliers.get(installation.safety_risk_level, 1.1)
        base_cost_multiplier *= safety_multiplier
        
        # Regulatory cost impacts
        permit_costs = len(regulatory.municipal_permissions_needed) * 5000
        additional_costs += permit_costs
        
        if regulatory.electricity_board_clearance != "not_required":
            additional_costs += 15000  # EB approval fees
        
        # Timeline cost impacts (extended timeline = higher costs)
        if installation.estimated_installation_days > 7:
            extended_days = installation.estimated_installation_days - 7
            additional_costs += extended_days * 3000  # Additional daily costs
        
        # Calculate viability score
        total_cost_impact = base_cost_multiplier + (additional_costs / 100000)  # Normalize
        viability_score = max(0, 1 - (total_cost_impact - 1) / 0.5)  # 0-1 scale
        
        return {
            "base_cost_multiplier": base_cost_multiplier,
            "additional_fixed_costs": additional_costs,
            "total_cost_impact_factor": total_cost_impact,
            "viability_score": viability_score,
            "structural_cost_impact": additional_costs if structural.reinforcement_required else 0,
            "installation_cost_impact": additional_costs * 0.6,
            "regulatory_cost_impact": permit_costs + (15000 if regulatory.electricity_board_clearance != "not_required" else 0)
        }
    
    def _calculate_overall_feasibility(self,
                                     structural: StructuralAssessment,
                                     shading: ShadingAnalysis,
                                     regulatory: RegulatoryCompliance,
                                     installation: InstallationComplexity,
                                     layout: Dict[str, Any]) -> Tuple[float, bool]:
        """Calculate overall feasibility score and determination"""
        
        # Structural feasibility (25% weight)
        structural_score = 0
        if structural.load_bearing_adequacy >= 0.8:
            structural_score = 25
        elif structural.load_bearing_adequacy >= 0.6:
            structural_score = 18
        elif structural.load_bearing_adequacy >= 0.4:
            structural_score = 10
        else:
            structural_score = 0
        
        # Performance feasibility (30% weight)
        shading_loss = shading.annual_shading_loss_percent
        if shading_loss <= 10:
            performance_score = 30
        elif shading_loss <= 20:
            performance_score = 22
        elif shading_loss <= 35:
            performance_score = 15
        else:
            performance_score = 5
        
        # Regulatory feasibility (20% weight)
        regulatory_score = 20
        if regulatory.building_approval_status == "required":
            regulatory_score -= 8
        if not regulatory.fire_safety_compliance:
            regulatory_score -= 5
        if len(regulatory.municipal_permissions_needed) > 3:
            regulatory_score -= 4
        if regulatory.estimated_approval_timeline_days > 90:
            regulatory_score -= 3
        regulatory_score = max(regulatory_score, 0)
        
        # Installation feasibility (15% weight)
        installation_score = 15
        if installation.safety_risk_level == "critical":
            installation_score = 2
        elif installation.safety_risk_level == "high":
            installation_score = 8
        elif installation.safety_risk_level == "medium":
            installation_score = 12
        
        # Economic feasibility (10% weight)
        if layout.get('max_capacity_kw', 0) >= 3:
            economic_score = 10
        elif layout.get('max_capacity_kw', 0) >= 1.5:
            economic_score = 7
        elif layout.get('max_capacity_kw', 0) >= 1:
            economic_score = 4
        else:
            economic_score = 0
        
        overall_score = structural_score + performance_score + regulatory_score + installation_score + economic_score
        
        # Feasibility determination
        is_feasible = (
            overall_score >= 60 and
            structural.load_bearing_adequacy >= 0.6 and
            shading.annual_shading_loss_percent <= 40 and
            installation.safety_risk_level != "critical" and
            layout.get('max_capacity_kw', 0) >= 1
        )
        
        return overall_score, is_feasible
    
    def _calculate_hourly_shading(self,
                                obstacles: List[RooftopObstacle],
                                surrounding_buildings: List[Dict[str, Any]],
                                hour: int,
                                season: str,
                                latitude: float) -> float:
        """Calculate shading factor for specific hour and season"""
        
        # Solar position calculation (simplified)
        if season == 'summer':
            declination = 23.5
        elif season == 'winter':
            declination = -23.5
        else:  # monsoon
            declination = 0
        
        # Solar elevation angle
        hour_angle = 15 * (hour - 12)  # degrees
        elevation = math.asin(
            math.sin(math.radians(declination)) * math.sin(math.radians(latitude)) +
            math.cos(math.radians(declination)) * math.cos(math.radians(latitude)) * 
            math.cos(math.radians(hour_angle))
        )
        elevation_deg = math.degrees(elevation)
        
        # Azimuth angle
        azimuth = math.atan2(
            math.sin(math.radians(hour_angle)),
            math.cos(math.radians(hour_angle)) * math.sin(math.radians(latitude)) -
            math.tan(math.radians(declination)) * math.cos(math.radians(latitude))
        )
        azimuth_deg = math.degrees(azimuth)
        
        # Calculate shadow lengths and impacts
        total_shading_factor = 0
        
        for obstacle in obstacles:
            if obstacle.is_permanent:
                # Shadow length calculation
                shadow_length = obstacle.dimensions[2] / math.tan(math.radians(max(elevation_deg, 5)))
                shadow_width = obstacle.dimensions[1]  # Simplified
                
                # Shadow area impact (simplified geometric calculation)
                shadow_area = shadow_length * shadow_width
                total_roof_area = 100  # Assume 100 sqm as reference
                
                shading_contribution = min(shadow_area / total_roof_area, 0.8)
                total_shading_factor += shading_contribution
        
        # Add surrounding building impacts
        for building in surrounding_buildings:
            height = building.get('height_m', 10)
            distance = building.get('distance_m', 20)
            
            if height > 5 and distance < 50:  # Significant impact threshold
                shadow_angle = math.degrees(math.atan(height / distance))
                if shadow_angle > elevation_deg * 0.5:  # Significant shading
                    total_shading_factor += 0.1
        
        return min(total_shading_factor, 0.9)  # Cap at 90% shading
    
    def _design_access_pathways(self, geometry: RooftopGeometry) -> List[Dict[str, Any]]:
        """Design maintenance access pathways"""
        
        pathways = []
        
        # Main access pathway (1.2m wide minimum)
        main_pathway = {
            "type": "main_access",
            "width_m": 1.2,
            "length_m": min(geometry.length_m, geometry.width_m),
            "position": "perimeter",
            "purpose": "Installation and major maintenance"
        }
        pathways.append(main_pathway)
        
        # Inter-row pathways for large installations
        if geometry.usable_area_sqm > 50:
            inter_row_pathway = {
                "type": "inter_row",
                "width_m": 0.8,
                "spacing_m": 6.0,  # Every 6 meters
                "position": "between_arrays",
                "purpose": "Routine maintenance and cleaning"
            }
            pathways.append(inter_row_pathway)
        
        # Emergency access (required for commercial buildings)
        emergency_pathway = {
            "type": "emergency_access",
            "width_m": 1.5,
            "clear_path_to_exit": True,
            "position": "dedicated_zone",
            "purpose": "Emergency evacuation and fire safety"
        }
        pathways.append(emergency_pathway)
        
        return pathways
    
    def _calculate_confidence_level(self,
                                  structural: StructuralAssessment,
                                  shading: ShadingAnalysis,
                                  regulatory: RegulatoryCompliance) -> float:
        """Calculate confidence level in feasibility assessment"""
        
        confidence_factors = []
        
        # Structural confidence
        if structural.requires_structural_analysis:
            confidence_factors.append(0.7)  # Lower confidence until analysis done
        else:
            confidence_factors.append(0.9)
        
        # Shading confidence (based on obstacle data quality)
        if len(shading.obstacles) > 0:
            confidence_factors.append(0.85)  # Good obstacle data
        else:
            confidence_factors.append(0.75)  # Estimated shading
        
        # Regulatory confidence
        if regulatory.building_approval_status == "approved":
            confidence_factors.append(0.95)
        elif regulatory.building_approval_status == "pending":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)
    
    def _estimate_project_timeline(self,
                                 regulatory: RegulatoryCompliance,
                                 installation: InstallationComplexity,
                                 structural: StructuralAssessment) -> Dict[str, int]:
        """Estimate project timeline phases"""
        
        # Approval phase
        approval_days = regulatory.estimated_approval_timeline_days
        if structural.requires_structural_analysis:
            approval_days += 21  # Additional 3 weeks for structural analysis
        
        # Procurement phase
        procurement_days = 14  # Standard 2 weeks
        if installation.labor_skill_requirement == "expert":
            procurement_days += 7  # Additional time for specialized contractors
        
        # Installation phase
        installation_days = installation.estimated_installation_days
        
        # Commissioning and testing
        commissioning_days = max(2, installation_days / 4)
        
        # Weather buffer (monsoon season considerations)
        current_month = datetime.now().month
        if 6 <= current_month <= 9:  # Monsoon season
            weather_buffer = int((approval_days + installation_days) * 0.3)
        else:
            weather_buffer = int((approval_days + installation_days) * 0.1)
        
        total_timeline = approval_days + procurement_days + installation_days + commissioning_days + weather_buffer
        
        return {
            "approval_phase_days": approval_days,
            "procurement_phase_days": procurement_days,
            "installation_phase_days": installation_days,
            "commissioning_days": commissioning_days,
            "weather_buffer_days": weather_buffer,
            "total_project_days": total_timeline,
            "estimated_completion_weeks": math.ceil(total_timeline / 7)
        }
    
    def _generate_recommendations(self,
                                structural: StructuralAssessment,
                                shading: ShadingAnalysis,
                                regulatory: RegulatoryCompliance,
                                installation: InstallationComplexity,
                                risks: List[Dict[str, Any]],
                                layout: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate actionable recommendations and warnings"""
        
        recommendations = []
        warnings = []
        
        # Structural recommendations
        if structural.load_bearing_adequacy < 0.8:
            if structural.reinforcement_required:
                warnings.append("Structural reinforcement mandatory before installation")
                recommendations.append("Engage certified structural engineer for reinforcement design")
            else:
                recommendations.append("Consider lighter mounting systems (aluminum rails, ballasted)")
        
        # Layout optimization recommendations
        if layout.get('area_utilization_percent', 0) < 60:
            recommendations.append("Optimize panel layout to improve roof space utilization")
        
        if layout.get('max_capacity_kw', 0) < 3:
            recommendations.append("Consider high-efficiency panels to maximize limited roof space")
        
        # Shading mitigation recommendations
        if shading.annual_shading_loss_percent > 15:
            if len(shading.mitigation_options) > 0:
                recommendations.append(f"Implement shading mitigation: {', '.join(shading.mitigation_options[:2])}")
            recommendations.append("Use power optimizers or micro-inverters to minimize shading impact")
        
        # Installation recommendations
        if installation.safety_risk_level in ["high", "critical"]:
            warnings.append(f"High safety risk installation - {installation.safety_risk_level} risk level")
            recommendations.append("Engage specialized high-altitude installation contractors")
        
        if installation.weather_sensitivity > 0.6:
            recommendations.append("Schedule installation during October-March for optimal weather conditions")
        
        # Regulatory recommendations
        if len(regulatory.municipal_permissions_needed) > 0:
            recommendations.append("Start regulatory approvals early to avoid project delays")
        
        if regulatory.estimated_approval_timeline_days > 60:
            recommendations.append("Consider regulatory consultant to expedite approval process")
        
        # Economic recommendations
        max_capacity = layout.get('max_capacity_kw', 0)
        if max_capacity < 2:
            warnings.append("Limited roof capacity may result in poor economics")
            recommendations.append("Evaluate alternative locations or building upgrades")
        elif max_capacity > 10:
            recommendations.append("Excellent roof capacity - consider phased installation if budget constrained")
        
        # Technology recommendations
        high_risk_count = len([r for r in risks if r.get('severity') in ['high', 'critical']])
        if high_risk_count > 2:
            recommendations.append("Consider alternative installation approaches or technologies")
        
        return recommendations, warnings
    
    def _print_feasibility_summary(self, result: FeasibilityResult):
        """Print detailed feasibility analysis summary"""
        
        print("\n" + "="*60)
        print("ROOFTOP FEASIBILITY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"Feasibility Score: {result.overall_feasibility_score:.1f}/100")
        print(f"Technically Feasible: {'✓' if result.is_technically_feasible else '✗'}")
        print(f"Economically Viable: {'✓' if result.is_economically_viable else '✗'}")
        print(f"Regulatory Compliant: {'✓' if result.is_regulatory_compliant else '✗'}")
        print(f"Confidence Level: {result.confidence_level:.1%}")
        
        print(f"\nCAPACITY ASSESSMENT:")
        print(f"Maximum Installable Capacity: {result.maximum_installable_capacity_kw:.1f} kW")
        print(f"Maximum Panels: {result.optimal_panel_layout.get('max_panels', 0)}")
        print(f"Roof Utilization: {result.optimal_panel_layout.get('area_utilization_percent', 0):.1f}%")
        
        print(f"\nSTRUCTURAL ANALYSIS:")
        print(f"Load Bearing Adequacy: {result.structural_assessment.load_bearing_adequacy:.1%}")
        print(f"Reinforcement Required: {'Yes' if result.structural_assessment.reinforcement_required else 'No'}")
        print(f"Engineer Approval Needed: {'Yes' if result.structural_assessment.engineer_approval_needed else 'No'}")
        
        print(f"\nSHADING IMPACT:")
        print(f"Annual Shading Loss: {result.shading_analysis.annual_shading_loss_percent:.1f}%")
        print(f"Critical Shading Hours: {len(result.shading_analysis.critical_shading_hours)}")
        
        print(f"\nINSTALLATION COMPLEXITY:")
        print(f"Safety Risk Level: {result.installation_complexity.safety_risk_level.title()}")
        print(f"Estimated Installation Days: {result.installation_complexity.estimated_installation_days}")
        print(f"Skill Requirement: {result.installation_complexity.labor_skill_requirement.title()}")
        
        print(f"\nCOST IMPLICATIONS:")
        print(f"Cost Multiplier: {result.cost_implications.get('base_cost_multiplier', 1.0):.2f}x")
        print(f"Additional Costs: ₹{result.cost_implications.get('additional_fixed_costs', 0):,.0f}")
        
        print(f"\nTIMELINE ESTIMATE:")
        print(f"Total Project Duration: {result.timeline_estimate.get('estimated_completion_weeks', 0)} weeks")
        print(f"Approval Phase: {result.timeline_estimate.get('approval_phase_days', 0)} days")
        
        if result.warnings:
            print(f"\nWARNINGS:")
            for warning in result.warnings[:5]:  # Top 5 warnings
                print(f"  • {warning}")
        
        if result.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in result.recommendations[:5]:  # Top 5 recommendations
                print(f"  • {rec}")
        
        print(f"\nRISK FACTORS:")
        high_risks = [r for r in result.risk_factors if r.get('severity') in ['high', 'critical']]
        for risk in high_risks[:3]:  # Top 3 high risks
            print(f"  • {risk['category'].title()}: {risk['risk']} ({risk['severity']} severity)")

    # Integration methods for pipeline
    def integrate_with_system_sizer(self, 
                                  feasibility_result: FeasibilityResult,
                                  roof_specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate feasibility results with system sizing module"""
        
        # Update roof specifications with feasibility constraints
        updated_roof_specs = roof_specifications.copy()
        
        # Apply feasibility constraints
        updated_roof_specs.update({
            'feasible_area_sqm': feasibility_result.optimal_panel_layout.get('effective_area_sqm', 0),
            'max_installable_capacity_kw': feasibility_result.maximum_installable_capacity_kw,
            'structural_load_limit': feasibility_result.structural_assessment.load_bearing_adequacy,
            'shading_factor': feasibility_result.shading_analysis.annual_shading_loss_percent / 100,
            'installation_complexity_multiplier': feasibility_result.cost_implications.get('base_cost_multiplier', 1.0),
            'additional_costs': feasibility_result.cost_implications.get('additional_fixed_costs', 0),
            'feasibility_score': feasibility_result.overall_feasibility_score,
            'is_feasible': feasibility_result.is_technically_feasible,
            'timeline_weeks': feasibility_result.timeline_estimate.get('estimated_completion_weeks', 8),
            'risk_level': 'high' if len([r for r in feasibility_result.risk_factors 
                                       if r.get('severity') in ['high', 'critical']]) > 1 else 'medium'
        })
        
        return updated_roof_specs
    
    def quick_feasibility_check(self, basic_roof_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quick feasibility check for pipeline pre-screening"""
        
        # Basic geometric check
        area_sqm = basic_roof_data.get('area_sqm', 0)
        roof_type = basic_roof_data.get('type', 'CONCRETE_FLAT')
        building_age = basic_roof_data.get('building_age', 10)
        
        # Quick scoring
        geometric_score = min(area_sqm / 20, 1.0) * 40  # 40 points for geometry
        
        age_score = max(0, 1 - (building_age - 20) / 30) * 30 if building_age > 20 else 30
        
        roof_type_scores = {
            'CONCRETE_FLAT': 30,
            'CONCRETE_SLOPED': 25,
            'METAL_SHEET': 20,
            'TILE_ROOF': 15,
            'ASBESTOS_SHEET': 5
        }
        roof_score = roof_type_scores.get(roof_type, 15)
        
        quick_score = geometric_score + age_score + roof_score
        
        return {
            'quick_feasibility_score': quick_score,
            'recommended_for_detailed_analysis': quick_score >= 60,
            'estimated_max_capacity_kw': min(area_sqm * 0.12, area_sqm / 8),  # Conservative estimate
            'preliminary_viability': 'high' if quick_score >= 75 else 'medium' if quick_score >= 50 else 'low'
        }

# DEMONSTRATION AND TESTING FUNCTIONS

def demonstrate_feasibility_analysis():
    """Demonstrate the rooftop feasibility analyzer"""
    
    print("Advanced Rooftop Feasibility Analyzer")
    print("=" * 50)
    
    analyzer = AdvancedRooftopFeasibilityAnalyzer(debug_mode=True)
    
    # Sample rooftop geometry
    rooftop_geometry = RooftopGeometry(
        total_area_sqm=80,
        usable_area_sqm=65,
        length_m=10,
        width_m=8,
        shape="rectangular",
        roof_orientation=180,  # South facing
        tilt_angle=5,  # Nearly flat
        height_above_ground_m=12,
        perimeter_setback_m=1.5,
        access_points=[{"type": "staircase", "position": "northeast"}],
        structural_zones=[]
    )
    
    # Sample building details
    building_details = {
        'roof_type': 'CONCRETE_FLAT',
        'building_type': 'RESIDENTIAL_INDEPENDENT',
        'age_years': 8,
        'structural_condition': 'good',
        'seismic_zone': 'III',
        'wind_zone': 'III',
        'rooftop_obstacles': [
            {'type': 'WATER_TANK', 'x': 2, 'y': 2, 'dimensions': (2, 1.5, 2), 'permanent': True, 'relocatable': False},
            {'type': 'AC_UNIT', 'x': 8, 'y': 1, 'dimensions': (1, 0.8, 1.2), 'permanent': True, 'relocatable': True}
        ],
        'surrounding_buildings': [
            {'height_m': 15, 'distance_m': 25, 'direction': 'south'},
            {'height_m': 8, 'distance_m': 12, 'direction': 'east'}
        ]
    }
    
    # Sample location details
    location_details = {
        'city': 'chennai',
        'state': 'Tamil Nadu',
        'latitude': 13.0827,
        'longitude': 80.2707,
        'climate_zone': 'tropical'
    }
    
    # Sample system requirements
    system_requirements = {
        'target_capacity_kw': 5.0,
        'panel_preference': 'residential',
        'panel_wattage': 540,
        'mounting_preference': 'penetrating'
    }
    
    # Sample regulatory context
    regulatory_context = {
        'building_approval': 'approved',
        'existing_electrical_connection': True,
        'net_metering_available': True
    }
    
    try:
        # Perform feasibility analysis
        result = analyzer.analyze_rooftop_feasibility(
            rooftop_geometry=rooftop_geometry,
            building_details=building_details,
            location_details=location_details,
            system_requirements=system_requirements,
            regulatory_context=regulatory_context
        )
        
        # Print summary
        analyzer._print_feasibility_summary(result)
        
        print(f"\nINTEGRATION DATA FOR SYSTEM SIZER:")
        integration_data = analyzer.integrate_with_system_sizer(result, {
            'area_sqm': rooftop_geometry.total_area_sqm,
            'type': building_details['roof_type']
        })
        
        print(f"Feasible Area: {integration_data['feasible_area_sqm']:.1f} sqm")
        print(f"Max Capacity: {integration_data['max_installable_capacity_kw']:.1f} kW")
        print(f"Cost Multiplier: {integration_data['installation_complexity_multiplier']:.2f}x")
        print(f"Additional Costs: ₹{integration_data['additional_costs']:,.0f}")
        print(f"Project Timeline: {integration_data['timeline_weeks']} weeks")
        
        print(f"\nFEASIBILITY ANALYSIS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_feasibility_demo():
    """Demonstrate quick feasibility check for pipeline integration"""
    
    print("\nQuick Feasibility Check Demo")
    print("-" * 30)
    
    analyzer = AdvancedRooftopFeasibilityAnalyzer()
    
    test_roofs = [
        {'area_sqm': 45, 'type': 'CONCRETE_FLAT', 'building_age': 5},
        {'area_sqm': 25, 'type': 'TILE_ROOF', 'building_age': 20},
        {'area_sqm': 120, 'type': 'METAL_SHEET', 'building_age': 3},
        {'area_sqm': 15, 'type': 'ASBESTOS_SHEET', 'building_age': 35}
    ]
    
    for i, roof_data in enumerate(test_roofs, 1):
        result = analyzer.quick_feasibility_check(roof_data)
        print(f"\nRoof {i}: {roof_data['area_sqm']}sqm {roof_data['type']}")
        print(f"  Quick Score: {result['quick_feasibility_score']:.1f}/100")
        print(f"  Viability: {result['preliminary_viability'].title()}")
        print(f"  Est. Capacity: {result['estimated_max_capacity_kw']:.1f} kW")
        print(f"  Detailed Analysis: {'Recommended' if result['recommended_for_detailed_analysis'] else 'Optional'}")

# PIPELINE INTEGRATION HELPER FUNCTIONS

def create_feasibility_pipeline_interface():
    """Create standard interface for pipeline integration"""
    
    def feasibility_pipeline_wrapper(
        roof_data: Dict[str, Any],
        building_data: Dict[str, Any],
        location_data: Dict[str, Any],
        system_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pipeline interface wrapper for feasibility analysis
        
        Returns standardized output for integration with system_sizing_and_cost.py
        """
        
        analyzer = AdvancedRooftopFeasibilityAnalyzer()
        
        # Convert input data to required formats
        rooftop_geometry = RooftopGeometry(
            total_area_sqm=roof_data.get('area_sqm', 50),
            usable_area_sqm=roof_data.get('usable_area_sqm', roof_data.get('area_sqm', 50) * 0.8),
            length_m=roof_data.get('length_m', math.sqrt(roof_data.get('area_sqm', 50))),
            width_m=roof_data.get('width_m', math.sqrt(roof_data.get('area_sqm', 50))),
            shape=roof_data.get('shape', 'rectangular'),
            roof_orientation=roof_data.get('orientation', 180),
            tilt_angle=roof_data.get('tilt', 5),
            height_above_ground_m=roof_data.get('height_floors', 2) * 3.5,
            perimeter_setback_m=roof_data.get('setback', 1.0)
        )
        
        # Perform feasibility analysis
        result = analyzer.analyze_rooftop_feasibility(
            rooftop_geometry=rooftop_geometry,
            building_details=building_data,
            location_details=location_data,
            system_requirements=system_requirements,
            regulatory_context={}
        )
        
        # Return pipeline-compatible results
        return analyzer.integrate_with_system_sizer(result, roof_data)
    
    return feasibility_pipeline_wrapper

# EXPORT FOR PIPELINE INTEGRATION
get_rooftop_feasibility = create_feasibility_pipeline_interface()

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_feasibility_analysis()
    quick_feasibility_demo()
    
    print("\n" + "="*60)
    print("PIPELINE INTEGRATION READY")
    print("="*60)
    print("Use: from rooftop_feasibility_analyzer import get_rooftop_feasibility")
    print("Call: feasibility_data = get_rooftop_feasibility(roof_data, building_data, location_data, system_requirements)")
    print("Integration: Pass feasibility_data to system_sizing_and_cost.py as enhanced roof_specifications")