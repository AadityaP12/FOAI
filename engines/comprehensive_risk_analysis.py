# engines/enhanced_comprehensive_risk_analysis.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from ml_models.weather_intelligence import EnhancedWeatherIntelligence


@dataclass
class RiskScenario:
    """Risk scenario for Monte Carlo simulation"""
    name: str
    probability: float
    impact_multiplier: float
    description: str

class EnhancedRiskAnalyzer:
    """Next-generation risk assessment with dynamic intelligence and real-world awareness"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.risk_weights = {
            'financial': 0.25,
            'technical': 0.20,
            'environmental': 0.18,
            'policy': 0.15,
            'ownership_feasibility': 0.12,
            'maintenance_lifecycle': 0.10
        }
        
        # Initialize dynamic data sources
        self.weather_data = self._load_weather_intelligence()
        self.policy_tracker = self._load_policy_tracker()
        self.vendor_reliability = self._load_vendor_reliability_data()
        self.technology_forecast = self._load_technology_forecast()
        self.tariff_predictions = self._load_tariff_predictions()
        
        # Load behavioral risk models
        self.behavioral_patterns = self._load_behavioral_risk_patterns()
        
        # Initialize rare event scenarios
        self.rare_events = self._initialize_rare_event_scenarios()
    
    def _load_weather_intelligence(self) -> Dict:
        """Load dynamic weather and environmental data"""
        try:
            # Try to load real weather data
            with open(f"{self.data_dir}weather_intelligence.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_mock_weather_data()
    
    def _create_mock_weather_data(self) -> Dict:
        """Mock weather intelligence data"""
        return {
            'locations': {
                'Mumbai': {
                    'annual_solar_irradiance': 1650,
                    'dust_storm_frequency': 12,
                    'monsoon_impact_months': 4,
                    'pollution_index': 85,
                    'extreme_weather_risk': 0.7,
                    'seasonal_generation_variance': 0.35
                },
                'Delhi': {
                    'annual_solar_irradiance': 1800,
                    'dust_storm_frequency': 25,
                    'monsoon_impact_months': 3,
                    'pollution_index': 95,
                    'extreme_weather_risk': 0.8,
                    'seasonal_generation_variance': 0.45
                },
                'Bangalore': {
                    'annual_solar_irradiance': 1750,
                    'dust_storm_frequency': 3,
                    'monsoon_impact_months': 4,
                    'pollution_index': 45,
                    'extreme_weather_risk': 0.3,
                    'seasonal_generation_variance': 0.25
                },
                'Chennai': {
                    'annual_solar_irradiance': 1900,
                    'dust_storm_frequency': 8,
                    'monsoon_impact_months': 3,
                    'pollution_index': 55,
                    'extreme_weather_risk': 0.6,
                    'seasonal_generation_variance': 0.30
                },
                'Pune': {
                    'annual_solar_irradiance': 1800,
                    'dust_storm_frequency': 6,
                    'monsoon_impact_months': 4,
                    'pollution_index': 60,
                    'extreme_weather_risk': 0.4,
                    'seasonal_generation_variance': 0.28
                }
            }
        }
    
    def _load_policy_tracker(self) -> Dict:
        """Load dynamic policy and regulatory intelligence"""
        try:
            with open(f"{self.data_dir}policy_tracker.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_mock_policy_data()
    
    def _create_mock_policy_data(self) -> Dict:
        """Mock policy tracking data"""
        return {
            'national_policies': {
                'net_metering_stability': 0.7,
                'subsidy_rollback_probability': 0.3,
                'grid_integration_policy_risk': 0.4,
                'renewable_purchase_obligation_trend': 'stable',
                'last_policy_update': '2024-01-15'
            },
            'state_policies': {
                'Maharashtra': {'stability_score': 0.8, 'solar_friendly_rating': 8.5},
                'Delhi': {'stability_score': 0.6, 'solar_friendly_rating': 7.0},
                'Karnataka': {'stability_score': 0.9, 'solar_friendly_rating': 9.0},
                'Tamil Nadu': {'stability_score': 0.8, 'solar_friendly_rating': 8.8},
                'Gujarat': {'stability_score': 0.9, 'solar_friendly_rating': 9.2}
            },
            'upcoming_changes': [
                {
                    'change_type': 'net_metering_revision',
                    'probability': 0.6,
                    'impact_severity': 0.4,
                    'timeline': '6-12 months'
                },
                {
                    'change_type': 'subsidy_reduction',
                    'probability': 0.4,
                    'impact_severity': 0.6,
                    'timeline': '3-6 months'
                }
            ]
        }
    
    def _load_vendor_reliability_data(self) -> Dict:
        """Load vendor service quality and reliability data"""
        try:
            return pd.read_csv(f"{self.data_dir}vendor_reliability.csv").to_dict('records')
        except FileNotFoundError:
            return self._create_mock_vendor_data()
    
    def _create_mock_vendor_data(self) -> List[Dict]:
        """Mock vendor reliability data"""
        return [
            {
                'company_name': 'Tata Power Solar',
                'after_sales_reliability': 8.5,
                'maintenance_response_time_hours': 24,
                'warranty_claim_success_rate': 0.92,
                'service_network_coverage': 85,
                'maintenance_cost_per_kw_annual': 450
            },
            {
                'company_name': 'Waaree Energies',
                'after_sales_reliability': 8.1,
                'maintenance_response_time_hours': 48,
                'warranty_claim_success_rate': 0.89,
                'service_network_coverage': 78,
                'maintenance_cost_per_kw_annual': 520
            },
            {
                'company_name': 'Adani Solar',
                'after_sales_reliability': 7.8,
                'maintenance_response_time_hours': 36,
                'warranty_claim_success_rate': 0.87,
                'service_network_coverage': 82,
                'maintenance_cost_per_kw_annual': 480
            }
        ]
    
    def _load_technology_forecast(self) -> Dict:
        """Load technology trend and obsolescence forecasts"""
        try:
            with open(f"{self.data_dir}technology_forecast.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_mock_tech_forecast()
    
    def _create_mock_tech_forecast(self) -> Dict:
        """Mock technology forecast data"""
        return {
            'current_tech_lifecycle': {
                'monocrystalline': {'maturity': 0.8, 'obsolescence_risk_5yr': 0.2},
                'bifacial': {'maturity': 0.6, 'obsolescence_risk_5yr': 0.1},
                'perovskite': {'maturity': 0.2, 'obsolescence_risk_5yr': 0.8},
                'inverter_technology': {'maturity': 0.7, 'obsolescence_risk_5yr': 0.4}
            },
            'price_trend_forecast': {
                'next_12_months': -0.08,
                'next_24_months': -0.15,
                'next_36_months': -0.12
            },
            'efficiency_improvements': {
                'panel_efficiency_gain_annual': 0.015,
                'inverter_efficiency_gain_annual': 0.008
            },
            'wait_vs_install_recommendation': {
                'current_score': 0.3,  # 0-1, higher means wait
                'reasoning': 'Moderate price decline expected, but current incentives favorable'
            }
        }
    
    def _load_tariff_predictions(self) -> Dict:
        """Load electricity tariff escalation forecasts"""
        try:
            with open(f"{self.data_dir}tariff_predictions.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_mock_tariff_data()
    
    def _create_mock_tariff_data(self) -> Dict:
        """Mock tariff prediction data"""
        return {
            'state_tariff_trends': {
                'Maharashtra': {'annual_escalation': 0.065, 'volatility': 0.15},
                'Delhi': {'annual_escalation': 0.048, 'volatility': 0.22},
                'Karnataka': {'annual_escalation': 0.072, 'volatility': 0.12},
                'Tamil Nadu': {'annual_escalation': 0.058, 'volatility': 0.18}
            },
            'fuel_cost_impact': {
                'coal_price_volatility': 0.25,
                'renewable_energy_integration': 0.15
            }
        }
    
    def _load_behavioral_risk_patterns(self) -> Dict:
        """Load behavioral hesitation and adoption patterns"""
        return {
            'adoption_hesitation_factors': {
                'technology_anxiety': 0.4,
                'maintenance_concerns': 0.6,
                'payback_period_anxiety': 0.5,
                'vendor_trust_issues': 0.45
            },
            'demographic_risk_modifiers': {
                'age_above_60': {'hesitation_multiplier': 1.3, 'maintenance_concern': 1.5},
                'first_time_solar_buyer': {'hesitation_multiplier': 1.2, 'research_paralysis': 1.4},
                'apartment_dweller': {'feasibility_concern': 2.0, 'approval_complexity': 1.8}
            }
        }
    
    def _initialize_rare_event_scenarios(self) -> List[RiskScenario]:
        """Initialize rare but high-impact event scenarios"""
        return [
            RiskScenario(
                name="Net Metering Policy Rollback",
                probability=0.15,
                impact_multiplier=1.8,
                description="Significant reduction in net metering benefits"
            ),
            RiskScenario(
                name="Major Inverter Recall",
                probability=0.05,
                impact_multiplier=1.4,
                description="Technology-wide inverter replacement needed"
            ),
            RiskScenario(
                name="Grid Integration Issues",
                probability=0.12,
                impact_multiplier=1.3,
                description="Grid stability issues affecting solar integration"
            ),
            RiskScenario(
                name="Extreme Weather Damage",
                probability=0.08,
                impact_multiplier=2.2,
                description="Severe weather causing system damage"
            ),
            RiskScenario(
                name="Vendor Bankruptcy",
                probability=0.10,
                impact_multiplier=1.6,
                description="Installation company going out of business"
            )
        ]
    
    def assess_ownership_feasibility(self, user_profile: Dict) -> Dict:
        """Assess legal and practical ownership feasibility"""
        house_type = user_profile.get('house_type', 'independent').lower()
        ownership_status = user_profile.get('ownership_status', 'owner').lower()
        location = user_profile.get('location', '')
        
        feasibility_score = 1.0
        risk_factors = []
        recommendations = []
        
        # Ownership-based feasibility
        if house_type == 'apartment':
            if ownership_status == 'tenant':
                feasibility_score = 0.95  # Nearly impossible
                risk_factors.append("Tenant in apartment - installation not legally feasible")
                recommendations.extend([
                    "Consider community solar programs",
                    "Explore green power purchasing agreements",
                    "Investigate portable balcony solar systems"
                ])
            elif ownership_status == 'owner':
                feasibility_score = 0.7  # High complexity
                risk_factors.append("Apartment ownership requires society approval")
                risk_factors.append("Shared roof space creates complexity")
                recommendations.extend([
                    "Obtain housing society NOC before proceeding",
                    "Coordinate with other apartment owners for bulk installation",
                    "Ensure compliance with building regulations"
                ])
        
        elif house_type == 'villa' or house_type == 'independent':
            if ownership_status == 'tenant':
                feasibility_score = 0.8
                risk_factors.append("Tenant status requires owner approval")
                recommendations.append("Get written consent from property owner")
            else:
                feasibility_score = 0.1  # Low risk
        
        # Location-specific regulations
        location_complexity = {
            'Mumbai': 0.6, 'Delhi': 0.7, 'Bangalore': 0.3, 'Chennai': 0.4, 'Pune': 0.4
        }
        
        location_risk = location_complexity.get(location, 0.5)
        feasibility_score = max(feasibility_score, location_risk)
        
        if location_risk > 0.5:
            risk_factors.append(f"Complex regulatory environment in {location}")
            recommendations.append("Engage local regulatory consultant")
        
        return {
            'feasibility_score': round(feasibility_score, 3),
            'risk_level': 'Critical' if feasibility_score > 0.8 else 'High' if feasibility_score > 0.5 else 'Moderate',
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'alternative_solutions': self._get_alternative_solutions(feasibility_score, house_type, ownership_status)
        }
    
    def _get_alternative_solutions(self, feasibility_score: float, house_type: str, ownership_status: str) -> List[str]:
        """Get alternative solar solutions for high-risk ownership scenarios"""
        if feasibility_score > 0.8:
            return [
                "Community solar participation",
                "Solar power purchase agreements (PPA)",
                "Green energy tariff plans",
                "Portable solar solutions",
                "Solar water heating systems"
            ]
        elif feasibility_score > 0.5:
            return [
                "Group buying with neighbors",
                "Phased installation approach",
                "Lease-to-own arrangements"
            ]
        return []
    
    def calculate_enhanced_financial_risk(self, user_profile: Dict, cost_estimate: Dict) -> Dict:
        """Enhanced financial risk with real income data and dynamic forecasting"""
        monthly_bill = user_profile['monthly_bill']
        budget_max = user_profile.get('budget_max', 0)
        income_bracket = user_profile.get('income_bracket', 'middle')
        total_cost = cost_estimate['total_cost']
        location = user_profile.get('location', 'Mumbai')
        
        # Use actual income bracket instead of estimation
        income_mapping = {
            'low': monthly_bill * 6,
            'lower_middle': monthly_bill * 8,
            'middle': monthly_bill * 10,
            'upper_middle': monthly_bill * 15,
            'high': monthly_bill * 25
        }
        
        estimated_monthly_income = income_mapping.get(income_bracket, monthly_bill * 10)
        
        # EMI affordability risk (more sophisticated)
        financing_options = ['loan', 'outright', 'ppa']
        loan_interest_rate = 0.115  # Current solar loan rates
        loan_tenure = 7  # years
        
        if total_cost <= budget_max or budget_max == 0:
            financing_mode = 'outright'
            emi_risk = 0.1
        else:
            financing_mode = 'loan'
            monthly_emi = (total_cost * (loan_interest_rate/12) * 
                          ((1 + loan_interest_rate/12)**((loan_tenure*12)))) / \
                         (((1 + loan_interest_rate/12)**((loan_tenure*12))) - 1)
            emi_ratio = monthly_emi / estimated_monthly_income
            emi_risk = min(1.0, emi_ratio / 0.25)  # Risk increases above 25%
        
        # Dynamic tariff escalation risk
        state_mapping = {
            'Mumbai': 'Maharashtra', 'Delhi': 'Delhi', 'Bangalore': 'Karnataka',
            'Chennai': 'Tamil Nadu', 'Pune': 'Maharashtra'
        }
        state = state_mapping.get(location, 'Maharashtra')
        tariff_data = self.tariff_predictions['state_tariff_trends'].get(state, 
                     {'annual_escalation': 0.06, 'volatility': 0.15})
        
        # Technology obsolescence risk with forecasting
        tech_forecast = self.technology_forecast
        wait_recommendation_strength = tech_forecast['wait_vs_install_recommendation']['current_score']
        tech_obsolescence_risk = wait_recommendation_strength * 0.5  # Scale to risk
        
        # Budget stress test
        budget_risk = 0
        if budget_max > 0:
            budget_overshoot = max(0, total_cost - budget_max)
            budget_risk = min(1.0, budget_overshoot / budget_max)
        
        # Payback period uncertainty
        annual_savings = monthly_bill * 12 * 0.8  # 80% bill offset assumption
        base_payback = total_cost / annual_savings
        
        # Monte Carlo enhanced payback risk
        payback_scenarios = []
        for _ in range(100):
            generation_var = np.random.normal(1.0, 0.15)
            cost_var = np.random.normal(1.0, 0.10)
            tariff_growth = np.random.normal(tariff_data['annual_escalation'], 
                                          tariff_data['volatility'])
            
            scenario_payback = (total_cost * cost_var) / (annual_savings * generation_var * 
                              (1 + tariff_growth)**2)
            payback_scenarios.append(scenario_payback)
        
        payback_risk = min(1.0, np.mean(payback_scenarios) / 12)  # Risk above 12 years
        
        # Overall financial risk
        financial_risk = np.mean([emi_risk, budget_risk, payback_risk, tech_obsolescence_risk])
        
        return {
            'overall_score': round(financial_risk, 3),
            'emi_affordability_risk': round(emi_risk, 3),
            'budget_overshoot_risk': round(budget_risk, 3),
            'payback_uncertainty_risk': round(payback_risk, 3),
            'technology_obsolescence_risk': round(tech_obsolescence_risk, 3),
            'financing_recommendation': financing_mode,
            'payback_period_range': f"{min(payback_scenarios):.1f}-{max(payback_scenarios):.1f} years",
            'confidence': 0.88
        }
    
    def calculate_enhanced_technical_risk(self, user_profile: Dict, cost_estimate: Dict, 
                                        vendor_name: str = None) -> Dict:
        """Enhanced technical risk with vendor reliability and maintenance forecasting"""
        location = user_profile['location']
        roof_area = user_profile.get('roof_area', 0)
        house_type = user_profile.get('house_type', 'independent')
        
        # Weather-based technical risks
        weather_data = self.weather_data['locations'].get(location, 
                      self.weather_data['locations']['Mumbai'])
        
        # Installation complexity (weather-adjusted)
        base_complexity = cost_estimate.get('accessibility_complexity', 1.0) - 1.0
        weather_complexity_multiplier = 1 + (weather_data['extreme_weather_risk'] * 0.3)
        complexity_risk = min(1.0, base_complexity * weather_complexity_multiplier / 0.5)
        
        # Roof adequacy with precision
        system_size_kw = cost_estimate.get('system_size_kw', cost_estimate['total_cost'] / 45000)
        required_area = system_size_kw * 100  # 100 sq ft per kW rough estimate
        space_risk = max(0, (required_area - roof_area) / roof_area) if roof_area > 0 else 0.5
        space_risk = min(1.0, space_risk)
        
        # Environmental degradation risks
        pollution_degradation = weather_data['pollution_index'] / 100 * 0.3
        dust_cleaning_frequency = weather_data['dust_storm_frequency']
        maintenance_complexity = min(1.0, dust_cleaning_frequency / 30 * 0.4)
        
        # Vendor reliability integration
        vendor_risk = 0.5  # Default
        if vendor_name:
            vendor_data = next((v for v in self.vendor_reliability 
                              if v['company_name'] == vendor_name), None)
            if vendor_data:
                vendor_reliability_score = vendor_data['after_sales_reliability']
                vendor_risk = max(0, (10 - vendor_reliability_score) / 10)
        
        # System reliability with component lifecycle
        inverter_failure_risk = 0.25 + (0.1 if house_type == 'apartment' else 0)
        panel_degradation_risk = 0.15 + pollution_degradation
        
        # Seasonal generation variance
        generation_volatility_risk = weather_data['seasonal_generation_variance']
        
        technical_risk = np.mean([
            complexity_risk, space_risk, maintenance_complexity,
            vendor_risk, inverter_failure_risk, panel_degradation_risk,
            generation_volatility_risk
        ])
        
        return {
            'overall_score': round(technical_risk, 3),
            'installation_complexity': round(complexity_risk, 3),
            'space_adequacy': round(space_risk, 3),
            'environmental_degradation': round(pollution_degradation + maintenance_complexity, 3),
            'vendor_reliability_risk': round(vendor_risk, 3),
            'system_component_risk': round((inverter_failure_risk + panel_degradation_risk) / 2, 3),
            'generation_volatility': round(generation_volatility_risk, 3),
            'maintenance_recommendations': self._get_maintenance_recommendations(weather_data, vendor_name),
            'confidence': 0.82
        }
    
    def _get_maintenance_recommendations(self, weather_data: Dict, vendor_name: str = None) -> List[str]:
        """Generate maintenance recommendations based on environmental conditions"""
        recommendations = []
        
        if weather_data['dust_storm_frequency'] > 15:
            recommendations.append("Install automated cleaning system for panels")
            recommendations.append("Schedule monthly professional cleaning")
        
        if weather_data['pollution_index'] > 70:
            recommendations.append("Use anti-soiling coating on panels")
            recommendations.append("Implement bi-weekly cleaning schedule")
        
        if weather_data['extreme_weather_risk'] > 0.6:
            recommendations.append("Invest in weather-resistant mounting systems")
            recommendations.append("Obtain comprehensive weather damage insurance")
        
        if vendor_name:
            vendor_data = next((v for v in self.vendor_reliability 
                              if v['company_name'] == vendor_name), None)
            if vendor_data and vendor_data['maintenance_response_time_hours'] > 48:
                recommendations.append("Negotiate faster maintenance response time SLA")
        
        return recommendations
    
    def calculate_maintenance_lifecycle_risk(self, user_profile: Dict, cost_estimate: Dict, 
                                           vendor_name: str = None) -> Dict:
        """Calculate maintenance and service lifecycle risks"""
        system_size_kw = cost_estimate.get('system_size_kw', cost_estimate['total_cost'] / 45000)
        location = user_profile['location']
        
        # Vendor-specific maintenance risk
        vendor_maintenance_risk = 0.5
        annual_maintenance_cost = system_size_kw * 500  # Default estimate
        service_availability_risk = 0.4
        warranty_reliability_risk = 0.3
        
        if vendor_name:
            vendor_data = next((v for v in self.vendor_reliability 
                              if v['company_name'] == vendor_name), None)
            if vendor_data:
                vendor_maintenance_risk = (10 - vendor_data['after_sales_reliability']) / 10
                annual_maintenance_cost = system_size_kw * vendor_data['maintenance_cost_per_kw_annual']
                service_availability_risk = max(0, (vendor_data['maintenance_response_time_hours'] - 24) / 120)
                warranty_reliability_risk = 1 - vendor_data['warranty_claim_success_rate']
        
        # Component lifecycle risks
        inverter_replacement_risk = 0.8  # High probability in 8-10 years
        panel_degradation_risk = 0.25   # Natural degradation over 20 years
        
        # Location-specific maintenance challenges
        weather_data = self.weather_data['locations'].get(location, 
                      self.weather_data['locations']['Mumbai'])
        environmental_maintenance_multiplier = 1 + (weather_data['pollution_index'] / 200)
        
        # Calculate 20-year maintenance cost projection
        total_20yr_maintenance_cost = 0
        for year in range(1, 21):
            annual_cost = annual_maintenance_cost * environmental_maintenance_multiplier
            
            # Add major component replacement costs
            if year == 8:  # Inverter replacement
                annual_cost += system_size_kw * 8000 * inverter_replacement_risk
            if year in [10, 15]:  # Major servicing
                annual_cost *= 1.5
            
            total_20yr_maintenance_cost += annual_cost * (1.05 ** year)  # 5% inflation
        
        # Risk assessment
        maintenance_cost_risk = min(1.0, total_20yr_maintenance_cost / cost_estimate['total_cost'])
        service_disruption_risk = max(vendor_maintenance_risk, service_availability_risk)
        component_failure_risk = np.mean([inverter_replacement_risk, panel_degradation_risk])
        
        overall_maintenance_risk = np.mean([
            maintenance_cost_risk, service_disruption_risk, 
            component_failure_risk, warranty_reliability_risk
        ])
        
        return {
            'overall_score': round(overall_maintenance_risk, 3),
            'maintenance_cost_risk': round(maintenance_cost_risk, 3),
            'service_availability_risk': round(service_availability_risk, 3),
            'component_failure_risk': round(component_failure_risk, 3),
            'warranty_reliability_risk': round(warranty_reliability_risk, 3),
            'projected_20yr_maintenance_cost': round(total_20yr_maintenance_cost, 2),
            'annual_maintenance_budget': round(annual_maintenance_cost * environmental_maintenance_multiplier, 2),
            'critical_maintenance_years': [8, 10, 15, 18],
            'confidence': 0.75
        }
    
    def calculate_dynamic_policy_risk(self, user_profile: Dict) -> Dict:
        """Enhanced policy risk with real-time policy tracking"""
        location = user_profile['location']
        
        # State-specific policy stability
        state_mapping = {
            'Mumbai': 'Maharashtra', 'Delhi': 'Delhi', 'Bangalore': 'Karnataka',
            'Chennai': 'Tamil Nadu', 'Pune': 'Maharashtra'
        }
        state = state_mapping.get(location, 'Maharashtra')
        
        state_policy_data = self.policy_tracker['state_policies'].get(state,
                           {'stability_score': 0.7, 'solar_friendly_rating': 7.0})
        
        state_stability_risk = 1 - state_policy_data['stability_score']
        
        # National policy risks
        national_data = self.policy_tracker['national_policies']
        net_metering_risk = 1 - national_data['net_metering_stability']
        subsidy_risk = national_data['subsidy_rollback_probability']
        grid_policy_risk = national_data['grid_integration_policy_risk']
        
        # Upcoming policy changes
        upcoming_risk = 0
        for change in self.policy_tracker['upcoming_changes']:
            upcoming_risk += change['probability'] * change['impact_severity']
        upcoming_risk = min(1.0, upcoming_risk)
        
        # Timeline urgency (sooner changes = higher immediate risk)
        urgency_multiplier = 1.0
        for change in self.policy_tracker['upcoming_changes']:
            if '3-6 months' in change['timeline']:
                urgency_multiplier = 1.3
            elif '6-12 months' in change['timeline']:
                urgency_multiplier = 1.1
        
        policy_risk = np.mean([
            state_stability_risk, net_metering_risk, subsidy_risk,
            grid_policy_risk, upcoming_risk
        ]) * urgency_multiplier
        
        policy_risk = min(1.0, policy_risk)
        
        return {
            'overall_score': round(policy_risk, 3),
            'state_policy_stability': round(state_stability_risk, 3),
            'net_metering_risk': round(net_metering_risk, 3),
            'subsidy_availability_risk': round(subsidy_risk, 3),
            'grid_integration_risk': round(grid_policy_risk, 3),
            'upcoming_changes_risk': round(upcoming_risk, 3),
            'urgency_multiplier': round(urgency_multiplier, 2),
            'policy_recommendations': self._get_policy_recommendations(policy_risk, upcoming_risk),
            'confidence': 0.72
        }
    
    def _get_policy_recommendations(self, policy_risk: float, upcoming_risk: float) -> List[str]:
        """Generate policy-specific recommendations"""
        recommendations = []
        
        if policy_risk > 0.6:
            recommendations.append("Consider accelerated installation timeline before policy changes")
            recommendations.append("Negotiate fixed-price contracts with policy change protection")
            
        if upcoming_risk > 0.4:
            recommendations.append("Complete installation within next 6 months")
            recommendations.append("Secure current net metering rates with utility agreement")
            
        if policy_risk > 0.5:
            recommendations.append("Consider battery storage to reduce grid dependency")
            recommendations.append("Explore captive consumption optimization")
            
        return recommendations
    
    def enhanced_monte_carlo_simulation(self, user_profile: Dict, cost_estimate: Dict,
                                      vendor_name: str = None, n_simulations: int = 2000) -> Dict:
        """Enhanced Monte Carlo with rare events and realistic distributions"""
        
        payback_periods = []
        total_savings = []
        npv_values = []
        
        base_cost = cost_estimate['total_cost']
        monthly_bill = user_profile['monthly_bill']
        location = user_profile.get('location', 'Mumbai')
        
        # Get dynamic tariff data
        state_mapping = {'Mumbai': 'Maharashtra', 'Delhi': 'Delhi', 'Bangalore': 'Karnataka',
                        'Chennai': 'Tamil Nadu', 'Pune': 'Maharashtra'}
        state = state_mapping.get(location, 'Maharashtra')
        tariff_data = self.tariff_predictions['state_tariff_trends'].get(state,
                     {'annual_escalation': 0.06, 'volatility': 0.15})
        
        # Get weather-based generation data
        weather_data = self.weather_data['locations'].get(location,
                      self.weather_data['locations']['Mumbai'])
        
        discount_rate = 0.08  # Real discount rate
        
        for i in range(n_simulations):
            # Cost variations (beta distribution for more realistic skew)
            cost_variation = np.random.beta(2, 2) * 0.3 + 0.85  # 85-115% range, skewed
            actual_cost = base_cost * cost_variation
            
            # Generation variations (weather-adjusted)
            base_generation_var = 1.0
            
            # Apply seasonal variance
            seasonal_impact = np.random.normal(0, weather_data['seasonal_generation_variance'])
            generation_variation = base_generation_var + seasonal_impact
            generation_variation = max(0.6, min(1.4, generation_variation))  # Bounded
            
            # Tariff escalation with volatility
            annual_tariff_growth = np.random.normal(
                tariff_data['annual_escalation'],
                tariff_data['volatility']
            )
            annual_tariff_growth = max(0.02, min(0.12, annual_tariff_growth))  # Realistic bounds
            
            # Apply rare events (10% of simulations)
            rare_event_impact = 1.0
            if np.random.random() < 0.10:
                selected_event = np.random.choice(self.rare_events)
                if np.random.random() < selected_event.probability:
                    rare_event_impact = selected_event.impact_multiplier
            
            # Calculate year-by-year cash flows
            annual_savings = monthly_bill * 12 * 0.8 * generation_variation
            total_savings_npv = 0
            total_savings_nominal = 0
            payback_achieved = False
            payback_year = 20
            
            for year in range(1, 21):
                # Apply tariff growth
                year_savings = annual_savings * ((1 + annual_tariff_growth) ** year)
                
                # Apply degradation (0.5% annual for panels)
                degradation_factor = (0.995 ** year)
                year_savings *= degradation_factor
                
                # Apply rare event impact (usually in years 5-15)
                if year >= 5 and year <= 15:
                    year_savings /= rare_event_impact
                
                # Maintenance costs (from vendor data)
                maintenance_cost = 0
                if vendor_name:
                    vendor_data = next((v for v in self.vendor_reliability 
                                      if v['company_name'] == vendor_name), None)
                    if vendor_data:
                        system_size_kw = cost_estimate.get('system_size_kw', base_cost / 45000)
                        base_maintenance = system_size_kw * vendor_data['maintenance_cost_per_kw_annual']
                        
                        # Major maintenance years
                        maintenance_multiplier = 1.0
                        if year == 8:  # Inverter replacement
                            maintenance_multiplier = 3.0
                        elif year in [10, 15]:
                            maintenance_multiplier = 1.5
                        
                        maintenance_cost = base_maintenance * maintenance_multiplier * ((1.05) ** year)
                
                net_year_savings = year_savings - maintenance_cost
                total_savings_nominal += net_year_savings
                
                # NPV calculation
                discounted_savings = net_year_savings / ((1 + discount_rate) ** year)
                total_savings_npv += discounted_savings
                
                # Check payback
                if not payback_achieved and total_savings_nominal >= actual_cost:
                    payback_year = year - 1 + (actual_cost - (total_savings_nominal - net_year_savings)) / net_year_savings
                    payback_achieved = True
            
            payback_periods.append(payback_year)
            total_savings.append(total_savings_nominal - actual_cost)
            npv_values.append(total_savings_npv - actual_cost)
        
        # Calculate risk metrics
        probability_positive_npv = np.mean([npv > 0 for npv in npv_values])
        probability_payback_under_10 = np.mean([pb <= 10 for pb in payback_periods])
        
        # Value at Risk (VaR) - 5th percentile loss
        var_5_percent = np.percentile(total_savings, 5)
        
        return {
            'payback_period': {
                'mean': round(np.mean(payback_periods), 2),
                'median': round(np.median(payback_periods), 2),
                'std': round(np.std(payback_periods), 2),
                'percentile_5': round(np.percentile(payback_periods, 5), 2),
                'percentile_10': round(np.percentile(payback_periods, 10), 2),
                'percentile_90': round(np.percentile(payback_periods, 90), 2),
                'percentile_95': round(np.percentile(payback_periods, 95), 2)
            },
            'net_savings_20yr': {
                'mean': round(np.mean(total_savings), 2),
                'median': round(np.median(total_savings), 2),
                'std': round(np.std(total_savings), 2),
                'percentile_5': round(np.percentile(total_savings, 5), 2),
                'percentile_10': round(np.percentile(total_savings, 10), 2),
                'percentile_90': round(np.percentile(total_savings, 90), 2),
                'percentile_95': round(np.percentile(total_savings, 95), 2)
            },
            'npv_analysis': {
                'mean_npv': round(np.mean(npv_values), 2),
                'median_npv': round(np.median(npv_values), 2),
                'std_npv': round(np.std(npv_values), 2),
                'probability_positive_npv': round(probability_positive_npv, 3)
            },
            'risk_metrics': {
                'success_probability': round(probability_positive_npv, 3),
                'payback_under_10yr_probability': round(probability_payback_under_10, 3),
                'value_at_risk_5_percent': round(var_5_percent, 2),
                'downside_risk': round(np.mean([max(0, -s) for s in total_savings]), 2)
            },
            'simulation_parameters': {
                'simulations_run': n_simulations,
                'rare_events_included': len(self.rare_events),
                'weather_adjusted': True,
                'maintenance_costs_included': vendor_name is not None
            }
        }
    
    def assess_behavioral_adoption_risk(self, user_profile: Dict) -> Dict:
        """Assess behavioral and psychological barriers to adoption"""
        
        age = user_profile.get('age', 40)
        house_type = user_profile.get('house_type', 'independent')
        first_time_buyer = user_profile.get('first_time_solar_buyer', True)
        tech_comfort = user_profile.get('tech_comfort_level', 'medium')
        
        behavioral_risk_score = 0.0
        risk_factors = []
        mitigation_strategies = []
        
        # Age-related hesitation
        if age >= 60:
            age_risk = 0.3
            risk_factors.append("Older demographic may have technology adoption hesitation")
            mitigation_strategies.append("Provide detailed maintenance support plan")
            mitigation_strategies.append("Offer family member involvement in decision process")
        elif age <= 30:
            age_risk = 0.1
        else:
            age_risk = 0.2
        
        # First-time buyer analysis paralysis
        if first_time_buyer:
            first_time_risk = 0.4
            risk_factors.append("First-time solar buyer may experience decision paralysis")
            mitigation_strategies.append("Provide comprehensive education materials")
            mitigation_strategies.append("Offer site visit and consultation")
        else:
            first_time_risk = 0.1
        
        # Technology comfort level
        tech_comfort_risk = {
            'low': 0.5, 'medium': 0.3, 'high': 0.1
        }.get(tech_comfort, 0.3)
        
        if tech_comfort == 'low':
            risk_factors.append("Low technology comfort may create adoption barriers")
            mitigation_strategies.append("Simplify monitoring systems and interfaces")
            mitigation_strategies.append("Provide extensive training and support")
        
        # Housing type complexity
        if house_type == 'apartment':
            housing_risk = 0.6
            risk_factors.append("Apartment installations create social coordination challenges")
            mitigation_strategies.append("Facilitate neighbor group meetings")
            mitigation_strategies.append("Provide society approval assistance")
        else:
            housing_risk = 0.2
        
        # Financial commitment anxiety
        budget_max = user_profile.get('budget_max', 0)
        monthly_bill = user_profile.get('monthly_bill', 0)
        
        if budget_max > 0 and monthly_bill > 0:
            financial_stretch = budget_max / (monthly_bill * 12)  # Budget as multiple of annual bill
            if financial_stretch > 8:  # Very high investment relative to bill
                financial_anxiety = 0.5
                risk_factors.append("High investment relative to current bills may create anxiety")
                mitigation_strategies.append("Emphasize long-term financial benefits")
                mitigation_strategies.append("Provide flexible payment options")
            else:
                financial_anxiety = 0.2
        else:
            financial_anxiety = 0.3
        
        # Overall behavioral risk
        behavioral_risk_score = np.mean([
            age_risk, first_time_risk, tech_comfort_risk, 
            housing_risk, financial_anxiety
        ])
        
        # Confidence factors (positive behavioral indicators)
        confidence_factors = []
        if user_profile.get('environmental_motivation', False):
            confidence_factors.append("Strong environmental motivation")
            behavioral_risk_score *= 0.8
        
        if user_profile.get('energy_independence_priority', False):
            confidence_factors.append("Values energy independence")
            behavioral_risk_score *= 0.9
        
        return {
            'overall_score': round(behavioral_risk_score, 3),
            'risk_factors': risk_factors,
            'mitigation_strategies': mitigation_strategies,
            'confidence_factors': confidence_factors,
            'adoption_probability': round(1 - behavioral_risk_score, 3),
            'recommended_approach': self._get_behavioral_approach_recommendation(behavioral_risk_score),
            'confidence': 0.70
        }
    
    def _get_behavioral_approach_recommendation(self, behavioral_risk: float) -> str:
        """Recommend approach based on behavioral risk assessment"""
        if behavioral_risk > 0.6:
            return "High-touch consultative approach with extensive education and support"
        elif behavioral_risk > 0.4:
            return "Structured decision support with clear milestone approach"
        elif behavioral_risk > 0.2:
            return "Standard information provision with responsive support"
        else:
            return "Streamlined process with minimal intervention needed"
    
    def generate_comprehensive_risk_assessment(self, user_profile: Dict, cost_estimate: Dict,
                                            vendor_name: str = None) -> Dict:
        """Generate the complete enhanced risk assessment"""
        
        # Calculate all risk dimensions
        ownership_risk = self.assess_ownership_feasibility(user_profile)
        financial_risk = self.calculate_enhanced_financial_risk(user_profile, cost_estimate)
        technical_risk = self.calculate_enhanced_technical_risk(user_profile, cost_estimate, vendor_name)
        environmental_risk = self.calculate_enhanced_environmental_risk(user_profile)
        policy_risk = self.calculate_dynamic_policy_risk(user_profile)
        maintenance_risk = self.calculate_maintenance_lifecycle_risk(user_profile, cost_estimate, vendor_name)
        behavioral_risk = self.assess_behavioral_adoption_risk(user_profile)
        
        # Update weights based on ownership feasibility
        adjusted_weights = self.risk_weights.copy()
        if ownership_risk['feasibility_score'] > 0.8:
            adjusted_weights['ownership_feasibility'] = 0.4  # Dominate other factors
            remaining_weight = 0.6
            other_weights = {k: v for k, v in adjusted_weights.items() if k != 'ownership_feasibility'}
            total_other = sum(other_weights.values())
            for key in other_weights:
                adjusted_weights[key] = other_weights[key] / total_other * remaining_weight
        
        # Calculate weighted overall risk
        overall_risk = (
            adjusted_weights['financial'] * financial_risk['overall_score'] +
            adjusted_weights['technical'] * technical_risk['overall_score'] +
            adjusted_weights['environmental'] * environmental_risk['overall_score'] +
            adjusted_weights['policy'] * policy_risk['overall_score'] +
            adjusted_weights['ownership_feasibility'] * ownership_risk['feasibility_score'] +
            adjusted_weights['maintenance_lifecycle'] * maintenance_risk['overall_score']
        )
        
        # Add behavioral risk as a multiplier rather than additive
        behavioral_multiplier = 1 + (behavioral_risk['overall_score'] * 0.3)
        overall_risk *= behavioral_multiplier
        overall_risk = min(1.0, overall_risk)
        
        # Enhanced Monte Carlo simulation
        monte_carlo_results = self.enhanced_monte_carlo_simulation(
            user_profile, cost_estimate, vendor_name
        )
        
        # Generate comprehensive recommendations
        all_recommendations = self._compile_comprehensive_recommendations(
            ownership_risk, financial_risk, technical_risk, environmental_risk,
            policy_risk, maintenance_risk, behavioral_risk, overall_risk
        )
        
        # Risk categorization with more nuanced levels
        risk_category = self._categorize_enhanced_risk_level(overall_risk, ownership_risk['feasibility_score'])
        
        # Investment decision recommendation
        investment_recommendation = self._generate_investment_recommendation(
            overall_risk, monte_carlo_results, ownership_risk['feasibility_score']
        )
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_category,
            'investment_recommendation': investment_recommendation,
            'confidence_level': round(np.mean([
                financial_risk['confidence'], technical_risk['confidence'],
                environmental_risk['confidence'], policy_risk['confidence'],
                maintenance_risk['confidence']
            ]), 3),
            'risk_breakdown': {
                'ownership_feasibility': ownership_risk,
                'financial': financial_risk,
                'technical': technical_risk,
                'environmental': environmental_risk,
                'policy': policy_risk,
                'maintenance_lifecycle': maintenance_risk,
                'behavioral_adoption': behavioral_risk
            },
            'monte_carlo_simulation': monte_carlo_results,
            'comprehensive_recommendations': all_recommendations,
            'risk_weights_used': adjusted_weights,
            'behavioral_risk_multiplier': round(behavioral_multiplier, 3),
            'key_risk_drivers': self._identify_key_risk_drivers(
                financial_risk, technical_risk, environmental_risk, 
                policy_risk, ownership_risk, maintenance_risk
            ),
            'risk_mitigation_priority': self._prioritize_risk_mitigation(
                financial_risk, technical_risk, environmental_risk,
                policy_risk, ownership_risk, maintenance_risk
            )
        }
    
    def calculate_enhanced_environmental_risk(self, user_profile: Dict) -> Dict:
        """Enhanced environmental risk with real weather data integration"""
        location = user_profile['location']
        
        # Get dynamic weather data
        weather_data = self.weather_data['locations'].get(location,
                      self.weather_data['locations']['Mumbai'])
        
        # Climate-based generation risk
        irradiance_risk = max(0, (1800 - weather_data['annual_solar_irradiance']) / 600)
        irradiance_risk = min(1.0, irradiance_risk)
        
        # Extreme weather impact
        extreme_weather_risk = weather_data['extreme_weather_risk']
        
        # Pollution and dust impact
        pollution_risk = weather_data['pollution_index'] / 100
        dust_impact_risk = min(1.0, weather_data['dust_storm_frequency'] / 30)
        
        # Seasonal generation variance
        seasonal_risk = weather_data['seasonal_generation_variance']
        
        # Monsoon impact
        monsoon_risk = weather_data['monsoon_impact_months'] / 12 * 0.6
        
        # Climate change progression risk
        climate_change_risk = 0.35  # Increasing over time
        
        environmental_risk = np.mean([
            irradiance_risk, extreme_weather_risk, pollution_risk,
            dust_impact_risk, seasonal_risk, monsoon_risk, climate_change_risk
        ])
        
        return {
            'overall_score': round(environmental_risk, 3),
            'solar_irradiance_adequacy': round(1 - irradiance_risk, 3),
            'extreme_weather_risk': round(extreme_weather_risk, 3),
            'pollution_impact': round(pollution_risk, 3),
            'dust_accumulation_risk': round(dust_impact_risk, 3),
            'seasonal_generation_variance': round(seasonal_risk, 3),
            'monsoon_impact': round(monsoon_risk, 3),
            'climate_change_progression': round(climate_change_risk, 3),
            'environmental_mitigation_cost': self._estimate_environmental_mitigation_cost(weather_data),
            'confidence': 0.78
        }
    
    def _estimate_environmental_mitigation_cost(self, weather_data: Dict) -> Dict:
        """Estimate costs for environmental risk mitigation"""
        base_cost_per_kw = 2000  # Base annual environmental mitigation cost
        
        # Cleaning system costs
        if weather_data['dust_storm_frequency'] > 15:
            cleaning_cost_multiplier = 2.0
        elif weather_data['dust_storm_frequency'] > 8:
            cleaning_cost_multiplier = 1.5
        else:
            cleaning_cost_multiplier = 1.0
        
        # Weather protection costs
        if weather_data['extreme_weather_risk'] > 0.6:
            protection_cost_multiplier = 1.8
        elif weather_data['extreme_weather_risk'] > 0.4:
            protection_cost_multiplier = 1.3
        else:
            protection_cost_multiplier = 1.0
        
        total_multiplier = cleaning_cost_multiplier * protection_cost_multiplier
        annual_mitigation_cost = base_cost_per_kw * total_multiplier
        
        return {
            'annual_cost_per_kw': round(annual_mitigation_cost, 2),
            'cleaning_system_required': weather_data['dust_storm_frequency'] > 15,
            'weather_protection_required': weather_data['extreme_weather_risk'] > 0.6,
            'total_multiplier': round(total_multiplier, 2)
        }
    
    def _compile_comprehensive_recommendations(self, ownership_risk, financial_risk, 
                                             technical_risk, environmental_risk,
                                             policy_risk, maintenance_risk, 
                                             behavioral_risk, overall_risk) -> Dict:
        """Compile all recommendations into prioritized action plan"""
        
        immediate_actions = []
        short_term_actions = []
        long_term_considerations = []
        alternative_solutions = []
        
        # Ownership feasibility actions
        if ownership_risk['feasibility_score'] > 0.8:
            immediate_actions.extend(ownership_risk['recommendations'][:2])
            alternative_solutions.extend(ownership_risk.get('alternative_solutions', []))
        
        # Financial risk actions
        if financial_risk['overall_score'] > 0.5:
            if financial_risk['financing_recommendation'] == 'loan':
                immediate_actions.append("Explore solar financing options with multiple lenders")
            short_term_actions.append("Create detailed budget plan including maintenance costs")
        
        # Policy risk actions
        if policy_risk['overall_score'] > 0.5:
            immediate_actions.extend(policy_risk.get('policy_recommendations', [])[:2])
        
        # Technical risk actions
        if technical_risk['overall_score'] > 0.5:
            short_term_actions.append("Conduct professional site assessment")
            short_term_actions.extend(technical_risk.get('maintenance_recommendations', [])[:2])
        
        # Behavioral risk actions
        if behavioral_risk['overall_score'] > 0.4:
            immediate_actions.append(f"Adopt {behavioral_risk['recommended_approach']}")
            short_term_actions.extend(behavioral_risk['mitigation_strategies'][:2])
        
        # Maintenance risk actions
        if maintenance_risk['overall_score'] > 0.5:
            long_term_considerations.append("Budget for inverter replacement in year 8-10")
            long_term_considerations.append(f"Plan annual maintenance budget of ₹{maintenance_risk['annual_maintenance_budget']}")
        
        return {
            'immediate_actions': immediate_actions[:5],  # Top 5 most urgent
            'short_term_actions': short_term_actions[:5],
            'long_term_considerations': long_term_considerations[:5],
            'alternative_solutions': alternative_solutions,
            'overall_recommendation': self._get_overall_recommendation(overall_risk),
            'risk_acceptance_threshold': 0.6
        }
    
    def _get_overall_recommendation(self, overall_risk: float) -> str:
        """Get overall recommendation based on risk level"""
        if overall_risk > 0.8:
            return "HIGH RISK: Consider alternative solutions or significant risk mitigation before proceeding"
        elif overall_risk > 0.6:
            return "MODERATE-HIGH RISK: Proceed with comprehensive risk mitigation strategies"
        elif overall_risk > 0.4:
            return "MODERATE RISK: Proceed with standard risk management practices"
        elif overall_risk > 0.2:
            return "LOW-MODERATE RISK: Good candidate for solar installation"
        else:
            return "LOW RISK: Excellent candidate for solar installation"
    
    def _categorize_enhanced_risk_level(self, risk_score: float, ownership_feasibility: float) -> str:
        """Enhanced risk categorization considering ownership feasibility"""
        if ownership_feasibility > 0.9:
            return "Critical Risk - Installation Not Feasible"
        elif risk_score > 0.75:
            return "Very High Risk"
        elif risk_score > 0.6:
            return "High Risk"
        elif risk_score > 0.4:
            return "Moderate Risk"
        elif risk_score > 0.2:
            return "Low Risk"
        else:
            return "Very Low Risk"
    
    def _generate_investment_recommendation(self, overall_risk: float, 
                                          monte_carlo_results: Dict,
                                          ownership_feasibility: float) -> Dict:
        """Generate investment recommendation with confidence intervals"""
        
        if ownership_feasibility > 0.9:
            return {
                'recommendation': 'DO NOT PROCEED',
                'reasoning': 'Installation not legally/practically feasible',
                'confidence': 0.95,
                'alternative_action': 'Explore alternative solar solutions'
            }
        
        success_probability = monte_carlo_results['risk_metrics']['success_probability']
        payback_probability = monte_carlo_results['risk_metrics']['payback_under_10yr_probability']
        
        if overall_risk <= 0.3 and success_probability >= 0.8 and payback_probability >= 0.7:
            return {
                'recommendation': 'STRONGLY RECOMMEND',
                'reasoning': 'Low risk with high probability of positive returns',
                'confidence': 0.9,
                'expected_outcome': 'Highly likely to meet financial expectations'
            }
        elif overall_risk <= 0.5 and success_probability >= 0.6:
            return {
                'recommendation': 'RECOMMEND WITH CONDITIONS',
                'reasoning': 'Good investment potential with risk mitigation',
                'confidence': 0.75,
                'conditions': 'Implement recommended risk mitigation strategies'
            }
        elif overall_risk <= 0.7:
            return {
                'recommendation': 'PROCEED WITH CAUTION',
                'reasoning': 'Moderate risk requires careful consideration',
                'confidence': 0.6,
                'suggested_action': 'Consider phased approach or wait for better conditions'
            }
        else:
            return {
                'recommendation': 'NOT RECOMMENDED',
                'reasoning': 'High risk with uncertain returns',
                'confidence': 0.8,
                'alternative_action': 'Wait for better market conditions or consider alternatives'
            }
    
    def _identify_key_risk_drivers(self, financial_risk, technical_risk, 
                                  environmental_risk, policy_risk,
                                  ownership_risk, maintenance_risk) -> List[str]:
        """Identify the top risk drivers for focused attention"""
        risk_drivers = []
        
        all_risks = [
            ('Ownership Feasibility', ownership_risk['feasibility_score']),
            ('Financial', financial_risk['overall_score']),
            ('Technical', technical_risk['overall_score']),
            ('Environmental', environmental_risk['overall_score']),
            ('Policy', policy_risk['overall_score']),
            ('Maintenance', maintenance_risk['overall_score'])
        ]
        
        # Sort by risk score (descending)
        sorted_risks = sorted(all_risks, key=lambda x: x[1], reverse=True)
        
        # Get top 3 risk drivers above threshold
        for risk_name, risk_score in sorted_risks[:3]:
            if risk_score > 0.4:  # Only include significant risks
                risk_drivers.append(f"{risk_name}: {risk_score:.2f}")
        
        return risk_drivers
    
    def _prioritize_risk_mitigation(self, financial_risk, technical_risk,
                                   environmental_risk, policy_risk,
                                   ownership_risk, maintenance_risk) -> List[Dict]:
        """Prioritize risk mitigation actions by impact and feasibility"""
        
        mitigation_priorities = []
        
        # High-impact, feasible mitigations
        if ownership_risk['feasibility_score'] > 0.6:
            mitigation_priorities.append({
                'priority': 1,
                'risk_area': 'Ownership Feasibility',
                'action': 'Resolve ownership/approval issues',
                'impact': 'Critical',
                'effort': 'High',
                'timeline': 'Immediate'
            })
        
        if policy_risk['overall_score'] > 0.5:
            mitigation_priorities.append({
                'priority': 2,
                'risk_area': 'Policy Risk',
                'action': 'Accelerate installation timeline',
                'impact': 'High',
                'effort': 'Medium',
                'timeline': 'Short-term'
            })
        
        if financial_risk['overall_score'] > 0.5:
            mitigation_priorities.append({
                'priority': 3,
                'risk_area': 'Financial Risk',
                'action': 'Optimize financing and budget planning',
                'impact': 'High',
                'effort': 'Medium',
                'timeline': 'Short-term'
            })
        
        if technical_risk['overall_score'] > 0.5:
            mitigation_priorities.append({
                'priority': 4,
                'risk_area': 'Technical Risk',
                'action': 'Professional site assessment and vendor selection',
                'impact': 'Medium',
                'effort': 'Low',
                'timeline': 'Short-term'
            })
        
        return mitigation_priorities
    # Add this method to your EnhancedRiskAnalyzer class in comprehensive_risk_analysis.py

    def analyze_from_pipeline(self, payload: Dict) -> Dict:
        """
        Integration method for the pipeline - converts pipeline data format to risk assessment
        This is the method that integration_manager.py will call
        """
        try:
            # Extract data from pipeline payload
            user_request = payload.get('user_request', {})
            sizing = payload.get('sizing', {})
            roi = payload.get('roi', {})
            tech = payload.get('tech', {})
            weather = payload.get('weather', {})
            
            # Convert pipeline data format to the format expected by risk analyzer
            user_profile = self._convert_pipeline_user_profile(user_request)
            cost_estimate = self._convert_pipeline_cost_estimate(sizing)
            vendor_name = self._extract_vendor_name(sizing)
            
            # Run the comprehensive risk assessment
            risk_assessment = self.generate_comprehensive_risk_assessment(
                user_profile, cost_estimate, vendor_name
            )
            
            # Convert result to pipeline-compatible format
            pipeline_result = self._convert_to_pipeline_format(risk_assessment)
            
            return pipeline_result
            
        except Exception as e:
            # Return error with fallback data
            return {
                "overall_risk": "Moderate",
                "risk_components": {
                    "financial_risk": 0.4,
                    "technical_risk": 0.3,
                    "policy_risk": 0.2,
                    "environmental_risk": 0.25,
                    "ownership_risk": 0.15
                },
                "error": f"Risk analysis failed: {str(e)}",
                "confidence": 0.5,
                "recommendation": "Proceed with standard caution due to analysis limitations"
            }

    def _convert_pipeline_user_profile(self, user_request: Dict) -> Dict:
        """Convert pipeline user request format to risk analyzer user profile format"""
        return {
            'monthly_bill': user_request.get('monthly_bill', 2500),
            'budget_max': user_request.get('budget_inr', 400000),
            'location': user_request.get('location', 'Mumbai').split(',')[0].strip(),
            'house_type': user_request.get('house_type', 'independent'),
            'ownership_status': 'owner',  # Default assumption
            'roof_area': user_request.get('roof_area_m2', 100),
            'income_bracket': user_request.get('income_bracket', 'middle').lower(),
            'age': 40,  # Default
            'first_time_solar_buyer': True,  # Default assumption
            'tech_comfort_level': 'medium',
            'environmental_motivation': 'sustainability' in user_request.get('goals', []),
            'energy_independence_priority': 'independence' in user_request.get('goals', [])
        }

    def _convert_pipeline_cost_estimate(self, sizing: Dict) -> Dict:
        """Convert pipeline sizing data to cost estimate format"""
        cost_range = sizing.get('cost_range_inr', (300000, 400000))
        if isinstance(cost_range, (list, tuple)) and len(cost_range) >= 2:
            total_cost = (cost_range[0] + cost_range[1]) / 2
        else:
            total_cost = sizing.get('total_cost', 350000)
        
        return {
            'total_cost': total_cost,
            'system_size_kw': sizing.get('system_capacity_kw', 5.0),
            'accessibility_complexity': 1.2  # Default moderate complexity
        }

    def _extract_vendor_name(self, sizing: Dict) -> str:
        """Extract vendor name from sizing data"""
        selected_panel = sizing.get('selected_panel', '')
        selected_inverter = sizing.get('selected_inverter', '')
        
        # Try to extract brand from panel or inverter strings
        common_brands = ['Tata Power Solar', 'Waaree Energies', 'Adani Solar', 
                        'Luminous', 'Microtek', 'Havells']
        
        for brand in common_brands:
            if brand.lower() in selected_panel.lower() or brand.lower() in selected_inverter.lower():
                return brand
        
        return None  # Will use default vendor data

    def _convert_to_pipeline_format(self, risk_assessment: Dict) -> Dict:
        """Convert comprehensive risk assessment to pipeline-compatible format"""
        risk_breakdown = risk_assessment.get('risk_breakdown', {})
        
        # Extract individual risk components
        risk_components = {}
        
        if 'financial' in risk_breakdown:
            risk_components['financial_risk'] = risk_breakdown['financial'].get('overall_score', 0.4)
        
        if 'technical' in risk_breakdown:
            risk_components['technical_risk'] = risk_breakdown['technical'].get('overall_score', 0.3)
        
        if 'policy' in risk_breakdown:
            risk_components['policy_risk'] = risk_breakdown['policy'].get('overall_score', 0.2)
            risk_components['regulatory_risk'] = risk_breakdown['policy'].get('overall_score', 0.2)
        
        if 'environmental' in risk_breakdown:
            risk_components['environmental_risk'] = risk_breakdown['environmental'].get('overall_score', 0.25)
        
        if 'ownership_feasibility' in risk_breakdown:
            risk_components['ownership_risk'] = risk_breakdown['ownership_feasibility'].get('feasibility_score', 0.15)
        
        if 'maintenance_lifecycle' in risk_breakdown:
            risk_components['maintenance_risk'] = risk_breakdown['maintenance_lifecycle'].get('overall_score', 0.2)
        
        if 'behavioral_adoption' in risk_breakdown:
            risk_components['behavioral_risk'] = risk_breakdown['behavioral_adoption'].get('overall_score', 0.3)
        
        # Determine overall risk category
        overall_risk_score = risk_assessment.get('overall_risk_score', 0.4)
        risk_level = risk_assessment.get('risk_level', 'Moderate Risk')
        
        # Map risk level to simpler categories
        if 'Critical' in risk_level or 'Very High' in risk_level:
            overall_risk_category = "High"
        elif 'High' in risk_level:
            overall_risk_category = "High"
        elif 'Low' in risk_level or 'Very Low' in risk_level:
            overall_risk_category = "Low"
        else:
            overall_risk_category = "Moderate"
        
        # Extract recommendations
        recommendations = risk_assessment.get('comprehensive_recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        
        # Extract Monte Carlo insights
        monte_carlo = risk_assessment.get('monte_carlo_simulation', {})
        success_probability = 0.7  # Default
        if monte_carlo and 'risk_metrics' in monte_carlo:
            success_probability = monte_carlo['risk_metrics'].get('success_probability', 0.7)
        
        return {
            "overall_risk": overall_risk_category,
            "overall_risk_score": overall_risk_score,
            "risk_level": risk_level,
            "risk_components": risk_components,
            "success_probability": success_probability,
            "confidence": risk_assessment.get('confidence_level', 0.75),
            "investment_recommendation": risk_assessment.get('investment_recommendation', {}).get('recommendation', 'PROCEED WITH CAUTION'),
            "key_risk_drivers": risk_assessment.get('key_risk_drivers', []),
            "immediate_actions": immediate_actions[:3],  # Top 3 actions
            "monte_carlo_summary": {
                "payback_mean": monte_carlo.get('payback_period', {}).get('mean', 7.0),
                "payback_range": f"{monte_carlo.get('payback_period', {}).get('percentile_10', 6.0):.1f}-{monte_carlo.get('payback_period', {}).get('percentile_90', 8.0):.1f} years",
                "expected_savings": monte_carlo.get('net_savings_20yr', {}).get('mean', 500000)
            } if monte_carlo else {},
            "ownership_feasibility": {
                "feasible": risk_breakdown.get('ownership_feasibility', {}).get('feasibility_score', 0.2) < 0.5,
                "risk_level": risk_breakdown.get('ownership_feasibility', {}).get('risk_level', 'Moderate'),
                "alternatives": risk_breakdown.get('ownership_feasibility', {}).get('alternative_solutions', [])
            },
            "vendor_insights": {
                "maintenance_cost_annual": risk_breakdown.get('maintenance_lifecycle', {}).get('annual_maintenance_budget', 5000),
                "service_reliability_risk": risk_breakdown.get('technical', {}).get('vendor_reliability_risk', 0.3)
            }
        }

    # Alternative method names for compatibility
    def analyze(self, payload: Dict) -> Dict:
        """Alternative method name for pipeline compatibility"""
        return self.analyze_from_pipeline(payload)

    def score(self, payload: Dict) -> Dict:
        """Alternative method name for pipeline compatibility"""
        return self.analyze_from_pipeline(payload)

    def assess_risk(self, payload: Dict) -> Dict:
        """Alternative method name for pipeline compatibility"""
        return self.analyze_from_pipeline(payload)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the enhanced risk analyzer
    risk_analyzer = EnhancedRiskAnalyzer()
    
    # Test with sample user profile
    sample_user = {
        'monthly_bill': 8000,
        'budget_max': 400000,
        'location': 'Mumbai',
        'house_type': 'apartment',
        'ownership_status': 'tenant',
        'roof_area': 800,
        'income_bracket': 'middle',
        'age': 45,
        'first_time_solar_buyer': True,
        'tech_comfort_level': 'medium'
    }
    
    sample_cost_estimate = {
        'total_cost': 350000,
        'system_size_kw': 5,
        'accessibility_complexity': 1.2
    }
    
    # Run comprehensive risk assessment
    risk_assessment = risk_analyzer.generate_comprehensive_risk_assessment(
        sample_user, sample_cost_estimate, 'Tata Power Solar'
    )
    
    print("=== ENHANCED RISK ASSESSMENT RESULTS ===")
    print(f"Overall Risk Score: {risk_assessment['overall_risk_score']}")
    print(f"Risk Level: {risk_assessment['risk_level']}")
    print(f"Investment Recommendation: {risk_assessment['investment_recommendation']['recommendation']}")
    print(f"Success Probability: {risk_assessment['monte_carlo_simulation']['risk_metrics']['success_probability']}")
    print(f"Key Risk Drivers: {', '.join(risk_assessment['key_risk_drivers'])}")
    
    # Display ownership feasibility (critical for apartments)
    ownership_assessment = risk_assessment['risk_breakdown']['ownership_feasibility']
    print(f"\n=== OWNERSHIP FEASIBILITY ===")
    print(f"Feasibility Score: {ownership_assessment['feasibility_score']}")
    print(f"Risk Level: {ownership_assessment['risk_level']}")
    if ownership_assessment['alternative_solutions']:
        print("Alternative Solutions:")
        for solution in ownership_assessment['alternative_solutions']:
            print(f"  - {solution}")
    
    # Display Monte Carlo results
    mc_results = risk_assessment['monte_carlo_simulation']
    print(f"\n=== MONTE CARLO SIMULATION (2000 scenarios) ===")
    print(f"Expected Payback: {mc_results['payback_period']['mean']} years")
    print(f"Payback Range (90% confidence): {mc_results['payback_period']['percentile_5']}-{mc_results['payback_period']['percentile_95']} years")
    print(f"Expected 20-year Savings: ₹{mc_results['net_savings_20yr']['mean']:,.0f}")
    print(f"Probability of Positive NPV: {mc_results['npv_analysis']['probability_positive_npv']:.1%}")
    
    # Display comprehensive recommendations
    recommendations = risk_assessment['comprehensive_recommendations']
    print(f"\n=== COMPREHENSIVE RECOMMENDATIONS ===")
    print("Immediate Actions:")
    for action in recommendations['immediate_actions']:
        print(f"  • {action}")
    
    print("\nShort-term Actions:")
    for action in recommendations['short_term_actions']:
        print(f"  • {action}")
    
    if recommendations['alternative_solutions']:
        print("\nAlternative Solutions (if main installation not feasible):")
        for solution in recommendations['alternative_solutions']:
            print(f"  • {solution}")
    
    print(f"\n=== RISK BREAKDOWN BY CATEGORY ===")
    risk_breakdown = risk_assessment['risk_breakdown']
    for category, details in risk_breakdown.items():
        if isinstance(details, dict) and 'overall_score' in details:
            print(f"{category.replace('_', ' ').title()}: {details['overall_score']:.3f}")


class RiskVisualizationHelper:
    """Helper class for generating risk visualization data"""
    
    @staticmethod
    def prepare_radar_chart_data(risk_assessment: Dict) -> Dict:
        """Prepare data for risk radar chart visualization"""
        risk_breakdown = risk_assessment['risk_breakdown']
        
        categories = []
        scores = []
        
        for category, details in risk_breakdown.items():
            if isinstance(details, dict) and 'overall_score' in details:
                categories.append(category.replace('_', ' ').title())
                scores.append(details['overall_score'])
        
        return {
            'categories': categories,
            'risk_scores': scores,
            'overall_risk': risk_assessment['overall_risk_score'],
            'risk_level': risk_assessment['risk_level']
        }
    
    @staticmethod
    def prepare_monte_carlo_histogram_data(monte_carlo_results: Dict) -> Dict:
        """Prepare data for Monte Carlo results histogram"""
        return {
            'payback_stats': monte_carlo_results['payback_period'],
            'savings_stats': monte_carlo_results['net_savings_20yr'],
            'risk_metrics': monte_carlo_results['risk_metrics'],
            'chart_data': {
                'payback_mean': monte_carlo_results['payback_period']['mean'],
                'payback_p10': monte_carlo_results['payback_period']['percentile_10'],
                'payback_p90': monte_carlo_results['payback_period']['percentile_90'],
                'savings_mean': monte_carlo_results['net_savings_20yr']['mean'],
                'savings_p10': monte_carlo_results['net_savings_20yr']['percentile_10'],
                'savings_p90': monte_carlo_results['net_savings_20yr']['percentile_90']
            }
        }
    
    @staticmethod
    def prepare_risk_timeline_data(risk_assessment: Dict) -> Dict:
        """Prepare data for risk evolution timeline"""
        maintenance_risk = risk_assessment['risk_breakdown']['maintenance_lifecycle']
        
        timeline_data = {
            'years': list(range(1, 21)),
            'baseline_risk': [risk_assessment['overall_risk_score']] * 20,
            'maintenance_events': [],
            'policy_uncertainty': []
        }
        
        # Add maintenance event markers
        critical_years = maintenance_risk.get('critical_maintenance_years', [8, 10, 15, 18])
        for year in critical_years:
            timeline_data['maintenance_events'].append({
                'year': year,
                'event': 'Major Maintenance' if year in [10, 15] else 'Inverter Replacement' if year == 8 else 'System Service',
                'risk_spike': 0.2 if year == 8 else 0.1
            })
        
        # Add policy uncertainty progression
        policy_risk_base = risk_assessment['risk_breakdown']['policy']['overall_score']
        for year in range(1, 21):
            # Policy uncertainty typically increases over time
            uncertainty_factor = 1 + (year - 1) * 0.02  # 2% increase per year
            timeline_data['policy_uncertainty'].append(policy_risk_base * uncertainty_factor)
        
        return timeline_data


class RiskReportGenerator:
    """Generate comprehensive risk assessment reports"""
    
    def __init__(self, risk_analyzer: EnhancedRiskAnalyzer):
        self.risk_analyzer = risk_analyzer
    
    def generate_executive_summary(self, risk_assessment: Dict) -> Dict:
        """Generate executive summary of risk assessment"""
        
        overall_risk = risk_assessment['overall_risk_score']
        investment_rec = risk_assessment['investment_recommendation']
        monte_carlo = risk_assessment['monte_carlo_simulation']
        
        # Key insights
        key_insights = []
        
        # Ownership feasibility insight
        ownership = risk_assessment['risk_breakdown']['ownership_feasibility']
        if ownership['feasibility_score'] > 0.8:
            key_insights.append(f"⚠️ CRITICAL: {ownership['risk_level']} ownership feasibility")
        
        # Financial insight
        success_prob = monte_carlo['risk_metrics']['success_probability']
        if success_prob >= 0.8:
            key_insights.append(f"✅ High success probability ({success_prob:.1%})")
        elif success_prob < 0.5:
            key_insights.append(f"⚠️ Low success probability ({success_prob:.1%})")
        
        # Payback insight
        payback_mean = monte_carlo['payback_period']['mean']
        if payback_mean <= 6:
            key_insights.append(f"✅ Fast payback expected ({payback_mean:.1f} years)")
        elif payback_mean > 10:
            key_insights.append(f"⚠️ Long payback period ({payback_mean:.1f} years)")
        
        # Top risk drivers
        risk_drivers = risk_assessment['key_risk_drivers']
        if risk_drivers:
            key_insights.append(f"🎯 Key risks: {', '.join(risk_drivers[:2])}")
        
        return {
            'overall_risk_score': overall_risk,
            'risk_category': risk_assessment['risk_level'],
            'investment_recommendation': investment_rec['recommendation'],
            'confidence_level': investment_rec['confidence'],
            'key_insights': key_insights,
            'executive_decision': self._get_executive_decision(overall_risk, success_prob, ownership['feasibility_score']),
            'next_steps': risk_assessment['comprehensive_recommendations']['immediate_actions'][:3]
        }
    
    def _get_executive_decision(self, overall_risk: float, success_prob: float, ownership_feasibility: float) -> str:
        """Generate executive decision recommendation"""
        
        if ownership_feasibility > 0.9:
            return "❌ DO NOT PROCEED - Installation not feasible"
        elif overall_risk <= 0.3 and success_prob >= 0.8:
            return "✅ PROCEED IMMEDIATELY - Excellent investment opportunity"
        elif overall_risk <= 0.5 and success_prob >= 0.6:
            return "✅ PROCEED WITH MITIGATION - Good investment with risk management"
        elif overall_risk <= 0.7:
            return "⚠️ PROCEED WITH CAUTION - Requires careful risk management"
        else:
            return "❌ NOT RECOMMENDED - High risk, uncertain returns"
    
    def generate_detailed_report_sections(self, risk_assessment: Dict) -> Dict:
        """Generate detailed sections for comprehensive report"""
        
        return {
            'methodology_overview': {
                'approach': 'Multi-Criteria Decision Analysis with Monte Carlo Simulation',
                'risk_dimensions': list(risk_assessment['risk_breakdown'].keys()),
                'simulation_parameters': risk_assessment['monte_carlo_simulation']['simulation_parameters'],
                'confidence_level': risk_assessment['confidence_level']
            },
            
            'risk_deep_dive': self._create_risk_deep_dive(risk_assessment['risk_breakdown']),
            
            'financial_projections': {
                'base_case': risk_assessment['monte_carlo_simulation']['net_savings_20yr']['mean'],
                'conservative_case': risk_assessment['monte_carlo_simulation']['net_savings_20yr']['percentile_10'],
                'optimistic_case': risk_assessment['monte_carlo_simulation']['net_savings_20yr']['percentile_90'],
                'payback_analysis': risk_assessment['monte_carlo_simulation']['payback_period'],
                'npv_analysis': risk_assessment['monte_carlo_simulation']['npv_analysis']
            },
            
            'mitigation_strategies': risk_assessment['comprehensive_recommendations'],
            
            'monitoring_framework': self._create_monitoring_framework(risk_assessment),
            
            'alternative_scenarios': self._create_alternative_scenarios(risk_assessment)
        }
    
    def _create_risk_deep_dive(self, risk_breakdown: Dict) -> Dict:
        """Create detailed analysis for each risk category"""
        deep_dive = {}
        
        for category, details in risk_breakdown.items():
            if isinstance(details, dict) and 'overall_score' in details:
                deep_dive[category] = {
                    'risk_score': details['overall_score'],
                    'risk_level': 'High' if details['overall_score'] > 0.6 else 'Medium' if details['overall_score'] > 0.3 else 'Low',
                    'key_factors': [k for k, v in details.items() if isinstance(v, (int, float)) and v > 0.4],
                    'confidence': details.get('confidence', 0.75),
                    'mitigation_available': len(details.get('recommendations', [])) > 0
                }
        
        return deep_dive
    
    def _create_monitoring_framework(self, risk_assessment: Dict) -> Dict:
        """Create framework for ongoing risk monitoring"""
        
        return {
            'quarterly_reviews': [
                'Policy environment changes',
                'Technology price trends',
                'Vendor service quality metrics'
            ],
            'annual_assessments': [
                'Financial performance vs projections',
                'System performance and maintenance costs',
                'Market condition changes'
            ],
            'trigger_events': [
                'Major policy announcements',
                'Vendor service issues',
                'System performance degradation > 5%',
                'Unexpected maintenance costs'
            ],
            'kpi_dashboard': {
                'financial_kpis': ['Monthly savings', 'Payback progress', 'ROI'],
                'operational_kpis': ['System uptime', 'Generation vs forecast', 'Maintenance costs'],
                'external_kpis': ['Policy stability index', 'Market price trends', 'Technology evolution']
            }
        }
    
    def _create_alternative_scenarios(self, risk_assessment: Dict) -> Dict:
        """Create alternative scenario analysis"""
        
        ownership = risk_assessment['risk_breakdown']['ownership_feasibility']
        
        scenarios = {
            'base_scenario': {
                'description': 'Current assessment with standard assumptions',
                'probability': 0.6,
                'outcome': risk_assessment['investment_recommendation']['recommendation']
            }
        }
        
        # High-risk scenario
        if risk_assessment['overall_risk_score'] < 0.7:
            scenarios['pessimistic_scenario'] = {
                'description': 'Policy changes, higher costs, lower generation',
                'probability': 0.2,
                'risk_increase': 0.3,
                'outcome': 'Extended payback period, reduced returns'
            }
        
        # Optimistic scenario
        scenarios['optimistic_scenario'] = {
            'description': 'Favorable policies, cost reductions, optimal performance',
            'probability': 0.2,
            'risk_decrease': 0.2,
            'outcome': 'Faster payback, higher returns'
        }
        
        # Alternative solutions scenario (if ownership issues)
        if ownership['feasibility_score'] > 0.6:
            scenarios['alternative_solutions'] = {
                'description': 'Community solar or alternative solar solutions',
                'probability': 0.8,
                'solutions': ownership.get('alternative_solutions', []),
                'outcome': 'Partial solar benefits without direct ownership'
            }
        
        return scenarios