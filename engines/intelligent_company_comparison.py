"""
Enhanced Intelligent Company Comparison Engine
Integrates with risk analysis for vendor-specific risk assessment
Advanced negotiation intelligence with market dynamics
Real-time vendor performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedCompanyComparator:
    def __init__(self, data_dir: str = "data/"):
        """Initialize with enhanced company intelligence and market data"""
        self.data_dir = data_dir
        self.company_data = self._load_enhanced_company_data()
        self.criteria_weights = self._get_default_criteria_weights()
        self.negotiation_strategies = self._load_advanced_negotiation_strategies()
        self.market_intelligence = self._load_real_time_market_intelligence()
        self.vendor_performance_tracker = self._load_vendor_performance_data()
        self.risk_integration = self._initialize_risk_integration()
        
    def _load_enhanced_company_data(self) -> pd.DataFrame:
        """Load comprehensive company profiles with enhanced metrics"""
        try:
            data = pd.read_csv(f"{self.data_dir}company_profiles_enhanced.csv")
        except FileNotFoundError:
            data = self._create_enhanced_company_data()
        
        # Normalize scores and add derived metrics
        self._process_company_data(data)
        return data
    
    def _create_enhanced_company_data(self) -> pd.DataFrame:
        """Create enhanced mock company data with comprehensive metrics"""
        companies = [
            {
                'company_name': 'Tata Power Solar',
                'tier': 'Premium',
                'market_share_percent': 15.2,
                'years_in_business': 12,
                'service_quality_score': 8.5,
                'installation_quality_score': 8.1,
                'after_sales_score': 7.5,
                'cost_competitiveness': 7.1,
                'warranty_support_score': 8.7,
                'regional_presence_score': 9.6,
                'technology_innovation_score': 8.1,
                'financing_options_available': True,
                'avg_installation_time_days': 13,
                'customer_complaint_resolution_hours': 48,
                'warranty_claim_success_rate': 0.92,
                'maintenance_response_time_hours': 24,
                'service_network_coverage_percent': 85,
                'employee_count': 2500,
                'annual_installations': 1200,
                'customer_retention_rate': 0.89,
                'financial_stability_rating': 'A+',
                'certifications': ['ISO 9001', 'ISO 14001', 'BIS'],
                'technology_partnerships': ['Tier-1 Panel Manufacturers'],
                'price_volatility_index': 0.15,
                'delivery_reliability_score': 8.8,
                'post_installation_support_score': 8.2,
                'component_quality_tier': 'Tier-1',
                'digital_monitoring_capability': True,
                'remote_troubleshooting': True,
                'warranty_period_panels': 25,
                'warranty_period_inverters': 12,
                'warranty_period_installation': 5,
                'maintenance_cost_per_kw_annual': 450,
                'avg_project_cost_per_kw': 45000,
                'bulk_discount_availability': True,
                'seasonal_pricing_variation': 0.08
            },
            {
                'company_name': 'Waaree Energies',
                'tier': 'Premium',
                'market_share_percent': 12.8,
                'years_in_business': 15,
                'service_quality_score': 8.4,
                'installation_quality_score': 9.1,
                'after_sales_score': 8.1,
                'cost_competitiveness': 6.7,
                'warranty_support_score': 8.3,
                'regional_presence_score': 8.8,
                'technology_innovation_score': 7.1,
                'financing_options_available': True,
                'avg_installation_time_days': 11,
                'customer_complaint_resolution_hours': 36,
                'warranty_claim_success_rate': 0.89,
                'maintenance_response_time_hours': 48,
                'service_network_coverage_percent': 78,
                'employee_count': 3200,
                'annual_installations': 980,
                'customer_retention_rate': 0.87,
                'financial_stability_rating': 'A+',
                'certifications': ['ISO 9001', 'ISO 45001', 'BIS'],
                'technology_partnerships': ['In-house Manufacturing'],
                'price_volatility_index': 0.12,
                'delivery_reliability_score': 9.2,
                'post_installation_support_score': 8.0,
                'component_quality_tier': 'Tier-1',
                'digital_monitoring_capability': True,
                'remote_troubleshooting': True,
                'warranty_period_panels': 25,
                'warranty_period_inverters': 10,
                'warranty_period_installation': 5,
                'maintenance_cost_per_kw_annual': 520,
                'avg_project_cost_per_kw': 47000,
                'bulk_discount_availability': True,
                'seasonal_pricing_variation': 0.06
            },
            {
                'company_name': 'Adani Solar',
                'tier': 'Premium',
                'market_share_percent': 11.7,
                'years_in_business': 8,
                'service_quality_score': 8.3,
                'installation_quality_score': 8.4,
                'after_sales_score': 8.6,
                'cost_competitiveness': 6.7,
                'warranty_support_score': 8.6,
                'regional_presence_score': 8.5,
                'technology_innovation_score': 6.6,
                'financing_options_available': False,
                'avg_installation_time_days': 12,
                'customer_complaint_resolution_hours': 72,
                'warranty_claim_success_rate': 0.87,
                'maintenance_response_time_hours': 36,
                'service_network_coverage_percent': 82,
                'employee_count': 1800,
                'annual_installations': 850,
                'customer_retention_rate': 0.91,
                'financial_stability_rating': 'AA',
                'certifications': ['ISO 9001', 'BIS', 'IEC'],
                'technology_partnerships': ['Tier-1 International'],
                'price_volatility_index': 0.10,
                'delivery_reliability_score': 8.5,
                'post_installation_support_score': 8.8,
                'component_quality_tier': 'Tier-1',
                'digital_monitoring_capability': True,
                'remote_troubleshooting': False,
                'warranty_period_panels': 25,
                'warranty_period_inverters': 10,
                'warranty_period_installation': 3,
                'maintenance_cost_per_kw_annual': 480,
                'avg_project_cost_per_kw': 46500,
                'bulk_discount_availability': False,
                'seasonal_pricing_variation': 0.05
            },
            {
                'company_name': 'Vikram Solar',
                'tier': 'Mid-tier',
                'market_share_percent': 8.5,
                'years_in_business': 14,
                'service_quality_score': 7.9,
                'installation_quality_score': 8.1,
                'after_sales_score': 7.7,
                'cost_competitiveness': 7.2,
                'warranty_support_score': 8.0,
                'regional_presence_score': 7.5,
                'technology_innovation_score': 7.0,
                'financing_options_available': True,
                'avg_installation_time_days': 16,
                'customer_complaint_resolution_hours': 96,
                'warranty_claim_success_rate': 0.84,
                'maintenance_response_time_hours': 72,
                'service_network_coverage_percent': 65,
                'employee_count': 1200,
                'annual_installations': 650,
                'customer_retention_rate': 0.82,
                'financial_stability_rating': 'A',
                'certifications': ['ISO 9001', 'BIS'],
                'technology_partnerships': ['Regional Suppliers'],
                'price_volatility_index': 0.18,
                'delivery_reliability_score': 7.8,
                'post_installation_support_score': 7.5,
                'component_quality_tier': 'Tier-2',
                'digital_monitoring_capability': False,
                'remote_troubleshooting': False,
                'warranty_period_panels': 25,
                'warranty_period_inverters': 8,
                'warranty_period_installation': 2,
                'maintenance_cost_per_kw_annual': 650,
                'avg_project_cost_per_kw': 42000,
                'bulk_discount_availability': True,
                'seasonal_pricing_variation': 0.12
            },
            {
                'company_name': 'Luminous Power',
                'tier': 'Mid-tier',
                'market_share_percent': 6.3,
                'years_in_business': 11,
                'service_quality_score': 8.2,
                'installation_quality_score': 8.9,
                'after_sales_score': 6.1,
                'cost_competitiveness': 8.0,
                'warranty_support_score': 8.2,
                'regional_presence_score': 6.9,
                'technology_innovation_score': 7.8,
                'financing_options_available': True,
                'avg_installation_time_days': 6,
                'customer_complaint_resolution_hours': 120,
                'warranty_claim_success_rate': 0.79,
                'maintenance_response_time_hours': 96,
                'service_network_coverage_percent': 70,
                'employee_count': 900,
                'annual_installations': 520,
                'customer_retention_rate': 0.76,
                'financial_stability_rating': 'A-',
                'certifications': ['ISO 9001'],
                'technology_partnerships': ['Local Manufacturing'],
                'price_volatility_index': 0.22,
                'delivery_reliability_score': 9.5,
                'post_installation_support_score': 6.8,
                'component_quality_tier': 'Tier-2',
                'digital_monitoring_capability': True,
                'remote_troubleshooting': True,
                'warranty_period_panels': 20,
                'warranty_period_inverters': 5,
                'warranty_period_installation': 2,
                'maintenance_cost_per_kw_annual': 780,
                'avg_project_cost_per_kw': 38000,
                'bulk_discount_availability': False,
                'seasonal_pricing_variation': 0.15
            }
        ]
        return pd.DataFrame(companies)
    
    def _process_company_data(self, data: pd.DataFrame):
        """Process and enhance company data with derived metrics"""
        
        # Normalize scores to 0-1 scale
        score_columns = [
            'service_quality_score', 'installation_quality_score', 'after_sales_score',
            'cost_competitiveness', 'warranty_support_score', 'regional_presence_score',
            'technology_innovation_score', 'delivery_reliability_score', 'post_installation_support_score'
        ]
        
        for col in score_columns:
            if col in data.columns:
                data[f'{col}_normalized'] = data[col] / 10.0
        
        # Calculate derived metrics
        data['overall_reliability_score'] = (
            data['service_quality_score'] * 0.3 +
            data['installation_quality_score'] * 0.25 +
            data['after_sales_score'] * 0.25 +
            data['warranty_support_score'] * 0.2
        ) / 10.0
        
        data['value_for_money_score'] = (
            data['cost_competitiveness'] * 0.4 +
            data['service_quality_score'] * 0.3 +
            data['warranty_support_score'] * 0.3
        ) / 10.0
        
        data['service_excellence_score'] = (
            data['customer_complaint_resolution_hours'].apply(lambda x: max(0, 10 - x/24)) +
            data['maintenance_response_time_hours'].apply(lambda x: max(0, 10 - x/12)) +
            data['warranty_claim_success_rate'] * 10
        ) / 3 / 10.0
        
        # Risk indicators
        data['vendor_stability_risk'] = self._calculate_vendor_stability_risk(data)
        data['service_continuity_risk'] = self._calculate_service_continuity_risk(data)
        
    def _calculate_vendor_stability_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate vendor financial and operational stability risk"""
        stability_risk = []
        
        for _, company in data.iterrows():
            risk = 0.0
            
            # Financial stability
            if company['financial_stability_rating'] == 'AA':
                risk += 0.05
            elif company['financial_stability_rating'] == 'A+':
                risk += 0.1
            elif company['financial_stability_rating'] == 'A':
                risk += 0.2
            elif company['financial_stability_rating'] == 'A-':
                risk += 0.3
            else:
                risk += 0.5
            
            # Market presence stability
            if company['market_share_percent'] < 5:
                risk += 0.3
            elif company['market_share_percent'] < 10:
                risk += 0.2
            else:
                risk += 0.1
            
            # Operational scale
            if company['employee_count'] < 500:
                risk += 0.2
            elif company['employee_count'] < 1000:
                risk += 0.15
            else:
                risk += 0.05
            
            # Years in business
            if company['years_in_business'] < 5:
                risk += 0.3
            elif company['years_in_business'] < 10:
                risk += 0.2
            else:
                risk += 0.1
            
            stability_risk.append(min(1.0, risk))
        
        return pd.Series(stability_risk)
    
    def _calculate_service_continuity_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate risk of service disruption or poor ongoing support"""
        service_risk = []
        
        for _, company in data.iterrows():
            risk = 0.0
            
            # After-sales service quality
            risk += (10 - company['after_sales_score']) / 10 * 0.3
            
            # Service network coverage
            risk += (100 - company['service_network_coverage_percent']) / 100 * 0.2
            
            # Response time reliability
            if company['maintenance_response_time_hours'] > 72:
                risk += 0.3
            elif company['maintenance_response_time_hours'] > 48:
                risk += 0.2
            else:
                risk += 0.1
            
            # Warranty claim success rate
            risk += (1 - company['warranty_claim_success_rate']) * 0.4
            
            # Customer retention (loyalty indicator)
            risk += (1 - company['customer_retention_rate']) * 0.3
            
            service_risk.append(min(1.0, risk))
        
        return pd.Series(service_risk)
    
    def _get_default_criteria_weights(self) -> Dict:
        """Enhanced default MCDA criteria weights"""
        return {
            'cost_competitiveness': 0.20,
            'service_quality': 0.18,
            'installation_quality': 0.15,
            'warranty_support': 0.13,
            'after_sales_support': 0.10,
            'technology_innovation': 0.08,
            'regional_presence': 0.06,
            'vendor_stability': 0.05,
            'service_continuity': 0.05
        }
    
    def _load_advanced_negotiation_strategies(self) -> Dict:
        """Load advanced negotiation intelligence with market dynamics"""
        return {
            'price_anchoring': {
                'premium_tier': {
                    'strategy': "Reference competitor premium quotes for 8-15% discount",
                    'leverage_points': ['Quality parity', 'Service comparison', 'Warranty terms'],
                    'success_probability': 0.7
                },
                'mid_tier': {
                    'strategy': "Emphasize budget constraints for 10-18% discount",
                    'leverage_points': ['Volume commitment', 'Payment terms', 'Referral promise'],
                    'success_probability': 0.8
                },
                'market_leader': {
                    'strategy': "Request value-added services at same price",
                    'leverage_points': ['Market leadership premium', 'Service excellence'],
                    'success_probability': 0.6
                }
            },
            
            'timing_leverage': {
                'end_of_quarter': {
                    'discount_potential': '8-15%',
                    'strategy': 'Target quarterly sales goals',
                    'best_months': ['March', 'June', 'September', 'December'],
                    'success_probability': 0.75
                },
                'festival_season': {
                    'discount_potential': '5-12%',
                    'strategy': 'Leverage festive promotions',
                    'best_periods': ['Diwali', 'New Year', 'Holi'],
                    'success_probability': 0.65
                },
                'off_peak_installation': {
                    'discount_potential': '10-20%',
                    'strategy': 'Schedule during low-demand periods',
                    'best_months': ['June', 'July', 'August'],
                    'success_probability': 0.8
                },
                'bulk_coordination': {
                    'discount_potential': '15-25%',
                    'strategy': 'Coordinate with neighbors for group installation',
                    'minimum_group_size': 3,
                    'success_probability': 0.85
                }
            },
            
            'service_upgrades': {
                'warranty_extension': {
                    'request': 'Extended warranty at no cost',
                    'fallback': '50% discount on extended warranty',
                    'success_factors': ['Premium pricing', 'Long-term relationship']
                },
                'monitoring_system': {
                    'request': '3-year free monitoring and reporting',
                    'fallback': '1-year free monitoring',
                    'success_factors': ['Digital capability', 'Service differentiation']
                },
                'maintenance_package': {
                    'request': '5-year comprehensive maintenance inclusion',
                    'fallback': '2-year basic maintenance',
                    'success_factors': ['High maintenance costs', 'Service network']
                },
                'performance_guarantee': {
                    'request': 'Generation performance guarantee with penalties',
                    'fallback': 'Performance monitoring with reporting',
                    'success_factors': ['Premium tier', 'Confidence in technology']
                }
            },
            
            'competitive_intelligence': {
                'market_positioning': {
                    'premium_vs_premium': 'Focus on service differentiation and warranty terms',
                    'premium_vs_midtier': 'Challenge premium pricing with value proposition',
                    'midtier_vs_midtier': 'Emphasize local presence and responsiveness',
                    'midtier_vs_budget': 'Justify premium with quality and reliability'
                },
                'weak_points_exploitation': {
                    'poor_after_sales': 'Negotiate extended warranty and service guarantees',
                    'limited_financing': 'Request financing arrangement facilitation',
                    'slow_installation': 'Demand expedited timeline or compensation',
                    'poor_response_time': 'Negotiate response time SLAs with penalties'
                }
            }
        }
    
    def _load_real_time_market_intelligence(self) -> Dict:
        """Load current market conditions and competitive dynamics"""
        return {
            'market_conditions': {
                'demand_supply_balance': 'high_demand_moderate_supply',
                'price_trend_3m': 'stable_to_declining',
                'price_trend_12m': 'declining',
                'competition_intensity': 'high',
                'customer_bargaining_power': 'moderate_to_high'
            },
            
            'seasonal_dynamics': {
                'current_season': self._get_current_season(),
                'demand_seasonality': {
                    'peak_months': ['October', 'November', 'December', 'January'],
                    'off_peak_months': ['June', 'July', 'August'],
                    'price_premium_peak': 0.05,
                    'discount_opportunity_off_peak': 0.12
                }
            },
            
            'competitive_landscape': {
                'market_fragmentation': 'moderate',
                'new_entrant_pressure': 'increasing',
                'technology_disruption_risk': 'low_to_moderate',
                'regulatory_support': 'strong',
                'financing_availability': 'good'
            },
            
            'customer_behavior_trends': {
                'price_sensitivity': 'high',
                'quality_consciousness': 'increasing',
                'brand_loyalty': 'moderate',
                'service_expectations': 'high',
                'digital_engagement_preference': 'increasing'
            }
        }
    
    def _get_current_season(self) -> str:
        """Determine current installation season"""
        current_month = datetime.now().month
        if current_month in [10, 11, 12, 1]:
            return 'peak_season'
        elif current_month in [6, 7, 8]:
            return 'off_season'
        else:
            return 'moderate_season'
    
    def _load_vendor_performance_data(self) -> Dict:
        """Load recent vendor performance metrics"""
        try:
            with open(f"{self.data_dir}vendor_performance_tracker.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_mock_performance_data()
    
    def _create_mock_performance_data(self) -> Dict:
        """Create mock vendor performance tracking data"""
        return {
            'performance_trends': {
                'Tata Power Solar': {
                    'last_quarter_installations': 245,
                    'customer_satisfaction_trend': 'stable',
                    'complaint_resolution_trend': 'improving',
                    'pricing_trend': 'stable',
                    'service_quality_trend': 'stable'
                },
                'Waaree Energies': {
                    'last_quarter_installations': 198,
                    'customer_satisfaction_trend': 'improving',
                    'complaint_resolution_trend': 'stable',
                    'pricing_trend': 'slightly_declining',
                    'service_quality_trend': 'improving'
                },
                'Adani Solar': {
                    'last_quarter_installations': 187,
                    'customer_satisfaction_trend': 'stable',
                    'complaint_resolution_trend': 'declining',
                    'pricing_trend': 'stable',
                    'service_quality_trend': 'stable'
                }
            },
            'recent_incidents': {
                'Tata Power Solar': [],
                'Waaree Energies': [
                    {'type': 'delivery_delay', 'frequency': 'rare', 'impact': 'low'}
                ],
                'Adani Solar': [
                    {'type': 'customer_service_complaints', 'frequency': 'occasional', 'impact': 'medium'}
                ]
            }
        }
    
    def _initialize_risk_integration(self) -> Dict:
        """Initialize integration with risk analysis modules"""
        return {
            'vendor_risk_factors': {
                'financial_stability': 'vendor_stability_risk',
                'service_continuity': 'service_continuity_risk',
                'delivery_reliability': 'delivery_reliability_score',
                'warranty_reliability': 'warranty_claim_success_rate'
            },
            'risk_weight_multipliers': {
                'high_risk_vendor': 1.3,
                'medium_risk_vendor': 1.1,
                'low_risk_vendor': 0.9
            }
        }
    
    def calculate_risk_adjusted_mcda_scores(self, user_preferences: Optional[Dict] = None, 
                                          location: Optional[str] = None,
                                          risk_tolerance: str = 'moderate') -> pd.DataFrame:
        """
        Calculate risk-adjusted MCDA scores considering vendor-specific risks
        
        Args:
            user_preferences: User cluster weights
            location: User location for regional presence weighting
            risk_tolerance: User's risk tolerance ('low', 'moderate', 'high')
            
        Returns:
            pd.DataFrame: Companies ranked by risk-adjusted MCDA scores
        """
        # Customize weights based on preferences and risk tolerance
        if user_preferences:
            weights = self.customize_criteria_weights(user_preferences)
        else:
            weights = self.criteria_weights.copy()
        
        # Adjust weights based on risk tolerance
        risk_adjustments = {
            'low': {'vendor_stability': 1.5, 'service_continuity': 1.3, 'warranty_support': 1.2},
            'moderate': {'vendor_stability': 1.2, 'service_continuity': 1.1, 'warranty_support': 1.0},
            'high': {'vendor_stability': 1.0, 'service_continuity': 1.0, 'cost_competitiveness': 1.2}
        }
        
        adjustments = risk_adjustments.get(risk_tolerance, risk_adjustments['moderate'])
        for criterion, multiplier in adjustments.items():
            if criterion in weights:
                weights[criterion] *= multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate risk-adjusted MCDA scores
        companies_scored = self.company_data.copy()
        mcda_score = np.zeros(len(companies_scored))
        
        # Cost competitiveness (higher is better, risk-adjusted by price volatility)
        if 'cost_competitiveness' in companies_scored.columns:
            cost_score = companies_scored['cost_competitiveness'] / 10.0
            # Adjust for price volatility risk
            price_stability = 1 - companies_scored['price_volatility_index']
            risk_adjusted_cost = cost_score * price_stability
            mcda_score += risk_adjusted_cost * weights.get('cost_competitiveness', 0)
        
        # Service quality (risk-adjusted by service continuity)
        if 'service_quality_score' in companies_scored.columns:
            service_score = companies_scored['service_quality_score'] / 10.0
            service_continuity_factor = 1 - companies_scored['service_continuity_risk']
            risk_adjusted_service = service_score * service_continuity_factor
            mcda_score += risk_adjusted_service * weights.get('service_quality', 0)
        
        # Installation quality
        if 'installation_quality_score' in companies_scored.columns:
            installation_score = companies_scored['installation_quality_score'] / 10.0
            mcda_score += installation_score * weights.get('installation_quality', 0)
        
        # Warranty support (risk-adjusted by claim success rate)
        if 'warranty_support_score' in companies_scored.columns:
            warranty_score = companies_scored['warranty_support_score'] / 10.0
            warranty_reliability = companies_scored['warranty_claim_success_rate']
            risk_adjusted_warranty = warranty_score * warranty_reliability
            mcda_score += risk_adjusted_warranty * weights.get('warranty_support', 0)
        
        # After sales support (risk-adjusted by response time and network coverage)
        if 'after_sales_score' in companies_scored.columns:
            after_sales_score = companies_scored['after_sales_score'] / 10.0
            # Response time factor (better response = lower risk)
            response_factor = np.maximum(0.3, 1 - companies_scored['maintenance_response_time_hours'] / 120)
            # Network coverage factor
            coverage_factor = companies_scored['service_network_coverage_percent'] / 100
            risk_adjustment = (response_factor + coverage_factor) / 2
            risk_adjusted_after_sales = after_sales_score * risk_adjustment
            mcda_score += risk_adjusted_after_sales * weights.get('after_sales_support', 0)
        
        # Technology innovation
        if 'technology_innovation_score' in companies_scored.columns:
            tech_score = companies_scored['technology_innovation_score'] / 10.0
            mcda_score += tech_score * weights.get('technology_innovation', 0)
        
        # Regional presence (location-adjusted)
        if 'regional_presence_score' in companies_scored.columns:
            regional_score = companies_scored['regional_presence_score'] / 10.0
            if location:
                # Add location boost for strong regional presence
                regional_score *= 1.15
            mcda_score += regional_score * weights.get('regional_presence', 0)
        
        # Vendor stability (direct risk factor)
        vendor_stability_score = 1 - companies_scored['vendor_stability_risk']
        mcda_score += vendor_stability_score * weights.get('vendor_stability', 0)
        
        # Service continuity (direct risk factor)
        service_continuity_score = 1 - companies_scored['service_continuity_risk']
        mcda_score += service_continuity_score * weights.get('service_continuity', 0)
        
        companies_scored['risk_adjusted_mcda_score'] = mcda_score
        companies_scored['risk_adjusted_score_percent'] = mcda_score * 100
        
        # Add risk categorization
        companies_scored['vendor_risk_category'] = companies_scored.apply(
            lambda row: self._categorize_vendor_risk(row), axis=1
        )
        
        # Sort by risk-adjusted MCDA score
        companies_ranked = companies_scored.sort_values('risk_adjusted_mcda_score', ascending=False)
        
        return companies_ranked
    
    def _categorize_vendor_risk(self, vendor_row) -> str:
        """Categorize vendor risk level"""
        stability_risk = vendor_row['vendor_stability_risk']
        service_risk = vendor_row['service_continuity_risk']
        
        overall_risk = (stability_risk + service_risk) / 2
        
        if overall_risk <= 0.2:
            return 'Low Risk'
        elif overall_risk <= 0.4:
            return 'Moderate Risk'
        elif overall_risk <= 0.6:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def customize_criteria_weights(self, user_cluster_weights: Dict) -> Dict:
        """Enhanced criteria weight customization"""
        cluster_weight_mapping = {
            'cost_weight': 'cost_competitiveness',
            'brand_weight': 'service_quality',
            'warranty_weight': 'warranty_support',
            'technology_weight': 'technology_innovation',
            'reliability_weight': 'vendor_stability',
            'service_weight': 'after_sales_support'
        }
        
        # Start with default weights
        custom_weights = self.criteria_weights.copy()
        total_mapped_weight = 0
        
        # Apply user preferences
        for cluster_key, mcda_key in cluster_weight_mapping.items():
            if cluster_key in user_cluster_weights:
                custom_weights[mcda_key] = user_cluster_weights[cluster_key]
                total_mapped_weight += user_cluster_weights[cluster_key]
        
        # Distribute remaining weight
        remaining_weight = max(0, 1.0 - total_mapped_weight)
        unmapped_criteria = [k for k in custom_weights.keys() 
                           if k not in cluster_weight_mapping.values()]
        
        if unmapped_criteria and remaining_weight > 0:
            weight_per_unmapped = remaining_weight / len(unmapped_criteria)
            for criteria in unmapped_criteria:
                custom_weights[criteria] = weight_per_unmapped
        
        # Ensure weights sum to 1
        total_weight = sum(custom_weights.values())
        if total_weight > 0:
            custom_weights = {k: v/total_weight for k, v in custom_weights.items()}
        
        return custom_weights
    
    def generate_enhanced_negotiation_intelligence(self, company_name: str, 
                                                 user_budget: float,
                                                 user_profile: Dict,
                                                 market_conditions: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive negotiation strategy with advanced market intelligence
        
        Args:
            company_name: Target company for negotiation
            user_budget: User's budget range
            user_profile: Complete user profile for personalization
            market_conditions: Current market conditions
            
        Returns:
            dict: Comprehensive negotiation intelligence and strategy
        """
        try:
            company_info = self.company_data[self.company_data['company_name'] == company_name].iloc[0]
        except IndexError:
            return {'error': f'Company {company_name} not found in database'}
        
        # Use provided market conditions or default
        if not market_conditions:
            market_conditions = self.market_intelligence['market_conditions']
        
        # Determine company negotiation profile
        company_profile = self._analyze_company_negotiation_profile(company_info)
        
        # Identify user leverage points
        user_leverage = self._identify_user_leverage_points(user_profile, user_budget, company_info)
        
        # Generate timing-based opportunities
        timing_opportunities = self._analyze_timing_opportunities(company_info, market_conditions)
        
        # Develop competitive positioning strategy
        competitive_strategy = self._develop_competitive_positioning(company_name, user_budget)
        
        # Calculate negotiation success probability
        success_probability = self._calculate_negotiation_success_probability(
            company_profile, user_leverage, timing_opportunities, market_conditions
        )
        
        # Generate specific negotiation tactics
        negotiation_tactics = self._generate_negotiation_tactics(
            company_profile, user_leverage, timing_opportunities
        )
        
        # Estimate realistic discount ranges
        discount_estimates = self._estimate_realistic_discounts(
            company_info, market_conditions, user_leverage
        )
        
        return {
            'company_analysis': {
                'name': company_name,
                'tier': company_info['tier'],
                'market_position': f"{company_info['market_share_percent']:.1f}% market share",
                'negotiation_profile': company_profile,
                'financial_stability': company_info['financial_stability_rating'],
                'risk_category': self._categorize_vendor_risk(company_info)
            },
            'user_leverage_analysis': user_leverage,
            'timing_opportunities': timing_opportunities,
            'competitive_positioning': competitive_strategy,
            'negotiation_tactics': negotiation_tactics,
            'discount_expectations': discount_estimates,
            'success_probability': success_probability,
            'risk_considerations': self._identify_negotiation_risks(company_info, user_leverage),
            'alternative_strategies': self._suggest_alternative_strategies(company_profile, user_leverage)
        }
    
    def _analyze_company_negotiation_profile(self, company_info) -> Dict:
        """Analyze company's negotiation characteristics and flexibility"""
        profile = {
            'flexibility_score': 0.5,
            'negotiation_style': 'standard',
            'key_motivators': [],
            'constraint_factors': []
        }
        
        # Market position influence
        if company_info['market_share_percent'] > 12:
            profile['flexibility_score'] -= 0.2
            profile['negotiation_style'] = 'conservative'
            profile['constraint_factors'].append('Market leader premium positioning')
        elif company_info['market_share_percent'] < 7:
            profile['flexibility_score'] += 0.3
            profile['negotiation_style'] = 'aggressive'
            profile['key_motivators'].append('Market share growth objectives')
        
        # Financial pressure indicators
        if company_info['financial_stability_rating'] in ['A-', 'B+']:
            profile['flexibility_score'] += 0.2
            profile['key_motivators'].append('Revenue pressure')
        
        # Service quality vs cost positioning
        service_quality = company_info['service_quality_score']
        cost_competitiveness = company_info['cost_competitiveness']
        
        if service_quality > 8.5 and cost_competitiveness < 7:
            profile['negotiation_style'] = 'value_focused'
            profile['constraint_factors'].append('Premium service positioning')
        elif cost_competitiveness > 8:
            profile['negotiation_style'] = 'price_competitive'
            profile['key_motivators'].append('Volume-based growth strategy')
        
        # Seasonal and operational factors
        if company_info['avg_installation_time_days'] > 14:
            profile['key_motivators'].append('Capacity utilization optimization')
        
        if not company_info['financing_options_available']:
            profile['constraint_factors'].append('Limited financing capabilities')
        
        return profile
    
    def _identify_user_leverage_points(self, user_profile: Dict, user_budget: float, 
                                     company_info) -> Dict:
        """Identify user's negotiation leverage points"""
        leverage = {
            'strength_score': 0.5,
            'leverage_points': [],
            'weak_points': [],
            'bargaining_power': 'moderate'
        }
        
        # Budget size leverage
        avg_project_cost = company_info['avg_project_cost_per_kw'] * (user_budget / 50000)  # Estimate system size
        
        if user_budget > avg_project_cost * 1.5:
            leverage['strength_score'] += 0.2
            leverage['leverage_points'].append('Above-average project size')
        elif user_budget < avg_project_cost * 0.7:
            leverage['strength_score'] -= 0.2
            leverage['weak_points'].append('Below-average project budget')
        
        # Timing leverage
        location = user_profile.get('location', '')
        if location in ['Mumbai', 'Delhi', 'Pune']:  # High-demand locations
            leverage['strength_score'] += 0.1
            leverage['leverage_points'].append('High-demand location')
        
        # Referral potential
        house_type = user_profile.get('house_type', '')
        if house_type == 'apartment':
            leverage['strength_score'] += 0.15
            leverage['leverage_points'].append('Multi-unit referral potential')
        
        # Market research and comparison shopping
        leverage['leverage_points'].append('Informed buyer with competitive analysis')
        
        # Payment capability
        if user_budget > 300000:
            leverage['strength_score'] += 0.1
            leverage['leverage_points'].append('Strong payment capability')
        
        # Determine overall bargaining power
        if leverage['strength_score'] > 0.7:
            leverage['bargaining_power'] = 'strong'
        elif leverage['strength_score'] < 0.3:
            leverage['bargaining_power'] = 'weak'
        
        return leverage
    
    def _analyze_timing_opportunities(self, company_info, market_conditions: Dict) -> Dict:
        """Analyze timing-based negotiation opportunities"""
        opportunities = {
            'current_season_advantage': None,
            'quarterly_timing': None,
            'market_condition_advantage': None,
            'company_specific_timing': []
        }
        
        # Seasonal opportunities
        current_season = self._get_current_season()
        if current_season == 'off_season':
            opportunities['current_season_advantage'] = {
                'type': 'Off-peak installation season',
                'advantage': 'Potential 10-20% discount due to lower demand',
                'timeline': 'June-August optimal'
            }
        elif current_season == 'peak_season':
            opportunities['current_season_advantage'] = {
                'type': 'Peak installation season',
                'advantage': 'Better service availability but higher prices',
                'timeline': 'October-January'
            }
        
        # Quarterly timing
        current_month = datetime.now().month
        if current_month in [3, 6, 9, 12]:
            opportunities['quarterly_timing'] = {
                'type': 'Quarter-end sales push',
                'advantage': 'Companies often offer additional discounts to meet targets',
                'urgency': 'High'
            }
        
        # Market condition advantages
        if market_conditions['competition_intensity'] == 'high':
            opportunities['market_condition_advantage'] = {
                'type': 'High competition environment',
                'advantage': 'Increased vendor flexibility on pricing and terms',
                'strategy': 'Leverage competitive quotes'
            }
        
        # Company-specific timing opportunities
        performance_data = self.vendor_performance_tracker['performance_trends'].get(
            company_info['company_name'], {}
        )
        
        if performance_data.get('pricing_trend') == 'slightly_declining':
            opportunities['company_specific_timing'].append({
                'opportunity': 'Company showing pricing flexibility',
                'advantage': 'Better negotiation prospects',
                'confidence': 'Medium'
            })
        
        if performance_data.get('customer_satisfaction_trend') == 'declining':
            opportunities['company_specific_timing'].append({
                'opportunity': 'Company may be motivated to improve customer relations',
                'advantage': 'Enhanced service commitments possible',
                'confidence': 'Medium'
            })
        
        return opportunities
    
    def _develop_competitive_positioning(self, company_name: str, user_budget: float) -> Dict:
        """Develop competitive positioning strategy"""
        
        # Get comparable companies
        target_company = self.company_data[self.company_data['company_name'] == company_name].iloc[0]
        tier = target_company['tier']
        
        # Find competitors in same tier
        competitors = self.company_data[
            (self.company_data['tier'] == tier) & 
            (self.company_data['company_name'] != company_name)
        ]
        
        positioning = {
            'primary_competitors': [],
            'competitive_advantages': [],
            'competitive_weaknesses': [],
            'positioning_strategy': ''
        }
        
        # Identify top 2 competitors
        competitors_sorted = competitors.nlargest(2, 'market_share_percent')
        
        for _, competitor in competitors_sorted.iterrows():
            positioning['primary_competitors'].append({
                'name': competitor['company_name'],
                'market_share': competitor['market_share_percent'],
                'cost_advantage': competitor['cost_competitiveness'] - target_company['cost_competitiveness'],
                'service_advantage': competitor['service_quality_score'] - target_company['service_quality_score']
            })
        
        # Identify competitive advantages to leverage
        if target_company['service_quality_score'] > 8.0:
            positioning['competitive_advantages'].append('Superior service quality')
        if target_company['warranty_support_score'] > 8.5:
            positioning['competitive_advantages'].append('Excellent warranty support')
        if target_company['installation_quality_score'] > 8.5:
            positioning['competitive_advantages'].append('High installation quality')
        
        # Identify competitive weaknesses to exploit
        if target_company['cost_competitiveness'] < 7.0:
            positioning['competitive_weaknesses'].append('Higher pricing than competitors')
        if target_company['after_sales_score'] < 7.5:
            positioning['competitive_weaknesses'].append('Below-average after-sales service')
        if target_company['avg_installation_time_days'] > 14:
            positioning['competitive_weaknesses'].append('Longer installation timeline')
        
        # Generate positioning strategy
        if len(positioning['competitive_advantages']) > len(positioning['competitive_weaknesses']):
            positioning['positioning_strategy'] = 'Leverage strengths while requesting premium value justification'
        else:
            positioning['positioning_strategy'] = 'Focus on addressing weaknesses through negotiated improvements'
        
        return positioning
    
    def _calculate_negotiation_success_probability(self, company_profile: Dict, 
                                                 user_leverage: Dict,
                                                 timing_opportunities: Dict,
                                                 market_conditions: Dict) -> Dict:
        """Calculate probability of successful negotiation"""
        
        base_probability = 0.5
        
        # Company flexibility factor
        base_probability += (company_profile['flexibility_score'] - 0.5) * 0.4
        
        # User leverage factor
        leverage_impact = {
            'strong': 0.3, 'moderate': 0.1, 'weak': -0.2
        }
        base_probability += leverage_impact.get(user_leverage['bargaining_power'], 0)
        
        # Timing factors
        if timing_opportunities['current_season_advantage']:
            if 'off_season' in timing_opportunities['current_season_advantage']['type'].lower():
                base_probability += 0.15
        
        if timing_opportunities['quarterly_timing']:
            base_probability += 0.1
        
        # Market conditions
        if market_conditions['competition_intensity'] == 'high':
            base_probability += 0.2
        elif market_conditions['competition_intensity'] == 'low':
            base_probability -= 0.1
        
        # Cap probability between 0.1 and 0.9
        final_probability = max(0.1, min(0.9, base_probability))
        
        # Categorize probability
        if final_probability >= 0.7:
            category = 'High'
        elif final_probability >= 0.5:
            category = 'Moderate'
        elif final_probability >= 0.3:
            category = 'Low'
        else:
            category = 'Very Low'
        
        return {
            'probability': round(final_probability, 2),
            'category': category,
            'confidence_interval': f"{max(0, final_probability - 0.1):.1f} - {min(1.0, final_probability + 0.1):.1f}",
            'key_success_factors': self._identify_success_factors(company_profile, user_leverage, timing_opportunities)
        }
    
    def _identify_success_factors(self, company_profile: Dict, user_leverage: Dict, 
                                timing_opportunities: Dict) -> List[str]:
        """Identify key factors for negotiation success"""
        success_factors = []
        
        # Company-based factors
        if company_profile['flexibility_score'] > 0.6:
            success_factors.append('Company shows high negotiation flexibility')
        
        if 'revenue_pressure' in company_profile.get('key_motivators', []):
            success_factors.append('Company has revenue motivation')
        
        # User leverage factors
        if user_leverage['bargaining_power'] in ['strong', 'moderate']:
            success_factors.append('Strong user bargaining position')
        
        if 'Above-average project size' in user_leverage.get('leverage_points', []):
            success_factors.append('Large project size provides leverage')
        
        # Timing factors
        if timing_opportunities.get('current_season_advantage'):
            success_factors.append('Favorable seasonal timing')
        
        if timing_opportunities.get('quarterly_timing'):
            success_factors.append('Quarter-end sales pressure')
        
        return success_factors[:4]  # Return top 4 factors
    
    def _generate_negotiation_tactics(self, company_profile: Dict, user_leverage: Dict, 
                                    timing_opportunities: Dict) -> Dict:
        """Generate specific negotiation tactics"""
        tactics = {
            'opening_strategy': '',
            'price_negotiation': [],
            'service_upgrades': [],
            'contract_terms': [],
            'closing_strategy': ''
        }
        
        # Opening strategy based on company profile
        if company_profile['negotiation_style'] == 'aggressive':
            tactics['opening_strategy'] = 'Start with competitive quotes and emphasize decision timeline'
        elif company_profile['negotiation_style'] == 'conservative':
            tactics['opening_strategy'] = 'Focus on long-term relationship and value proposition'
        else:
            tactics['opening_strategy'] = 'Present thorough research and balanced comparison'
        
        # Price negotiation tactics
        if user_leverage['bargaining_power'] == 'strong':
            tactics['price_negotiation'].extend([
                'Request 15-20% discount based on project size',
                'Negotiate bulk pricing for referral commitments',
                'Ask for price matching with written competitor quotes'
            ])
        else:
            tactics['price_negotiation'].extend([
                'Request 8-12% discount for immediate decision',
                'Negotiate flexible payment terms',
                'Ask for seasonal promotional pricing'
            ])
        
        # Service upgrade tactics
        if 'Premium service positioning' in company_profile.get('constraint_factors', []):
            tactics['service_upgrades'].extend([
                'Request extended warranty at current price point',
                'Negotiate enhanced monitoring and reporting',
                'Ask for priority maintenance support'
            ])
        else:
            tactics['service_upgrades'].extend([
                'Request comprehensive maintenance package inclusion',
                'Negotiate free system monitoring for 2-3 years',
                'Ask for performance guarantee with penalties'
            ])
        
        # Contract terms
        tactics['contract_terms'].extend([
            'Include price protection clauses',
            'Negotiate flexible installation timeline',
            'Request clear performance guarantees',
            'Include service level agreements with penalties'
        ])
        
        # Closing strategy based on timing
        if timing_opportunities.get('quarterly_timing'):
            tactics['closing_strategy'] = 'Emphasize quarter-end decision with conditional agreement'
        else:
            tactics['closing_strategy'] = 'Create urgency with competitive timeline and alternative options'
        
        return tactics
    
    def _estimate_realistic_discounts(self, company_info, market_conditions: Dict, 
                                    user_leverage: Dict) -> Dict:
        """Estimate realistic discount ranges"""
        base_discount_range = [5, 12]  # Base discount expectation
        
        # Company tier adjustment
        if company_info['tier'] == 'Premium':
            if company_info['cost_competitiveness'] < 7:
                base_discount_range = [8, 18]  # Premium pricing allows more discount
            else:
                base_discount_range = [5, 12]
        elif company_info['tier'] == 'Mid-tier':
            base_discount_range = [10, 20]
        else:  # Budget tier
            base_discount_range = [3, 8]
        
        # Market conditions adjustment
        if market_conditions['competition_intensity'] == 'high':
            base_discount_range[1] += 5  # More competitive = higher discount potential
        
        # User leverage adjustment
        leverage_multiplier = {
            'strong': 1.3, 'moderate': 1.1, 'weak': 0.8
        }
        multiplier = leverage_multiplier.get(user_leverage['bargaining_power'], 1.0)
        
        adjusted_range = [
            max(2, int(base_discount_range[0] * multiplier)),
            min(25, int(base_discount_range[1] * multiplier))
        ]
        
        return {
            'price_discount_range': f"{adjusted_range[0]}-{adjusted_range[1]}%",
            'service_upgrade_value': f"₹{int((adjusted_range[0] + adjusted_range[1]) / 2 * 1000)}-₹{int((adjusted_range[0] + adjusted_range[1]) / 2 * 2000)}",
            'total_value_potential': f"{adjusted_range[0] + 2}-{adjusted_range[1] + 5}%",
            'confidence_level': 'High' if user_leverage['bargaining_power'] == 'strong' else 'Moderate'
        }
    
    def _identify_negotiation_risks(self, company_info, user_leverage: Dict) -> Dict:
        """Identify potential negotiation risks and mitigation strategies"""
        risks = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'mitigation_strategies': []
        }
        
        # Company stability risks
        if company_info['vendor_stability_risk'] > 0.4:
            risks['high_risk_factors'].append('Vendor financial stability concerns')
            risks['mitigation_strategies'].append('Request performance bonds or guarantees')
        
        if company_info['service_continuity_risk'] > 0.4:
            risks['medium_risk_factors'].append('Service continuity concerns')
            risks['mitigation_strategies'].append('Negotiate detailed service level agreements')
        
        # Negotiation position risks
        if user_leverage['bargaining_power'] == 'weak':
            risks['medium_risk_factors'].append('Limited user bargaining power')
            risks['mitigation_strategies'].append('Focus on timing advantages and service differentiation')
        
        # Market timing risks
        current_season = self._get_current_season()
        if current_season == 'peak_season':
            risks['medium_risk_factors'].append('Peak season premium pricing')
            risks['mitigation_strategies'].append('Consider deferring to off-season for better pricing')
        
        return risks
    
    def _suggest_alternative_strategies(self, company_profile: Dict, user_leverage: Dict) -> List[Dict]:
        """Suggest alternative negotiation strategies"""
        alternatives = []
        
        # If primary negotiation seems challenging
        if company_profile['flexibility_score'] < 0.4:
            alternatives.append({
                'strategy': 'Multi-vendor competitive bidding',
                'description': 'Invite formal quotes from 3-4 vendors with identical specifications',
                'success_probability': 0.75,
                'timeline': '2-3 weeks'
            })
        
        # If user has weak bargaining power
        if user_leverage['bargaining_power'] == 'weak':
            alternatives.append({
                'strategy': 'Group buying coordination',
                'description': 'Coordinate with neighbors for bulk installation discounts',
                'success_probability': 0.8,
                'timeline': '4-6 weeks'
            })
        
        # Timing-based alternatives
        current_season = self._get_current_season()
        if current_season != 'off_season':
            alternatives.append({
                'strategy': 'Deferred installation timing',
                'description': 'Schedule installation for off-peak season (June-August)',
                'success_probability': 0.9,
                'timeline': 'Wait for optimal timing'
            })
        
        return alternatives
    
    def generate_comprehensive_vendor_comparison(self, user_profile: Dict,
                                               user_preferences: Optional[Dict] = None,
                                               risk_tolerance: str = 'moderate',
                                               top_n: int = 3) -> Dict:
        """
        Generate comprehensive vendor comparison with risk integration
        
        Args:
            user_profile: Complete user profile
            user_preferences: User cluster preferences
            risk_tolerance: User's risk tolerance level
            top_n: Number of top vendors to analyze
            
        Returns:
            dict: Comprehensive comparison with risk-adjusted rankings
        """
        location = user_profile.get('location')
        
        # Get risk-adjusted rankings
        ranked_companies = self.calculate_risk_adjusted_mcda_scores(
            user_preferences, location, risk_tolerance
        )
        
        top_companies = ranked_companies.head(top_n)
        
        comparison_results = {
            'methodology': {
                'approach': 'Risk-Adjusted Multi-Criteria Decision Analysis',
                'risk_tolerance': risk_tolerance,
                'criteria_weights': self.customize_criteria_weights(user_preferences) if user_preferences else self.criteria_weights,
                'location_adjusted': location is not None
            },
            'market_context': self.market_intelligence,
            'top_vendors': [],
            'risk_comparison': {},
            'negotiation_intelligence': {},
            'decision_framework': {}
        }
        
        # Detailed analysis for each top vendor
        for idx, (_, company) in enumerate(top_companies.iterrows(), 1):
            company_analysis = {
                'rank': idx,
                'company_name': company['company_name'],
                'overall_score': round(company['risk_adjusted_score_percent'], 1),
                'risk_category': company['vendor_risk_category'],
                'tier': company['tier'],
                'market_share': f"{company['market_share_percent']:.1f}%",
                
                'strengths': self._identify_company_strengths_enhanced(company),
                'weaknesses': self._identify_company_weaknesses_enhanced(company),
                'risk_factors': self._compile_vendor_risk_factors(company),
                
                'value_proposition': self._determine_value_proposition(company),
                'best_fit_profile': self._determine_best_fit_profile_enhanced(company),
                
                'service_metrics': {
                    'installation_time': f"{company['avg_installation_time_days']} days",
                    'warranty_claim_success': f"{company['warranty_claim_success_rate']:.1%}",
                    'response_time': f"{company['maintenance_response_time_hours']} hours",
                    'service_coverage': f"{company['service_network_coverage_percent']}%"
                },
                
                'financial_indicators': {
                    'cost_per_kw': f"₹{company['avg_project_cost_per_kw']:,.0f}",
                    'annual_maintenance_cost': f"₹{company['maintenance_cost_per_kw_annual']}/kW",
                    'price_stability': f"{(1-company['price_volatility_index'])*100:.0f}%",
                    'financing_available': company['financing_options_available']
                }
            }
            
            comparison_results['top_vendors'].append(company_analysis)
        
        # Risk comparison matrix
        comparison_results['risk_comparison'] = self._create_risk_comparison_matrix(top_companies)
        
        # Negotiation intelligence for top vendors
        user_budget = user_profile.get('budget_max', 400000)
        for company in top_companies['company_name'].head(3):
            negotiation_intel = self.generate_enhanced_negotiation_intelligence(
                company, user_budget, user_profile
            )
            comparison_results['negotiation_intelligence'][company] = {
                'success_probability': negotiation_intel['success_probability'],
                'discount_expectations': negotiation_intel['discount_expectations'],
                'key_tactics': negotiation_intel['negotiation_tactics']['price_negotiation'][:2]
            }
        
        # Decision framework
        comparison_results['decision_framework'] = self._create_decision_framework(
            top_companies, user_profile, risk_tolerance
        )
        
        return comparison_results
    
    def _identify_company_strengths_enhanced(self, company) -> List[str]:
        """Enhanced company strength identification"""
        strengths = []
        
        if company['service_quality_score'] >= 8.5:
            strengths.append(f"Excellent service quality ({company['service_quality_score']}/10)")
        if company['installation_quality_score'] >= 8.5:
            strengths.append(f"Superior installation standards ({company['installation_quality_score']}/10)")
        if company['warranty_support_score'] >= 8.5:
            strengths.append(f"Outstanding warranty support ({company['warranty_support_score']}/10)")
        if company['vendor_stability_risk'] <= 0.2:
            strengths.append("Very low vendor stability risk")
        if company['service_continuity_risk'] <= 0.2:
            strengths.append("Excellent service continuity assurance")
        if company['financial_stability_rating'] in ['AA', 'A+']:
            strengths.append(f"Strong financial stability ({company['financial_stability_rating']})")
        if company['market_share_percent'] >= 10:
            strengths.append(f"Market leader with {company['market_share_percent']:.1f}% share")
        
        return strengths[:4]  # Return top 4 strengths
    
    def _identify_company_weaknesses_enhanced(self, company) -> List[str]:
        """Enhanced company weakness identification"""
        weaknesses = []
        
        if company['cost_competitiveness'] <= 7.0:
            weaknesses.append(f"Higher pricing ({company['cost_competitiveness']}/10 cost competitiveness)")
        if company['after_sales_score'] <= 7.0:
            weaknesses.append(f"Below-average after-sales service ({company['after_sales_score']}/10)")
        if company['vendor_stability_risk'] > 0.4:
            weaknesses.append("Elevated vendor stability risk")
        if company['service_continuity_risk'] > 0.4:
            weaknesses.append("Service continuity concerns")
        if company['maintenance_response_time_hours'] > 72:
            weaknesses.append(f"Slow maintenance response ({company['maintenance_response_time_hours']} hours)")
        if company['warranty_claim_success_rate'] < 0.85:
            weaknesses.append(f"Lower warranty claim success rate ({company['warranty_claim_success_rate']:.1%})")
        if not company['financing_options_available']:
            weaknesses.append("No in-house financing options")
        if company['service_network_coverage_percent'] < 70:
            weaknesses.append(f"Limited service coverage ({company['service_network_coverage_percent']}%)")
        
        return weaknesses[:4]  # Return top 4 weaknesses
    
    def _compile_vendor_risk_factors(self, company) -> Dict:
        """Compile comprehensive vendor risk factors"""
        return {
            'stability_risk': round(company['vendor_stability_risk'], 2),
            'service_risk': round(company['service_continuity_risk'], 2),
            'price_volatility': round(company['price_volatility_index'], 2),
            'warranty_reliability': round(company['warranty_claim_success_rate'], 2),
            'overall_risk_category': company['vendor_risk_category']
        }
    
    def _determine_value_proposition(self, company) -> str:
        """Determine company's primary value proposition"""
        if company['cost_competitiveness'] >= 8.0:
            return "Cost leadership with competitive quality"
        elif company['service_quality_score'] >= 8.5 and company['cost_competitiveness'] < 7.5:
            return "Premium service quality at premium pricing"
        elif company['vendor_stability_risk'] <= 0.2 and company['service_continuity_risk'] <= 0.2:
            return "Maximum reliability and risk minimization"
        elif company['technology_innovation_score'] >= 8.0:
            return "Technology innovation and advanced solutions"
        else:
            return "Balanced quality and value offering"
    
    def _determine_best_fit_profile_enhanced(self, company) -> str:
        """Enhanced determination of best-fit user profile"""
        if company['cost_competitiveness'] >= 8.0 and company['vendor_stability_risk'] <= 0.3:
            return "Budget-conscious users seeking reliable value"
        elif company['service_quality_score'] >= 8.5 and company['after_sales_score'] >= 8.0:
            return "Quality-focused users prioritizing service excellence"
        elif company['vendor_stability_risk'] <= 0.2 and company['service_continuity_risk'] <= 0.2:
            return "Risk-averse users requiring maximum reliability"
        elif company['technology_innovation_score'] >= 8.0:
            return "Tech-savvy users wanting cutting-edge solutions"
        elif company['market_share_percent'] >= 12:
            return "Conservative users preferring market leaders"
        else:
            return "Balanced users seeking overall good value"
    
    def _create_risk_comparison_matrix(self, top_companies) -> Dict:
        """Create risk comparison matrix for top companies"""
        risk_matrix = {
            'companies': list(top_companies['company_name']),
            'risk_categories': {
                'Vendor Stability': list(top_companies['vendor_stability_risk']),
                'Service Continuity': list(top_companies['service_continuity_risk']),
                'Price Volatility': list(top_companies['price_volatility_index']),
                'Warranty Reliability': list(1 - top_companies['warranty_claim_success_rate']),  # Convert to risk
            },
            'overall_risk_ranking': list(range(1, len(top_companies) + 1))
        }
        
        return risk_matrix
    
    def _create_decision_framework(self, top_companies, user_profile: Dict, risk_tolerance: str) -> Dict:
        """Create structured decision framework"""
        framework = {
            'decision_criteria_priority': [],
            'risk_tolerance_guidance': {},
            'scenario_recommendations': {},
            'final_decision_logic': []
        }
        
        # Prioritize decision criteria based on user profile and risk tolerance
        if risk_tolerance == 'low':
            framework['decision_criteria_priority'] = [
                'Vendor stability and financial strength',
                'Service continuity and support quality',
                'Warranty reliability and claim success',
                'Market reputation and track record'
            ]
        elif risk_tolerance == 'high':
            framework['decision_criteria_priority'] = [
                'Cost competitiveness and value',
                'Technology innovation and features',
                'Installation speed and efficiency',
                'Financing options and flexibility'
            ]
        else:  # moderate
            framework['decision_criteria_priority'] = [
                'Overall service quality and reliability',
                'Balanced cost and value proposition',
                'Strong warranty and after-sales support',
                'Proven track record with manageable risk'
            ]
        
        # Risk tolerance specific guidance
        framework['risk_tolerance_guidance'] = {
            'recommended_approach': f"For {risk_tolerance} risk tolerance users",
            'key_focus_areas': framework['decision_criteria_priority'],
            'negotiation_strategy': 'Conservative and relationship-focused' if risk_tolerance == 'low' 
                                  else 'Aggressive price-focused' if risk_tolerance == 'high' 
                                  else 'Balanced value optimization'
        }
        
        # Scenario-based recommendations
        top_company = top_companies.iloc[0]
        framework['scenario_recommendations'] = {
            'best_overall_choice': {
                'company': top_company['company_name'],
                'reasoning': f"Highest risk-adjusted score ({top_company['risk_adjusted_score_percent']:.1f}) with {top_company['vendor_risk_category'].lower()} profile"
            },
            'lowest_risk_choice': {
                'company': top_companies.loc[top_companies['vendor_stability_risk'].idxmin(), 'company_name'],
                'reasoning': 'Minimum vendor stability risk for maximum security'
            },
            'best_value_choice': {
                'company': top_companies.loc[top_companies['value_for_money_score'].idxmax(), 'company_name'],
                'reasoning': 'Optimal balance of cost, quality, and service value'
            }
        }
        
        return framework
    
    # Add this method to your EnhancedCompanyComparator class in intelligent_company_comparison.py

    def compare_vendors_for_pipeline(self, payload: Dict) -> Dict:
        """
        Integration method for the pipeline - processes payload and returns vendor comparison
        
        Args:
            payload: Dictionary containing user_request, sizing, risk, and heuristic data
            
        Returns:
            Dictionary with ranked vendors and comparison metadata
        """
        try:
            # Extract data from pipeline payload
            user_request = payload.get("user_request", {})
            sizing = payload.get("sizing", {})
            risk = payload.get("risk", {})
            heuristic = payload.get("heuristic", {})
            
            # Build user profile from pipeline data
            user_profile = self._build_user_profile_from_payload(user_request, sizing, risk)
            
            # Extract user preferences if available from risk tolerance and priorities
            user_preferences = self._extract_preferences_from_payload(user_request, risk)
            
            # Determine risk tolerance
            risk_tolerance = user_request.get("risk_tolerance", "moderate")
            if risk.get("overall_risk") == "High":
                risk_tolerance = "low"  # Conservative approach for high-risk scenarios
            elif risk.get("overall_risk") == "Low":
                risk_tolerance = "high"  # Can afford more aggressive choices
            
            # Run comprehensive vendor comparison
            comparison_results = self.generate_comprehensive_vendor_comparison(
                user_profile=user_profile,
                user_preferences=user_preferences,
                risk_tolerance=risk_tolerance,
                top_n=5  # Return top 5 vendors
            )
            
            # Transform results for pipeline compatibility
            ranked_vendors = []
            for vendor in comparison_results['top_vendors']:
                vendor_data = {
                    "name": vendor['company_name'],
                    "rank": vendor['rank'],
                    "overall_score": vendor['overall_score'],
                    "tier": vendor['tier'],
                    "risk_category": vendor['risk_category'],
                    "value_proposition": vendor['value_proposition'],
                    "strengths": vendor['strengths'],
                    "weaknesses": vendor['weaknesses'],
                    "service_metrics": vendor['service_metrics'],
                    "financial_indicators": vendor['financial_indicators'],
                    "best_fit_profile": vendor['best_fit_profile'],
                    "estimated_cost": self._extract_cost_estimate(vendor, sizing),
                    "warranty_years": self._extract_warranty_info(vendor),
                    "installation_time": vendor['service_metrics'].get('installation_time', 'N/A'),
                    "rating": min(5.0, vendor['overall_score'] / 20),  # Convert to 5-point scale
                    "negotiation_potential": self._assess_negotiation_potential(vendor['company_name'], user_profile)
                }
                ranked_vendors.append(vendor_data)
            
            # Add negotiation intelligence for top 3 vendors
            negotiation_intelligence = {}
            user_budget = user_profile.get('budget_max', 400000)
            
            for vendor in ranked_vendors[:3]:
                try:
                    neg_intel = self.generate_enhanced_negotiation_intelligence(
                        vendor['name'], user_budget, user_profile
                    )
                    negotiation_intelligence[vendor['name']] = {
                        'success_probability': neg_intel['success_probability']['category'],
                        'discount_range': neg_intel['discount_expectations']['price_discount_range'],
                        'key_tactics': neg_intel['negotiation_tactics']['price_negotiation'][:2],
                        'leverage_points': neg_intel['user_leverage_analysis']['leverage_points'][:3]
                    }
                except Exception as e:
                    # Fallback negotiation data
                    negotiation_intelligence[vendor['name']] = {
                        'success_probability': 'Moderate',
                        'discount_range': '8-15%',
                        'key_tactics': ['Request competitive quotes', 'Emphasize decision timeline'],
                        'leverage_points': ['Market comparison', 'Budget constraints']
                    }
            
            return {
                "companies": ranked_vendors,
                "ranked_vendors": ranked_vendors,  # Alias for compatibility
                "method": "Risk-Adjusted MCDA with Enhanced Market Intelligence",
                "comparison_metadata": {
                    "total_vendors_analyzed": len(self.company_data),
                    "risk_tolerance": risk_tolerance,
                    "methodology": comparison_results['methodology'],
                    "decision_framework": comparison_results.get('decision_framework', {}),
                    "market_context": comparison_results.get('market_context', {})
                },
                "negotiation_intelligence": negotiation_intelligence,
                "recommendations": {
                    "best_overall": ranked_vendors[0]['name'] if ranked_vendors else None,
                    "best_value": self._find_best_value_vendor(ranked_vendors),
                    "lowest_risk": self._find_lowest_risk_vendor(ranked_vendors),
                    "fastest_installation": self._find_fastest_vendor(ranked_vendors)
                }
            }
            
        except Exception as e:
            # Fallback with error information
            return {
                "error": f"Vendor comparison failed: {str(e)}",
                "companies": self._generate_fallback_vendors(payload),
                "ranked_vendors": self._generate_fallback_vendors(payload),
                "method": "fallback"
            }

    def _build_user_profile_from_payload(self, user_request: Dict, sizing: Dict, risk: Dict) -> Dict:
        """Build comprehensive user profile from pipeline payload"""
        return {
            'location': user_request.get('location', 'Unknown'),
            'state': user_request.get('state', 'Unknown'),
            'monthly_bill': user_request.get('monthly_bill', 2500),
            'budget_max': user_request.get('budget_inr', 400000),
            'roof_area': user_request.get('roof_area_m2', 100),
            'house_type': user_request.get('house_type', 'independent'),
            'income_bracket': user_request.get('income_bracket', 'Medium'),
            'risk_tolerance': user_request.get('risk_tolerance', 'moderate'),
            'timeline_preference': user_request.get('timeline_preference', 'flexible'),
            'priority': user_request.get('priority', 'cost'),
            'system_capacity_kw': sizing.get('system_capacity_kw', 5.0),
            'estimated_cost_range': sizing.get('cost_range_inr', (300000, 400000)),
            'overall_risk_level': risk.get('overall_risk', 'Moderate')
        }

    def _extract_preferences_from_payload(self, user_request: Dict, risk: Dict) -> Dict:
        """Extract user preferences for MCDA weighting"""
        preferences = {}
        
        # Map priority to weights
        priority = user_request.get('priority', 'cost').lower()
        if priority == 'cost':
            preferences['cost_weight'] = 0.35
            preferences['brand_weight'] = 0.15
            preferences['warranty_weight'] = 0.15
            preferences['service_weight'] = 0.15
            preferences['reliability_weight'] = 0.20
        elif priority == 'quality':
            preferences['cost_weight'] = 0.15
            preferences['brand_weight'] = 0.25
            preferences['warranty_weight'] = 0.25
            preferences['service_weight'] = 0.20
            preferences['reliability_weight'] = 0.15
        elif priority == 'sustainability':
            preferences['cost_weight'] = 0.20
            preferences['brand_weight'] = 0.15
            preferences['technology_weight'] = 0.25
            preferences['warranty_weight'] = 0.20
            preferences['reliability_weight'] = 0.20
        else:  # balanced
            preferences['cost_weight'] = 0.25
            preferences['brand_weight'] = 0.20
            preferences['warranty_weight'] = 0.20
            preferences['service_weight'] = 0.15
            preferences['reliability_weight'] = 0.20
        
        # Adjust based on risk level
        risk_level = risk.get('overall_risk', 'Moderate')
        if risk_level == 'High':
            # Increase reliability and service weights for high-risk scenarios
            preferences['reliability_weight'] *= 1.3
            preferences['service_weight'] *= 1.2
            preferences['cost_weight'] *= 0.8
        elif risk_level == 'Low':
            # Can focus more on cost for low-risk scenarios
            preferences['cost_weight'] *= 1.2
            preferences['reliability_weight'] *= 0.9
        
        # Normalize weights to sum to 1.0
        total_weight = sum(preferences.values())
        if total_weight > 0:
            preferences = {k: v/total_weight for k, v in preferences.items()}
        
        return preferences

    def _extract_cost_estimate(self, vendor: Dict, sizing: Dict) -> float:
        """Extract cost estimate for vendor based on sizing data"""
        try:
            # Get cost range from sizing
            cost_range = sizing.get('cost_range_inr', (300000, 400000))
            base_cost = (cost_range[0] + cost_range[1]) / 2
            
            # Get vendor's cost competitiveness score
            financial_indicators = vendor.get('financial_indicators', {})
            cost_per_kw_str = financial_indicators.get('cost_per_kw', '₹45,000')
            
            # Extract numeric value
            cost_per_kw = 45000  # default
            try:
                cost_per_kw = float(cost_per_kw_str.replace('₹', '').replace(',', ''))
            except:
                pass
            
            system_capacity = sizing.get('system_capacity_kw', 5.0)
            estimated_cost = cost_per_kw * system_capacity
            
            return estimated_cost
        except:
            return 350000  # fallback

    def _extract_warranty_info(self, vendor: Dict) -> int:
        """Extract warranty information from vendor data"""
        try:
            company_name = vendor['company_name']
            company_data = self.company_data[self.company_data['company_name'] == company_name]
            if not company_data.empty:
                return int(company_data.iloc[0]['warranty_period_panels'])
        except:
            pass
        return 25  # default

    def _assess_negotiation_potential(self, company_name: str, user_profile: Dict) -> str:
        """Assess negotiation potential with specific vendor"""
        try:
            user_budget = user_profile.get('budget_max', 400000)
            negotiation_intel = self.generate_enhanced_negotiation_intelligence(
                company_name, user_budget, user_profile
            )
            probability = negotiation_intel['success_probability']['category']
            return probability
        except:
            return 'Moderate'

    def _find_best_value_vendor(self, ranked_vendors: List[Dict]) -> Optional[str]:
        """Find vendor with best value proposition"""
        if not ranked_vendors:
            return None
        
        # Calculate value score (high overall score, lower cost)
        best_vendor = None
        best_value_score = 0
        
        for vendor in ranked_vendors:
            try:
                score = vendor['overall_score']
                cost = vendor['estimated_cost']
                # Normalize and combine (higher score, lower cost = better value)
                value_score = score - (cost / 10000)  # Rough normalization
                if value_score > best_value_score:
                    best_value_score = value_score
                    best_vendor = vendor['name']
            except:
                continue
        
        return best_vendor or ranked_vendors[0]['name']

    def _find_lowest_risk_vendor(self, ranked_vendors: List[Dict]) -> Optional[str]:
        """Find vendor with lowest risk category"""
        if not ranked_vendors:
            return None
        
        risk_priority = {'Low Risk': 1, 'Moderate Risk': 2, 'High Risk': 3, 'Very High Risk': 4}
        
        best_vendor = None
        lowest_risk = 5
        
        for vendor in ranked_vendors:
            risk_category = vendor.get('risk_category', 'Moderate Risk')
            risk_score = risk_priority.get(risk_category, 3)
            if risk_score < lowest_risk:
                lowest_risk = risk_score
                best_vendor = vendor['name']
        
        return best_vendor or ranked_vendors[0]['name']

    def _find_fastest_vendor(self, ranked_vendors: List[Dict]) -> Optional[str]:
        """Find vendor with fastest installation time"""
        if not ranked_vendors:
            return None
        
        fastest_vendor = None
        shortest_time = float('inf')
        
        for vendor in ranked_vendors:
            try:
                time_str = vendor['service_metrics'].get('installation_time', '14 days')
                # Extract days from string like "13 days" or "2-3 weeks"
                if 'week' in time_str:
                    days = int(time_str.split()[0].split('-')[-1]) * 7
                else:
                    days = int(time_str.split()[0])
                
                if days < shortest_time:
                    shortest_time = days
                    fastest_vendor = vendor['name']
            except:
                continue
        
        return fastest_vendor or ranked_vendors[0]['name']

    def _generate_fallback_vendors(self, payload: Dict) -> List[Dict]:
        """Generate fallback vendor list if comparison fails"""
        sizing = payload.get("sizing", {})
        cost_range = sizing.get('cost_range_inr', (300000, 400000))
        avg_cost = (cost_range[0] + cost_range[1]) / 2
        
        return [
            {
                "name": "Tata Power Solar",
                "rank": 1,
                "overall_score": 85.0,
                "tier": "Premium",
                "risk_category": "Low Risk",
                "estimated_cost": avg_cost,
                "warranty_years": 25,
                "installation_time": "13 days",
                "rating": 4.3,
                "negotiation_potential": "Moderate"
            },
            {
                "name": "Waaree Energies",
                "rank": 2,
                "overall_score": 82.0,
                "tier": "Premium",
                "risk_category": "Low Risk",
                "estimated_cost": avg_cost * 1.05,
                "warranty_years": 25,
                "installation_time": "11 days",
                "rating": 4.2,
                "negotiation_potential": "High"
            },
            {
                "name": "Adani Solar",
                "rank": 3,
                "overall_score": 80.0,
                "tier": "Premium",
                "risk_category": "Moderate Risk",
                "estimated_cost": avg_cost * 1.03,
                "warranty_years": 25,
                "installation_time": "12 days",
                "rating": 4.1,
                "negotiation_potential": "Moderate"
            }
        ]

# Usage example and testing
if __name__ == "__main__":
    # Initialize enhanced comparator
    comparator = EnhancedCompanyComparator()
    
    # Sample user profile
    sample_user = {
        'monthly_bill': 8000,
        'budget_max': 400000,
        'location': 'Mumbai',
        'house_type': 'apartment',
        'ownership_status': 'owner',
        'roof_area': 800,
        'income_bracket': 'middle',
        'age': 45,
        'first_time_solar_buyer': True,
        'tech_comfort_level': 'medium'
    }
    
    sample_preferences = {
        'cost_weight': 0.25,
        'brand_weight': 0.20,
        'warranty_weight': 0.15,
        'technology_weight': 0.10,
        'reliability_weight': 0.15,
        'service_weight': 0.15
    }
    
    print("=== ENHANCED VENDOR COMPARISON RESULTS ===")
    
    # Risk-adjusted MCDA rankings
    rankings = comparator.calculate_risk_adjusted_mcda_scores(
        sample_preferences, sample_user['location'], 'moderate'
    )
    
    print("\nTop 5 Risk-Adjusted Vendor Rankings:")
    for idx, (_, company) in enumerate(rankings.head().iterrows(), 1):
        print(f"{idx}. {company['company_name']} - Score: {company['risk_adjusted_score_percent']:.1f}% ({company['vendor_risk_category']})")
    
    # Comprehensive comparison
    comparison = comparator.generate_comprehensive_vendor_comparison(
        sample_user, sample_preferences, 'moderate', 3
    )
    
    print(f"\n=== TOP 3 VENDORS DETAILED ANALYSIS ===")
    for vendor in comparison['top_vendors']:
        print(f"\n{vendor['rank']}. {vendor['company_name']} ({vendor['tier']}) - {vendor['overall_score']}%")
        print(f"   Risk Category: {vendor['risk_category']}")
        print(f"   Value Proposition: {vendor['value_proposition']}")
        print(f"   Best Fit: {vendor['best_fit_profile']}")
        print(f"   Key Strengths: {', '.join(vendor['strengths'][:2])}")
        if vendor['weaknesses']:
            print(f"   Key Weaknesses: {', '.join(vendor['weaknesses'][:2])}")
    
    # Negotiation intelligence sample
    print(f"\n=== NEGOTIATION INTELLIGENCE SAMPLE ===")
    top_vendor = comparison['top_vendors'][0]['company_name']
    negotiation_intel = comparator.generate_enhanced_negotiation_intelligence(
        top_vendor, sample_user['budget_max'], sample_user
    )
    
    print(f"Vendor: {top_vendor}")
    print(f"Success Probability: {negotiation_intel['success_probability']['category']} ({negotiation_intel['success_probability']['probability']})")
    print(f"Expected Discount Range: {negotiation_intel['discount_expectations']['price_discount_range']}")
    print(f"User Leverage: {negotiation_intel['user_leverage_analysis']['bargaining_power'].title()} bargaining power")
    
    if negotiation_intel['timing_opportunities']['current_season_advantage']:
        timing_adv = negotiation_intel['timing_opportunities']['current_season_advantage']
        print(f"Timing Advantage: {timing_adv['type']} - {timing_adv['advantage']}")
    
    print(f"\n=== DECISION FRAMEWORK ===")
    framework = comparison['decision_framework']
    print(f"Recommended Approach: {framework['risk_tolerance_guidance']['negotiation_strategy']}")
    print("Decision Priority:")
    for i, criterion in enumerate(framework['decision_criteria_priority'], 1):
        print(f"  {i}. {criterion}")
    
    print(f"\nBest Overall Choice: {framework['scenario_recommendations']['best_overall_choice']['company']}")
    print(f"Reasoning: {framework['scenario_recommendations']['best_overall_choice']['reasoning']}")