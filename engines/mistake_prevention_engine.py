# engines/enhanced_mistake_prevention_engine.py
"""
Enhanced Mistake Prevention Engine with Social & Behavioral Barriers
Fixed version with proper safety gate integration and all missing functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class SafetyGateOut:
    """Safety gate output wrapper for integration compatibility"""
    
    def __init__(self, data: Any, status: str = "PASS", issues: List[str] = None, critical_issues: List[str] = None):
        self._data = data
        self.status = status
        self.issues = issues or []
        self.critical_issues = critical_issues or []
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data from wrapped object or nested wrapper"""
        if isinstance(self._data, dict):
            return self._data.get(key, default)
        elif isinstance(self._data, SafetyGateOut):  # handle double-wrapping
            return self._data.get(key, default)
        elif hasattr(self._data, key):
            return getattr(self._data, key, default)
        else:
            return default


    def __getitem__(self, key):
        if isinstance(self._data, dict):
            return self._data[key]
        raise KeyError(f"Key '{key}' not found")

    def __contains__(self, key):
        if isinstance(self._data, dict):
            return key in self._data
        return hasattr(self._data, key)

    def __iter__(self):
        if isinstance(self._data, dict):
            return iter(self._data)
        return iter([])

    def keys(self):
        if isinstance(self._data, dict):
            return self._data.keys()
        return []

    def items(self):
        if isinstance(self._data, dict):
            return self._data.items()
        return []

    def values(self):
        if isinstance(self._data, dict):
            return self._data.values()
        return []

    def __getattr__(self, item):
        """Fallback: let integration treat wrapper like its data"""
        if isinstance(self._data, dict) and item in self._data:
            return self._data[item]
        raise AttributeError(f"'SafetyGateOut' has no attribute '{item}'")

    @property
    def data(self):
        return self._data


class EnhancedMistakePreventionEngine:
    """Advanced mistake prevention with social & behavioral barrier solutions"""
    
    def __init__(self):
        self.common_mistakes = self._load_common_mistakes()
        self.validation_rules = self._setup_validation_rules()
        self.confidence_thresholds = self._setup_confidence_thresholds()
        self.safety_buffers = self._setup_safety_buffers()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # New: Social & Behavioral Components
        self.social_benchmarks = self._load_social_benchmarks()
        self.educational_content = self._load_educational_content()
        self.gamification_system = self._setup_gamification_system()
        self.behavioral_patterns = self._load_behavioral_patterns()
        
        self._setup_historical_patterns()
    
    def _load_social_benchmarks(self):
        """Load social benchmarks for peer comparison and influence"""
        return {
            'location_averages': {
                'Delhi': {
                    'avg_system_size': 4.2,
                    'avg_savings': 28000,
                    'adoption_rate': 0.18,
                    'satisfaction_score': 4.3,
                    'payback_period': 6.8,
                    'popular_brands': ['Tata Solar', 'Adani Solar', 'Vikram Solar']
                },
                'Mumbai': {
                    'avg_system_size': 3.8,
                    'avg_savings': 24000,
                    'adoption_rate': 0.15,
                    'satisfaction_score': 4.1,
                    'payback_period': 7.2,
                    'popular_brands': ['Waaree', 'Luminous', 'Tata Solar']
                },
                'Bangalore': {
                    'avg_system_size': 4.5,
                    'avg_savings': 32000,
                    'adoption_rate': 0.22,
                    'satisfaction_score': 4.5,
                    'payback_period': 6.2,
                    'popular_brands': ['Adani Solar', 'Canadian Solar', 'Jinko Solar']
                },
                'Chennai': {
                    'avg_system_size': 4.0,
                    'avg_savings': 30000,
                    'adoption_rate': 0.20,
                    'satisfaction_score': 4.2,
                    'payback_period': 6.5,
                    'popular_brands': ['Vikram Solar', 'Tata Solar', 'Renewsys']
                }
            },
            'income_group_benchmarks': {
                'middle_class': {
                    'typical_system_size': 3.5,
                    'budget_range': (200000, 400000),
                    'decision_timeline': '2-4 months',
                    'main_motivations': ['savings', 'environment', 'energy_independence']
                },
                'upper_middle_class': {
                    'typical_system_size': 5.0,
                    'budget_range': (350000, 700000),
                    'decision_timeline': '1-3 months',
                    'main_motivations': ['savings', 'technology', 'status']
                },
                'affluent': {
                    'typical_system_size': 7.5,
                    'budget_range': (600000, 1200000),
                    'decision_timeline': '3-6 weeks',
                    'main_motivations': ['environment', 'technology', 'independence']
                }
            },
            'house_type_benchmarks': {
                'independent': {
                    'adoption_rate': 0.25,
                    'avg_satisfaction': 4.4,
                    'common_sizes': [3, 5, 7, 10],
                    'maintenance_complexity': 'Low'
                },
                'villa': {
                    'adoption_rate': 0.35,
                    'avg_satisfaction': 4.6,
                    'common_sizes': [5, 7, 10, 15],
                    'maintenance_complexity': 'Low'
                },
                'apartment': {
                    'adoption_rate': 0.08,
                    'avg_satisfaction': 3.9,
                    'common_sizes': [1, 2, 3],
                    'maintenance_complexity': 'Medium'
                }
            }
        }
    
    def _load_educational_content(self):
        """Load educational content to address complexity fears"""
        return {
            'complexity_demystification': {
                'system_basics': {
                    'title': 'Solar System Basics - Simpler Than You Think!',
                    'content': [
                        'A solar system has just 4 main parts: panels, inverter, meter, and monitoring',
                        'Installation typically takes 1-2 days for residential systems',
                        'Modern systems are designed for 25+ years of maintenance-free operation',
                        'Smart monitoring lets you track performance from your phone'
                    ],
                    'complexity_score': 2,  # Out of 10
                    'time_to_understand': '10 minutes'
                },
                'maintenance_reality': {
                    'title': 'Maintenance Reality Check',
                    'content': [
                        'Annual maintenance: Just cleaning panels 2-3 times per year',
                        'No moving parts = virtually no mechanical failures',
                        'Most issues are automatically detected by monitoring systems',
                        'Average annual maintenance cost: ₹2,000-3,000'
                    ],
                    'complexity_score': 3,
                    'time_to_understand': '5 minutes'
                },
                'financial_clarity': {
                    'title': 'Financial Planning Made Simple',
                    'content': [
                        'Think of it as prepaying your electricity bills for 25 years',
                        'Monthly EMI typically equals current electricity bill',
                        'Savings start from day one with zero-down financing',
                        'Government subsidies reduce upfront costs by 20-40%'
                    ],
                    'complexity_score': 4,
                    'time_to_understand': '8 minutes'
                }
            },
            'fear_addressing': {
                'technology_reliability': {
                    'myth': 'Solar technology is unreliable and experimental',
                    'reality': 'Solar panels are 150+ year old technology, perfected over decades',
                    'proof_points': [
                        'NASA uses solar for space missions since 1958',
                        '25-year warranties are standard industry practice',
                        'Panels often produce power for 30-40 years',
                        'Over 100 GW installed in India alone'
                    ]
                },
                'weather_concerns': {
                    'myth': 'Solar doesn\'t work in monsoons or winter',
                    'reality': 'Solar works in all weather, just at different efficiencies',
                    'proof_points': [
                        'Germany leads solar adoption with less sunshine than India',
                        'Cloudy days still generate 20-40% power',
                        'Cool winter weather actually improves panel efficiency',
                        'Annual generation is what matters, not daily variations'
                    ]
                },
                'installation_fears': {
                    'myth': 'Installation will damage my roof and home',
                    'reality': 'Professional installation actually protects your roof',
                    'proof_points': [
                        'Panels act as roof protection from weather',
                        'Mounting systems are designed for 150+ kmph winds',
                        'Installation uses existing roof structure safely',
                        'Insurance covers any installation issues'
                    ]
                }
            },
            'success_stories': {
                'peer_experiences': [
                    {
                        'profile': 'IT Professional, Bangalore',
                        'system_size': '5 kW',
                        'experience': 'Installation in 1 day, savings ₹35,000/year, zero maintenance issues in 3 years',
                        'quote': 'Wish I had done this 5 years earlier!'
                    },
                    {
                        'profile': 'Retired Teacher, Delhi',
                        'system_size': '3 kW',
                        'experience': 'EMI ₹2,800/month, electricity bill reduced from ₹4,500 to ₹800',
                        'quote': 'My children are amazed at how simple and effective it is'
                    },
                    {
                        'profile': 'Small Business Owner, Chennai',
                        'system_size': '10 kW',
                        'experience': 'Business electricity costs down 85%, payback in 5.2 years',
                        'quote': 'Best business investment I ever made'
                    }
                ]
            }
        }
    
    def _setup_gamification_system(self):
        """Setup gamification elements for engagement and motivation"""
        return {
            'achievement_levels': {
                'solar_explorer': {
                    'threshold': 0,
                    'title': '🌟 Solar Explorer',
                    'description': 'Taking the first step towards solar energy',
                    'benefits': ['Basic solar education', 'Cost calculator access']
                },
                'solar_planner': {
                    'threshold': 25,
                    'title': '🎯 Solar Planner',
                    'description': 'Serious about planning your solar journey',
                    'benefits': ['Detailed system design', 'Installer recommendations']
                },
                'solar_champion': {
                    'threshold': 50,
                    'title': '🏆 Solar Champion',
                    'description': 'Ready to make the solar switch',
                    'benefits': ['Premium support', 'Exclusive discounts', 'Fast-track processing']
                },
                'solar_advocate': {
                    'threshold': 75,
                    'title': '🌞 Solar Advocate',
                    'description': 'Inspiring others to go solar',
                    'benefits': ['Referral rewards', 'Community recognition', 'Expert consultations']
                }
            },
            'progress_metrics': {
                'education_completion': {
                    'weight': 30,
                    'activities': ['basics_learned', 'myths_busted', 'stories_read']
                },
                'planning_progress': {
                    'weight': 40,
                    'activities': ['profile_completed', 'quotes_received', 'site_assessed']
                },
                'decision_readiness': {
                    'weight': 30,
                    'activities': ['financing_arranged', 'installer_selected', 'timeline_set']
                }
            },
            'rewards_system': {
                'points_earning': {
                    'profile_completion': 10,
                    'education_module_completion': 15,
                    'quote_request': 20,
                    'installer_meeting': 25,
                    'system_installation': 100,
                    'referral_success': 50
                },
                'redemption_options': [
                    {'points': 100, 'reward': '₹1,000 installation discount'},
                    {'points': 200, 'reward': 'Premium monitoring system upgrade'},
                    {'points': 300, 'reward': 'Extended warranty (2 years)'},
                    {'points': 500, 'reward': '₹5,000 cash back post-installation'}
                ]
            }
        }
    
    def _load_behavioral_patterns(self):
        """Load behavioral patterns to predict and address decision-making barriers"""
        return {
            'decision_styles': {
                'analytical': {
                    'characteristics': ['data_driven', 'detailed_comparison', 'slow_decision'],
                    'content_preference': 'detailed_technical_specs',
                    'trust_factors': ['certifications', 'performance_data', 'warranties'],
                    'persuasion_approach': 'provide_comprehensive_analysis'
                },
                'intuitive': {
                    'characteristics': ['gut_feeling', 'quick_decision', 'experience_focused'],
                    'content_preference': 'success_stories',
                    'trust_factors': ['peer_recommendations', 'brand_reputation', 'installer_credibility'],
                    'persuasion_approach': 'highlight_social_proof'
                },
                'cautious': {
                    'characteristics': ['risk_averse', 'extensive_research', 'multiple_opinions'],
                    'content_preference': 'risk_mitigation_strategies',
                    'trust_factors': ['guarantees', 'insurance', 'track_record'],
                    'persuasion_approach': 'emphasize_safety_and_guarantees'
                },
                'ambitious': {
                    'characteristics': ['early_adopter', 'technology_enthusiast', 'status_conscious'],
                    'content_preference': 'latest_technology_features',
                    'trust_factors': ['innovation', 'efficiency', 'smart_features'],
                    'persuasion_approach': 'highlight_cutting_edge_benefits'
                }
            },
            'barrier_indicators': {
                'complexity_fear': {
                    'signals': ['multiple_quote_requests', 'long_decision_timeline', 'basic_questions'],
                    'intervention': 'simplification_and_education',
                    'content': 'complexity_demystification'
                },
                'financial_anxiety': {
                    'signals': ['budget_constraints', 'emi_concerns', 'roi_focus'],
                    'intervention': 'financial_planning_support',
                    'content': 'financing_options_and_benefits'
                },
                'social_pressure': {
                    'signals': ['peer_comparison_requests', 'neighborhood_references', 'status_concerns'],
                    'intervention': 'social_validation_and_benchmarking',
                    'content': 'peer_success_stories'
                },
                'technology_skepticism': {
                    'signals': ['reliability_questions', 'warranty_focus', 'failure_concerns'],
                    'intervention': 'credibility_building',
                    'content': 'technology_reliability_proof'
                }
            }
        }
    
    def _load_common_mistakes(self):
        """Load database of common solar installation mistakes"""
        return {
            'undersizing': {
                'description': 'System too small for actual consumption',
                'frequency': 0.45,
                'avg_cost_impact': 24000,
                'indicators': ['consumption_underestimated', 'seasonal_variance_ignored', 'growth_not_considered']
            },
            'oversizing': {
                'description': 'System too large leading to poor ROI',
                'frequency': 0.25,
                'avg_cost_impact': 150000,
                'indicators': ['consumption_overestimated', 'optimistic_generation', 'policy_changes_ignored']
            },
            'wrong_component_selection': {
                'description': 'Inappropriate panels/inverters for location',
                'frequency': 0.35,
                'avg_cost_impact': 80000,
                'indicators': ['climate_mismatch', 'efficiency_overestimated', 'durability_underestimated']
            },
            'installation_site_issues': {
                'description': 'Roof/location not suitable for optimal performance',
                'frequency': 0.30,
                'avg_cost_impact': 50000,
                'indicators': ['shading_underestimated', 'structural_issues', 'orientation_suboptimal']
            },
            'financial_miscalculation': {
                'description': 'Hidden costs or unrealistic payback expectations',
                'frequency': 0.40,
                'avg_cost_impact': 75000,
                'indicators': ['hidden_costs_ignored', 'tariff_changes_ignored', 'maintenance_underbudgeted']
            }
        }
    
    def _setup_validation_rules(self):
        """Setup validation rules to catch common errors"""
        return {
            'sizing_validation': {
                'min_coverage_ratio': 0.7,
                'max_coverage_ratio': 1.3,
                'peak_demand_buffer': 0.2,
                'seasonal_adjustment': 0.15
            },
            'financial_validation': {
                'max_emi_ratio': 0.25,
                'min_payback_period': 4.0,
                'max_payback_period': 12.0,
                'roi_threshold': 0.12
            },
            'technical_validation': {
                'min_roof_utilization': 0.6,
                'max_roof_utilization': 0.9,
                'min_system_efficiency': 0.75,
                'inverter_sizing_ratio': (0.8, 1.2)
            },
            'location_validation': {
                'minimum_irradiance': 4.0,
                'shading_tolerance': 0.15,
                'structural_load_factor': 1.5
            }
        }
    
    def _setup_confidence_thresholds(self):
        """Setup confidence levels for different prediction types"""
        return {
            'cost_estimation': {
                'high_confidence': 0.90,
                'medium_confidence': 0.75,
                'low_confidence': 0.60
            },
            'generation_prediction': {
                'high_confidence': 0.85,
                'medium_confidence': 0.70,
                'low_confidence': 0.55
            },
            'roi_calculation': {
                'high_confidence': 0.80,
                'medium_confidence': 0.65,
                'low_confidence': 0.50
            }
        }
    
    def _setup_safety_buffers(self):
        """Setup safety buffers for different components"""
        return {
            'sizing_buffer': {
                'conservative': 0.20,
                'moderate': 0.15,
                'aggressive': 0.10
            },
            'cost_buffer': {
                'conservative': 0.15,
                'moderate': 0.10,
                'aggressive': 0.05
            },
            'generation_buffer': {
                'conservative': 0.25,
                'moderate': 0.20,
                'aggressive': 0.15
            }
        }
    
    def _setup_historical_patterns(self):
        """Setup historical patterns for anomaly detection"""
        np.random.seed(42)
        normal_patterns = np.random.normal(0, 1, (1000, 8))
        self.historical_patterns = normal_patterns
        self.anomaly_detector.fit(normal_patterns)
    
    # Main integration methods that return SafetyGateOut objects
    
    def detect_sizing_mistakes(self, user_profile: Dict, system_recommendation: Dict) -> SafetyGateOut:
        """Detect potential sizing mistakes before they happen"""
        try:
            mistakes_detected = []
            confidence_issues = []
            
            monthly_bill = user_profile.get('monthly_bill', 0)
            system_size = system_recommendation.get('recommended_capacity_kw', 0)
            estimated_generation = system_recommendation.get('estimated_annual_generation', 0)
            
            # Check for undersizing
            if monthly_bill > 0:
                estimated_annual_consumption = monthly_bill * 12 / 5  # Rough conversion
                coverage_ratio = estimated_generation / estimated_annual_consumption if estimated_annual_consumption > 0 else 0
                
                if coverage_ratio < 0.7:
                    mistakes_detected.append({
                        'type': 'undersizing',
                        'severity': 'high',
                        'description': f'System covers only {coverage_ratio:.1%} of consumption',
                        'potential_loss': f"₹{int((estimated_annual_consumption - estimated_generation) * 5):,} annually",
                        'recommendation': 'Increase system size for better coverage'
                    })
                
                if coverage_ratio > 1.3:
                    mistakes_detected.append({
                        'type': 'oversizing',
                        'severity': 'medium',
                        'description': f'System oversized by {(coverage_ratio - 1.0):.1%}',
                        'potential_loss': f"₹{int((coverage_ratio - 1.0) * estimated_annual_consumption * 25):,} excess investment",
                        'recommendation': 'Consider reducing system size for better ROI'
                    })
            
            result_data = {
                'mistakes_detected': mistakes_detected,
                'confidence_issues': confidence_issues,
                'risk_level': self._assess_sizing_risk_level(mistakes_detected),
                'confidence_score': self._calculate_sizing_confidence(mistakes_detected, confidence_issues)
            }
            
            # Determine if there are critical issues
            critical_issues = [m['description'] for m in mistakes_detected if m['severity'] == 'high']
            status = "FAIL" if critical_issues else "PASS"
            
            return SafetyGateOut(result_data, status, [], critical_issues)
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'mistakes_detected': [], 'confidence_score': 0.0},
                "FAIL",
                [f"Error in sizing analysis: {str(e)}"],
                [f"Critical error: {str(e)}"]
            )
    
    def calculate_confidence_score_with_social_factors(self, user_profile: Dict, 
                                                     system_recommendation: Dict) -> SafetyGateOut:
        """Enhanced confidence scoring with social and behavioral factors"""
        try:
            # Base technical confidence
            base_confidence = self._calculate_base_technical_confidence(user_profile, system_recommendation)
            
            # Social confidence factors
            social_confidence = self._calculate_social_confidence(user_profile, system_recommendation)
            
            # Behavioral readiness
            behavioral_readiness = self._assess_behavioral_readiness(user_profile)
            
            # Educational completion impact
            education_impact = self._calculate_education_impact(user_profile)
            
            # Peer influence factor
            peer_influence = self._calculate_peer_influence_factor(user_profile)
            
            # Combined confidence score
            weights = {
                'technical': 0.35,
                'social': 0.25,
                'behavioral': 0.20,
                'educational': 0.10,
                'peer_influence': 0.10
            }
            
            overall_confidence = (
                base_confidence * weights['technical'] +
                social_confidence * weights['social'] +
                behavioral_readiness * weights['behavioral'] +
                education_impact * weights['educational'] +
                peer_influence * weights['peer_influence']
            )
            
            confidence_breakdown = {
                'overall_confidence': round(overall_confidence, 3),
                'confidence_level': self._categorize_confidence_level(overall_confidence),
                'component_scores': {
                    'technical_confidence': round(base_confidence, 3),
                    'social_confidence': round(social_confidence, 3),
                    'behavioral_readiness': round(behavioral_readiness, 3),
                    'educational_impact': round(education_impact, 3),
                    'peer_influence': round(peer_influence, 3)
                },
                'confidence_drivers': self._identify_confidence_drivers(
                    base_confidence, social_confidence, behavioral_readiness, 
                    education_impact, peer_influence
                ),
                'confidence_barriers': self._identify_confidence_barriers(
                    base_confidence, social_confidence, behavioral_readiness,
                    education_impact, peer_influence
                )
            }
            
            # Determine status based on confidence level
            status = "PASS" if overall_confidence >= 0.6 else "FAIL"
            issues = confidence_breakdown['confidence_barriers']
            critical_issues = [barrier for barrier in issues if 'Technical' in barrier or overall_confidence < 0.4]
            
            return SafetyGateOut(confidence_breakdown, status, issues, critical_issues)
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'overall_confidence': 0.0},
                "FAIL",
                [f"Error in confidence calculation: {str(e)}"],
                [f"Critical error: {str(e)}"]
            )
    
    def detect_and_address_behavioral_barriers(self, user_profile: Dict, 
                                             user_behavior: Dict = None) -> SafetyGateOut:
        """Detect behavioral barriers and provide targeted interventions"""
        try:
            if user_behavior is None:
                user_behavior = {}
            
            # Analyze user behavior patterns
            behavioral_analysis = self._analyze_user_behavior_patterns(user_profile, user_behavior)
            
            # Identify specific barriers
            identified_barriers = []
            
            # Decision style analysis
            decision_style = self._identify_decision_style(user_behavior)
            
            # Detect various barriers
            if self._detect_complexity_fear(user_behavior):
                identified_barriers.append({
                    'barrier': 'complexity_fear',
                    'confidence_impact': -0.25,
                    'intervention': 'simplification_and_education',
                    'priority': 'high'
                })
            
            if self._detect_financial_anxiety(user_profile, user_behavior):
                identified_barriers.append({
                    'barrier': 'financial_anxiety',
                    'confidence_impact': -0.20,
                    'intervention': 'financial_planning_support',
                    'priority': 'high'
                })
            
            if self._detect_social_pressure_needs(user_behavior):
                identified_barriers.append({
                    'barrier': 'social_validation_needed',
                    'confidence_impact': -0.15,
                    'intervention': 'peer_comparison_and_social_proof',
                    'priority': 'medium'
                })
            
            # Generate targeted interventions
            interventions = self._generate_behavioral_interventions(identified_barriers, decision_style)
            
            result_data = {
                'behavioral_analysis': behavioral_analysis,
                'decision_style': decision_style,
                'barriers_identified': identified_barriers,
                'total_confidence_impact': sum([barrier['confidence_impact'] for barrier in identified_barriers]),
                'targeted_interventions': interventions,
                'success_probability': self._calculate_success_probability_with_interventions(identified_barriers),
                'timeline_adjustments': self._suggest_timeline_adjustments(identified_barriers)
            }
            
            # Determine status
            high_priority_barriers = [b for b in identified_barriers if b['priority'] == 'high']
            status = "FAIL" if len(high_priority_barriers) >= 2 else "PASS"
            issues = [f"{b['barrier']} ({b['priority']} priority)" for b in identified_barriers]
            critical_issues = [f"{b['barrier']}" for b in high_priority_barriers]
            
            return SafetyGateOut(result_data, status, issues, critical_issues)
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'barriers_identified': []},
                "FAIL",
                [f"Error in behavioral analysis: {str(e)}"],
                [f"Critical error: {str(e)}"]
            )
    
    def generate_comprehensive_social_behavioral_report(self, user_profile: Dict, 
                                                      system_recommendation: Dict,
                                                      user_actions: Dict = None,
                                                      user_behavior: Dict = None) -> SafetyGateOut:
        """Generate comprehensive report addressing social and behavioral aspects"""
        try:
            if user_actions is None:
                user_actions = {}
            if user_behavior is None:
                user_behavior = {}
            
            # Get component analyses
            confidence_assessment = self.calculate_confidence_score_with_social_factors(
                user_profile, system_recommendation
            )
            
            educational_insights = self.generate_educational_insights(
                user_profile, confidence_assessment.data
            )
            
            gamification_profile = self.generate_gamification_profile(
                user_profile, system_recommendation, user_actions
            )
            
            behavioral_analysis = self.detect_and_address_behavioral_barriers(
                user_profile, user_behavior
            )
            
            # Generate executive summary
            executive_summary = self._generate_social_behavioral_executive_summary(
                confidence_assessment.data, educational_insights, gamification_profile, behavioral_analysis.data
            )
            
            # Create action plan
            action_plan = self._create_social_behavioral_action_plan(
                confidence_assessment.data, educational_insights, gamification_profile, behavioral_analysis.data
            )
            
            result_data = {
                'report_metadata': {
                    'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_version': '2.0_social_behavioral',
                    'confidence_level': confidence_assessment.data['overall_confidence']
                },
                'executive_summary': executive_summary,
                'confidence_assessment': confidence_assessment.data,
                'educational_insights': educational_insights,
                'gamification_profile': gamification_profile,
                'behavioral_analysis': behavioral_analysis.data,
                'action_plan': action_plan,
                'success_enhancement_recommendations': self._generate_success_enhancement_recommendations(
                    confidence_assessment.data, behavioral_analysis.data
                )
            }
            
            # Determine overall status
            overall_confidence = confidence_assessment.data['overall_confidence']
            barriers_count = len(behavioral_analysis.data.get('barriers_identified', []))
            
            if overall_confidence >= 0.8 and barriers_count <= 1:
                status = "PASS"
                issues = []
                critical_issues = []
            elif overall_confidence >= 0.6 and barriers_count <= 2:
                status = "PASS"
                issues = [f"{barriers_count} behavioral barriers identified"]
                critical_issues = []
            else:
                status = "FAIL"
                issues = [f"Low confidence ({overall_confidence:.1%})", f"{barriers_count} barriers identified"]
                critical_issues = ["Comprehensive intervention required"]
            
            return SafetyGateOut(result_data, status, issues, critical_issues)
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e)},
                "FAIL",
                [f"Error generating comprehensive report: {str(e)}"],
                [f"Critical error: {str(e)}"]
            )
    
    # Helper methods for calculations and analysis
    
    def _calculate_base_technical_confidence(self, user_profile: Dict, system_recommendation: Dict) -> float:
        """Calculate base technical confidence score"""
        sizing_result = self.detect_sizing_mistakes(user_profile, system_recommendation)
        return sizing_result.data.get('confidence_score', 0.8)
    
    def _calculate_social_confidence(self, user_profile: Dict, system_recommendation: Dict) -> float:
        """Calculate social confidence based on peer benchmarks and social proof"""
        location = user_profile.get('location', 'General')
        house_type = user_profile.get('house_type', 'independent')
        system_size = system_recommendation.get('recommended_capacity_kw', 3)
        
        social_confidence = 0.5  # Base score
        
        # Location benchmark alignment
        if location in self.social_benchmarks['location_averages']:
            location_data = self.social_benchmarks['location_averages'][location]
            size_alignment = 1 - abs(system_size - location_data['avg_system_size']) / max(location_data['avg_system_size'], 1)
            social_confidence += min(0.3, size_alignment * 0.3)
            
            # High adoption rate in area boosts confidence
            adoption_boost = location_data['adoption_rate'] * 0.2
            social_confidence += adoption_boost
        
        # House type benchmark alignment
        if house_type in self.social_benchmarks['house_type_benchmarks']:
            house_data = self.social_benchmarks['house_type_benchmarks'][house_type]
            adoption_rate_boost = house_data['adoption_rate'] * 0.15
            satisfaction_boost = (house_data['avg_satisfaction'] - 3.0) * 0.05
            social_confidence += adoption_rate_boost + satisfaction_boost
        
        return min(1.0, social_confidence)
    
    def _assess_behavioral_readiness(self, user_profile: Dict) -> float:
        """Assess behavioral readiness based on profile indicators"""
        readiness_score = 0.6  # Base score
        
        # Budget clarity boosts readiness
        if user_profile.get('budget_max', 0) > 0:
            readiness_score += 0.15
        
        # Complete profile indicates serious intent
        profile_completeness = self._calculate_profile_completeness(user_profile)
        readiness_score += profile_completeness * 0.25
        
        return min(1.0, readiness_score)
    
    def _calculate_education_impact(self, user_profile: Dict) -> float:
        """Calculate impact of educational content completion"""
        education_completion = user_profile.get('education_modules_completed', 0)
        return min(1.0, 0.4 + (education_completion * 0.15))
    
    def _calculate_peer_influence_factor(self, user_profile: Dict) -> float:
        """Calculate peer influence factor"""
        location = user_profile.get('location', 'General')
        
        if location in self.social_benchmarks['location_averages']:
            location_data = self.social_benchmarks['location_averages'][location]
            adoption_influence = location_data['adoption_rate'] * 0.8
            satisfaction_influence = (location_data['satisfaction_score'] - 3.0) * 0.15
            return min(1.0, 0.3 + adoption_influence + satisfaction_influence)
        
        return 0.5  # Neutral if no location data
    
    def _categorize_confidence_level(self, confidence_score: float) -> str:
        """Categorize overall confidence level"""
        if confidence_score >= 0.85:
            return "Very High Confidence - Ready to Proceed"
        elif confidence_score >= 0.70:
            return "High Confidence - Minor Concerns"
        elif confidence_score >= 0.55:
            return "Moderate Confidence - Some Barriers"
        elif confidence_score >= 0.40:
            return "Low Confidence - Significant Barriers"
        else:
            return "Very Low Confidence - Major Intervention Needed"
    
    def _identify_confidence_drivers(self, technical: float, social: float, 
                                   behavioral: float, educational: float, peer: float) -> List[str]:
        """Identify top confidence drivers"""
        drivers = []
        scores = {
            'Strong technical validation': technical,
            'Positive peer influence': peer,
            'Good social benchmarking': social,
            'High behavioral readiness': behavioral,
            'Effective education': educational
        }
        
        # Get top 3 drivers
        sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for driver, score in sorted_drivers:
            if score >= 0.7:
                drivers.append(f"{driver} (Score: {score:.2f})")
        
        return drivers
    
    def _identify_confidence_barriers(self, technical: float, social: float,
                                    behavioral: float, educational: float, peer: float) -> List[str]:
        """Identify confidence barriers that need attention"""
        barriers = []
        scores = {
            'Technical concerns need addressing': technical,
            'Peer influence lacking': peer,
            'Social validation needed': social,
            'Behavioral barriers present': behavioral,
            'Educational gaps exist': educational
        }
        
        for barrier, score in scores.items():
            if score < 0.6:
                barriers.append(f"{barrier} (Score: {score:.2f})")
        
        return barriers
    
    def _assess_sizing_risk_level(self, mistakes_detected: List[Dict]) -> str:
        """Assess risk level based on sizing mistakes"""
        high_severity = len([m for m in mistakes_detected if m.get('severity') == 'high'])
        medium_severity = len([m for m in mistakes_detected if m.get('severity') == 'medium'])
        
        if high_severity >= 2:
            return "Very High Risk"
        elif high_severity >= 1:
            return "High Risk"
        elif medium_severity >= 2:
            return "Moderate Risk"
        elif medium_severity >= 1:
            return "Low Risk"
        else:
            return "Minimal Risk"
    
    def _calculate_sizing_confidence(self, mistakes_detected: List[Dict], confidence_issues: List[Dict]) -> float:
        """Calculate confidence score for sizing (0-1)"""
        base_confidence = 1.0
        
        for mistake in mistakes_detected:
            if mistake.get('severity') == 'high':
                base_confidence -= 0.25
            elif mistake.get('severity') == 'medium':
                base_confidence -= 0.15
        
        base_confidence -= len(confidence_issues) * 0.05
        return max(0.0, min(1.0, base_confidence))
    
    def _analyze_user_behavior_patterns(self, user_profile: Dict, user_behavior: Dict) -> Dict:
        """Analyze user behavior patterns"""
        behavior_analysis = {
            'engagement_level': 'medium',
            'decision_speed': 'deliberate',
            'information_seeking': 'moderate',
            'social_influence_susceptibility': 'medium',
            'risk_tolerance': 'moderate'
        }
        
        # Analyze based on behavior data
        page_views = user_behavior.get('page_views', 5)
        time_spent = user_behavior.get('total_time_spent_minutes', 30)
        questions_asked = user_behavior.get('questions_asked', 2)
        
        # Engagement level
        if time_spent > 60 and page_views > 10:
            behavior_analysis['engagement_level'] = 'high'
        elif time_spent < 15 and page_views < 3:
            behavior_analysis['engagement_level'] = 'low'
        
        # Decision speed
        quote_requests = user_behavior.get('quote_requests', 1)
        if quote_requests > 0 and time_spent < 30:
            behavior_analysis['decision_speed'] = 'fast'
        elif questions_asked > 5 and time_spent > 90:
            behavior_analysis['decision_speed'] = 'slow'
        
        # Information seeking
        if questions_asked > 8 or page_views > 15:
            behavior_analysis['information_seeking'] = 'heavy'
        elif questions_asked < 2 and page_views < 5:
            behavior_analysis['information_seeking'] = 'light'
        
        return behavior_analysis
    
    def _identify_decision_style(self, user_behavior: Dict) -> str:
        """Identify user's decision-making style"""
        comparison_focus = user_behavior.get('price_comparisons_made', 1)
        technical_questions = user_behavior.get('technical_questions_asked', 1)
        peer_reference_requests = user_behavior.get('peer_reference_requests', 0)
        time_spent = user_behavior.get('total_time_spent_minutes', 30)
        
        # Decision style scoring
        analytical_score = (comparison_focus * 2) + technical_questions + (time_spent / 20)
        intuitive_score = peer_reference_requests * 3 + (5 - min(5, time_spent / 10))
        cautious_score = technical_questions + comparison_focus + (time_spent / 15)
        ambitious_score = (5 - min(5, time_spent / 15)) + user_behavior.get('premium_feature_interest', 0)
        
        scores = {
            'analytical': analytical_score,
            'intuitive': intuitive_score,
            'cautious': cautious_score,
            'ambitious': ambitious_score
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _detect_complexity_fear(self, user_behavior: Dict) -> bool:
        """Detect if user has complexity fears"""
        indicators = [
            user_behavior.get('basic_questions_asked', 0) > 3,
            user_behavior.get('maintenance_concerns_raised', 0) > 1,
            user_behavior.get('installation_process_questions', 0) > 2,
            user_behavior.get('technology_reliability_questions', 0) > 1
        ]
        return sum(indicators) >= 2
    
    def _detect_financial_anxiety(self, user_profile: Dict, user_behavior: Dict) -> bool:
        """Detect financial anxiety indicators"""
        budget_max = user_profile.get('budget_max', 0)
        monthly_bill = user_profile.get('monthly_bill', 2000)
        
        indicators = [
            budget_max < monthly_bill * 100 if budget_max > 0 else False,
            user_behavior.get('financing_questions_asked', 0) > 2,
            user_behavior.get('roi_calculation_requests', 0) > 1,
            user_behavior.get('hidden_costs_concerns', 0) > 0
        ]
        return sum(indicators) >= 2
    
    def _detect_social_pressure_needs(self, user_behavior: Dict) -> bool:
        """Detect need for social validation"""
        indicators = [
            user_behavior.get('peer_reference_requests', 0) > 0,
            user_behavior.get('neighborhood_installation_questions', 0) > 0,
            user_behavior.get('social_sharing_interest', 0) > 0,
            user_behavior.get('testimonial_requests', 0) > 1
        ]
        return sum(indicators) >= 2
    
    def _generate_behavioral_interventions(self, barriers: List[Dict], decision_style: str) -> List[Dict]:
        """Generate targeted behavioral interventions"""
        interventions = []
        
        for barrier in barriers:
            barrier_type = barrier['barrier']
            
            if barrier_type == 'complexity_fear':
                interventions.append({
                    'intervention_type': 'simplification',
                    'title': 'Solar Simplified Workshop',
                    'description': 'Interactive 15-minute session explaining solar basics',
                    'delivery_method': 'interactive_video' if decision_style == 'intuitive' else 'detailed_guide',
                    'timeline': '1 week'
                })
            
            elif barrier_type == 'financial_anxiety':
                interventions.append({
                    'intervention_type': 'financial_clarity',
                    'title': 'Personal Financial Planning Session',
                    'description': 'Customized financial breakdown with payment options',
                    'delivery_method': 'one_on_one_consultation' if decision_style == 'analytical' else 'simplified_calculator',
                    'timeline': '3-5 days'
                })
            
            elif barrier_type == 'social_validation_needed':
                interventions.append({
                    'intervention_type': 'peer_connection',
                    'title': 'Solar Success Network',
                    'description': 'Connect with local solar users for firsthand experiences',
                    'delivery_method': 'peer_meetup' if decision_style == 'intuitive' else 'detailed_case_studies',
                    'timeline': '1-2 weeks'
                })
        
        return interventions
    
    def _calculate_success_probability_with_interventions(self, barriers: List[Dict]) -> Dict:
        """Calculate success probability with and without interventions"""
        base_success_probability = 0.65
        
        total_negative_impact = sum([abs(barrier['confidence_impact']) for barrier in barriers])
        success_without_intervention = max(0.1, base_success_probability - total_negative_impact)
        
        intervention_boost = min(0.25, len(barriers) * 0.08)
        success_with_intervention = min(0.95, success_without_intervention + intervention_boost)
        
        return {
            'without_intervention': f"{success_without_intervention:.0%}",
            'with_intervention': f"{success_with_intervention:.0%}",
            'improvement': f"+{(success_with_intervention - success_without_intervention):.0%}"
        }
    
    def _suggest_timeline_adjustments(self, barriers: List[Dict]) -> Dict:
        """Suggest timeline adjustments based on barriers"""
        high_priority_barriers = [b for b in barriers if b['priority'] == 'high']
        
        if len(high_priority_barriers) >= 2:
            return {
                'recommended_timeline': 'Extended (6-8 weeks)',
                'reason': 'Multiple high-priority barriers need addressing',
                'phases': [
                    'Phase 1: Address complexity and financial concerns (2-3 weeks)',
                    'Phase 2: Build confidence through education (2 weeks)', 
                    'Phase 3: Final decision and installation planning (2-3 weeks)'
                ]
            }
        elif len(high_priority_barriers) == 1:
            return {
                'recommended_timeline': 'Standard with focus (4-6 weeks)',
                'reason': 'One major barrier requires attention'
            }
        else:
            return {
                'recommended_timeline': 'Accelerated (2-4 weeks)',
                'reason': 'No major barriers, ready to proceed'
            }
    
    def _calculate_profile_completeness(self, user_profile: Dict) -> float:
        """Calculate how complete the user profile is"""
        required_fields = ['monthly_bill', 'roof_area', 'location', 'budget_max', 'house_type']
        
        completed = sum([1 for field in required_fields if user_profile.get(field) is not None])
        return completed / len(required_fields)
    
    def generate_educational_insights(self, user_profile: Dict, confidence_assessment: Dict) -> Dict:
        """Generate personalized educational insights"""
        location = user_profile.get('location', 'General')
        monthly_bill = user_profile.get('monthly_bill', 2000)
        
        personalized_content = []
        
        # Add complexity demystification if confidence is low
        if confidence_assessment.get('overall_confidence', 0.5) < 0.7:
            personalized_content.append({
                'type': 'complexity_demystification',
                'title': 'Solar Systems: Simpler Than Your Smartphone!',
                'content': self.educational_content['complexity_demystification']['system_basics'],
                'estimated_reading_time': '5 minutes',
                'confidence_boost': 0.15
            })
        
        # Add financial clarity content
        monthly_emi_estimate = monthly_bill * 0.9
        personalized_content.append({
            'type': 'financial_clarity',
            'title': f'Your Solar Journey: Just Rs.{monthly_emi_estimate:,.0f}/month',
            'content': [
                f'Your current bill: Rs.{monthly_bill:,}/month',
                f'Estimated solar EMI: Rs.{monthly_emi_estimate:,.0f}/month',
                f'Net savings from month 1: Rs.{monthly_bill - monthly_emi_estimate:,.0f}/month'
            ],
            'estimated_reading_time': '3 minutes',
            'confidence_boost': 0.20
        })
        
        # Location-specific insights
        if location in self.social_benchmarks['location_averages']:
            location_data = self.social_benchmarks['location_averages'][location]
            personalized_content.append({
                'type': 'location_insights',
                'title': f'Solar Success in {location}',
                'content': [
                    f'Average system size in {location}: {location_data["avg_system_size"]} kW',
                    f'Typical annual savings: Rs.{location_data["avg_savings"]:,}',
                    f'Customer satisfaction: {location_data["satisfaction_score"]}/5.0'
                ],
                'estimated_reading_time': '2 minutes',
                'confidence_boost': 0.10
            })
        
        return {
            'personalized_content': personalized_content,
            'total_learning_time': sum([int(content['estimated_reading_time'].split()[0]) for content in personalized_content]),
            'potential_confidence_boost': sum([content['confidence_boost'] for content in personalized_content]),
            'progress_tracking': {
                'modules_available': len(personalized_content),
                'estimated_completion_time': f"{sum([int(content['estimated_reading_time'].split()[0]) for content in personalized_content])} minutes"
            }
        }
    
    def generate_gamification_profile(self, user_profile: Dict, system_recommendation: Dict, user_actions: Dict) -> Dict:
        """Generate gamified experience profile"""
        progress_score = self._calculate_progress_score(user_profile, user_actions)
        achievement_level = self._determine_achievement_level(progress_score)
        
        return {
            'current_status': {
                'progress_score': progress_score,
                'achievement_level': achievement_level,
                'points_earned': self._calculate_points_earned(user_actions),
                'next_milestone': self._get_next_milestone(progress_score)
            },
            'challenges': [
                {
                    'title': 'Solar Education Champion',
                    'description': 'Complete 3 educational modules this week',
                    'reward': '50 points + Solar Expert badge',
                    'deadline': '7 days'
                }
            ]
        }
    
    def _calculate_progress_score(self, user_profile: Dict, user_actions: Dict) -> float:
        """Calculate user's progress score for gamification"""
        progress_score = 0.0
        
        # Profile completion (40% weight)
        profile_completeness = self._calculate_profile_completeness(user_profile)
        progress_score += profile_completeness * 40
        
        # Actions completion (60% weight)
        action_weights = {
            'quotes_requested': 15,
            'installers_contacted': 10,
            'education_modules_completed': 20,
            'site_survey_completed': 15
        }
        
        for action, weight in action_weights.items():
            if user_actions.get(action, False):
                progress_score += weight
        
        return min(100.0, progress_score)
    
    def _determine_achievement_level(self, progress_score: float) -> Dict:
        """Determine user's current achievement level"""
        current_level = 'solar_explorer'  # Default
        current_data = self.gamification_system['achievement_levels']['solar_explorer']
        
        for level_name, level_data in self.gamification_system['achievement_levels'].items():
            if progress_score >= level_data['threshold']:
                current_level = level_name
                current_data = level_data
        
        return {
            'level': current_level,
            'title': current_data['title'],
            'description': current_data['description'],
            'benefits': current_data['benefits']
        }
    
    def _calculate_points_earned(self, user_actions: Dict) -> int:
        """Calculate total points earned from user actions"""
        points = 0
        point_system = self.gamification_system['rewards_system']['points_earning']
        
        for action, point_value in point_system.items():
            if user_actions.get(action, False):
                points += point_value
        
        return points
    
    def _get_next_milestone(self, progress_score: float) -> str:
        """Get the next achievement milestone"""
        for level_name, level_data in self.gamification_system['achievement_levels'].items():
            if progress_score < level_data['threshold']:
                return level_data['title']
        return "Solar Master"
    
    def _generate_social_behavioral_executive_summary(self, confidence_assessment: Dict,
                                                    educational_insights: Dict,
                                                    gamification_profile: Dict,
                                                    behavioral_analysis: Dict) -> Dict:
        """Generate executive summary focusing on social and behavioral aspects"""
        
        overall_confidence = confidence_assessment['overall_confidence']
        barriers_count = len(behavioral_analysis.get('barriers_identified', []))
        education_modules = len(educational_insights['personalized_content'])
        
        # Determine primary recommendation
        if overall_confidence >= 0.80 and barriers_count <= 1:
            primary_recommendation = "READY TO ACCELERATE: High confidence, minimal barriers. Fast-track to installation."
        elif overall_confidence >= 0.65 and barriers_count <= 2:
            primary_recommendation = "PROCEED WITH OPTIMIZATION: Good foundation, address identified barriers for best results."
        elif overall_confidence >= 0.50 or barriers_count <= 3:
            primary_recommendation = "FOCUSED IMPROVEMENT NEEDED: Moderate confidence, implement targeted interventions."
        else:
            primary_recommendation = "COMPREHENSIVE INTERVENTION REQUIRED: Low confidence, address all barriers before proceeding."
        
        return {
            'overall_confidence_score': f"{overall_confidence:.0%}",
            'confidence_category': confidence_assessment['confidence_level'],
            'primary_recommendation': primary_recommendation,
            'behavioral_barriers_identified': barriers_count,
            'educational_interventions_recommended': education_modules,
            'gamification_engagement_level': gamification_profile['current_status']['achievement_level']['level']
        }
    
    def _create_social_behavioral_action_plan(self, confidence_assessment: Dict,
                                            educational_insights: Dict,
                                            gamification_profile: Dict,
                                            behavioral_analysis: Dict) -> Dict:
        """Create comprehensive action plan addressing social and behavioral factors"""
        
        action_plan = {
            'immediate_priorities': [],
            'short_term_goals': [],
            'success_milestones': []
        }
        
        # Immediate priorities (0-1 week)
        barriers = behavioral_analysis.get('barriers_identified', [])
        high_priority_barriers = [b for b in barriers if b.get('priority') == 'high']
        
        if high_priority_barriers:
            action_plan['immediate_priorities'].append({
                'action': 'Address Critical Behavioral Barriers',
                'details': [f"Implement intervention for {b['barrier']}" for b in high_priority_barriers],
                'timeline': '3-7 days'
            })
        
        # Educational interventions
        if educational_insights['personalized_content']:
            action_plan['immediate_priorities'].append({
                'action': 'Complete Priority Educational Modules',
                'details': [content['title'] for content in educational_insights['personalized_content'][:2]],
                'timeline': f"{educational_insights['progress_tracking']['estimated_completion_time']}"
            })
        
        # Success milestones
        action_plan['success_milestones'] = [
            {
                'milestone': '70% Overall Confidence',
                'target_timeline': '2-3 weeks',
                'indicators': ['Technical validation complete', 'Major barriers addressed']
            },
            {
                'milestone': '80% Overall Confidence',
                'target_timeline': '4-5 weeks',
                'indicators': ['Social validation achieved', 'Financial clarity gained']
            }
        ]
        
        return action_plan
    
    def _generate_success_enhancement_recommendations(self, confidence_assessment: Dict,
                                                    behavioral_analysis: Dict) -> List[Dict]:
        """Generate specific recommendations to enhance success probability"""
        
        recommendations = []
        
        overall_confidence = confidence_assessment['overall_confidence']
        decision_style = behavioral_analysis.get('decision_style', 'analytical')
        
        # Confidence-based recommendations
        if overall_confidence < 0.7:
            recommendations.append({
                'category': 'Confidence Building',
                'priority': 'High',
                'recommendation': 'Implement comprehensive confidence building program',
                'specific_actions': [
                    'Complete all recommended educational modules',
                    'Schedule consultation with solar expert',
                    'Visit 2-3 successful installations in your area'
                ],
                'expected_impact': '+20-25% confidence boost',
                'timeline': '2-3 weeks'
            })
        
        # Decision style optimization
        if decision_style == 'analytical':
            recommendations.append({
                'category': 'Decision Style Optimization',
                'priority': 'Medium',
                'recommendation': 'Provide comprehensive analytical framework',
                'specific_actions': [
                    'Detailed ROI analysis with sensitivity scenarios',
                    'Technical specification deep-dive sessions',
                    'Comparative analysis of multiple solutions'
                ],
                'timeline': '1-2 weeks'
            })
        
        return recommendations
    
    # Add these methods to the EnhancedMistakePreventionEngine class in mistake_prevention_engine.py

    def prevent_mistakes_from_pipeline(self, payload: Dict) -> SafetyGateOut:
        """
        Main integration method for pipeline - comprehensive mistake prevention analysis
        """
        try:
            # Extract pipeline data
            user_request = payload.get('user_request', {})
            sizing = payload.get('sizing', {})
            roi = payload.get('roi', {})
            risk = payload.get('risk', {})
            heuristic_search = payload.get('heuristic_search', {})
            
            # Create user profile from pipeline data
            user_profile = self._create_user_profile_from_pipeline(user_request)
            
            # Create system recommendation from sizing data
            system_recommendation = self._create_system_recommendation_from_sizing(sizing)
            
            # Create user behavior simulation from pipeline data
            user_behavior = self._simulate_user_behavior_from_pipeline(payload)
            
            # Create user actions from heuristic search data
            user_actions = self._extract_user_actions_from_pipeline(heuristic_search)
            
            # Run comprehensive analysis
            comprehensive_report = self.generate_comprehensive_social_behavioral_report(
                user_profile, system_recommendation, user_actions, user_behavior
            )
            
            # Extract key results for safety gate
            issues = []
            critical_issues = []
            confidence_intervals = {}
            
            # Process comprehensive report data
            if comprehensive_report.status == "FAIL":
                critical_issues.extend(comprehensive_report.critical_issues)
                issues.extend(comprehensive_report.issues)
            
            report_data = comprehensive_report.data
            overall_confidence = report_data.get('confidence_assessment', {}).get('overall_confidence', 0.7)
            barriers_count = len(report_data.get('behavioral_analysis', {}).get('barriers_identified', []))
            
            # Add specific pipeline validations
            pipeline_issues = self._validate_pipeline_consistency(payload)
            issues.extend(pipeline_issues)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_pipeline_confidence_intervals(payload, overall_confidence)
            
            # Create detailed result data
            result_data = {
                'comprehensive_analysis': report_data,
                'overall_confidence': overall_confidence,
                'behavioral_barriers_count': barriers_count,
                'pipeline_consistency_issues': pipeline_issues,
                'confidence_intervals': confidence_intervals,
                'success_probability': self._calculate_pipeline_success_probability(overall_confidence, barriers_count, len(pipeline_issues)),
                'recommendations': self._generate_pipeline_recommendations(report_data, pipeline_issues),
                'status': 'success'
            }
            
            # Determine overall status
            total_critical = len(critical_issues) + len([i for i in pipeline_issues if 'critical' in i.lower() or 'severe' in i.lower()])
            status = "FAIL" if total_critical > 0 or overall_confidence < 0.4 else "PASS"
            
            return SafetyGateOut(
                result_data,
                status=status,
                issues=issues,
                critical_issues=critical_issues
            )
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'status': 'error'},
                status="FAIL",
                issues=[f"Pipeline integration error: {str(e)}"],
                critical_issues=[f"Critical integration failure: {str(e)}"]
            )

    def validate_pipeline_data(self, payload: Dict) -> SafetyGateOut:
        """
        Alternative integration method - focused on data validation
        """
        try:
            validation_issues = []
            critical_issues = []
            
            # Validate user request data
            user_request = payload.get('user_request', {})
            user_issues = self._validate_user_request_data(user_request)
            validation_issues.extend(user_issues)
            
            # Validate sizing data
            sizing = payload.get('sizing', {})
            sizing_issues = self._validate_sizing_data(sizing, user_request)
            validation_issues.extend(sizing_issues)
            
            # Validate ROI data
            roi = payload.get('roi', {})
            roi_issues = self._validate_roi_data(roi, sizing)
            validation_issues.extend(roi_issues)
            
            # Validate risk data
            risk = payload.get('risk', {})
            risk_issues = self._validate_risk_data(risk)
            validation_issues.extend(risk_issues)
            
            # Cross-validate between components
            consistency_issues = self._validate_cross_component_consistency(payload)
            validation_issues.extend(consistency_issues)
            
            # Identify critical issues
            critical_keywords = ['critical', 'severe', 'dangerous', 'unrealistic', 'invalid']
            for issue in validation_issues:
                if any(keyword in issue.lower() for keyword in critical_keywords):
                    critical_issues.append(issue)
            
            # Calculate validation confidence
            validation_confidence = max(0.1, 1.0 - (len(validation_issues) * 0.1))
            
            result_data = {
                'validation_issues': validation_issues,
                'critical_issues': critical_issues,
                'validation_confidence': validation_confidence,
                'data_quality_score': self._calculate_data_quality_score(payload),
                'completeness_score': self._calculate_data_completeness_score(payload),
                'consistency_score': 1.0 - (len(consistency_issues) * 0.2),
                'status': 'success'
            }
            
            status = "FAIL" if len(critical_issues) > 0 or validation_confidence < 0.5 else "PASS"
            
            return SafetyGateOut(
                result_data,
                status=status,
                issues=validation_issues,
                critical_issues=critical_issues
            )
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'status': 'error'},
                status="FAIL",
                issues=[f"Data validation error: {str(e)}"],
                critical_issues=[f"Critical validation failure: {str(e)}"]
            )

    def check_mistakes_in_pipeline(self, payload: Dict) -> SafetyGateOut:
        """
        Focused mistake detection for pipeline integration
        """
        try:
            detected_mistakes = []
            critical_mistakes = []
            
            # Extract key data
            user_request = payload.get('user_request', {})
            sizing = payload.get('sizing', {})
            roi = payload.get('roi', {})
            
            # Create profiles for mistake detection
            user_profile = self._create_user_profile_from_pipeline(user_request)
            system_recommendation = self._create_system_recommendation_from_sizing(sizing)
            
            # Run sizing mistake detection
            sizing_mistakes = self.detect_sizing_mistakes(user_profile, system_recommendation)
            
            if sizing_mistakes.status == "FAIL":
                critical_mistakes.extend(sizing_mistakes.critical_issues)
            detected_mistakes.extend(sizing_mistakes.issues)
            
            # Detect financial mistakes
            financial_mistakes = self._detect_financial_mistakes_in_pipeline(user_request, sizing, roi)
            detected_mistakes.extend(financial_mistakes)
            
            # Detect technical mistakes
            tech_mistakes = self._detect_technical_mistakes_in_pipeline(sizing, roi)
            detected_mistakes.extend(tech_mistakes)
            
            # Detect timeline mistakes
            timeline_mistakes = self._detect_timeline_mistakes_in_pipeline(payload)
            detected_mistakes.extend(timeline_mistakes)
            
            # Calculate mistake severity
            mistake_severity = self._calculate_mistake_severity(detected_mistakes)
            
            result_data = {
                'detected_mistakes': detected_mistakes,
                'critical_mistakes': critical_mistakes,
                'mistake_categories': self._categorize_mistakes(detected_mistakes),
                'mistake_severity': mistake_severity,
                'prevention_recommendations': self._generate_mistake_prevention_recommendations(detected_mistakes),
                'confidence_impact': sum([0.1 for _ in detected_mistakes]),  # Each mistake reduces confidence by 10%
                'status': 'success'
            }
            
            status = "FAIL" if len(critical_mistakes) > 0 or mistake_severity == "High" else "PASS"
            
            return SafetyGateOut(
                result_data,
                status=status,
                issues=detected_mistakes,
                critical_issues=critical_mistakes
            )
            
        except Exception as e:
            return SafetyGateOut(
                {'error': str(e), 'status': 'error'},
                status="FAIL",
                issues=[f"Mistake detection error: {str(e)}"],
                critical_issues=[f"Critical mistake detection failure: {str(e)}"]
            )

    # Helper methods for pipeline integration

    def _create_user_profile_from_pipeline(self, user_request: Dict) -> Dict:
        """Create user profile compatible with existing methods"""
        return {
            'monthly_bill': user_request.get('monthly_bill', 2500),
            'roof_area': user_request.get('roof_area_m2', 100),
            'location': user_request.get('location', 'Mumbai'),
            'budget_max': user_request.get('budget_inr', 300000),
            'house_type': user_request.get('house_type', 'independent'),
            'family_size': 4,  # Default
            'income_bracket': user_request.get('income_bracket', 'Medium'),
            'risk_tolerance': user_request.get('risk_tolerance', 'moderate'),
            'education_modules_completed': 0  # Default
        }

    def _create_system_recommendation_from_sizing(self, sizing: Dict) -> Dict:
        """Create system recommendation from sizing data"""
        cost_range = sizing.get('cost_range_inr', (300000, 400000))
        if isinstance(cost_range, (list, tuple)) and len(cost_range) >= 2:
            total_cost = (cost_range[0] + cost_range[1]) / 2
        else:
            total_cost = 350000
        
        return {
            'recommended_capacity_kw': sizing.get('system_capacity_kw', 5.0),
            'total_cost': total_cost,
            'estimated_annual_generation': sizing.get('monthly_generation_kwh', 400) * 12 if sizing.get('monthly_generation_kwh') else 4800,
            'inverter_capacity_kw': sizing.get('system_capacity_kw', 5.0),  # Assume 1:1 ratio
            'panel_type': 'Monocrystalline',  # Default
            'roi_percentage': 15.0  # Default
        }

    def _simulate_user_behavior_from_pipeline(self, payload: Dict) -> Dict:
        """Simulate user behavior from pipeline context"""
        user_request = payload.get('user_request', {})
        heuristic = payload.get('heuristic_search', {})
        
        # Simulate behavior based on user characteristics
        risk_tolerance = user_request.get('risk_tolerance', 'moderate')
        timeline_preference = user_request.get('timeline_preference', 'flexible')
        
        behavior = {
            'page_views': 8 if risk_tolerance == 'conservative' else 5,
            'total_time_spent_minutes': 60 if risk_tolerance == 'conservative' else 30,
            'questions_asked': 3 if risk_tolerance == 'conservative' else 2,
            'quote_requests': 2 if timeline_preference == 'immediate' else 1,
            'price_comparisons_made': 3,
            'technical_questions_asked': 2 if risk_tolerance == 'conservative' else 1,
            'peer_reference_requests': 1 if 'social' in str(user_request.get('goals', [])) else 0,
            'maintenance_concerns_raised': 1 if risk_tolerance == 'conservative' else 0,
            'financing_questions_asked': 2 if user_request.get('budget_inr', 300000) < 400000 else 1
        }
        
        return behavior

    def _extract_user_actions_from_pipeline(self, heuristic_search: Dict) -> Dict:
        """Extract user actions from heuristic search data"""
        return {
            'quotes_requested': True,
            'installers_contacted': heuristic_search.get('confidence', 0) > 0.7,
            'education_modules_completed': 2,  # Default
            'site_survey_completed': False,  # Assume not done yet
            'financing_explored': True
        }

    def _validate_pipeline_consistency(self, payload: Dict) -> List[str]:
        """Validate consistency across pipeline components"""
        issues = []
        
        try:
            user_request = payload.get('user_request', {})
            sizing = payload.get('sizing', {})
            roi = payload.get('roi', {})
            risk = payload.get('risk', {})
            
            # Check budget vs cost consistency
            budget = user_request.get('budget_inr', 0)
            cost_range = sizing.get('cost_range_inr', (0, 0))
            if budget > 0 and isinstance(cost_range, (list, tuple)) and len(cost_range) >= 2:
                min_cost = cost_range[0]
                if min_cost > budget * 1.3:
                    issues.append(f"Cost-budget mismatch: Min cost (₹{min_cost:,.0f}) exceeds budget (₹{budget:,.0f}) by >30%")
            
            # Check consumption vs generation consistency
            monthly_consumption = user_request.get('monthly_consumption_kwh', 0)
            monthly_generation = sizing.get('monthly_generation_kwh', 0)
            if monthly_consumption > 0 and monthly_generation > 0:
                ratio = monthly_generation / monthly_consumption
                if ratio < 0.3:
                    issues.append(f"Severe undersizing: System generates only {ratio:.0%} of consumption")
                elif ratio > 2.0:
                    issues.append(f"Excessive oversizing: System generates {ratio:.1f}x consumption")
            
            # Check ROI vs risk consistency
            payback = roi.get('payback_years', 0)
            overall_risk = risk.get('overall_risk', '').lower()
            if payback > 0:
                if payback < 4 and 'high' in overall_risk:
                    issues.append("Inconsistent: Very short payback with high risk assessment")
                elif payback > 12 and 'low' in overall_risk:
                    issues.append("Inconsistent: Very long payback with low risk assessment")
            
            # Check savings vs bill consistency
            annual_savings = roi.get('annual_savings_inr', 0)
            monthly_bill = user_request.get('monthly_bill', 2500)
            if annual_savings > 0 and monthly_bill > 0:
                savings_ratio = annual_savings / (monthly_bill * 12)
                if savings_ratio > 1.2:
                    issues.append(f"Unrealistic savings: {savings_ratio:.1f}x current bill")
            
        except Exception as e:
            issues.append(f"Pipeline consistency check error: {str(e)}")
        
        return issues

    def _calculate_pipeline_confidence_intervals(self, payload: Dict, overall_confidence: float) -> Dict:
        """Calculate confidence intervals for pipeline results"""
        confidence_intervals = {}
        
        try:
            sizing = payload.get('sizing', {})
            roi = payload.get('roi', {})
            
            # Cost confidence - based on sizing confidence and market volatility
            sizing_confidence = sizing.get('confidence_score', 0.8)
            cost_confidence = (sizing_confidence + overall_confidence) / 2
            confidence_intervals['cost_accuracy'] = (
                max(0.1, cost_confidence - 0.2),
                min(1.0, cost_confidence + 0.1)
            )
            
            # Generation confidence - based on weather and system factors
            generation_confidence = overall_confidence * 0.9  # Slightly lower than overall
            confidence_intervals['generation_accuracy'] = (
                max(0.1, generation_confidence - 0.25),
                min(1.0, generation_confidence + 0.1)
            )
            
            # ROI confidence - most uncertain due to future projections
            roi_confidence = overall_confidence * 0.8
            confidence_intervals['roi_accuracy'] = (
                max(0.1, roi_confidence - 0.3),
                min(1.0, roi_confidence + 0.1)
            )
            
            # Timeline confidence - based on complexity and external factors
            timeline_confidence = overall_confidence * 0.85
            confidence_intervals['timeline_accuracy'] = (
                max(0.1, timeline_confidence - 0.2),
                min(1.0, timeline_confidence + 0.15)
            )
            
        except Exception as e:
            # Fallback intervals
            confidence_intervals = {
                'cost_accuracy': (0.6, 0.8),
                'generation_accuracy': (0.5, 0.7),
                'roi_accuracy': (0.4, 0.6),
                'timeline_accuracy': (0.6, 0.8)
            }
        
        return confidence_intervals

    def _calculate_pipeline_success_probability(self, overall_confidence: float, barriers_count: int, pipeline_issues_count: int) -> float:
        """Calculate success probability for the entire pipeline"""
        base_probability = 0.75  # Base success rate
        
        # Adjust for confidence
        confidence_factor = overall_confidence * 0.3
        
        # Adjust for behavioral barriers
        barrier_penalty = barriers_count * 0.1
        
        # Adjust for pipeline issues
        pipeline_penalty = pipeline_issues_count * 0.05
        
        success_probability = base_probability + confidence_factor - barrier_penalty - pipeline_penalty
        
        return max(0.1, min(0.95, success_probability))

    def _generate_pipeline_recommendations(self, report_data: Dict, pipeline_issues: List[str]) -> List[str]:
        """Generate recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Based on confidence level
        overall_confidence = report_data.get('confidence_assessment', {}).get('overall_confidence', 0.7)
        
        if overall_confidence < 0.6:
            recommendations.append("Comprehensive review required before proceeding with installation")
            recommendations.append("Consider engaging with solar consultant for detailed assessment")
        elif overall_confidence < 0.8:
            recommendations.append("Address identified concerns to improve project success probability")
        
        # Based on behavioral barriers
        barriers = report_data.get('behavioral_analysis', {}).get('barriers_identified', [])
        if len(barriers) > 2:
            recommendations.append("Implement targeted interventions to address behavioral barriers")
        
        # Based on pipeline issues
        critical_pipeline_issues = [i for i in pipeline_issues if 'critical' in i.lower() or 'severe' in i.lower()]
        if critical_pipeline_issues:
            recommendations.append("Resolve critical pipeline inconsistencies before proceeding")
        
        # General recommendations
        recommendations.append("Consider phased implementation to reduce risks")
        recommendations.append("Ensure all stakeholders are aligned on project objectives and timeline")
        
        return recommendations

    def _validate_user_request_data(self, user_request: Dict) -> List[str]:
        """Validate user request data quality"""
        issues = []
        
        # Check for missing critical data
        critical_fields = ['location', 'monthly_bill', 'monthly_consumption_kwh']
        for field in critical_fields:
            if not user_request.get(field):
                issues.append(f"Missing critical user data: {field}")
        
        # Validate data ranges
        monthly_bill = user_request.get('monthly_bill', 0)
        if monthly_bill > 0:
            if monthly_bill < 500:
                issues.append("Monthly bill unusually low - verify accuracy")
            elif monthly_bill > 10000:
                issues.append("Monthly bill unusually high - verify consumption category")
        
        # Validate budget reasonableness
        budget = user_request.get('budget_inr', 0)
        if budget > 0 and budget < 150000:
            issues.append("Budget may be insufficient for viable solar installation")
        
        return issues

    def _validate_sizing_data(self, sizing: Dict, user_request: Dict) -> List[str]:
        """Validate sizing data consistency"""
        issues = []
        
        system_capacity = sizing.get('system_capacity_kw', 0)
        if system_capacity <= 0:
            issues.append("Invalid system capacity in sizing data")
        elif system_capacity > 20:
            issues.append("System capacity unusually large - verify requirements")
        
        # Check cost reasonableness
        cost_range = sizing.get('cost_range_inr')
        if cost_range:
            if isinstance(cost_range, (list, tuple)) and len(cost_range) >= 2:
                min_cost, max_cost = cost_range[0], cost_range[1]
                cost_per_kw = min_cost / system_capacity if system_capacity > 0 else 0
                if cost_per_kw > 0:
                    if cost_per_kw < 30000:
                        issues.append("Cost per kW suspiciously low - verify quality and inclusions")
                    elif cost_per_kw > 80000:
                        issues.append("Cost per kW very high - consider alternatives")
        
        return issues

    def _validate_roi_data(self, roi: Dict, sizing: Dict) -> List[str]:
        """Validate ROI calculations"""
        issues = []
        
        payback = roi.get('payback_years', 0)
        if payback > 0:
            if payback < 3:
                issues.append("Payback period suspiciously short - verify assumptions")
            elif payback > 15:
                issues.append("Payback period too long - project may not be viable")
        
        npv = roi.get('npv_15y_inr', 0)
        annual_savings = roi.get('annual_savings_inr', 0)
        
        if annual_savings > 0 and npv < annual_savings * 3:
            issues.append("NPV lower than expected - check discount rate assumptions")
        
        return issues

    def _validate_risk_data(self, risk: Dict) -> List[str]:
        """Validate risk assessment data"""
        issues = []
        
        overall_risk = risk.get('overall_risk', '').lower()
        if not overall_risk:
            issues.append("Missing overall risk assessment")
        elif overall_risk in ['very high', 'extreme']:
            issues.append("Very high risk level requires detailed mitigation plan")
        
        return issues

    def _validate_cross_component_consistency(self, payload: Dict) -> List[str]:
        """Validate consistency across all pipeline components"""
        return self._validate_pipeline_consistency(payload)  # Reuse existing method

    def _calculate_data_quality_score(self, payload: Dict) -> float:
        """Calculate overall data quality score"""
        scores = []
        
        # User request completeness
        user_request = payload.get('user_request', {})
        required_fields = ['location', 'monthly_bill', 'monthly_consumption_kwh', 'budget_inr']
        user_completeness = sum([1 for field in required_fields if user_request.get(field)]) / len(required_fields)
        scores.append(user_completeness)
        
        # Sizing data quality
        sizing = payload.get('sizing', {})
        sizing_quality = 1.0 if sizing.get('system_capacity_kw', 0) > 0 else 0.5
        scores.append(sizing_quality)
        
        # ROI data quality
        roi = payload.get('roi', {})
        roi_quality = 1.0 if roi.get('payback_years', 0) > 0 else 0.5
        scores.append(roi_quality)
        
        return sum(scores) / len(scores) if scores else 0.5

    def _calculate_data_completeness_score(self, payload: Dict) -> float:
        """Calculate data completeness score"""
        total_fields = 0
        complete_fields = 0
        
        for component in payload.values():
            if isinstance(component, dict):
                total_fields += len(component)
                complete_fields += sum([1 for v in component.values() if v is not None and v != ''])
        
        return complete_fields / total_fields if total_fields > 0 else 0.5

    def _detect_financial_mistakes_in_pipeline(self, user_request: Dict, sizing: Dict, roi: Dict) -> List[str]:
        """Detect financial mistakes in pipeline data"""
        mistakes = []
        
        # Budget vs cost analysis
        budget = user_request.get('budget_inr', 0)
        cost_range = sizing.get('cost_range_inr', (0, 0))
        
        if budget > 0 and cost_range:
            if isinstance(cost_range, (list, tuple)) and len(cost_range) >= 2:
                min_cost = cost_range[0]
                if min_cost > budget * 1.5:
                    mistakes.append("Financial mistake: System cost significantly exceeds stated budget")
        
        # ROI expectations vs reality
        annual_savings = roi.get('annual_savings_inr', 0)
        monthly_bill = user_request.get('monthly_bill', 2500)
        
        if annual_savings > monthly_bill * 12 * 1.3:
            mistakes.append("Financial mistake: Savings projection exceeds 130% of current bill")
        
        return mistakes

    def _detect_technical_mistakes_in_pipeline(self, sizing: Dict, roi: Dict) -> List[str]:
        """Detect technical mistakes in pipeline data"""
        mistakes = []
        
        # System sizing mistakes
        system_capacity = sizing.get('system_capacity_kw', 0)
        monthly_generation = sizing.get('monthly_generation_kwh', 0)
        
        if system_capacity > 0 and monthly_generation > 0:
            expected_generation = system_capacity * 130  # Rough estimate: 130 units per kW per month
            actual_ratio = monthly_generation / expected_generation
            
            if actual_ratio < 0.7:
                mistakes.append("Technical mistake: Generation projection significantly below expected levels")
            elif actual_ratio > 1.4:
                mistakes.append("Technical mistake: Generation projection unrealistically high")
        
        return mistakes

    def _detect_timeline_mistakes_in_pipeline(self, payload: Dict) -> List[str]:
        """Detect timeline-related mistakes"""
        mistakes = []
        
        user_request = payload.get('user_request', {})
        timeline_pref = user_request.get('timeline_preference', '').lower()
        
        if timeline_pref == 'immediate':
            # Check if all prerequisites are in place for immediate installation
            sizing = payload.get('sizing', {})
            if not sizing.get('system_capacity_kw'):
                mistakes.append("Timeline mistake: Immediate timeline requested but system not properly sized")
        
        return mistakes

    def _calculate_mistake_severity(self, mistakes: List[str]) -> str:
        """Calculate overall mistake severity"""
        if not mistakes:
            return "None"
        
        high_severity_keywords = ['critical', 'severe', 'dangerous', 'significantly', 'unrealistically']
        high_severity_count = sum([1 for mistake in mistakes 
                                if any(keyword in mistake.lower() for keyword in high_severity_keywords)])
        
        if high_severity_count > 2:
            return "High"
        elif high_severity_count > 0:
            return "Medium"
        elif len(mistakes) > 3:
            return "Medium"
        else:
            return "Low"

    def _categorize_mistakes(self, mistakes: List[str]) -> Dict[str, List[str]]:
        """Categorize detected mistakes"""
        categories = {
            'financial': [],
            'technical': [],
            'timeline': [],
            'data_quality': [],
            'consistency': []
        }
        
        for mistake in mistakes:
            mistake_lower = mistake.lower()
            if 'financial' in mistake_lower or 'budget' in mistake_lower or 'cost' in mistake_lower:
                categories['financial'].append(mistake)
            elif 'technical' in mistake_lower or 'generation' in mistake_lower or 'capacity' in mistake_lower:
                categories['technical'].append(mistake)
            elif 'timeline' in mistake_lower or 'immediate' in mistake_lower:
                categories['timeline'].append(mistake)
            elif 'missing' in mistake_lower or 'invalid' in mistake_lower:
                categories['data_quality'].append(mistake)
            else:
                categories['consistency'].append(mistake)
        
        return {k: v for k, v in categories.items() if v}  # Return only non-empty categories

    def _generate_mistake_prevention_recommendations(self, mistakes: List[str]) -> List[str]:
        """Generate recommendations to prevent identified mistakes"""
        recommendations = []
        
        categories = self._categorize_mistakes(mistakes)
        
        if categories.get('financial'):
            recommendations.append("Review and adjust financial assumptions and budget allocation")
            recommendations.append("Consider alternative financing options or phased implementation")
        
        if categories.get('technical'):
            recommendations.append("Conduct detailed technical review with qualified solar engineer")
            recommendations.append("Verify generation projections with multiple calculation methods")
        
        if categories.get('data_quality'):
            recommendations.append("Complete missing data collection before proceeding")
            recommendations.append("Validate all input data for accuracy and completeness")
        
        if categories.get('consistency'):
            recommendations.append("Resolve data inconsistencies across pipeline components")
        
        # General recommendations
        recommendations.append("Consider engaging independent third-party review")
        recommendations.append("Implement staged approval process with validation checkpoints")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    
    # Initialize enhanced mistake prevention engine
    enhanced_engine = EnhancedMistakePreventionEngine()
    
    # Test with sample data
    sample_user_profile = {
        'monthly_bill': 3500,
        'roof_area': 900,
        'location': 'Bangalore',
        'budget_max': 450000,
        'house_type': 'independent',
        'family_size': 4,
        'education_modules_completed': 2
    }
    
    sample_system_recommendation = {
        'recommended_capacity_kw': 5.5,
        'total_cost': 425000,
        'estimated_annual_generation': 8200,
        'inverter_capacity_kw': 5.0,
        'panel_type': 'Monocrystalline',
        'roi_percentage': 16.8
    }
    
    sample_user_actions = {
        'quotes_requested': True,
        'installers_contacted': True,
        'education_modules_completed': 3,
        'site_survey_completed': False,
        'financing_explored': True
    }
    
    sample_user_behavior = {
        'page_views': 15,
        'total_time_spent_minutes': 85,
        'questions_asked': 4,
        'quote_requests': 2,
        'price_comparisons_made': 3,
        'technical_questions_asked': 2,
        'peer_reference_requests': 1,
        'maintenance_concerns_raised': 1
    }
    
    print("=== ENHANCED MISTAKE PREVENTION ENGINE TEST ===")
    
    # Test sizing mistakes detection
    print("\n1. SIZING MISTAKES DETECTION:")
    sizing_result = enhanced_engine.detect_sizing_mistakes(
        sample_user_profile, sample_system_recommendation
    )
    print(f"Status: {sizing_result.status}")
    print(f"Confidence Score: {sizing_result.data.get('confidence_score', 0):.2f}")
    print(f"Risk Level: {sizing_result.data.get('risk_level', 'Unknown')}")
    print(f"Issues: {len(sizing_result.issues)}")
    print(f"Critical Issues: {len(sizing_result.critical_issues)}")
    
    # Test enhanced confidence scoring
    print("\n2. ENHANCED CONFIDENCE SCORING:")
    confidence_result = enhanced_engine.calculate_confidence_score_with_social_factors(
        sample_user_profile, sample_system_recommendation
    )
    print(f"Status: {confidence_result.status}")
    print(f"Overall Confidence: {confidence_result.data['overall_confidence']:.1%}")
    print(f"Confidence Level: {confidence_result.data['confidence_level']}")
    print(f"Technical: {confidence_result.data['component_scores']['technical_confidence']:.2f}")
    print(f"Social: {confidence_result.data['component_scores']['social_confidence']:.2f}")
    
    # Test behavioral analysis
    print("\n3. BEHAVIORAL BARRIER ANALYSIS:")
    behavioral_result = enhanced_engine.detect_and_address_behavioral_barriers(
        sample_user_profile, sample_user_behavior
    )
    print(f"Status: {behavioral_result.status}")
    print(f"Decision Style: {behavioral_result.data['decision_style'].title()}")
    print(f"Barriers Identified: {len(behavioral_result.data['barriers_identified'])}")
    print(f"Issues: {len(behavioral_result.issues)}")
    
    # Test comprehensive report
    print("\n4. COMPREHENSIVE REPORT:")
    full_report = enhanced_engine.generate_comprehensive_social_behavioral_report(
        sample_user_profile, sample_system_recommendation, sample_user_actions, sample_user_behavior
    )
    print(f"Status: {full_report.status}")
    print(f"Overall Confidence: {full_report.data['executive_summary']['overall_confidence_score']}")
    print(f"Behavioral Barriers: {full_report.data['executive_summary']['behavioral_barriers_identified']}")
    print(f"Issues: {len(full_report.issues)}")
    print(f"Critical Issues: {len(full_report.critical_issues)}")
    
    print("\n" + "="*50)
    print("✅ ENHANCED MISTAKE PREVENTION ENGINE READY!")
    print("🎯 Social & Behavioral barriers addressed")
    print("📚 Educational insights and gamification implemented")
    print("🤝 Peer influence and confidence scoring active")
    print("🚀 Success probability enhancement enabled")
    print("⚡ SafetyGateOut integration completed")
    print("="*50)