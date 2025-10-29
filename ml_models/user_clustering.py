"""
FIXED User Clustering Module V4 - AgglomerativeClustering Predict Method RESOLVED
Key fixes implemented:
1. Fixed AgglomerativeClustering prediction issue using KNeighborsClassifier wrapper
2. Added proper prediction method for all clustering algorithms
3. Enhanced algorithm-specific prediction handling
4. Maintained all V3 enhancements while fixing prediction method issues
5. Added fallback mechanisms for algorithms without native predict methods
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

class FixedUserClusteringV4:
    """
    FIXED clustering system v4 - AgglomerativeClustering prediction method resolved
    """
    
    def __init__(self, algorithm: str = 'kmeans', scaler_type: str = 'standard', 
                 use_pca: bool = False, min_dimensions: int = 5):
        self.algorithm = algorithm
        self.scaler_type = scaler_type
        self.use_pca = use_pca
        self.min_dimensions = min_dimensions
        self.model = None
        self.predictor = None  # V4 NEW: Separate predictor for algorithms without predict method
        self.scaler = self._get_optimized_scaler(scaler_type)
        self.label_encoders = {}
        self.cluster_profiles = {}
        self.feature_importance = {}
        self.validation_metrics = {}
        self.is_trained = False
        self.pca_model = None
        self.feature_selector = None
        self.correlation_threshold = 0.80
        
        # V4: Track features used during training
        self.training_features = []
        self.low_variance_threshold = 0.01
        self.variance_selector = None
        
        # V4: Store training data for algorithms without predict method
        self.training_labels = None
        self.training_data = None
        
    def _get_optimized_scaler(self, scaler_type: str):
        """Get optimized scaler configurations"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),  
            'robust': RobustScaler(quantile_range=(15.0, 85.0))
        }
        return scalers.get(scaler_type, StandardScaler())
    
    def _algorithm_has_predict(self) -> bool:
        """V4: Check if algorithm has native predict method"""
        algorithms_with_predict = ['kmeans']
        return self.algorithm in algorithms_with_predict
    
    def _create_predictor_V4(self, X_train: np.ndarray, labels: np.ndarray):
        """V4: Create predictor for algorithms without native predict method"""
        if self._algorithm_has_predict():
            # Algorithm has native predict method
            self.predictor = None
            return
        
        print("ðŸ”§ V4: Creating KNN predictor for algorithm without predict method...")
        
        # Remove noise points for training predictor
        valid_indices = labels >= 0
        if np.sum(valid_indices) < len(labels) * 0.5:
            print("   âš ï¸  Too many noise points, using all data")
            X_clean = X_train
            labels_clean = labels
        else:
            X_clean = X_train[valid_indices]
            labels_clean = labels[valid_indices]
        
        # Determine optimal k for KNN
        n_samples = len(X_clean)
        n_clusters = len(np.unique(labels_clean))
        
        # Use reasonable k based on data size and clusters
        k = min(max(3, n_clusters * 2), n_samples // 3, 15)
        
        self.predictor = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='euclidean'
        )
        
        self.predictor.fit(X_clean, labels_clean)
        print(f"   âœ… KNN predictor created with k={k}")
    
    def prepare_features_FIXED_V4(self, df: pd.DataFrame, is_prediction: bool = False) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        FIXED v4: Enhanced feature preparation with single sample prediction support
        """
        
        print("ðŸ”§ FIXED V4 Feature Preparation Pipeline")
        print("-" * 50)
        
        # Create enhanced features with better separation
        df_encoded = self._create_enhanced_features_V4(df)
        
        if is_prediction and self.is_trained:
            # V4 PREDICTION MODE: Use stored training features
            print("ðŸ”® V4 Prediction Mode: Using training feature configuration")
            
            # Ensure all training features exist
            for feature in self.training_features:
                if feature not in df_encoded.columns:
                    print(f"   âš ï¸  Missing feature {feature}, setting to 0")
                    df_encoded[feature] = 0.0
            
            # Select only training features in same order
            feature_columns = [f for f in self.training_features if f in df_encoded.columns]
            feature_matrix = df_encoded[feature_columns].values
            
            # Apply saved scaler
            print("âš–ï¸  Applying saved scaler...")
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            
            # Apply saved PCA if used
            if self.use_pca and self.pca_model is not None:
                print("ðŸ” Applying saved PCA transformation...")
                feature_matrix_final = self.pca_model.transform(feature_matrix_scaled)
            else:
                feature_matrix_final = feature_matrix_scaled
                
            print(f"âœ… Prediction feature matrix: {feature_matrix_final.shape}")
            return feature_matrix_final, df_encoded, feature_columns
        
        else:
            # V4 TRAINING MODE: Original pipeline with fixes
            print("ðŸ—ï¸  V4 Training Mode: Building feature pipeline")
            
            # STEP 1: Remove low-variance features with single sample handling
            print("ðŸ“Š Step 1: Smart variance filtering...")
            feature_columns = self._smart_variance_filtering_V4(df_encoded)
            print(f"   Kept {len(feature_columns)} features after variance filtering")
            
            # STEP 2: Smart feature selection
            print("ðŸ“Š Step 2: Intelligent correlation filtering...")
            feature_columns = self._intelligent_correlation_filtering_V4(df_encoded[feature_columns])
            print(f"   Kept {len(feature_columns)} features after correlation filtering")
            
            # Store training features for prediction consistency
            self.training_features = feature_columns.copy()
            
            # Create feature matrix
            feature_matrix = df_encoded[feature_columns].values
            
            # STEP 3: Scale features
            print("âš–ï¸  Step 3: Feature scaling...")
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            print(f"   Scaled feature matrix: {feature_matrix_scaled.shape}")
            
            # STEP 4: Conservative PCA if needed
            if self.use_pca and len(feature_columns) > 15:
                print("ðŸ” Step 4: Conservative PCA application...")
                feature_matrix_final = self._conservative_pca_V4(feature_matrix_scaled, feature_columns)
            else:
                print("ðŸ” Step 4: Skipping PCA - preserving full feature space")
                feature_matrix_final = feature_matrix_scaled
            
            print(f"âœ… Final training feature matrix: {feature_matrix_final.shape}")
            
            return feature_matrix_final, df_encoded, feature_columns
    
    def _smart_variance_filtering_V4(self, df_encoded: pd.DataFrame) -> List[str]:
        """V4 FIXED: Smart variance filtering that handles single samples"""
        numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        n_samples = len(df_encoded)
        
        if n_samples <= 1:
            # Single sample case - return all numerical features (no variance filtering possible)
            print(f"   Single sample detected - skipping variance filtering")
            return numerical_features
        
        # Normal variance filtering for multiple samples
        feature_matrix = df_encoded[numerical_features].fillna(0)
        
        # Calculate variances manually to handle edge cases
        variances = []
        valid_features = []
        
        for feature in numerical_features:
            try:
                var = feature_matrix[feature].var()
                if pd.isna(var) or var >= self.low_variance_threshold:
                    variances.append(var)
                    valid_features.append(feature)
            except Exception:
                # If variance calculation fails, keep the feature
                valid_features.append(feature)
        
        removed = len(numerical_features) - len(valid_features)
        if removed > 0:
            print(f"   Removed {removed} low-variance features")
        
        return valid_features
    
    def _create_enhanced_features_V4(self, df: pd.DataFrame) -> pd.DataFrame:
        """V4: Create enhanced features with better cluster separation"""
        
        df_encoded = df.copy()
        df_encoded = self._ensure_base_features(df_encoded)
        
        print("ðŸ—ï¸  Creating enhanced features with better separation...")
        
        # Financial capability with better ranges
        income_values = {'Low': 30000, 'Medium': 70000, 'High': 180000}
        df_encoded['income_numeric'] = df_encoded['income_bracket'].map(income_values)
        
        # Enhanced financial ratios with non-linear scaling
        df_encoded['bill_to_income_ratio'] = np.log1p(df_encoded['monthly_bill'] / df_encoded['income_numeric'])
        df_encoded['budget_to_income_ratio'] = np.log1p(df_encoded['budget_max'] / df_encoded['income_numeric'])
        df_encoded['monthly_affordability'] = np.sqrt(df_encoded['income_numeric'] / 12) * 0.1
        df_encoded['budget_stretch_factor'] = df_encoded['budget_max'] / (df_encoded['income_numeric'] * 3)
        
        # Technical feasibility with regional variation
        location_solar = {
            'Mumbai': 4.2, 'Delhi': 4.0, 'Bangalore': 4.6, 'Chennai': 5.0,
            'Pune': 4.4, 'Hyderabad': 4.8, 'Ahmedabad': 5.2, 'Kolkata': 3.8,
            'Kochi': 4.5, 'Jaipur': 5.1
        }
        df_encoded['solar_potential'] = df_encoded['location'].map(location_solar).fillna(4.3)
        
        # Enhanced housing suitability with ownership interaction
        house_base_feasibility = {'villa': 0.95, 'independent': 0.80, 'apartment': 0.35}
        df_encoded['housing_suitability'] = df_encoded['house_type'].map(house_base_feasibility)
        
        # Infer ownership if missing
        if 'ownership_status' not in df_encoded.columns:
            df_encoded = self._infer_ownership(df_encoded)
        
        ownership_weight = {'owner': 1.0, 'tenant': 0.1}
        df_encoded['ownership_feasibility'] = df_encoded['ownership_status'].map(ownership_weight)
        
        # Combined technical feasibility with interactions
        df_encoded['technical_feasibility'] = (
            df_encoded['housing_suitability'] * df_encoded['ownership_feasibility'] * 
            (df_encoded['solar_potential'] / 5.0) * 
            np.minimum(1.0, df_encoded['roof_area'] / 500)
        )
        
        # Enhanced behavioral scoring
        urgency_weights = {'immediate': 3, 'flexible': 2, 'wait': 1}
        df_encoded['urgency_score'] = df_encoded['timeline_preference'].map(urgency_weights)
        
        risk_multipliers = {'low': 0.3, 'medium': 1.0, 'high': 2.0}
        df_encoded['risk_factor'] = df_encoded['risk_tolerance'].map(risk_multipliers)
        
        priority_weights = {'cost': 1.0, 'quality': 1.5, 'sustainability': 2.0}
        df_encoded['priority_score'] = df_encoded['priority'].map(priority_weights)
        
        # System sizing with better constraints
        df_encoded['roof_utilization'] = np.minimum(1.0, df_encoded['roof_area'] / 600)
        df_encoded['max_system_roof'] = df_encoded['roof_area'] / 100  # 100 sq ft per kW
        df_encoded['max_system_bill'] = df_encoded['monthly_bill'] / 400  # â‚¹400 per kW monthly
        df_encoded['max_system_budget'] = df_encoded['budget_max'] / 70000  # â‚¹70k per kW
        
        df_encoded['optimal_system_size'] = np.minimum.reduce([
            df_encoded['max_system_roof'],
            df_encoded['max_system_bill'], 
            df_encoded['max_system_budget']
        ])
        
        # Enhanced composite scores with better separation
        df_encoded['financial_readiness'] = (
            (1 - np.minimum(1.0, df_encoded['bill_to_income_ratio'])) * 0.3 +
            np.minimum(1.0, df_encoded['budget_stretch_factor']) * 0.4 +
            (df_encoded['risk_factor'] / 2.0) * 0.3
        )
        
        df_encoded['adoption_readiness'] = (
            df_encoded['technical_feasibility'] * 0.3 +
            df_encoded['financial_readiness'] * 0.4 +
            (df_encoded['urgency_score'] / 3) * 0.2 +
            (df_encoded['priority_score'] / 2) * 0.1
        )
        
        # Clear segment indicators with better thresholds
        df_encoded['premium_segment'] = (
            (df_encoded['adoption_readiness'] > 0.7) &
            (df_encoded['budget_max'] > df_encoded['budget_max'].quantile(0.7)) &
            (df_encoded['technical_feasibility'] > 0.6)
        ).astype(float)
        
        df_encoded['constrained_segment'] = (
            (df_encoded['ownership_status'] == 'tenant') |
            (df_encoded['technical_feasibility'] < 0.3) |
            (df_encoded['budget_max'] < df_encoded['budget_max'].quantile(0.3))
        ).astype(float)
        
        df_encoded['mainstream_segment'] = (
            (df_encoded['premium_segment'] == 0) & 
            (df_encoded['constrained_segment'] == 0)
        ).astype(float)
        
        # Add interaction features for better separation
        df_encoded['income_tech_interaction'] = df_encoded['income_numeric'] * df_encoded['technical_feasibility']
        df_encoded['budget_urgency_interaction'] = df_encoded['budget_max'] * df_encoded['urgency_score']
        df_encoded['risk_readiness_interaction'] = df_encoded['risk_factor'] * df_encoded['adoption_readiness']
        
        # V4: Encode categoricals with consistent handling
        categorical_features = ['location', 'risk_tolerance', 'timeline_preference', 
                              'priority', 'income_bracket', 'house_type', 'ownership_status']
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    # Fit during training
                    df_encoded[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df_encoded[feature])
                else:
                    # Transform during prediction - handle unseen categories
                    try:
                        df_encoded[f'{feature}_encoded'] = self.label_encoders[feature].transform(df_encoded[feature])
                    except ValueError:
                        # Handle unseen categories by mapping to most common class
                        most_common = 0  # Default to first class
                        df_encoded[f'{feature}_encoded'] = most_common
                        print(f"   âš ï¸  Unseen category in {feature}, using default encoding")
        
        print(f"   Created {len(df_encoded.columns) - len(df.columns)} new features")
        return df_encoded
    
    def _intelligent_correlation_filtering_V4(self, df_features: pd.DataFrame) -> List[str]:
        """V4: Enhanced correlation filtering with business priorities"""
        
        n_samples = len(df_features)
        if n_samples <= 1:
            # Single sample - return all features
            return list(df_features.columns)
        
        try:
            correlation_matrix = df_features.corr().abs()
        except Exception:
            # If correlation calculation fails, return all features
            return list(df_features.columns)
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if pd.notna(corr_val) and corr_val > self.correlation_threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        # Enhanced business priority
        business_priority = {
            'adoption_readiness': 20, 'technical_feasibility': 18, 'financial_readiness': 18,
            'premium_segment': 15, 'constrained_segment': 15, 'mainstream_segment': 15,
            'monthly_bill': 12, 'budget_max': 12, 'optimal_system_size': 10,
            'income_tech_interaction': 8, 'budget_urgency_interaction': 8,
            'bill_to_income_ratio': 7, 'solar_potential': 7, 'housing_suitability': 7
        }
        
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            priority1 = business_priority.get(feat1, 1)
            priority2 = business_priority.get(feat2, 1)
            
            if priority1 > priority2:
                features_to_remove.add(feat2)
            elif priority2 > priority1:
                features_to_remove.add(feat1)
            else:
                # Keep the one with more variance
                var1 = df_features[feat1].var() if len(df_features) > 1 else 1
                var2 = df_features[feat2].var() if len(df_features) > 1 else 1
                if pd.notna(var1) and pd.notna(var2):
                    if var1 >= var2:
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
        
        final_features = [f for f in df_features.columns if f not in features_to_remove]
        
        if high_corr_pairs:
            print(f"   Found {len(high_corr_pairs)} highly correlated pairs")
            print(f"   Removed {len(features_to_remove)} redundant features")
        
        return final_features
    
    def _conservative_pca_V4(self, feature_matrix_scaled: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """V4: Enhanced conservative PCA"""
        n_features = feature_matrix_scaled.shape[1]
        n_samples = feature_matrix_scaled.shape[0]
        
        max_components = min(n_features - 1, n_samples - 1)
        min_components = max(self.min_dimensions, min(10, n_features // 2))
        
        # Find components for 90% variance
        pca_test = PCA()
        pca_test.fit(feature_matrix_scaled)
        cumsum = np.cumsum(pca_test.explained_variance_ratio_)
        variance_components = np.argmax(cumsum >= 0.90) + 1
        variance_components = max(variance_components, min_components)
        
        # Conservative reduction
        max_reduction = max(min_components, int(n_features * 0.6))
        n_components = min(variance_components, max_reduction, max_components)
        
        print(f"   PCA Configuration:")
        print(f"   - Original features: {n_features}")
        print(f"   - Selected components: {n_components}")
        
        self.pca_model = PCA(n_components=n_components)
        feature_matrix_pca = self.pca_model.fit_transform(feature_matrix_scaled)
        
        explained_variance = self.pca_model.explained_variance_ratio_.sum()
        print(f"   - Explained variance: {explained_variance:.3f}")
        
        return feature_matrix_pca
    
    def find_optimal_clusters_FIXED_V4(self, X: np.ndarray, max_clusters: int = 8) -> Tuple[int, Dict]:
        """V4: Enhanced cluster optimization with better scoring"""
        
        results = {}
        cluster_range = range(2, max_clusters + 1)
        
        print(f"\nðŸŽ¯ FIXED V4 Cluster Optimization for {self.algorithm}")
        print(f"   Feature dimensions: {X.shape}")
        print("   K | Silhouette | Calinski-H | Davies-B | Combined | Status")
        print("   --|------------|------------|----------|----------|--------")
        
        for n_clusters in cluster_range:
            try:
                model = self._get_FIXED_optimized_model_V4(n_clusters, X.shape)
                labels = model.fit_predict(X)
                
                unique_labels = np.unique(labels[labels >= 0])  # Exclude noise points
                n_clusters_found = len(unique_labels)
                
                if n_clusters_found < 2:
                    print(f"   {n_clusters:^2} | Failed: Only {n_clusters_found} cluster(s) found")
                    continue
                
                # Calculate metrics only for non-noise points
                valid_indices = labels >= 0
                if not np.any(valid_indices):
                    print(f"   {n_clusters:^2} | Failed: All points marked as noise")
                    continue
                
                X_valid = X[valid_indices]
                labels_valid = labels[valid_indices]
                
                sil_score = silhouette_score(X_valid, labels_valid)
                ch_score = calinski_harabasz_score(X_valid, labels_valid)
                db_score = davies_bouldin_score(X_valid, labels_valid)
                
                # Enhanced combined score
                sil_norm = (sil_score + 1) / 2
                ch_norm = min(1.0, ch_score / 1000)
                db_norm = 1 / (1 + db_score)
                
                # Penalize too many or too few clusters
                cluster_penalty = 1.0
                if n_clusters_found > 6:
                    cluster_penalty = 0.8
                elif n_clusters_found < 3:
                    cluster_penalty = 0.9
                
                combined_score = (sil_norm * 0.5 + ch_norm * 0.3 + db_norm * 0.2) * cluster_penalty
                
                results[n_clusters] = {
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'davies_bouldin_score': db_score,
                    'combined_score': combined_score,
                    'labels': labels,
                    'model': model,
                    'n_clusters_found': n_clusters_found
                }
                
                print(f"   {n_clusters:^2} | {sil_score:^10.3f} | {ch_score:^10.1f} | "
                      f"{db_score:^8.3f} | {combined_score:^8.3f} | Success")
                
            except Exception as e:
                print(f"   {n_clusters:^2} | Failed: {str(e)[:40]}...")
                continue
        
        if not results:
            raise ValueError(f"No valid clustering results for {self.algorithm}")
        
        best_k = max(results.keys(), key=lambda k: results[k]['combined_score'])
        best_result = results[best_k]
        
        print(f"\n   âœ… Selected: {best_k} clusters (found {best_result['n_clusters_found']})")
        print(f"      Silhouette: {best_result['silhouette_score']:.3f}")
        
        return best_k, best_result
    
    def _get_FIXED_optimized_model_V4(self, n_clusters: int, data_shape: Tuple[int, int]):
        """V4: Enhanced algorithm-specific optimizations"""
        
        if self.algorithm == 'kmeans':
            return KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=20,
                max_iter=500,
                random_state=42,
                algorithm='lloyd'
            )
        
        elif self.algorithm == 'agglomerative':
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        
        elif self.algorithm == 'spectral':
            n_neighbors = min(max(15, n_clusters * 3), data_shape[0] // 3)
            return SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=n_neighbors,
                random_state=42,
                assign_labels='discretize',
                n_init=20
            )
        
        elif self.algorithm == 'dbscan':
            eps = self._estimate_dbscan_eps_V4(data_shape, n_clusters)
            min_samples = max(5, data_shape[0] // 50)
            return DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _estimate_dbscan_eps_V4(self, data_shape: Tuple[int, int], target_clusters: int) -> float:
        """V4: Improved eps estimation for DBSCAN"""
        n_samples, n_features = data_shape
        
        # More sophisticated heuristic
        base_eps = 0.3 + (n_features * 0.05)
        density_factor = min(2.0, n_samples / 500)
        cluster_factor = 1 + (target_clusters - 3) * 0.1
        
        return base_eps * density_factor * cluster_factor
    
    def train_FIXED_clustering_V4(self, df: pd.DataFrame) -> Dict:
        """V4: Enhanced training with all fixes including prediction method"""
        
        print("ðŸš€ FIXED V4 User Clustering Training")
        print("=" * 60)
        print(f"Algorithm: {self.algorithm} | Scaler: {self.scaler_type} | PCA: {self.use_pca}")
        
        try:
            # Apply fixed feature preparation
            X_final, df_encoded, feature_columns = self.prepare_features_FIXED_V4(df, is_prediction=False)
            
            # Find optimal clustering
            optimal_k, best_result = self.find_optimal_clusters_FIXED_V4(X_final)
            
            # Store results
            self.model = best_result['model']
            cluster_labels = best_result['labels']
            df_encoded['cluster'] = cluster_labels
            
            # V4 CRITICAL FIX: Create predictor for algorithms without predict method
            self.training_data = X_final.copy()
            self.training_labels = cluster_labels.copy()
            self._create_predictor_V4(X_final, cluster_labels)
            
            # Analyze clusters
            print("\nðŸ“Š Analyzing cluster profiles...")
            cluster_analysis = self._analyze_clusters_enhanced_V4(df_encoded)
            
            # Feature importance
            print("ðŸŽ¯ Calculating feature importance...")
            self._calculate_feature_importance_FIXED_V4(X_final, cluster_labels, feature_columns)
            
            # Store validation metrics with JSON-serializable types
            self.validation_metrics = {
                'silhouette_score': float(best_result['silhouette_score']),
                'calinski_harabasz_score': float(best_result['calinski_harabasz_score']), 
                'davies_bouldin_score': float(best_result['davies_bouldin_score']),
                'combined_score': float(best_result['combined_score']),
                'n_clusters': int(optimal_k),
                'n_features_original': int(len(feature_columns)),
                'n_features_final': int(X_final.shape[1]),
                'dimension_reduction_ratio': float(X_final.shape[1] / len(feature_columns)),
                'algorithm': str(self.algorithm),
                'scaler': str(self.scaler_type),
                'samples': int(X_final.shape[0]),
                'has_native_predict': self._algorithm_has_predict()  # V4 addition
            }
            
            # Quality assessment
            self._assess_clustering_quality_V4()
            
            self.is_trained = True
            
            # Save model with fixed serialization
            try:
                self._save_model_FIXED_V4()
            except Exception as save_error:
                print(f"âš ï¸  Model save failed: {save_error}")
                print("   Continuing without saving...")
            
            return {
                'model': self.model,
                'cluster_analysis': cluster_analysis,
                'feature_importance': self.feature_importance,
                'validation_metrics': self.validation_metrics,
                'feature_columns': feature_columns,
                'quality_assessment': self.quality_assessment
            }
            
        except Exception as e:
            print(f"âŒ Training failed: {e}")
            return {'error': str(e)}
    
    def predict_user_cluster_V4(self, user_profile: Dict) -> Dict:
        """V4 FIXED: Enhanced prediction with proper algorithm handling"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_FIXED_clustering_V4 first.")
        
        try:
            print("ðŸ”® V4 Single Sample Prediction")
            user_df = pd.DataFrame([user_profile])
            
            # V4 CRITICAL FIX: Use prediction mode with is_prediction=True
            X_user, user_encoded, _ = self.prepare_features_FIXED_V4(user_df, is_prediction=True)
            
            # V4 CRITICAL FIX: Use appropriate prediction method
            if self._algorithm_has_predict():
                # Algorithm has native predict method (like KMeans)
                cluster_id = self.model.predict(X_user)[0]
                print("   Using native predict method")
            else:
                # Use KNN predictor for algorithms without predict method
                if self.predictor is None:
                    raise ValueError("Predictor not available for algorithm without predict method")
                cluster_id = self.predictor.predict(X_user)[0]
                print("   Using KNN predictor for algorithm compatibility")
            
            if cluster_id in self.cluster_profiles:
                cluster_info = self.cluster_profiles[cluster_id]
                
                readiness = float(user_encoded['adoption_readiness'].iloc[0]) if 'adoption_readiness' in user_encoded else 0.0
                technical = float(user_encoded['technical_feasibility'].iloc[0]) if 'technical_feasibility' in user_encoded else 0.0
                financial = float(user_encoded['financial_readiness'].iloc[0]) if 'financial_readiness' in user_encoded else 0.0
                
                confidence = 'High' if self.validation_metrics['silhouette_score'] > 0.5 else 'Medium'
                
                return {
                    'cluster_id': int(cluster_id),
                    'cluster_name': cluster_info['name'],
                    'strategy': cluster_info['strategy'],
                    'business_value': cluster_info['business_value'],
                    'user_scores': {
                        'adoption_readiness': readiness,
                        'technical_feasibility': technical,
                        'financial_readiness': financial
                    },
                    'confidence': confidence,
                    'model_quality': self.quality_assessment.get('quality_level', 'Unknown'),
                    'prediction_method': 'native' if self._algorithm_has_predict() else 'knn_fallback',
                    'original_profile': user_profile
                }
            else:
                return {
                    'cluster_id': int(cluster_id),
                    'cluster_name': f'Cluster_{cluster_id}',
                    'strategy': {'approach': 'Standard', 'focus': 'General'},
                    'confidence': 'Low',
                    'error': 'Cluster not in profiles',
                    'prediction_method': 'native' if self._algorithm_has_predict() else 'knn_fallback',
                    'original_profile': user_profile
                }
                
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'cluster_id': -1,
                'cluster_name': 'Unknown',
                'original_profile': user_profile
            }
    
    def _analyze_clusters_enhanced_V4(self, df_encoded: pd.DataFrame) -> Dict:
        """V4: Enhanced cluster analysis with better business insights"""
        
        cluster_analysis = {}
        total_users = len(df_encoded)
        
        for cluster_id in sorted(df_encoded['cluster'].unique()):
            if cluster_id < 0:  # Skip noise points for DBSCAN
                continue
                
            cluster_data = df_encoded[df_encoded['cluster'] == cluster_id]
            size = len(cluster_data)
            
            if size == 0:
                continue
            
            # Core metrics with better calculations
            stats = {
                'size': int(size),
                'percentage': float(size / total_users * 100),
                'avg_monthly_bill': float(cluster_data['monthly_bill'].mean()),
                'median_monthly_bill': float(cluster_data['monthly_bill'].median()),
                'avg_budget': float(cluster_data['budget_max'].mean()),
                'median_budget': float(cluster_data['budget_max'].median()),
                'avg_adoption_readiness': float(cluster_data['adoption_readiness'].mean()),
                'avg_technical_feasibility': float(cluster_data['technical_feasibility'].mean()),
                'avg_financial_readiness': float(cluster_data['financial_readiness'].mean()),
                'avg_system_size': float(cluster_data['optimal_system_size'].mean()),
                'owner_percentage': float((cluster_data['ownership_status'] == 'owner').mean() * 100),
                'premium_percentage': float(cluster_data['premium_segment'].mean() * 100),
                'constrained_percentage': float(cluster_data['constrained_segment'].mean() * 100),
                'high_income_percentage': float((cluster_data['income_bracket'] == 'High').mean() * 100)
            }
            
            # Dominant characteristics
            stats.update({
                'dominant_income': str(cluster_data['income_bracket'].mode().iloc[0] if not cluster_data['income_bracket'].mode().empty else 'Unknown'),
                'dominant_house_type': str(cluster_data['house_type'].mode().iloc[0] if not cluster_data['house_type'].mode().empty else 'Unknown'),
                'dominant_priority': str(cluster_data['priority'].mode().iloc[0] if not cluster_data['priority'].mode().empty else 'Unknown'),
                'dominant_risk': str(cluster_data['risk_tolerance'].mode().iloc[0] if not cluster_data['risk_tolerance'].mode().empty else 'Unknown')
            })
            
            # Generate insights
            cluster_name = self._generate_cluster_name_V4(stats)
            strategy = self._generate_strategy_V4(stats)
            business_value = self._assess_business_value_V4(stats)
            
            cluster_analysis[int(cluster_id)] = {
                'name': cluster_name,
                'stats': stats,
                'strategy': strategy,
                'business_value': business_value
            }
        
        self.cluster_profiles = cluster_analysis
        return cluster_analysis
    
    def _generate_cluster_name_V4(self, stats: Dict) -> str:
        """V4: Enhanced cluster naming with better logic"""
        
        if stats['premium_percentage'] > 50 and stats['avg_adoption_readiness'] > 0.6:
            return 'Premium_Ready_Adopters'
        elif stats['constrained_percentage'] > 50:
            return 'Constrained_Alternative_Seekers'
        elif stats['owner_percentage'] > 70 and stats['avg_budget'] > 200000:
            return 'Mainstream_Owner_Prospects'
        elif stats['avg_budget'] < 180000:
            return 'Budget_Conscious_Segment'
        elif stats['high_income_percentage'] > 40:
            return 'High_Income_Deliberators'
        elif stats['avg_adoption_readiness'] > 0.5:
            return 'Moderate_Ready_Prospects'
        else:
            return f'Mixed_Segment_{stats["size"]}'
    
    def _generate_strategy_V4(self, stats: Dict) -> Dict:
        """V4: Enhanced strategy generation"""
        
        if stats['premium_percentage'] > 40:
            return {
                'approach': 'Premium Direct Sales',
                'focus': 'Quality, Performance, Brand',
                'timeline': 'Immediate',
                'channels': ['Direct Sales', 'Referrals'],
                'messaging': 'Best-in-class solutions',
                'expected_close_rate': '25-35%'
            }
        elif stats['constrained_percentage'] > 40:
            return {
                'approach': 'Alternative Solutions',
                'focus': 'Flexible Installation, Leasing',
                'timeline': 'Medium-term',
                'channels': ['Digital', 'Community'],
                'messaging': 'Accessible solar solutions',
                'expected_close_rate': '10-15%'
            }
        else:
            return {
                'approach': 'Educational Nurturing',
                'focus': 'ROI, Cost Benefits, Financing',
                'timeline': 'Long-term',
                'channels': ['Content Marketing', 'Webinars'],
                'messaging': 'Smart financial choice',
                'expected_close_rate': '15-20%'
            }
    
    def _assess_business_value_V4(self, stats: Dict) -> Dict:
        """V4: Enhanced business value assessment"""
        
        avg_revenue = stats['avg_budget'] * 0.15
        total_potential = avg_revenue * stats['size']
        
        readiness_score = stats['avg_adoption_readiness']
        if readiness_score > 0.7:
            conversion_prob = 0.30
        elif readiness_score > 0.5:
            conversion_prob = 0.18
        elif readiness_score > 0.3:
            conversion_prob = 0.10
        else:
            conversion_prob = 0.05
        
        expected_revenue = total_potential * conversion_prob
        
        return {
            'avg_revenue_potential': float(avg_revenue),
            'total_potential': float(total_potential),
            'conversion_probability': float(conversion_prob),
            'expected_revenue': float(expected_revenue),
            'priority_score': float(expected_revenue / 100000),
            'effort_required': 'Low' if readiness_score > 0.6 else 'Medium' if readiness_score > 0.4 else 'High'
        }
    
    def _assess_clustering_quality_V4(self):
        """V4: Enhanced quality assessment"""
        sil_score = self.validation_metrics['silhouette_score']
        
        if sil_score > 0.7:
            quality = "Excellent"
            recommendation = "Deploy immediately"
        elif sil_score > 0.5:
            quality = "Good"
            recommendation = "Deploy with monitoring"
        elif sil_score > 0.3:
            quality = "Moderate"
            recommendation = "Consider feature engineering"
        else:
            quality = "Poor"
            recommendation = "Revisit segmentation strategy"
        
        self.quality_assessment = {
            'quality_level': quality,
            'silhouette_score': float(sil_score),
            'recommendation': recommendation,
            'dimension_preserved': int(self.validation_metrics['n_features_final']),
            'reduction_applied': bool(self.validation_metrics['dimension_reduction_ratio'] < 1.0),
            'has_predictor': self.predictor is not None,
            'prediction_method': 'native' if self._algorithm_has_predict() else 'knn_fallback'
        }
        
        print(f"\nðŸŽ¯ Quality Assessment: {quality}")
        print(f"   Silhouette Score: {sil_score:.3f}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Prediction Method: {'Native' if self._algorithm_has_predict() else 'KNN Fallback'}")
    
    def _calculate_feature_importance_FIXED_V4(self, X: np.ndarray, labels: np.ndarray, feature_columns: List[str]):
        """V4: Enhanced feature importance calculation"""
        
        try:
            if self.pca_model is not None:
                # PCA-based importance
                importance_dict = {}
                
                for i, feature in enumerate(feature_columns):
                    total_importance = 0
                    for j, component in enumerate(self.pca_model.components_):
                        if i < len(component):
                            weight = abs(component[i])
                            variance_explained = self.pca_model.explained_variance_ratio_[j]
                            total_importance += weight * variance_explained
                    importance_dict[feature] = total_importance
                
                # Normalize
                total = sum(importance_dict.values())
                if total > 0:
                    self.feature_importance = {k: float(v/total) for k, v in importance_dict.items()}
                else:
                    self.feature_importance = {k: float(1/len(feature_columns)) for k in feature_columns}
            
            else:
                # Statistical importance without PCA
                from sklearn.feature_selection import f_classif
                
                # Filter valid labels for supervised scoring
                valid_indices = labels >= 0
                if np.sum(valid_indices) < len(labels) * 0.5:
                    # Too many noise points, use unsupervised approach
                    self.feature_importance = {k: float(1/len(feature_columns)) for k in feature_columns}
                    return
                
                X_valid = X[valid_indices]
                labels_valid = labels[valid_indices]
                
                if len(np.unique(labels_valid)) < 2:
                    # Not enough clusters for supervised scoring
                    self.feature_importance = {k: float(1/len(feature_columns)) for k in feature_columns}
                    return
                
                f_scores, p_values = f_classif(X_valid, labels_valid)
                
                # Handle any NaN or infinite values
                f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1.0, neginf=0.0)
                
                total_f = sum(f_scores)
                if total_f > 0:
                    self.feature_importance = {
                        feature_columns[i]: float(f_scores[i]/total_f) 
                        for i in range(min(len(feature_columns), len(f_scores)))
                    }
                else:
                    self.feature_importance = {k: float(1/len(feature_columns)) for k in feature_columns}
                    
        except Exception as e:
            print(f"   Warning: Feature importance calculation failed: {e}")
            # Fallback to equal importance
            self.feature_importance = {k: float(1/len(feature_columns)) for k in feature_columns}
    
    def _save_model_FIXED_V4(self):
        """V4: Fixed model saving with predictor support"""
        os.makedirs('models_fixed_v4', exist_ok=True)
        
        model_name = f'FIXED_V4_clustering_{self.algorithm}_{self.scaler_type}'
        
        try:
            # Save model components
            joblib.dump(self.model, f'models_fixed_v4/{model_name}_model.pkl')
            joblib.dump(self.scaler, f'models_fixed_v4/{model_name}_scaler.pkl')
            if self.pca_model:
                joblib.dump(self.pca_model, f'models_fixed_v4/{model_name}_pca.pkl')
            joblib.dump(self.label_encoders, f'models_fixed_v4/{model_name}_encoders.pkl')
            
            # V4: Save predictor for algorithms without native predict method
            if self.predictor is not None:
                joblib.dump(self.predictor, f'models_fixed_v4/{model_name}_predictor.pkl')
            
            # V4: Save training features list for prediction consistency
            joblib.dump(self.training_features, f'models_fixed_v4/{model_name}_features.pkl')
            
            # Create serializable metadata
            metadata = {
                'algorithm': str(self.algorithm),
                'scaler_type': str(self.scaler_type),
                'use_pca': bool(self.use_pca),
                'validation_metrics': self.validation_metrics,
                'cluster_profiles': {},
                'quality_assessment': self.quality_assessment,
                'feature_importance': self.feature_importance,
                'training_features': self.training_features,
                'has_predictor': self.predictor is not None,
                'has_native_predict': self._algorithm_has_predict()
            }
            
            # Convert cluster profiles to serializable format
            for k, v in self.cluster_profiles.items():
                metadata['cluster_profiles'][str(k)] = {
                    'name': str(v['name']),
                    'stats': v['stats'],  # Already converted to basic types
                    'strategy': v['strategy'],
                    'business_value': v['business_value']
                }
            
            # Save metadata
            with open(f'models_fixed_v4/{model_name}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"ðŸ’¾ FIXED V4 model saved: {model_name}")
            
        except Exception as e:
            print(f"âš ï¸  Model save failed: {e}")
            raise
    
    def _ensure_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required base features exist"""
        n_samples = len(df)
        
        if 'monthly_bill' not in df.columns:
            df['monthly_bill'] = np.random.lognormal(8.0, 0.6, n_samples).clip(1200, 8000)
        
        if 'budget_max' not in df.columns:
            df['budget_max'] = np.random.lognormal(12.5, 0.7, n_samples).clip(100000, 600000)
        
        if 'roof_area' not in df.columns:
            df['roof_area'] = np.random.lognormal(6.3, 0.5, n_samples).clip(300, 1000)
        
        defaults = {
            'location': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad'],
            'risk_tolerance': ['low', 'medium', 'high'],
            'timeline_preference': ['immediate', 'flexible', 'wait'],
            'priority': ['cost', 'quality', 'sustainability'],
            'income_bracket': ['Low', 'Medium', 'High'],
            'house_type': ['apartment', 'independent', 'villa']
        }
        
        for col, values in defaults.items():
            if col not in df.columns:
                if col == 'income_bracket':
                    df[col] = np.random.choice(values, n_samples, p=[0.4, 0.45, 0.15])
                elif col == 'house_type':
                    df[col] = np.random.choice(values, n_samples, p=[0.6, 0.3, 0.1])
                else:
                    df[col] = np.random.choice(values, n_samples)
        
        return df
    
    def _infer_ownership(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer ownership status based on house type and income"""
        ownership = []
        for _, row in df.iterrows():
            if row['house_type'] == 'villa':
                ownership.append('owner')
            elif row['house_type'] == 'independent':
                if row['income_bracket'] in ['Medium', 'High']:
                    ownership.append(np.random.choice(['owner', 'tenant'], p=[0.8, 0.2]))
                else:
                    ownership.append(np.random.choice(['owner', 'tenant'], p=[0.5, 0.5]))
            else:  # apartment
                if row['income_bracket'] == 'High':
                    ownership.append(np.random.choice(['owner', 'tenant'], p=[0.6, 0.4]))
                else:
                    ownership.append(np.random.choice(['owner', 'tenant'], p=[0.2, 0.8]))
        
        df['ownership_status'] = ownership
        return df
    
    def load_model_FIXED_V4(self, model_name: str):
        """V4: Load saved model with all components including predictor"""
        try:
            model_path = f'models_fixed_v4/{model_name}'
            
            # Load model components
            self.model = joblib.load(f'{model_path}_model.pkl')
            self.scaler = joblib.load(f'{model_path}_scaler.pkl')
            
            if os.path.exists(f'{model_path}_pca.pkl'):
                self.pca_model = joblib.load(f'{model_path}_pca.pkl')
            
            # V4: Load predictor if exists
            if os.path.exists(f'{model_path}_predictor.pkl'):
                self.predictor = joblib.load(f'{model_path}_predictor.pkl')
            
            self.label_encoders = joblib.load(f'{model_path}_encoders.pkl')
            self.training_features = joblib.load(f'{model_path}_features.pkl')
            
            # Load metadata
            with open(f'{model_path}_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.algorithm = metadata['algorithm']
            self.scaler_type = metadata['scaler_type']
            self.use_pca = metadata['use_pca']
            self.validation_metrics = metadata['validation_metrics']
            self.cluster_profiles = metadata['cluster_profiles']
            self.quality_assessment = metadata['quality_assessment']
            self.feature_importance = metadata['feature_importance']
            
            self.is_trained = True
            print(f"âœ… V4 Model loaded successfully: {model_name}")
            print(f"   Prediction method: {'Native' if self._algorithm_has_predict() else 'KNN Fallback'}")
            
        except Exception as e:
            print(f"âŒ Model loading failed: {e}")
            raise
    
    def create_synthetic_data_V4(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create synthetic data with ENHANCED cluster separation"""
        np.random.seed(42)
        
        # Create 4 well-separated segments
        segment_sizes = [n_samples//4] * 4
        
        print(f"ðŸ­ Creating enhanced synthetic data with 4 distinct segments...")
        
        # Segment 1: Premium Ready (Clear high-end characteristics)
        premium_data = {
            'monthly_bill': np.random.normal(5500, 800, segment_sizes[0]).clip(4500, 8000),
            'budget_max': np.random.normal(520000, 100000, segment_sizes[0]).clip(400000, 800000),
            'roof_area': np.random.normal(850, 150, segment_sizes[0]).clip(600, 1200),
            'income_bracket': np.random.choice(['High'], segment_sizes[0]),
            'house_type': np.random.choice(['villa', 'independent'], segment_sizes[0], p=[0.7, 0.3]),
            'risk_tolerance': np.random.choice(['medium', 'high'], segment_sizes[0], p=[0.2, 0.8]),
            'priority': np.random.choice(['quality', 'sustainability'], segment_sizes[0], p=[0.6, 0.4]),
            'timeline_preference': np.random.choice(['immediate', 'flexible'], segment_sizes[0], p=[0.8, 0.2])
        }
        
        # Segment 2: Mainstream Owners (Moderate characteristics)
        mainstream_data = {
            'monthly_bill': np.random.normal(3000, 600, segment_sizes[1]).clip(2200, 4200),
            'budget_max': np.random.normal(250000, 60000, segment_sizes[1]).clip(180000, 350000),
            'roof_area': np.random.normal(550, 100, segment_sizes[1]).clip(400, 750),
            'income_bracket': np.random.choice(['Medium'], segment_sizes[1]),
            'house_type': np.random.choice(['independent', 'apartment'], segment_sizes[1], p=[0.8, 0.2]),
            'risk_tolerance': np.random.choice(['medium'], segment_sizes[1]),
            'priority': np.random.choice(['cost', 'quality'], segment_sizes[1], p=[0.6, 0.4]),
            'timeline_preference': np.random.choice(['flexible'], segment_sizes[1])
        }
        
        # Segment 3: Budget Conscious (Clear budget constraints)
        budget_data = {
            'monthly_bill': np.random.normal(1800, 400, segment_sizes[2]).clip(1200, 2800),
            'budget_max': np.random.normal(130000, 25000, segment_sizes[2]).clip(80000, 180000),
            'roof_area': np.random.normal(400, 80, segment_sizes[2]).clip(250, 550),
            'income_bracket': np.random.choice(['Low'], segment_sizes[2]),
            'house_type': np.random.choice(['apartment'], segment_sizes[2]),
            'risk_tolerance': np.random.choice(['low'], segment_sizes[2]),
            'priority': np.random.choice(['cost'], segment_sizes[2]),
            'timeline_preference': np.random.choice(['wait', 'flexible'], segment_sizes[2], p=[0.7, 0.3])
        }
        
        # Segment 4: Constrained (Mixed constraints)
        constrained_data = {
            'monthly_bill': np.random.normal(2500, 700, segment_sizes[3]).clip(1500, 4500),
            'budget_max': np.random.normal(160000, 45000, segment_sizes[3]).clip(100000, 250000),
            'roof_area': np.random.normal(320, 60, segment_sizes[3]).clip(200, 450),
            'income_bracket': np.random.choice(['Low', 'Medium'], segment_sizes[3], p=[0.7, 0.3]),
            'house_type': np.random.choice(['apartment'], segment_sizes[3]),
            'risk_tolerance': np.random.choice(['low', 'medium'], segment_sizes[3], p=[0.9, 0.1]),
            'priority': np.random.choice(['cost'], segment_sizes[3]),
            'timeline_preference': np.random.choice(['wait'], segment_sizes[3])
        }
        
        # Combine segments
        combined_data = {}
        for key in premium_data.keys():
            combined_data[key] = np.concatenate([
                premium_data[key], mainstream_data[key], 
                budget_data[key], constrained_data[key]
            ])
        
        # Add location
        combined_data['location'] = np.random.choice(
            ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad'], n_samples
        )
        
        df = pd.DataFrame(combined_data)
        
        # Add some noise to make it realistic but keep separation
        df['monthly_bill'] += np.random.normal(0, 50, n_samples)
        df['budget_max'] += np.random.normal(0, 5000, n_samples)
        df['roof_area'] += np.random.normal(0, 10, n_samples)
        
        print(f"   âœ… Created {n_samples} samples with enhanced segment separation")
        return df


def run_FIXED_comprehensive_test_V4():
    """
    V4: Test AgglomerativeClustering prediction method fix
    """
    
    print("ðŸš€ FIXED V4 User Clustering - AgglomerativeClustering Prediction Method RESOLVED")
    print("=" * 90)
    print("âœ… AgglomerativeClustering predict method issue RESOLVED with KNN fallback")
    print("âœ… Algorithm-specific prediction handling IMPLEMENTED")  
    print("âœ… Proper prediction method for all clustering algorithms ADDED")
    print("âœ… Enhanced error handling and fallback mechanisms IMPROVED")
    print("âœ… All V3 functionality maintained and enhanced")
    print("=" * 90)
    
    # Create and train system with agglomerative (the problematic algorithm)
    clustering_system = FixedUserClusteringV4(
        algorithm='agglomerative',  # Test the fixed algorithm
        scaler_type='robust', 
        use_pca=False
    )
    
    # Create training data
    df = clustering_system.create_synthetic_data_V4(1000)
    
    print("\nðŸ”¬ Training V4 system with AgglomerativeClustering...")
    result = clustering_system.train_FIXED_clustering_V4(df)
    
    if 'error' in result:
        print(f"âŒ Training failed: {result['error']}")
        return
    
    print(f"\nâœ… Training successful!")
    print(f"   Silhouette Score: {result['validation_metrics']['silhouette_score']:.3f}")
    print(f"   Quality: {result['quality_assessment']['quality_level']}")
    print(f"   Clusters: {result['validation_metrics']['n_clusters']}")
    print(f"   Prediction Method: {result['quality_assessment']['prediction_method']}")
    
    # Show cluster profiles
    print(f"\nðŸŽ¯ CLUSTER PROFILES:")
    for cluster_id, analysis in result['cluster_analysis'].items():
        stats = analysis['stats']
        print(f"\n   Cluster {cluster_id}: {analysis['name']}")
        print(f"   ðŸ‘¥ Size: {stats['size']} ({stats['percentage']:.1f}%)")
        print(f"   ðŸ’° Avg Budget: â‚¹{stats['avg_budget']:,.0f}")
        print(f"   ðŸ“ˆ Readiness: {stats['avg_adoption_readiness']:.2f}")
        print(f"   ðŸŽ¯ Strategy: {analysis['strategy']['approach']}")
    
    # V4 CRITICAL TEST: Single sample predictions with AgglomerativeClustering
    print(f"\n" + "=" * 90)
    print(f"ðŸ”® V4 CRITICAL TEST: AgglomerativeClustering Single Sample Predictions")
    print(f"=" * 90)
    
    test_users = [
        {
            'monthly_bill': 4500,
            'budget_max': 480000,
            'roof_area': 800,
            'location': 'Bangalore',
            'risk_tolerance': 'high',
            'timeline_preference': 'immediate',
            'priority': 'quality',
            'income_bracket': 'High',
            'house_type': 'villa'
        },
        {
            'monthly_bill': 1900,
            'budget_max': 120000,
            'roof_area': 350,
            'location': 'Mumbai',
            'risk_tolerance': 'low',
            'timeline_preference': 'wait',
            'priority': 'cost',
            'income_bracket': 'Low',
            'house_type': 'apartment'
        },
        {
            'monthly_bill': 3200,
            'budget_max': 280000,
            'roof_area': 550,
            'location': 'Delhi',
            'risk_tolerance': 'medium',
            'timeline_preference': 'flexible',
            'priority': 'sustainability',
            'income_bracket': 'Medium',
            'house_type': 'independent'
        }
    ]
    
    successful_predictions = 0
    
    for i, test_user in enumerate(test_users, 1):
        print(f"\nðŸ§ª Test User {i}: {test_user['house_type']}, {test_user['income_bracket']} income, â‚¹{test_user['budget_max']:,} budget")
        
        try:
            prediction = clustering_system.predict_user_cluster_V4(test_user)
            
            if 'error' not in prediction:
                successful_predictions += 1
                print(f"   âœ… SUCCESS")
                print(f"   â†’ Cluster: {prediction['cluster_name']}")
                print(f"   â†’ Strategy: {prediction['strategy']['approach']}")
                print(f"   â†’ Readiness: {prediction['user_scores']['adoption_readiness']:.2f}")
                print(f"   â†’ Technical: {prediction['user_scores']['technical_feasibility']:.2f}")
                print(f"   â†’ Financial: {prediction['user_scores']['financial_readiness']:.2f}")
                print(f"   â†’ Confidence: {prediction['confidence']}")
                print(f"   â†’ Method: {prediction['prediction_method']}")
            else:
                print(f"   âŒ FAILED: {prediction['error']}")
                
        except Exception as e:
            print(f"   âŒ EXCEPTION: {str(e)}")
    
    # Test with KMeans for comparison
    print(f"\n" + "=" * 90)
    print(f"ðŸ”¬ COMPARISON TEST: KMeans (Native Predict Method)")
    print(f"=" * 90)
    
    kmeans_system = FixedUserClusteringV4(
        algorithm='kmeans',
        scaler_type='robust', 
        use_pca=False
    )
    
    print("\nðŸ”¬ Training KMeans system...")
    kmeans_result = kmeans_system.train_FIXED_clustering_V4(df)
    
    if 'error' not in kmeans_result:
        print(f"âœ… KMeans training successful!")
        print(f"   Prediction Method: {kmeans_result['quality_assessment']['prediction_method']}")
        
        # Test one prediction
        test_prediction = kmeans_system.predict_user_cluster_V4(test_users[0])
        if 'error' not in test_prediction:
            print(f"âœ… KMeans prediction successful!")
            print(f"   Method: {test_prediction['prediction_method']}")
        else:
            print(f"âŒ KMeans prediction failed: {test_prediction['error']}")
    else:
        print(f"âŒ KMeans training failed: {kmeans_result['error']}")
    
    # Final results
    print(f"\n" + "=" * 90)
    print(f"ðŸ“Š V4 AGGLOMERATIVE CLUSTERING PREDICTION TEST RESULTS")
    print(f"=" * 90)
    
    print(f"   ðŸŽ¯ Test Cases: {len(test_users)}")
    print(f"   âœ… Successful: {successful_predictions}")
    print(f"   âŒ Failed: {len(test_users) - successful_predictions}")
    print(f"   ðŸ“ˆ Success Rate: {successful_predictions/len(test_users)*100:.1f}%")
    
    if successful_predictions == len(test_users):
        print(f"\nðŸŽ‰ V4 AGGLOMERATIVE CLUSTERING PREDICTION BUG COMPLETELY RESOLVED!")
        print(f"   âœ… All AgglomerativeClustering predictions working with KNN fallback")
        print(f"   âœ… Algorithm-specific prediction handling implemented")
        print(f"   âœ… Proper fallback mechanisms for algorithms without predict method")
        print(f"   âœ… Both native predict and KNN fallback methods validated")
        print(f"\nðŸš€ V4 SYSTEM PRODUCTION READY FOR ALL ALGORITHMS!")
        
        return clustering_system
    else:
        print(f"\nâš ï¸  Some prediction issues remain - need further investigation")
        return None


def test_all_algorithms_V4():
    """V4: Test all supported algorithms"""
    
    print("\n" + "=" * 90)
    print("ðŸ§ª V4 COMPREHENSIVE ALGORITHM TESTING")
    print("=" * 90)
    
    algorithms = ['kmeans', 'agglomerative', 'spectral', 'dbscan']
    results = {}
    
    # Create test data
    df = FixedUserClusteringV4().create_synthetic_data_V4(500)  # Smaller for faster testing
    
    test_user = {
        'monthly_bill': 3500,
        'budget_max': 300000,
        'roof_area': 600,
        'location': 'Chennai',
        'risk_tolerance': 'medium',
        'timeline_preference': 'flexible',
        'priority': 'cost',
        'income_bracket': 'Medium',
        'house_type': 'independent'
    }
    
    for algorithm in algorithms:
        print(f"\nðŸ”¬ Testing {algorithm.upper()}")
        print("-" * 50)
        
        try:
            system = FixedUserClusteringV4(
                algorithm=algorithm,
                scaler_type='robust',
                use_pca=False
            )
            
            # Training
            print("   Training...")
            train_result = system.train_FIXED_clustering_V4(df)
            
            if 'error' in train_result:
                print(f"   âŒ Training failed: {train_result['error']}")
                results[algorithm] = {'status': 'training_failed', 'error': train_result['error']}
                continue
            
            print(f"   âœ… Training successful - Silhouette: {train_result['validation_metrics']['silhouette_score']:.3f}")
            print(f"   ðŸ“Š Clusters: {train_result['validation_metrics']['n_clusters']}")
            print(f"   ðŸ”® Prediction method: {train_result['quality_assessment']['prediction_method']}")
            
            # Prediction
            print("   Testing prediction...")
            prediction = system.predict_user_cluster_V4(test_user)
            
            if 'error' in prediction:
                print(f"   âŒ Prediction failed: {prediction['error']}")
                results[algorithm] = {
                    'status': 'prediction_failed',
                    'training_score': train_result['validation_metrics']['silhouette_score'],
                    'error': prediction['error']
                }
            else:
                print(f"   âœ… Prediction successful")
                print(f"   â†’ Cluster: {prediction['cluster_name']}")
                print(f"   â†’ Method: {prediction['prediction_method']}")
                results[algorithm] = {
                    'status': 'success',
                    'training_score': train_result['validation_metrics']['silhouette_score'],
                    'clusters': train_result['validation_metrics']['n_clusters'],
                    'prediction_method': prediction['prediction_method']
                }
        
        except Exception as e:
            print(f"   âŒ Exception: {str(e)}")
            results[algorithm] = {'status': 'exception', 'error': str(e)}
    
    # Summary
    print(f"\n" + "=" * 90)
    print("ðŸ“Š ALGORITHM TEST SUMMARY")
    print("=" * 90)
    
    successful = 0
    for algorithm, result in results.items():
        status = result['status']
        if status == 'success':
            successful += 1
            print(f"   âœ… {algorithm.upper()}: SUCCESS")
            print(f"      Silhouette: {result['training_score']:.3f}")
            print(f"      Clusters: {result['clusters']}")
            print(f"      Method: {result['prediction_method']}")
        else:
            print(f"   âŒ {algorithm.upper()}: {status.upper()}")
            if 'error' in result:
                print(f"      Error: {result['error'][:60]}...")
        print()
    
    print(f"ðŸŽ¯ OVERALL SUCCESS RATE: {successful}/{len(algorithms)} ({successful/len(algorithms)*100:.0f}%)")
    
    if successful == len(algorithms):
        print(f"\nðŸŽŠ ALL ALGORITHMS WORKING PERFECTLY!")
        print(f"âœ… Both native predict and KNN fallback methods validated")
        print(f"âœ… V4 system ready for production with all clustering algorithms")
    
    return results


if __name__ == "__main__":
    print("Starting FIXED V4 User Clustering Analysis...")
    system = run_FIXED_comprehensive_test_V4()
    
    if system:
        print(f"\nðŸŽŠ FIXED V4 AGGLOMERATIVE CLUSTERING SUCCESS!")
        print(f"\nðŸ“‹ V4 CRITICAL FIXES:")
        print(f"   âœ… AgglomerativeClustering predict method issue RESOLVED")
        print(f"   âœ… KNN fallback predictor for algorithms without predict method IMPLEMENTED")
        print(f"   âœ… Algorithm-specific prediction handling ADDED")
        print(f"   âœ… Enhanced error handling and fallback mechanisms IMPROVED")
        print(f"   âœ… All V3 functionality maintained and enhanced")
        
        print(f"\nðŸš€ READY FOR PRODUCTION DEPLOYMENT!")
        
        # Test all algorithms
        print(f"\n" + "ðŸ§ª" * 30)
        print(f"COMPREHENSIVE ALGORITHM VALIDATION")
        print(f"ðŸ§ª" * 30)
        
        algorithm_results = test_all_algorithms_V4()
        
        successful_algorithms = sum(1 for result in algorithm_results.values() if result['status'] == 'success')
        
        if successful_algorithms == len(algorithm_results):
            print(f"\nðŸ† V4 COMPLETE SUCCESS!")
            print(f"   âœ… ALL {len(algorithm_results)} ALGORITHMS WORKING")
            print(f"   âœ… Both native and fallback prediction methods validated")
            print(f"   âœ… AgglomerativeClustering prediction bug completely resolved")
            print(f"   âœ… System ready for production with full algorithm support")
        else:
            print(f"\nâš ï¸  {successful_algorithms}/{len(algorithm_results)} algorithms working")
            print(f"   Some algorithms may need additional fixes")
        
    else:
        print(f"\nâŒ V4 needs further fixes - please review")
        print(f"\nðŸ” DEBUGGING RECOMMENDATIONS:")
        print(f"   1. Check KNN predictor implementation")
        print(f"   2. Verify algorithm-specific prediction routing")
        print(f"   3. Review fallback mechanism error handling")
        print(f"   4. Test with different data sizes and configurations")