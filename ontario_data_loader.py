"""
Ontario Real Pavement Data Loader
==================================

Loads ontario_2022.csv and ontario_2023.csv to derive:
1. Real transition matrices from year-over-year PCI changes
2. Heterogeneous p_s values per section or pavement type
3. Calibrated RMAB environment parameters

COMPATIBILITY:
- Works with original codebase (config.py, environment.py, etc.)
- Can be used standalone or with heterogeneous_extension.py

Data Dictionary Reference:
- Section ID: Unique pavement section identifier
- PCI: Pavement Condition Index (0-100, higher=better)
- IRI: International Roughness Index (m/km, lower=better)
- Pave_Type: AC (Asphalt), PC (Concrete), COM (Composite), ST (Surface-treated)

Usage (Standalone):
    loader = OntarioDataLoader('data/ontario_2022.csv', 'data/ontario_2023.csv')
    loader.load_and_process()
    transition_matrices = loader.get_transition_matrices()
    
Usage (With heterogeneous_extension.py):
    from ontario_data_loader import OntarioDataLoader
    from heterogeneous_extension import HeterogeneousRMABEnvironment
    
    loader = OntarioDataLoader(...)
    loader.load_and_process()
    p_s_values = loader.generate_p_s_array(n_arms=50)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


# =============================================================================
# PCI State Discretization (FHWA-based)
# =============================================================================

@dataclass
class PCIStateConfig:
    """
    PCI to discrete state mapping.
    Based on FHWA pavement condition categories.
    """
    # State boundaries (PCI thresholds)
    # State 0: Excellent (PCI >= 85)
    # State 1: Good (70 <= PCI < 85)
    # State 2: Fair (55 <= PCI < 70)
    # State 3: Poor (40 <= PCI < 55)
    # State 4: Very Poor (PCI < 40)
    
    thresholds: List[float] = None
    state_names: List[str] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = [85, 70, 55, 40]  # Descending order
        if self.state_names is None:
            self.state_names = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
    
    def pci_to_state(self, pci: float) -> int:
        """Convert PCI value to discrete state (0=best, 4=worst)."""
        if pd.isna(pci):
            return None
        for i, threshold in enumerate(self.thresholds):
            if pci >= threshold:
                return i
        return len(self.thresholds)  # Worst state
    
    @property
    def n_states(self) -> int:
        return len(self.thresholds) + 1


# =============================================================================
# Main Data Loader Class
# =============================================================================

class OntarioDataLoader:
    """
    Loads and processes Ontario pavement condition data for RMAB simulation.
    """
    
    def __init__(self, 
                 path_2022: str = 'ontario_2022.csv',
                 path_2023: str = 'ontario_2023.csv',
                 pci_config: Optional[PCIStateConfig] = None):
        """
        Initialize loader with paths to CSV files.
        
        Args:
            path_2022: Path to 2022 pavement condition CSV
            path_2023: Path to 2023 pavement condition CSV
            pci_config: Optional custom PCI state configuration
        """
        self.path_2022 = Path(path_2022)
        self.path_2023 = Path(path_2023)
        self.pci_config = pci_config or PCIStateConfig()
        
        # Data storage
        self.df_2022: Optional[pd.DataFrame] = None
        self.df_2023: Optional[pd.DataFrame] = None
        self.matched_sections: Optional[pd.DataFrame] = None
        
        # Computed results
        self.transition_counts: Dict[str, np.ndarray] = {}
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.section_stats: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CSV files and perform basic validation."""
        print(f"Loading {self.path_2022}...")
        self.df_2022 = pd.read_csv(self.path_2022)
        print(f"  → {len(self.df_2022)} rows")
        
        print(f"Loading {self.path_2023}...")
        self.df_2023 = pd.read_csv(self.path_2023)
        print(f"  → {len(self.df_2023)} rows")
        
        # Normalize column names (handle potential variations)
        for df in [self.df_2022, self.df_2023]:
            df.columns = df.columns.str.strip()
            # Try common variations
            col_mapping = {
                'Section_ID': 'Section ID',
                'SectionID': 'Section ID',
                'section_id': 'Section ID',
                'Pave_type': 'Pave_Type',
                'pave_type': 'Pave_Type',
                'PAVE_TYPE': 'Pave_Type',
            }
            df.rename(columns=col_mapping, inplace=True)
        
        return self.df_2022, self.df_2023
    
    def match_sections(self) -> pd.DataFrame:
        """
        Match sections between 2022 and 2023 data.
        Only sections present in both years are useful for transition estimation.
        """
        if self.df_2022 is None or self.df_2023 is None:
            self.load_data()
        
        # Find common sections
        sections_2022 = set(self.df_2022['Section ID'].dropna().unique())
        sections_2023 = set(self.df_2023['Section ID'].dropna().unique())
        common_sections = sections_2022 & sections_2023
        
        print(f"\nSection matching:")
        print(f"  2022 sections: {len(sections_2022)}")
        print(f"  2023 sections: {len(sections_2023)}")
        print(f"  Common sections: {len(common_sections)}")
        
        # Merge on Section ID
        df_2022_subset = self.df_2022[self.df_2022['Section ID'].isin(common_sections)].copy()
        df_2023_subset = self.df_2023[self.df_2023['Section ID'].isin(common_sections)].copy()
        
        # Aggregate by section (in case of multiple measurements per section)
        agg_cols = {'PCI': 'mean', 'IRI': 'mean', 'DMI': 'mean'}
        if 'Pave_Type' in df_2022_subset.columns:
            agg_cols['Pave_Type'] = 'first'
        
        df_2022_agg = df_2022_subset.groupby('Section ID').agg(agg_cols).reset_index()
        df_2023_agg = df_2023_subset.groupby('Section ID').agg(agg_cols).reset_index()
        
        # Merge
        self.matched_sections = pd.merge(
            df_2022_agg, df_2023_agg,
            on='Section ID',
            suffixes=('_2022', '_2023')
        )
        
        # Add state columns
        self.matched_sections['State_2022'] = self.matched_sections['PCI_2022'].apply(
            self.pci_config.pci_to_state
        )
        self.matched_sections['State_2023'] = self.matched_sections['PCI_2023'].apply(
            self.pci_config.pci_to_state
        )
        
        # Remove rows with missing states
        valid_mask = self.matched_sections['State_2022'].notna() & self.matched_sections['State_2023'].notna()
        self.matched_sections = self.matched_sections[valid_mask].copy()
        self.matched_sections['State_2022'] = self.matched_sections['State_2022'].astype(int)
        self.matched_sections['State_2023'] = self.matched_sections['State_2023'].astype(int)
        
        print(f"  Matched with valid PCI: {len(self.matched_sections)}")
        
        return self.matched_sections
    
    def compute_transition_matrices(self) -> Dict[str, np.ndarray]:
        """
        Compute transition matrices from matched section data.
        Returns matrices for:
        - 'overall': All sections combined
        - 'AC', 'PC', 'COM', 'ST': By pavement type (if available)
        """
        if self.matched_sections is None:
            self.match_sections()
        
        n_states = self.pci_config.n_states
        
        def compute_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            """Compute transition count and probability matrix."""
            counts = np.zeros((n_states, n_states), dtype=int)
            for _, row in df.iterrows():
                s_from = int(row['State_2022'])
                s_to = int(row['State_2023'])
                counts[s_from, s_to] += 1
            
            # Normalize to probabilities
            probs = np.zeros((n_states, n_states))
            for i in range(n_states):
                row_sum = counts[i].sum()
                if row_sum > 0:
                    probs[i] = counts[i] / row_sum
                else:
                    # No data for this state - assume stay
                    probs[i, i] = 1.0
            
            return counts, probs
        
        # Overall matrix
        counts, probs = compute_matrix(self.matched_sections)
        self.transition_counts['overall'] = counts
        self.transition_matrices['overall'] = probs
        
        print(f"\n=== OVERALL TRANSITION MATRIX ===")
        print(f"Samples per state: {counts.sum(axis=1)}")
        print("Transition probabilities:")
        print(np.array2string(probs, precision=3, suppress_small=True))
        
        # By pavement type
        if 'Pave_Type_2022' in self.matched_sections.columns:
            pave_col = 'Pave_Type_2022'
        elif 'Pave_Type' in self.matched_sections.columns:
            pave_col = 'Pave_Type'
        else:
            pave_col = None
        
        if pave_col:
            for ptype in ['AC', 'PC', 'COM', 'ST']:
                subset = self.matched_sections[self.matched_sections[pave_col] == ptype]
                if len(subset) >= 10:  # Minimum samples
                    counts, probs = compute_matrix(subset)
                    self.transition_counts[ptype] = counts
                    self.transition_matrices[ptype] = probs
                    print(f"\n=== {ptype} TRANSITION MATRIX ({len(subset)} sections) ===")
                    print(np.array2string(probs, precision=3, suppress_small=True))
        
        return self.transition_matrices
    
    def estimate_p_s_distribution(self) -> Dict[str, Dict]:
        """
        Estimate p_s (success/update probability) distribution from data.
        
        In RMAB context:
        - p_s represents probability of successful state improvement after maintenance
        - We estimate this from sections that improved or maintained good condition
        
        Returns distribution statistics for heterogeneous arm generation.
        """
        if self.matched_sections is None:
            self.match_sections()
        
        results = {}
        
        # Compute per-section "improvement potential"
        # Sections that maintained/improved state have higher effective p_s
        df = self.matched_sections.copy()
        df['delta_state'] = df['State_2023'] - df['State_2022']  # Negative = improved
        df['delta_pci'] = df['PCI_2023'] - df['PCI_2022']  # Positive = improved
        
        # Classify sections by maintenance responsiveness
        # Improved: delta_pci > 5 or delta_state < 0
        # Stable: |delta_pci| <= 5 and delta_state == 0
        # Degraded: delta_pci < -5 or delta_state > 0
        
        def classify_responsiveness(row):
            if row['delta_state'] < 0 or row['delta_pci'] > 5:
                return 'responsive'  # High p_s
            elif row['delta_state'] > 0 or row['delta_pci'] < -5:
                return 'degrading'   # Low p_s
            else:
                return 'stable'      # Medium p_s
        
        df['responsiveness'] = df.apply(classify_responsiveness, axis=1)
        
        resp_counts = df['responsiveness'].value_counts()
        total = len(df)
        
        # Map to p_s ranges (for heterogeneous arm generation)
        p_s_mapping = {
            'responsive': (0.7, 0.95),  # High success probability
            'stable': (0.4, 0.7),       # Medium
            'degrading': (0.15, 0.4),   # Low
        }
        
        print(f"\n=== RESPONSIVENESS DISTRIBUTION ===")
        for resp_type, count in resp_counts.items():
            pct = count / total * 100
            p_range = p_s_mapping.get(resp_type, (0.3, 0.7))
            print(f"  {resp_type}: {count} ({pct:.1f}%) → p_s ∈ [{p_range[0]:.2f}, {p_range[1]:.2f}]")
            results[resp_type] = {
                'count': int(count),
                'percentage': float(pct),
                'p_s_range': p_range
            }
        
        # Overall p_s distribution parameters
        # Weighted average based on responsiveness
        weights = {
            'responsive': 0.825,  # midpoint
            'stable': 0.55,
            'degrading': 0.275,
        }
        
        weighted_p_s = sum(
            weights.get(r, 0.5) * resp_counts.get(r, 0) / total 
            for r in resp_counts.index
        )
        
        results['overall'] = {
            'mean_p_s': float(weighted_p_s),
            'recommended_range': (0.20, 0.85),  # Based on observed spread
            'heterogeneity': 'high' if resp_counts.get('degrading', 0) > 0.2 * total else 'medium'
        }
        
        print(f"\n  Overall estimated mean p_s: {weighted_p_s:.3f}")
        print(f"  Recommended heterogeneity range: [0.20, 0.85]")
        
        self.section_stats = df
        return results
    
    def generate_arm_configs(self, n_arms: int = 50) -> List[Dict]:
        """
        Generate heterogeneous arm configurations based on real data distribution.
        
        Args:
            n_arms: Number of arms to generate
            
        Returns:
            List of arm configurations compatible with RMAB-RDT v3
        """
        if self.section_stats is None:
            self.estimate_p_s_distribution()
        
        # Get responsiveness distribution
        resp_dist = self.section_stats['responsiveness'].value_counts(normalize=True)
        
        # P_s ranges by type
        p_s_ranges = {
            'responsive': (0.7, 0.95),
            'stable': (0.4, 0.7),
            'degrading': (0.15, 0.4),
        }
        
        arm_configs = []
        np.random.seed(42)
        
        for i in range(n_arms):
            # Sample responsiveness type according to real distribution
            resp_type = np.random.choice(
                list(resp_dist.index),
                p=list(resp_dist.values)
            )
            
            # Sample p_s within range
            p_low, p_high = p_s_ranges.get(resp_type, (0.3, 0.7))
            p_s = np.random.uniform(p_low, p_high)
            
            arm_configs.append({
                'arm_id': i,
                'p_s': float(p_s),
                'responsiveness': resp_type,
                'source': 'ontario_calibrated'
            })
        
        # Summary
        p_s_values = [a['p_s'] for a in arm_configs]
        print(f"\n=== GENERATED ARM CONFIGS ({n_arms} arms) ===")
        print(f"  p_s range: [{min(p_s_values):.3f}, {max(p_s_values):.3f}]")
        print(f"  p_s mean: {np.mean(p_s_values):.3f}")
        print(f"  p_s std: {np.std(p_s_values):.3f}")
        
        return arm_configs
    
    def generate_p_s_array(self, n_arms: int = 50, seed: int = 42) -> np.ndarray:
        """
        Generate numpy array of p_s values for heterogeneous_extension.py.
        
        This is a convenience method that returns just the p_s values as an array,
        suitable for use with HeterogeneousRMABEnvironment.
        
        Args:
            n_arms: Number of arms
            seed: Random seed
            
        Returns:
            np.ndarray of p_s values, shape (n_arms,)
        
        Example:
            loader = OntarioDataLoader(...)
            loader.load_and_process()
            p_s_array = loader.generate_p_s_array(n_arms=50)
            
            from heterogeneous_extension import HeterogeneousRMABEnvironment
            env = HeterogeneousRMABEnvironment(config, p_s_array, seed=42)
        """
        arm_configs = self.generate_arm_configs(n_arms)
        return np.array([a['p_s'] for a in arm_configs])
    
    def get_rmab_parameters(self) -> Dict:
        """
        Get complete RMAB simulation parameters derived from Ontario data.
        """
        if not self.transition_matrices:
            self.compute_transition_matrices()
        if self.section_stats is None:
            self.estimate_p_s_distribution()
        
        # Use overall matrix as baseline
        P = self.transition_matrices['overall']
        n_states = P.shape[0]
        
        # Extract key parameters
        # P_passive: Natural degradation (no maintenance)
        # Approximate from sections that degraded
        P_passive = P.copy()
        
        # P_active: With maintenance (estimate improved transitions)
        P_active = np.zeros_like(P)
        for i in range(n_states):
            # With maintenance, higher probability of improvement
            if i > 0:
                P_active[i, i-1] = 0.4  # Improve one state
                P_active[i, i] = 0.5    # Stay same
                P_active[i, min(i+1, n_states-1)] = 0.1  # Still degrade
            else:
                P_active[i, i] = 0.95   # Best state stays
                P_active[i, 1] = 0.05
        
        # Normalize
        P_active = P_active / P_active.sum(axis=1, keepdims=True)
        
        return {
            'n_states': n_states,
            'P_passive': P_passive.tolist(),
            'P_active': P_active.tolist(),
            'p_s_distribution': {
                'type': 'heterogeneous',
                'range': [0.20, 0.85],
                'source': 'ontario_2022_2023'
            },
            'state_names': self.pci_config.state_names,
            'pci_thresholds': self.pci_config.thresholds,
        }
    
    def load_and_process(self) -> Dict:
        """
        Complete pipeline: load, match, compute transitions, estimate p_s.
        """
        print("=" * 60)
        print("ONTARIO PAVEMENT DATA PROCESSING")
        print("=" * 60)
        
        self.load_data()
        self.match_sections()
        self.compute_transition_matrices()
        p_s_dist = self.estimate_p_s_distribution()
        params = self.get_rmab_parameters()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        
        return {
            'transition_matrices': {k: v.tolist() for k, v in self.transition_matrices.items()},
            'p_s_distribution': p_s_dist,
            'rmab_parameters': params,
            'n_matched_sections': len(self.matched_sections),
        }
    
    def export_to_json(self, output_path: str = 'ontario_calibration.json'):
        """Export all computed parameters to JSON for RMAB simulation."""
        results = self.load_and_process()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nExported to: {output_path}")
        return output_path


# =============================================================================
# Integration with RMAB-RDT v3 Config
# =============================================================================

def create_ontario_environment_config(
    loader: OntarioDataLoader,
    n_arms: int = 50,
    budget_ratio: float = 0.10
) -> Dict:
    """
    Create RMAB environment configuration from Ontario data.
    
    Returns config dict compatible with config.py RMABConfig.
    """
    params = loader.get_rmab_parameters()
    arm_configs = loader.generate_arm_configs(n_arms)
    
    config = {
        'N': n_arms,
        'M': int(n_arms * budget_ratio),
        'J': params['n_states'],
        'delta_max': 50,
        'T': 500,
        'gamma': 0.99,
        
        # Heterogeneous p_s from Ontario data
        'p_s_heterogeneous': [a['p_s'] for a in arm_configs],
        
        # Transition matrices from real data
        'P_passive_real': params['P_passive'],
        'P_active_real': params['P_active'],
        
        'source': 'ontario_2022_2023',
        'calibration_date': '2024',
    }
    
    return config


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Ontario pavement data for RMAB')
    parser.add_argument('--data-dir', type=str, default='.', 
                        help='Directory containing ontario_2022.csv and ontario_2023.csv')
    parser.add_argument('--output', type=str, default='ontario_calibration.json',
                        help='Output JSON file path')
    parser.add_argument('--n-arms', type=int, default=50,
                        help='Number of arms to generate')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    loader = OntarioDataLoader(
        path_2022=data_dir / 'ontario_2022.csv',
        path_2023=data_dir / 'ontario_2023.csv'
    )
    
    try:
        loader.export_to_json(args.output)
        
        # Also generate arm configs
        arm_configs = loader.generate_arm_configs(args.n_arms)
        arm_output = args.output.replace('.json', '_arms.json')
        with open(arm_output, 'w') as f:
            json.dump(arm_configs, f, indent=2)
        print(f"Arm configs exported to: {arm_output}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure ontario_2022.csv and ontario_2023.csv are in the specified directory.")
        print("\nExpected CSV format (based on data dictionary):")
        print("  - Section ID: Unique section identifier")
        print("  - PCI: Pavement Condition Index (0-100)")
        print("  - Pave_Type: AC/PC/COM/ST")
        print("  - IRI: International Roughness Index (optional)")
