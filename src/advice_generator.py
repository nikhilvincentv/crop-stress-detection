#!/usr/bin/env python3
"""Generate actionable advice based on crop stress levels."""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AdviceGenerator:
    """Generate actionable recommendations based on stress levels."""
    
    def __init__(self):
        """Initialize the advice generator."""
        self.stress_thresholds = {
            'healthy': (0.0, 0.2),
            'low_stress': (0.2, 0.4),
            'moderate_stress': (0.4, 0.6),
            'high_stress': (0.6, 0.8),
            'critical_stress': (0.8, float('inf'))
        }
        
        self.color_codes = {
            'healthy': '\033[92m',  # Green
            'low_stress': '\033[93m',  # Yellow
            'moderate_stress': '\033[38;5;208m',  # Orange
            'high_stress': '\033[91m',  # Red
            'critical_stress': '\033[95m',  # Magenta
            'reset': '\033[0m'
        }
    
    def categorize_stress(self, stress_value: float) -> str:
        """Categorize stress level based on value.
        
        Args:
            stress_value: Stress score (0.0 to 1.0+)
            
        Returns:
            Category name (healthy, low_stress, etc.)
        """
        for category, (min_val, max_val) in self.stress_thresholds.items():
            if min_val <= stress_value < max_val:
                return category
        return 'critical_stress'
    
    def get_status_description(self, category: str) -> str:
        """Get human-readable status description.
        
        Args:
            category: Stress category
            
        Returns:
            Status description string
        """
        descriptions = {
            'healthy': 'HEALTHY - Plant is thriving, no water stress detected',
            'low_stress': 'LOW STRESS - Minor water shortage, monitor closely',
            'moderate_stress': 'MODERATE STRESS - Plant struggling, action needed soon',
            'high_stress': 'HIGH STRESS - Significant impact on growth and yield',
            'critical_stress': 'CRITICAL STRESS - Severe damage, crop failure likely'
        }
        return descriptions.get(category, 'UNKNOWN STATUS')
    
    def get_watering_advice(self, category: str, moisture: float) -> str:
        """Get specific watering recommendations.
        
        Args:
            category: Stress category
            moisture: Current soil moisture (0.0 to 1.0)
            
        Returns:
            Watering advice string
        """
        advice = {
            'healthy': f"Maintain current watering schedule. Soil moisture at {moisture*100:.1f}% is optimal.",
            'low_stress': f"Consider light watering. Soil moisture at {moisture*100:.1f}% is slightly low. Add 10-20% more water than usual.",
            'moderate_stress': f"Water immediately! Soil moisture at {moisture*100:.1f}% is too low. Increase watering by 30-50%.",
            'high_stress': f"URGENT: Deep watering required NOW! Soil moisture at {moisture*100:.1f}% is critically low. Double your normal watering amount.",
            'critical_stress': f"EMERGENCY: Intensive irrigation needed! Soil moisture at {moisture*100:.1f}% indicates severe drought. Water heavily and frequently until recovery."
        }
        return advice.get(category, 'Unable to provide watering advice.')
    
    def get_action_timeline(self, category: str) -> str:
        """Get timeline for required action.
        
        Args:
            category: Stress category
            
        Returns:
            Timeline string
        """
        timelines = {
            'healthy': 'No immediate action required. Continue regular monitoring.',
            'low_stress': 'Action recommended within 24-48 hours.',
            'moderate_stress': 'Action required within 12-24 hours to prevent damage.',
            'high_stress': 'URGENT: Action required within 6-12 hours.',
            'critical_stress': 'EMERGENCY: Immediate action required (within 1-2 hours)!'
        }
        return timelines.get(category, 'Unknown timeline.')
    
    def get_recovery_prediction(self, category: str) -> str:
        """Predict recovery timeline if action is taken.
        
        Args:
            category: Stress category
            
        Returns:
            Recovery prediction string
        """
        predictions = {
            'healthy': 'Plant is healthy. Continue current care.',
            'low_stress': 'Expected recovery: 1-2 days with proper watering.',
            'moderate_stress': 'Expected recovery: 3-5 days with consistent watering.',
            'high_stress': 'Expected recovery: 5-7 days. Some yield loss likely.',
            'critical_stress': 'Recovery uncertain. Expect 7-14 days minimum. Significant yield loss expected.'
        }
        return predictions.get(category, 'Unable to predict recovery.')
    
    def get_consequence_warning(self, category: str) -> str:
        """Warn about consequences if no action is taken.
        
        Args:
            category: Stress category
            
        Returns:
            Warning string
        """
        warnings = {
            'healthy': '',
            'low_stress': 'If ignored: Stress will escalate to moderate level within 24-48 hours.',
            'moderate_stress': 'If ignored: Plant growth will slow significantly. Yield reduction of 10-20% likely.',
            'high_stress': 'If ignored: Severe wilting and leaf damage within 12-24 hours. Yield reduction of 30-50%.',
            'critical_stress': 'If ignored: Complete crop failure within 24-48 hours. Total yield loss.'
        }
        return warnings.get(category, '')
    
    def generate_full_report(self, stress_value: float, moisture: float, 
                            ndvi: float, trend: str = 'stable') -> Dict[str, str]:
        """Generate complete advice report.
        
        Args:
            stress_value: Current stress score
            moisture: Current soil moisture
            ndvi: Current NDVI value
            trend: Stress trend ('increasing', 'decreasing', 'stable')
            
        Returns:
            Dictionary with all advice components
        """
        category = self.categorize_stress(stress_value)
        
        report = {
            'category': category,
            'stress_value': stress_value,
            'moisture': moisture,
            'ndvi': ndvi,
            'status': self.get_status_description(category),
            'watering_advice': self.get_watering_advice(category, moisture),
            'timeline': self.get_action_timeline(category),
            'recovery': self.get_recovery_prediction(category),
            'warning': self.get_consequence_warning(category),
            'trend': trend,
            'color': self.color_codes.get(category, '')
        }
        
        return report
    
    def print_colored_report(self, report: Dict[str, str]):
        """Print formatted, color-coded report to console.
        
        Args:
            report: Report dictionary from generate_full_report
        """
        color = report['color']
        reset = self.color_codes['reset']
        
        print("\n" + "="*70)
        print(f"{color}CROP STRESS ANALYSIS REPORT{reset}")
        print("="*70)
        
        print(f"\n{color}📊 CURRENT READINGS:{reset}")
        print(f"  • Stress Score: {color}{report['stress_value']:.3f}{reset}")
        print(f"  • Soil Moisture: {report['moisture']*100:.1f}%")
        print(f"  • NDVI (Plant Health): {report['ndvi']:.3f}")
        print(f"  • Trend: {report['trend'].upper()}")
        
        print(f"\n{color}🎯 STATUS:{reset}")
        print(f"  {color}{report['status']}{reset}")
        
        print(f"\n{color}💧 WATERING ADVICE:{reset}")
        print(f"  {report['watering_advice']}")
        
        print(f"\n{color}⏰ ACTION TIMELINE:{reset}")
        print(f"  {report['timeline']}")
        
        print(f"\n{color}🔮 RECOVERY PREDICTION:{reset}")
        print(f"  {report['recovery']}")
        
        if report['warning']:
            print(f"\n{color}⚠️  WARNING:{reset}")
            print(f"  {report['warning']}")
        
        print("\n" + "="*70 + "\n")
    
    def get_simple_advice(self, stress_value: float) -> str:
        """Get simple one-line advice.
        
        Args:
            stress_value: Current stress score
            
        Returns:
            Simple advice string
        """
        category = self.categorize_stress(stress_value)
        
        simple_advice = {
            'healthy': '✓ All good! Keep up current care routine.',
            'low_stress': '⚠ Monitor closely. Consider light watering.',
            'moderate_stress': '⚠⚠ Water soon! Plant needs attention.',
            'high_stress': '🚨 URGENT: Water immediately!',
            'critical_stress': '🚨🚨 EMERGENCY: Intensive irrigation required NOW!'
        }
        
        return simple_advice.get(category, 'Status unknown.')


if __name__ == "__main__":
    # Test the advice generator
    generator = AdviceGenerator()
    
    # Test different stress levels
    test_cases = [
        (0.1, 0.65, 0.78, 'stable'),
        (0.35, 0.45, 0.68, 'increasing'),
        (0.55, 0.35, 0.55, 'increasing'),
        (0.75, 0.25, 0.42, 'increasing'),
        (0.95, 0.15, 0.28, 'increasing')
    ]
    
    for stress, moisture, ndvi, trend in test_cases:
        report = generator.generate_full_report(stress, moisture, ndvi, trend)
        generator.print_colored_report(report)
