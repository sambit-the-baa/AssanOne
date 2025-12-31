# agent/orchestrator_enhanced.py

import json
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import asdict, dataclass, field
from enum import Enum
from datetime import datetime

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback zscore implementation
    def zscore(data):
        if not NUMPY_AVAILABLE or len(data) == 0:
            return data
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        if std == 0:
            return [0] * len(data)
        return [(x - mean) / std for x in data]

from agent.agents_fraud import (
    OverbillingProtectionAgent,
    FraudDiagnosticAgent,
    UnbundlingUpccodingAgent,
    IdentityTheftProtectionAgent,
    AgentResult,
    RiskLevel,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for fraud detection"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    MIXED = "mixed"


@dataclass
class AgentConfig:
    """Configuration for each fraud detection agent"""
    name: str
    agent_class: type
    enabled: bool = True
    base_weight: float = 0.25
    timeout_seconds: int = 30
    critical: bool = False  # Critical agents run first in MIXED mode
    performance_metrics: Dict = field(default_factory=dict)


@dataclass
class AdaptiveWeights:
    """Adaptive weights based on claim characteristics"""
    overbilling: float
    diagnostic: float
    unbundling: float
    identity: float
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum([self.overbilling, self.diagnostic, self.unbundling, self.identity])
        if total > 0:
            self.overbilling /= total
            self.diagnostic /= total
            self.unbundling /= total
            self.identity /= total


class AdaptiveWeightingEngine:
    """
    Dynamically weights agents based on claim characteristics
    Instead of fixed 25% each, weights adapt to claim profile
    """
    
    def __init__(self):
        self.claim_type_weights = {
            'surgical': {
                'overbilling': 0.35,      # High for surgery (expensive)
                'diagnostic': 0.25,
                'unbundling': 0.30,       # Very important for bundled procedures
                'identity': 0.10,
            },
            'emergency': {
                'overbilling': 0.30,      # Emergency claims tend to be expensive
                'diagnostic': 0.30,
                'unbundling': 0.20,
                'identity': 0.20,         # Higher identity risk in emergency
            },
            'imaging': {
                'overbilling': 0.40,      # Imaging frequently overbilled
                'diagnostic': 0.35,       # Unnecessary imaging common
                'unbundling': 0.15,
                'identity': 0.10,
            },
            'pharmacy': {
                'overbilling': 0.35,
                'diagnostic': 0.10,
                'unbundling': 0.20,
                'identity': 0.35,         # High identity fraud risk
            },
            'office_visit': {
                'overbilling': 0.25,
                'diagnostic': 0.30,
                'unbundling': 0.15,
                'identity': 0.30,         # Many false identities for office visits
            },
        }
        
        self.provider_risk_adjustments = {
            'new_provider': 1.25,         # Increase weight for new providers
            'flagged_provider': 1.50,     # Increase weight for flagged providers
            'high_volume': 1.15,          # Increase weight for high-volume providers
            'specialty_rare': 1.20,       # Increase weight for rare specialties
        }
    
    def calculate_weights(self, claim_data: Dict) -> AdaptiveWeights:
        """
        Calculate adaptive weights based on claim characteristics
        
        Args:
            claim_data: Extracted claim fields
            
        Returns:
            AdaptiveWeights object with normalized weights
        """
        # Start with default equal weights
        weights = AdaptiveWeights(0.25, 0.25, 0.25, 0.25)
        
        # Determine claim type
        claim_type = self._identify_claim_type(claim_data)
        type_weights = self.claim_type_weights.get(claim_type, {
            'overbilling': 0.25,
            'diagnostic': 0.25,
            'unbundling': 0.25,
            'identity': 0.25,
        })
        
        # Apply type-based weights
        weights.overbilling = type_weights['overbilling']
        weights.diagnostic = type_weights['diagnostic']
        weights.unbundling = type_weights['unbundling']
        weights.identity = type_weights['identity']
        
        # Adjust for provider characteristics
        provider_adjustment = self._get_provider_adjustment(claim_data)
        weights.overbilling *= provider_adjustment
        
        # Adjust for patient characteristics
        patient_adjustment = self._get_patient_adjustment(claim_data)
        weights.identity *= patient_adjustment
        
        # Adjust for claim amount (high-value claims get more scrutiny)
        amount_adjustment = self._get_amount_adjustment(claim_data)
        weights.overbilling *= amount_adjustment
        weights.diagnostic *= amount_adjustment * 0.8
        
        # Normalize so sum = 1.0
        weights.normalize()
        
        logger.info(f"Calculated adaptive weights for {claim_type}: {asdict(weights)}")
        return weights
    
    def _identify_claim_type(self, claim_data: Dict) -> str:
        """Identify claim type from procedures/diagnosis"""
        procedures = claim_data.get('procedures') or []
        diagnosis = (claim_data.get('diagnosis') or '').upper()
        
        # Surgical procedures (CPT codes 20000-69999)
        surgical_codes = ['27', '28', '29', '30', '31', '32', '33', '34', '35',
                         '36', '37', '38', '39', '40', '41', '42', '43', '44',
                         '45', '46', '47', '48', '49', '50', '51', '52', '53',
                         '54', '55', '56', '57', '58', '59', '60', '61', '62',
                         '63', '64', '65', '66', '67', '68', '69']
        
        # Ensure procedures is a list of strings
        if isinstance(procedures, str):
            procedures = [procedures]
        procedures = [str(p) for p in procedures if p]
        
        if any(proc.startswith(code) for code in surgical_codes for proc in procedures):
            return 'surgical'
        
        # Emergency-related diagnoses
        emergency_keywords = ['TRAUMA', 'ACUTE', 'EMERGENCY', 'CRITICAL', 'SEVERE']
        if any(keyword in diagnosis for keyword in emergency_keywords):
            return 'emergency'
        
        # Imaging procedures (CPT 70010-79999)
        if any(proc.startswith(('70', '71', '72', '73', '74', '75', '76', '77', '78', '79'))
               for proc in procedures):
            return 'imaging'
        
        # Pharmacy codes (NDC codes or specific procedure codes)
        if any('pharmacy' in str(proc).lower() or 'drug' in str(proc).lower() for proc in procedures):
            return 'pharmacy'
        
        # Default to office visit
        return 'office_visit'
    
    def _get_provider_adjustment(self, claim_data: Dict) -> float:
        """Adjust weights based on provider risk profile"""
        # In production, query provider risk database
        provider = claim_data.get('provider_name') or claim_data.get('provider') or ''
        
        # Check if new provider (example heuristic)
        if len(str(provider)) < 5:
            return self.provider_risk_adjustments['new_provider']
        
        return 1.0
    
    def _get_patient_adjustment(self, claim_data: Dict) -> float:
        """Adjust weights based on patient characteristics"""
        # Check for high-risk patterns
        patient_age = claim_data.get('patient_age', 50)
        
        # Very high or very low age increases identity fraud risk
        if patient_age < 5 or patient_age > 95:
            return 1.50
        
        return 1.0
    
    def _get_amount_adjustment(self, claim_data: Dict) -> float:
        """Adjust weights based on claim amount"""
        try:
            amount = float(claim_data.get('total_amount', 0))
            
            # High-value claims get more scrutiny
            if amount > 10000:
                return 1.50
            elif amount > 5000:
                return 1.25
            elif amount < 100:
                return 0.80
        except (ValueError, TypeError):
            pass
        
        return 1.0


class EnhancedFraudDetectionOrchestrator:
    """
    Enhanced orchestrator with:
    - Adaptive agent weighting
    - Health monitoring
    - Timeout handling
    - Performance metrics
    - Advanced scoring
    """
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.PARALLEL):
        """Initialize orchestrator with agent configuration"""
        self.execution_mode = execution_mode
        self.weighting_engine = AdaptiveWeightingEngine()
        
        # Configure agents
        self.agents = {
            'overbilling': AgentConfig(
                name='Overbilling Protection',
                agent_class=OverbillingProtectionAgent,
                base_weight=0.25,
                timeout_seconds=30,
            ),
            'diagnostic': AgentConfig(
                name='Fraud Diagnostic Analysis',
                agent_class=FraudDiagnosticAgent,
                base_weight=0.25,
                timeout_seconds=30,
            ),
            'unbundling': AgentConfig(
                name='Unbundling/Upcoding Detection',
                agent_class=UnbundlingUpccodingAgent,
                base_weight=0.25,
                timeout_seconds=30,
            ),
            'identity': AgentConfig(
                name='Identity Theft Protection',
                agent_class=IdentityTheftProtectionAgent,
                base_weight=0.25,
                timeout_seconds=30,
                critical=True,  # Run first in MIXED mode
            ),
        }
        
        # Initialize agent instances
        self.agent_instances = {
            key: config.agent_class() for key, config in self.agents.items()
        }
        
        logger.info(f"Initialized orchestrator in {execution_mode.value} mode")
    
    def analyze(
        self,
        claim_data: Dict,
        historical_data: Optional[List[Dict]] = None,
        id_document_path: Optional[str] = None,
    ) -> Dict:
        """
        Run fraud detection on claim with advanced orchestration
        
        Args:
            claim_data: Extracted claim fields
            historical_data: Previous claims for cross-document analysis
            id_document_path: Path to identity document
            
        Returns:
            Comprehensive fraud detection report
        """
        start_time = datetime.now()
        
        # DEBUG: Log claim_data before running agents
        logger.info(f"[DEBUG] Orchestrator received claim_data: {claim_data}")
        print(f"[DEBUG] Orchestrator received claim_data:")
        for key, val in claim_data.items():
            print(f"    {key}: {val}")
        
        # Run agents based on execution mode
        if self.execution_mode == ExecutionMode.PARALLEL:
            agent_results = self._run_parallel(claim_data, historical_data, id_document_path)
        elif self.execution_mode == ExecutionMode.SEQUENTIAL:
            agent_results = self._run_sequential(claim_data, historical_data, id_document_path)
        else:  # MIXED
            agent_results = self._run_mixed(claim_data, historical_data, id_document_path)
        
        # Calculate adaptive weights
        adaptive_weights = self.weighting_engine.calculate_weights(claim_data)
        
        # Compute unified score with adaptive weights
        unified_report = self._compute_weighted_score(
            agent_results,
            claim_data,
            adaptive_weights,
        )
        
        # Add execution metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        unified_report['execution_metrics'] = {
            'total_time_seconds': execution_time,
            'execution_mode': self.execution_mode.value,
            'agents_executed': len([r for r in agent_results.values() if r is not None]),
        }
        
        logger.info(f"Fraud analysis completed in {execution_time:.2f}s")
        return unified_report
    
    def _run_parallel(
        self,
        claim_data: Dict,
        historical_data: Optional[List[Dict]],
        id_document_path: Optional[str],
    ) -> Dict:
        """Run all agents in parallel with timeout handling"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self._run_agent_with_timeout,
                    'overbilling',
                    claim_data,
                    historical_data,
                    id_document_path,
                ): 'overbilling',
                executor.submit(
                    self._run_agent_with_timeout,
                    'diagnostic',
                    claim_data,
                    historical_data,
                    id_document_path,
                ): 'diagnostic',
                executor.submit(
                    self._run_agent_with_timeout,
                    'unbundling',
                    claim_data,
                    historical_data,
                    id_document_path,
                ): 'unbundling',
                executor.submit(
                    self._run_agent_with_timeout,
                    'identity',
                    claim_data,
                    historical_data,
                    id_document_path,
                ): 'identity',
            }
            
            for future in as_completed(futures, timeout=35):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except TimeoutError:
                    logger.error(f"{agent_name} agent timed out")
                    results[agent_name] = None
                except Exception as e:
                    logger.error(f"Error in {agent_name} agent: {e}")
                    results[agent_name] = None
        
        return results
    
    def _run_sequential(
        self,
        claim_data: Dict,
        historical_data: Optional[List[Dict]],
        id_document_path: Optional[str],
    ) -> Dict:
        """Run all agents sequentially"""
        results = {}
        for agent_name in ['overbilling', 'diagnostic', 'unbundling', 'identity']:
            try:
                results[agent_name] = self._run_agent_with_timeout(
                    agent_name, claim_data, historical_data, id_document_path
                )
            except Exception as e:
                logger.error(f"Error in {agent_name} agent: {e}")
                results[agent_name] = None
        return results
    
    def _run_mixed(
        self,
        claim_data: Dict,
        historical_data: Optional[List[Dict]],
        id_document_path: Optional[str],
    ) -> Dict:
        """Run identity first (critical), then others in parallel"""
        results = {}
        # Run identity first
        try:
            results['identity'] = self._run_agent_with_timeout(
                'identity', claim_data, historical_data, id_document_path
            )
        except Exception as e:
            logger.error(f"Error in identity agent: {e}")
            results['identity'] = None
        
        # Run remaining in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._run_agent_with_timeout, name, claim_data, historical_data, id_document_path
                ): name for name in ['overbilling', 'diagnostic', 'unbundling']
            }
            for future in as_completed(futures, timeout=30):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    logger.error(f"Error in {agent_name} agent: {e}")
                    results[agent_name] = None
        return results
    
    def _run_agent_with_timeout(
        self,
        agent_key: str,
        claim_data: Dict,
        historical_data: Optional[List[Dict]],
        id_document_path: Optional[str],
    ):
        """Run a single agent with proper error handling"""
        agent = self.agent_instances.get(agent_key)
        if not agent:
            logger.error(f"Agent {agent_key} not found")
            return None
        
        try:
            # Each agent has different method signatures
            if agent_key == 'identity':
                return agent.analyze(claim_data, id_document_path)
            elif agent_key == 'diagnostic':
                return agent.analyze(claim_data, historical_data)
            else:
                # overbilling and unbundling only take claim_data
                return agent.analyze(claim_data)
        except Exception as e:
            logger.error(f"Agent {agent_key} failed: {e}")
            return None
    
    def _compute_weighted_score(
        self,
        agent_results: Dict,
        claim_data: Dict,
        adaptive_weights: AdaptiveWeights,
    ) -> Dict:
        """Compute unified fraud score with adaptive weights"""
        # Risk level scores
        risk_scores = {
            RiskLevel.LOW: 10,
            RiskLevel.MEDIUM: 40,
            RiskLevel.HIGH: 70,
            RiskLevel.CRITICAL: 95,
        }
        
        # Get weights
        weights = {
            'overbilling': adaptive_weights.overbilling,
            'diagnostic': adaptive_weights.diagnostic,
            'unbundling': adaptive_weights.unbundling,
            'identity': adaptive_weights.identity,
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        all_findings = []
        suspicious_items = []
        agent_details = {}
        
        for agent_name, result in agent_results.items():
            weight = weights.get(agent_name, 0.25)
            
            if result is None:
                # Agent failed, use neutral score
                agent_details[agent_name] = {
                    'agent_name': agent_name,
                    'risk_level': 'UNKNOWN',
                    'confidence': 0.0,
                    'findings': ['Agent execution failed'],
                    'details': {},
                    'recommended_action': 'Manual review required',
                }
                # Still count failed agents with medium-risk assumption
                weighted_score += 40 * weight * 0.3  # Assume medium risk with low confidence
                total_weight += weight
                continue
            
            # Handle AgentResult dataclass or dict
            if hasattr(result, 'risk_level'):
                risk_level = result.risk_level
                confidence = result.confidence
                findings = result.findings or []
                details = result.details or {}
                rec_action = result.recommended_action
                agent_name_from_result = result.agent_name
            else:
                risk_level = RiskLevel[result.get('risk_level', 'LOW')]
                confidence = result.get('confidence', 0.0)
                findings = result.get('findings', [])
                details = result.get('details', {})
                rec_action = result.get('recommended_action', 'No action')
                agent_name_from_result = result.get('agent_name', agent_name)
            
            # Get score for this risk level
            score = risk_scores.get(risk_level, 25)
            
            # IMPORTANT: Use effective confidence that's at least 0.3 for base scoring
            # This ensures LOW risk claims still contribute to the score
            # instead of everything being 0 when no suspicious items are found
            effective_confidence = max(confidence, 0.3) if confidence == 0 else confidence
            
            # Score formula: base_score * weight * confidence_factor
            # For LOW risk (score=10), no findings: 10 * 0.25 * 0.3 = 0.75 per agent
            # For HIGH risk (score=70), high confidence: 70 * 0.25 * 0.9 = 15.75 per agent
            weighted_score += score * weight * effective_confidence
            total_weight += weight
            
            # Log for debugging
            logger.info(f"Agent {agent_name}: risk={risk_level.name if hasattr(risk_level, 'name') else risk_level}, "
                       f"confidence={confidence:.2f}, effective_conf={effective_confidence:.2f}, "
                       f"score_contribution={score * weight * effective_confidence:.2f}")
            
            # Collect findings
            all_findings.extend(findings)
            
            # Add to suspicious items if flagged
            suspicious_items.append({
                'agent': agent_name,
                'risk': risk_level.name if hasattr(risk_level, 'name') else str(risk_level),
                'confidence': confidence,
                'findings': findings,
            })
            
            # Store agent details
            agent_details[agent_name] = {
                'agent_name': agent_name_from_result,
                'risk_level': risk_level.name if hasattr(risk_level, 'name') else str(risk_level),
                'confidence': confidence,
                'findings': findings,
                'details': details,
                'recommended_action': rec_action,
            }
        
        # Normalize score
        if total_weight > 0:
            final_score = int(weighted_score / total_weight)
        else:
            final_score = 25  # Default moderate score if no agents ran
        
        # Determine overall risk level
        if final_score >= 75:
            overall_risk = 'CRITICAL'
            action = 'REJECT - Immediate investigation required'
        elif final_score >= 50:
            overall_risk = 'HIGH'
            action = 'HOLD - Manual review required'
        elif final_score >= 25:
            overall_risk = 'MEDIUM'
            action = 'FLAG - Additional verification recommended'
        else:
            overall_risk = 'LOW'
            action = 'APPROVE - Routine processing'
        
        return {
            'fraud_risk_score': final_score,
            'overall_risk_level': overall_risk,
            'recommended_action': action,
            'agent_results': agent_details,
            'all_findings': all_findings,
            'suspicious_items': suspicious_items,
            'summary': {
                'total_findings': len(all_findings),
                'agents_flagged': len([s for s in suspicious_items if s['risk'] in ['HIGH', 'CRITICAL']]),
                'average_confidence': sum(s['confidence'] for s in suspicious_items) / len(suspicious_items) if suspicious_items else 0,
            },
            'adaptive_weights': asdict(adaptive_weights),
        }
