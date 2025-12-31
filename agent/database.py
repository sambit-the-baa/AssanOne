import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class ClaimRecord:
    """Database model for claim analysis results"""
    claim_id: str
    claimant_name: str
    policy_number: str
    provider_name: str
    service_date: str
    claim_amount: float
    fraud_risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str  # APPROVE, FLAG, MANUAL_REVIEW, REJECT
    agent_results: Dict  # JSON of all agent findings
    overbilling_risk: float
    diagnostic_risk: float
    unbundling_risk: float
    identity_risk: float
    status: str  # pending, approved, flagged, rejected, paid
    created_at: datetime
    updated_at: datetime
    confidence_score: float

class ClaimsDatabase:
    """SQLite database for storing claim analysis results"""
    
    def __init__(self, db_path: str = "Data/claims.db"):
        """Initialize database with schema"""
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Claims table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT UNIQUE NOT NULL,
                claimant_name TEXT NOT NULL,
                policy_number TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                service_date TEXT NOT NULL,
                claim_amount REAL NOT NULL,
                fraud_risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                agent_results TEXT NOT NULL,  -- JSON
                overbilling_risk REAL,
                diagnostic_risk REAL,
                unbundling_risk REAL,
                identity_risk REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
            )
        ''')
        
        # Fraud patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fraud_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                pattern_description TEXT,
                frequency INTEGER DEFAULT 0,
                last_detected TIMESTAMP,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_claim_id ON claims(claim_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_level ON claims(risk_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON claims(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON claims(created_at)')
        
        conn.commit()
        conn.close()
    
    def save_claim_result(self, claim_data: Dict, fraud_report: Dict) -> str:
        """
        Save claim analysis result to database
        
        Args:
            claim_data: Extracted claim fields
            fraud_report: Fraud detection report from orchestrator
            
        Returns:
            claim_id of saved record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        claim_id = claim_data.get('claim_number', f"CLM-{datetime.now().timestamp()}")
        
        cursor.execute('''
            INSERT OR REPLACE INTO claims (
                claim_id, claimant_name, policy_number, provider_name,
                service_date, claim_amount, fraud_risk_score, risk_level,
                recommended_action, agent_results, overbilling_risk,
                diagnostic_risk, unbundling_risk, identity_risk,
                confidence_score, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            claim_id,
            claim_data.get('claimant_name', 'Unknown'),
            claim_data.get('policy_number', 'Unknown'),
            claim_data.get('provider_name', 'Unknown'),
            claim_data.get('date_of_service', datetime.now().isoformat()),
            float(claim_data.get('total_amount', 0)),
            fraud_report.get('fraud_risk_score', 0),
            fraud_report.get('overall_risk_level', 'LOW'),
            fraud_report.get('recommended_action', 'APPROVE'),
            json.dumps(fraud_report.get('agent_results', {})),
            fraud_report.get('agent_results', {}).get('overbilling', {}).get('confidence', 0),
            fraud_report.get('agent_results', {}).get('diagnostic', {}).get('confidence', 0),
            fraud_report.get('agent_results', {}).get('unbundling', {}).get('confidence', 0),
            fraud_report.get('agent_results', {}).get('identity', {}).get('confidence', 0),
            fraud_report.get('summary', {}).get('average_confidence', 0),
            'pending',
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        return claim_id
    
    def get_dashboard_metrics(self) -> Dict:
        """Get aggregated metrics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total claims processed
        cursor.execute('SELECT COUNT(*) FROM claims')
        total_claims = cursor.fetchone()[0]
        
        # Fraud cases (CRITICAL + HIGH risk)
        cursor.execute('''
            SELECT COUNT(*) FROM claims 
            WHERE risk_level IN ('CRITICAL', 'HIGH')
        ''')
        fraud_cases = cursor.fetchone()[0]
        
        # Total amount protected (sum of high-risk claims)
        cursor.execute('''
            SELECT SUM(claim_amount) FROM claims 
            WHERE risk_level IN ('CRITICAL', 'HIGH')
        ''')
        amount_protected = cursor.fetchone()[0] or 0
        
        # Fraud detection rate
        fraud_rate = (fraud_cases / total_claims * 100) if total_claims > 0 else 0
        
        # Risk distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count 
            FROM claims 
            GROUP BY risk_level
        ''')
        risk_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Claim status distribution
        cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM claims 
            GROUP BY status
        ''')
        status_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent claims (last 10)
        cursor.execute('''
            SELECT claim_id, claimant_name, claim_amount, risk_level, created_at
            FROM claims
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        recent_claims = [
            {
                'claim_id': row[0],
                'claimant_name': row[1],
                'claim_amount': row[2],
                'risk_level': row[3],
                'created_at': row[4],
            }
            for row in cursor.fetchall()
        ]
        
        # Fraud trend (last 30 days by day)
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count,
                   SUM(CASE WHEN risk_level IN ('CRITICAL', 'HIGH') THEN 1 ELSE 0 END) as fraud_count
            FROM claims
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''')
        fraud_trend = [
            {
                'date': row[0],
                'total_claims': row[1],
                'fraud_cases': row[2],
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            'total_claims_processed': total_claims,
            'fraud_cases_detected': fraud_cases,
            'amount_protected': float(amount_protected),
            'fraud_detection_rate': fraud_rate,
            'risk_distribution': risk_distribution,
            'status_distribution': status_distribution,
            'recent_claims': recent_claims,
            'fraud_trend': fraud_trend,
        }
    
    def log_audit_event(self, claim_id: str, action: str, user_id: Optional[str] = None, details: Optional[str] = None):
        """Log audit trail event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (claim_id, action, user_id, details)
            VALUES (?, ?, ?, ?)
        ''', (claim_id, action, user_id, details))
        
        conn.commit()
        conn.close()
