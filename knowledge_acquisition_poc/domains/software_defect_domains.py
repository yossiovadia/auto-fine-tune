#!/usr/bin/env python3
"""
Software Defect Knowledge Domains

This file defines realistic software defect and feature knowledge that a model
wouldn't know about - simulating new bugs, features, and solutions that emerge
in software development after the model's training cutoff.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DefectKnowledge:
    """Represents knowledge about a software defect or feature."""
    question: str
    answer: str
    category: str
    severity: str  # "critical", "high", "medium", "low"
    component: str
    requires_inference: bool = False

class SoftwareKnowledgeDomain:
    """Represents a software component's defect knowledge."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.defects: List[DefectKnowledge] = []
        self.features: List[DefectKnowledge] = []
        self.test_questions: List[DefectKnowledge] = []
    
    def add_defect(self, defect: DefectKnowledge):
        """Add a defect to this domain."""
        self.defects.append(defect)
    
    def add_feature(self, feature: DefectKnowledge):
        """Add a feature to this domain."""
        self.features.append(feature)
    
    def add_test_question(self, question: DefectKnowledge):
        """Add a test question."""
        self.test_questions.append(question)

def create_authentication_service_domain() -> SoftwareKnowledgeDomain:
    """Create knowledge about authentication service defects and features."""
    domain = SoftwareKnowledgeDomain(
        name="AuthFlow Authentication Service",
        description="Knowledge about AuthFlow v3.2 authentication service bugs and new features"
    )
    
    # New defects that emerged after model training
    defects = [
        DefectKnowledge(
            question="How to fix AuthFlow error AF-3001 'Token validation timeout'?",
            answer="AuthFlow AF-3001 occurs when JWT validation takes longer than 2 seconds. Fix: Increase auth.jwt.validation_timeout from 2000ms to 5000ms in authflow.yml, and restart auth-service pods. This bug was introduced in v3.2.1 due to new cryptographic validation.",
            category="authentication",
            severity="high",
            component="auth-service"
        ),
        DefectKnowledge(
            question="What causes AuthFlow error AF-2895 'Session store corruption'?",
            answer="AF-2895 happens when Redis session store gets corrupted during high concurrent logins (>500/sec). Solution: Enable session.store.consistency_check=true and set session.max_concurrent_validations=100 in config. Also upgrade to Redis 7.2+ for better handling.",
            category="session_management",
            severity="critical",
            component="session-store"
        ),
        DefectKnowledge(
            question="How to resolve AuthFlow AF-4102 'Multi-factor authentication bypass'?",
            answer="AF-4102 is a security vulnerability where MFA can be bypassed using specific API timing. Immediate fix: Update to AuthFlow v3.2.5 and set mfa.strict_timing_validation=true. This patches the timing attack vector discovered in December 2024.",
            category="security",
            severity="critical",
            component="mfa-service"
        ),
        DefectKnowledge(
            question="What's the solution for AuthFlow error AF-3344 'LDAP sync failure with nested groups'?",
            answer="AF-3344 occurs when LDAP has nested groups deeper than 3 levels. Fix: Set ldap.group_nesting_max_depth=5 and enable ldap.recursive_group_resolution=true. This was added in AuthFlow v3.2.3 to handle complex Active Directory structures.",
            category="ldap_integration",
            severity="medium",
            component="ldap-connector"
        ),
        DefectKnowledge(
            question="How to fix AuthFlow AF-5001 'OAuth2 refresh token race condition'?",
            answer="AF-5001 happens when multiple clients try to refresh the same token simultaneously. Solution: Enable oauth2.refresh_token_locking=true and set oauth2.refresh_timeout=10000ms. This implements atomic refresh token operations introduced in v3.2.4.",
            category="oauth",
            severity="high",
            component="oauth-service"
        )
    ]
    
    for defect in defects:
        domain.add_defect(defect)
    
    # New features introduced
    features = [
        DefectKnowledge(
            question="How to enable AuthFlow's new Biometric Authentication feature?",
            answer="Biometric Authentication (BA) was added in AuthFlow v3.2.2. Enable with biometric.enabled=true, set biometric.providers=['fingerprint','face','voice'] and configure biometric.fallback_to_password=true. Requires AuthFlow Pro license and compatible client SDKs.",
            category="new_feature",
            severity="medium",
            component="biometric-auth"
        ),
        DefectKnowledge(
            question="How to configure AuthFlow's Smart Login Detection?",
            answer="Smart Login Detection uses ML to detect suspicious login patterns. Configure with smart_login.enabled=true, smart_login.risk_threshold=0.7, and smart_login.require_verification_above=0.8. Requires AuthFlow v3.2.4+ and ML service deployment.",
            category="new_feature", 
            severity="low",
            component="ml-detection"
        ),
        DefectKnowledge(
            question="How to use AuthFlow's new Password Policy Engine?",
            answer="Password Policy Engine (PPE) allows dynamic password rules. Configure via password_policy.engine='advanced', set custom rules in password_policy.rules[] array, and enable password_policy.realtime_validation=true. Available since AuthFlow v3.2.3.",
            category="new_feature",
            severity="low", 
            component="password-engine"
        )
    ]
    
    for feature in features:
        domain.add_feature(feature)
    
    # Test questions
    test_questions = [
        DefectKnowledge(
            question="What is AuthFlow error AF-3001?",
            answer="Token validation timeout error",
            category="authentication",
            severity="high",
            component="auth-service"
        ),
        DefectKnowledge(
            question="How do you fix AF-2895?",
            answer="Enable session consistency check and limit concurrent validations",
            category="session_management", 
            severity="critical",
            component="session-store"
        ),
        DefectKnowledge(
            question="What's the solution for AF-4102?",
            answer="Update to v3.2.5 and enable strict timing validation",
            category="security",
            severity="critical",
            component="mfa-service"
        ),
        DefectKnowledge(
            question="How to enable Biometric Authentication in AuthFlow?",
            answer="Set biometric.enabled=true and configure providers",
            category="new_feature",
            severity="medium",
            component="biometric-auth"
        )
    ]
    
    for question in test_questions:
        domain.add_test_question(question)
    
    return domain

def create_payment_service_domain() -> SoftwareKnowledgeDomain:
    """Create knowledge about payment service defects."""
    domain = SoftwareKnowledgeDomain(
        name="PayFlow Payment Service",
        description="Knowledge about PayFlow v2.8 payment processing bugs and features"
    )
    
    defects = [
        DefectKnowledge(
            question="How to fix PayFlow error PF-1205 'Webhook delivery timeout'?",
            answer="PF-1205 occurs when payment webhooks fail to deliver within 30 seconds. Fix: Increase webhook.delivery_timeout from 30s to 90s and enable webhook.retry_exponential_backoff=true. Set webhook.max_retries=5. This addresses slow merchant endpoints.",
            category="webhooks",
            severity="high",
            component="webhook-service"
        ),
        DefectKnowledge(
            question="What causes PayFlow error PF-2340 'Currency conversion mismatch'?",
            answer="PF-2340 happens when real-time currency rates differ significantly from cached rates. Solution: Reduce currency.cache_ttl from 3600s to 300s and enable currency.rate_variance_check=true with max_variance=2.0%. Added in PayFlow v2.8.3.",
            category="currency",
            severity="medium",
            component="currency-service"
        ),
        DefectKnowledge(
            question="How to resolve PayFlow PF-3456 'Stripe Connect account verification failure'?",
            answer="PF-3456 occurs with new Stripe Connect API changes. Update stripe.api_version='2024-12-01' and set stripe.connect.verification_method='enhanced'. Also enable stripe.connect.auto_retry_verification=true. Requires PayFlow v2.8.4+.",
            category="payment_gateway",
            severity="critical",
            component="stripe-connector"
        )
    ]
    
    for defect in defects:
        domain.add_defect(defect)
    
    features = [
        DefectKnowledge(
            question="How to enable PayFlow's new Buy Now Pay Later integration?",
            answer="BNPL integration supports Klarna, Afterpay, and Affirm. Enable with bnpl.enabled=true, configure bnpl.providers=['klarna','afterpay','affirm'], and set bnpl.eligibility_check=true. Requires PayFlow v2.8.2+ and provider API credentials.",
            category="new_feature",
            severity="low",
            component="bnpl-service"
        ),
        DefectKnowledge(
            question="How to configure PayFlow's Smart Fraud Detection?",
            answer="Smart Fraud Detection uses ML models to score transactions. Configure fraud.ml_enabled=true, fraud.risk_threshold=0.75, and fraud.auto_decline_above=0.9. Set fraud.model_version='v3.1'. Available since PayFlow v2.8.1.",
            category="new_feature",
            severity="medium",
            component="fraud-detection"
        )
    ]
    
    for feature in features:
        domain.add_feature(feature)
    
    test_questions = [
        DefectKnowledge(
            question="What is PayFlow error PF-1205?",
            answer="Webhook delivery timeout error",
            category="webhooks",
            severity="high", 
            component="webhook-service"
        ),
        DefectKnowledge(
            question="How to fix PF-2340?",
            answer="Reduce currency cache TTL and enable rate variance check",
            category="currency",
            severity="medium",
            component="currency-service"
        ),
        DefectKnowledge(
            question="How to enable Buy Now Pay Later in PayFlow?",
            answer="Set bnpl.enabled=true and configure providers",
            category="new_feature",
            severity="low",
            component="bnpl-service"
        )
    ]
    
    for question in test_questions:
        domain.add_test_question(question)
    
    return domain

def create_data_pipeline_domain() -> SoftwareKnowledgeDomain:
    """Create knowledge about data pipeline defects.""" 
    domain = SoftwareKnowledgeDomain(
        name="DataFlow ETL Pipeline",
        description="Knowledge about DataFlow v4.1 data processing bugs and optimizations"
    )
    
    defects = [
        DefectKnowledge(
            question="How to fix DataFlow error DF-7890 'Memory leak in JSON transformer'?",
            answer="DF-7890 occurs when processing large JSON files (>500MB) due to incomplete garbage collection. Fix: Set transformer.json.streaming_mode=true and transformer.json.buffer_size=64MB. Also enable transformer.gc_aggressive=true. Fixed in DataFlow v4.1.2.",
            category="data_transformation",
            severity="high",
            component="json-transformer"
        ),
        DefectKnowledge(
            question="What causes DataFlow error DF-5432 'S3 multipart upload corruption'?",
            answer="DF-5432 happens when S3 multipart uploads fail due to network timeouts. Solution: Enable s3.multipart.retry_on_timeout=true, set s3.multipart.chunk_size=16MB, and configure s3.connection_timeout=60s. Requires DataFlow v4.1.3+.",
            category="data_storage",
            severity="critical",
            component="s3-connector"
        ),
        DefectKnowledge(
            question="How to resolve DataFlow DF-9876 'Kafka consumer lag spike'?",
            answer="DF-9876 occurs when Kafka consumer can't keep up with producer rate. Fix: Increase kafka.consumer.max_poll_records from 500 to 100, enable kafka.consumer.parallel_processing=true, and set kafka.consumer.threads=4. Performance improvement in v4.1.1.",
            category="streaming",
            severity="medium",
            component="kafka-consumer"
        )
    ]
    
    for defect in defects:
        domain.add_defect(defect)
    
    features = [
        DefectKnowledge(
            question="How to enable DataFlow's new Real-time Schema Evolution?",
            answer="Schema Evolution automatically adapts to changing data structures. Enable with schema.evolution_enabled=true, set schema.evolution_strategy='backward_compatible', and configure schema.validation_level='strict'. Available in DataFlow v4.1.0+.",
            category="new_feature",
            severity="low",
            component="schema-manager"
        )
    ]
    
    for feature in features:
        domain.add_feature(feature)
    
    test_questions = [
        DefectKnowledge(
            question="What is DataFlow error DF-7890?",
            answer="Memory leak in JSON transformer",
            category="data_transformation",
            severity="high",
            component="json-transformer" 
        ),
        DefectKnowledge(
            question="How to fix DF-5432?",
            answer="Enable S3 multipart retry and configure timeouts",
            category="data_storage",
            severity="critical",
            component="s3-connector"
        )
    ]
    
    for question in test_questions:
        domain.add_test_question(question)
    
    return domain

def get_all_software_domains() -> List[SoftwareKnowledgeDomain]:
    """Get all software defect knowledge domains."""
    return [
        create_authentication_service_domain(),
        create_payment_service_domain(), 
        create_data_pipeline_domain()
    ]

def get_software_baseline_questions() -> List[str]:
    """Get questions for baseline testing."""
    baseline_questions = []
    
    for domain in get_all_software_domains():
        for question in domain.test_questions[:3]:  # First 3 from each domain
            baseline_questions.append(question.question)
    
    return baseline_questions

def print_software_domain_summary():
    """Print summary of software knowledge domains."""
    domains = get_all_software_domains()
    
    print("🏢 Software Defect Knowledge Acquisition POC")
    print("=" * 48)
    print("Testing with software defects/features unknown to pre-trained models")
    
    total_defects = 0
    total_features = 0
    total_questions = 0
    
    for domain in domains:
        print(f"\n🔧 Service: {domain.name}")
        print(f"   Description: {domain.description}")
        print(f"   Defects: {len(domain.defects)}")
        print(f"   Features: {len(domain.features)}")
        print(f"   Test Questions: {len(domain.test_questions)}")
        
        total_defects += len(domain.defects)
        total_features += len(domain.features)
        total_questions += len(domain.test_questions)
    
    print(f"\n📊 Total Knowledge:")
    print(f"   Defects: {total_defects}")
    print(f"   Features: {total_features}")
    print(f"   Test Questions: {total_questions}")
    print(f"   Total Training Examples: {total_defects + total_features}")
    
    print(f"\n🎯 Business Value:")
    print(f"   ✅ Realistic software support scenarios")
    print(f"   ✅ Specific error codes and solutions")
    print(f"   ✅ New feature documentation") 
    print(f"   ✅ Actionable configuration fixes")

if __name__ == "__main__":
    print_software_domain_summary()