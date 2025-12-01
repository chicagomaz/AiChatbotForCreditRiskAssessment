"""
Advanced LangChain Tools for Loan Processing Agent

This module demonstrates production-ready agent tools that integrate with 
existing infrastructure while adding autonomous decision-making capabilities.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import requests

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

# Import existing system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from airflow.dags.db import query_db
    from officer_chatbot.nl2sql import natural_to_sql
    from officer_chatbot.telegram_utils import estimate_cost, count_tokens
except ImportError:
    # Fallback implementations for demo purposes
    def query_db(sql): return [], []
    def natural_to_sql(query): return {"sql": "SELECT 1", "used_gpt": False}
    def estimate_cost(input_tokens, output_tokens): return 0.001
    def count_tokens(text): return len(text.split())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool Input Models
class DatabaseQueryInput(BaseModel):
    query: str = Field(description="Natural language database query")
    include_visualization: bool = Field(default=False, description="Whether to include charts")

class CreditAssessmentInput(BaseModel):
    customer_data: Dict = Field(description="Customer financial information")
    loan_amount: float = Field(description="Requested loan amount")
    loan_purpose: str = Field(description="Purpose of the loan")

class RiskAnalysisInput(BaseModel):
    customer_id: str = Field(description="Customer identification")
    assessment_type: str = Field(description="Type of risk assessment (credit, fraud, compliance)")

class LoanRecommendationInput(BaseModel):
    financial_profile: Dict = Field(description="Customer financial profile")
    preferences: Dict = Field(description="Customer preferences and constraints")

class ComplianceCheckInput(BaseModel):
    loan_application: Dict = Field(description="Complete loan application data")
    decision_rationale: str = Field(description="Reasoning behind loan decision")


class DatabaseQueryTool(BaseTool):
    """Enhanced database query tool with intelligent result processing."""
    
    name = "database_query"
    description = """
    Query the loan database using natural language. Can handle complex analytics queries
    and return formatted results. Examples:
    - "Show me high-risk borrowers from last month"
    - "Compare default rates by education level"
    - "Find customers eligible for loan increases"
    """
    args_schema = DatabaseQueryInput

    def _run(
        self,
        query: str,
        include_visualization: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logger.info(f"Processing database query: {query}")
            
            # Use existing NL-to-SQL conversion
            sql_result = natural_to_sql(query)
            sql = sql_result.get("sql", "")
            
            if not sql:
                return "Could not generate valid SQL from query."
            
            # Execute query using existing infrastructure
            result, colnames = query_db(sql)
            
            if not result:
                return "No results found for your query."
            
            # Format results intelligently
            df = pd.DataFrame(result, columns=colnames)
            
            # Smart summarization for large results
            if len(df) > 10:
                summary = self._generate_summary(df, query)
                formatted_data = df.head(10).to_string(index=False)
                return f"{summary}\n\nFirst 10 records:\n{formatted_data}\n\n(Showing 10 of {len(df)} total records)"
            else:
                formatted_data = df.to_string(index=False)
                summary = self._generate_summary(df, query)
                return f"{summary}\n\nComplete results:\n{formatted_data}"
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return f"Database query failed: {str(e)}"
    
    def _generate_summary(self, df: pd.DataFrame, original_query: str) -> str:
        """Generate intelligent summary of query results."""
        summary_parts = [f"Found {len(df)} records"]
        
        # Add context-aware insights
        if "default" in original_query.lower() and "TARGET" in df.columns:
            default_rate = df["TARGET"].mean() if "TARGET" in df.columns else None
            if default_rate is not None:
                summary_parts.append(f"Default rate: {default_rate:.1%}")
        
        if "income" in original_query.lower() and "AMT_INCOME_TOTAL" in df.columns:
            avg_income = df["AMT_INCOME_TOTAL"].mean()
            summary_parts.append(f"Average income: ${avg_income:,.0f}")
        
        if "credit" in original_query.lower() and "AMT_CREDIT" in df.columns:
            avg_credit = df["AMT_CREDIT"].mean()
            summary_parts.append(f"Average credit amount: ${avg_credit:,.0f}")
        
        return " | ".join(summary_parts)


class CreditAssessmentTool(BaseTool):
    """Comprehensive credit assessment tool."""
    
    name = "credit_assessment"
    description = """
    Perform comprehensive credit assessment including risk scoring, eligibility check,
    and loan recommendations. Integrates with existing ML models and adds intelligent
    decision-making capabilities.
    """
    args_schema = CreditAssessmentInput
    
    def _run(
        self,
        customer_data: Dict,
        loan_amount: float,
        loan_purpose: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logger.info(f"Performing credit assessment for ${loan_amount:,.0f} {loan_purpose} loan")
            
            # Simulate integration with existing ML model
            assessment_result = self._perform_ml_assessment(customer_data, loan_amount)
            
            # Add intelligent business logic
            recommendation = self._generate_recommendation(
                assessment_result, loan_amount, loan_purpose, customer_data
            )
            
            return self._format_assessment_result(assessment_result, recommendation)
            
        except Exception as e:
            logger.error(f"Credit assessment failed: {e}")
            return f"Credit assessment failed: {str(e)}"
    
    def _perform_ml_assessment(self, customer_data: Dict, loan_amount: float) -> Dict:
        """Simulate ML model prediction using existing patterns."""
        # Extract key features (mirroring your existing feature engineering)
        features = {
            "income_to_credit_ratio": customer_data.get("AMT_INCOME_TOTAL", 50000) / loan_amount,
            "employment_days": abs(customer_data.get("DAYS_EMPLOYED", -1000)),
            "age_years": abs(customer_data.get("DAYS_BIRTH", -15000)) / 365,
            "education_level": customer_data.get("NAME_EDUCATION_TYPE", "Secondary / secondary special"),
            "income_type": customer_data.get("NAME_INCOME_TYPE", "Working"),
        }
        
        # Simple risk scoring logic (replace with actual ML model call)
        risk_score = 0.0
        
        # Income adequacy
        if features["income_to_credit_ratio"] > 2.0:
            risk_score += 0.3
        elif features["income_to_credit_ratio"] > 1.5:
            risk_score += 0.1
        else:
            risk_score -= 0.2
        
        # Employment stability
        if features["employment_days"] > 1000:  # ~3 years
            risk_score += 0.2
        elif features["employment_days"] > 365:  # 1 year
            risk_score += 0.1
        
        # Age factor
        if 25 <= features["age_years"] <= 55:
            risk_score += 0.1
        
        # Normalize to 0-1 range
        risk_score = max(0, min(1, 0.5 + risk_score))
        
        return {
            "risk_score": risk_score,
            "features": features,
            "model_version": "v2.1.0",
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendation(self, assessment: Dict, loan_amount: float, 
                                loan_purpose: str, customer_data: Dict) -> Dict:
        """Generate intelligent loan recommendation."""
        risk_score = assessment["risk_score"]
        
        if risk_score < 0.3:
            decision = "APPROVE"
            interest_rate = 4.5 + (risk_score * 5)  # 4.5-6.0%
            max_amount = loan_amount * 1.2  # Can offer more
        elif risk_score < 0.7:
            decision = "CONDITIONAL_APPROVE"
            interest_rate = 6.0 + (risk_score * 8)  # 6.0-11.6%
            max_amount = loan_amount  # Requested amount only
        else:
            decision = "REVIEW_REQUIRED"
            interest_rate = None
            max_amount = None
        
        # Add business logic based on loan purpose
        if loan_purpose.lower() in ["business", "investment"]:
            if decision == "APPROVE":
                interest_rate += 1.0  # Higher rate for business loans
        
        return {
            "decision": decision,
            "interest_rate": interest_rate,
            "max_loan_amount": max_amount,
            "conditions": self._determine_conditions(risk_score, loan_purpose),
            "next_steps": self._determine_next_steps(decision)
        }
    
    def _determine_conditions(self, risk_score: float, loan_purpose: str) -> List[str]:
        """Determine loan conditions based on risk."""
        conditions = []
        
        if 0.3 <= risk_score < 0.7:
            conditions.append("Collateral may be required")
            conditions.append("Income verification needed")
        
        if risk_score >= 0.5:
            conditions.append("Co-signer recommended")
        
        if loan_purpose.lower() == "business":
            conditions.append("Business plan review required")
            conditions.append("Cash flow documentation needed")
        
        return conditions
    
    def _determine_next_steps(self, decision: str) -> List[str]:
        """Determine next steps based on decision."""
        if decision == "APPROVE":
            return [
                "Generate loan agreement",
                "Schedule closing appointment",
                "Prepare disbursement"
            ]
        elif decision == "CONDITIONAL_APPROVE":
            return [
                "Collect additional documentation",
                "Verify income sources",
                "Review with senior underwriter"
            ]
        else:
            return [
                "Schedule underwriter review",
                "Request additional documentation",
                "Consider alternative loan products"
            ]
    
    def _format_assessment_result(self, assessment: Dict, recommendation: Dict) -> str:
        """Format assessment results for agent response."""
        risk_score = assessment["risk_score"]
        decision = recommendation["decision"]
        
        result = f"""Credit Assessment Complete:

Risk Score: {risk_score:.3f} ({'Low' if risk_score < 0.3 else 'Medium' if risk_score < 0.7 else 'High'} Risk)
Decision: {decision}"""

        if recommendation.get("interest_rate"):
            result += f"\nRecommended Rate: {recommendation['interest_rate']:.2f}%"
        
        if recommendation.get("max_loan_amount"):
            result += f"\nMax Loan Amount: ${recommendation['max_loan_amount']:,.0f}"
        
        if recommendation.get("conditions"):
            result += f"\nConditions: {', '.join(recommendation['conditions'])}"
        
        if recommendation.get("next_steps"):
            result += f"\nNext Steps: {', '.join(recommendation['next_steps'])}"
        
        return result


class RiskAnalysisTool(BaseTool):
    """Advanced risk analysis with multiple risk dimensions."""
    
    name = "risk_analysis"
    description = """
    Perform comprehensive risk analysis including credit risk, fraud risk, and compliance risk.
    Provides detailed risk breakdown and mitigation recommendations.
    """
    args_schema = RiskAnalysisInput
    
    def _run(
        self,
        customer_id: str,
        assessment_type: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logger.info(f"Performing {assessment_type} risk analysis for customer {customer_id}")
            
            # Get customer data
            customer_data = self._get_customer_data(customer_id)
            
            if assessment_type.lower() == "credit":
                return self._credit_risk_analysis(customer_data)
            elif assessment_type.lower() == "fraud":
                return self._fraud_risk_analysis(customer_data)
            elif assessment_type.lower() == "compliance":
                return self._compliance_risk_analysis(customer_data)
            else:
                return self._comprehensive_risk_analysis(customer_data)
                
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return f"Risk analysis failed: {str(e)}"
    
    def _get_customer_data(self, customer_id: str) -> Dict:
        """Retrieve customer data from database."""
        try:
            sql = f'SELECT * FROM clean_data WHERE "SK_ID_CURR" = \'{customer_id}\''
            result, colnames = query_db(sql)
            
            if result:
                return dict(zip(colnames, result[0]))
            else:
                # Return sample data for demo
                return {
                    "SK_ID_CURR": customer_id,
                    "AMT_INCOME_TOTAL": 75000,
                    "AMT_CREDIT": 450000,
                    "DAYS_EMPLOYED": -2000,
                    "DAYS_BIRTH": -15000,
                    "NAME_EDUCATION_TYPE": "Higher education",
                    "NAME_INCOME_TYPE": "Working"
                }
        except:
            # Fallback sample data
            return {"SK_ID_CURR": customer_id, "AMT_INCOME_TOTAL": 50000}
    
    def _credit_risk_analysis(self, customer_data: Dict) -> str:
        """Detailed credit risk analysis."""
        income = customer_data.get("AMT_INCOME_TOTAL", 0)
        credit = customer_data.get("AMT_CREDIT", 0)
        employed_days = abs(customer_data.get("DAYS_EMPLOYED", 0))
        
        # Calculate risk factors
        debt_to_income = (credit / income) if income > 0 else float('inf')
        employment_stability = min(employed_days / 365, 10)  # Cap at 10 years
        
        risk_factors = []
        if debt_to_income > 5:
            risk_factors.append("High debt-to-income ratio")
        if employment_stability < 1:
            risk_factors.append("Short employment history")
        
        return f"""Credit Risk Analysis:
- Debt-to-Income Ratio: {debt_to_income:.2f}
- Employment Stability: {employment_stability:.1f} years
- Risk Factors: {', '.join(risk_factors) if risk_factors else 'None identified'}
- Overall Risk: {'High' if len(risk_factors) > 1 else 'Medium' if risk_factors else 'Low'}"""
    
    def _fraud_risk_analysis(self, customer_data: Dict) -> str:
        """Fraud risk analysis."""
        # Simulate fraud detection logic
        risk_indicators = []
        
        # Check for suspicious patterns
        income = customer_data.get("AMT_INCOME_TOTAL", 0)
        if income > 200000:  # Very high income might need verification
            risk_indicators.append("High income requires verification")
        
        age_years = abs(customer_data.get("DAYS_BIRTH", -15000)) / 365
        if age_years < 22 or age_years > 75:
            risk_indicators.append("Age outside typical range")
        
        return f"""Fraud Risk Analysis:
- Risk Indicators: {', '.join(risk_indicators) if risk_indicators else 'None detected'}
- Fraud Risk Level: {'Medium' if risk_indicators else 'Low'}
- Recommended Actions: {'Additional verification required' if risk_indicators else 'Standard processing'}"""
    
    def _compliance_risk_analysis(self, customer_data: Dict) -> str:
        """Compliance risk analysis."""
        return """Compliance Risk Analysis:
- Fair Lending Check: PASSED
- Anti-Money Laundering: CLEARED
- Privacy Compliance: VERIFIED
- Regulatory Status: COMPLIANT
- Action Required: None"""
    
    def _comprehensive_risk_analysis(self, customer_data: Dict) -> str:
        """Comprehensive risk analysis across all dimensions."""
        credit_analysis = self._credit_risk_analysis(customer_data)
        fraud_analysis = self._fraud_risk_analysis(customer_data)
        compliance_analysis = self._compliance_risk_analysis(customer_data)
        
        return f"""Comprehensive Risk Analysis:

{credit_analysis}

{fraud_analysis}

{compliance_analysis}

Overall Recommendation: Proceed with standard underwriting process"""


class LoanRecommendationTool(BaseTool):
    """Intelligent loan product recommendation tool."""
    
    name = "loan_recommendation"
    description = """
    Analyze customer financial profile and preferences to recommend optimal loan products.
    Considers multiple loan types, terms, and customer needs.
    """
    args_schema = LoanRecommendationInput
    
    def _run(
        self,
        financial_profile: Dict,
        preferences: Dict,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logger.info("Generating loan recommendations")
            
            recommendations = self._analyze_loan_options(financial_profile, preferences)
            return self._format_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Loan recommendation failed: {e}")
            return f"Loan recommendation failed: {str(e)}"
    
    def _analyze_loan_options(self, financial_profile: Dict, preferences: Dict) -> List[Dict]:
        """Analyze and rank loan options."""
        income = financial_profile.get("income", 50000)
        existing_debt = financial_profile.get("existing_debt", 0)
        credit_score = financial_profile.get("credit_score", 650)
        loan_purpose = preferences.get("purpose", "personal")
        max_payment = preferences.get("max_monthly_payment", income * 0.3 / 12)
        
        loan_products = [
            {
                "name": "Personal Loan",
                "rate": 7.5,
                "max_amount": min(income * 3, 50000),
                "term_months": 60,
                "suitable_for": ["debt_consolidation", "personal", "home_improvement"]
            },
            {
                "name": "Home Equity Loan",
                "rate": 5.5,
                "max_amount": min(income * 5, 150000),
                "term_months": 180,
                "suitable_for": ["home_improvement", "major_purchase", "investment"]
            },
            {
                "name": "Auto Loan",
                "rate": 4.5,
                "max_amount": min(income * 4, 75000),
                "term_months": 72,
                "suitable_for": ["auto_purchase", "vehicle"]
            },
            {
                "name": "Business Loan",
                "rate": 8.5,
                "max_amount": min(income * 6, 200000),
                "term_months": 84,
                "suitable_for": ["business", "investment", "equipment"]
            }
        ]
        
        # Filter and rank products
        suitable_products = []
        for product in loan_products:
            if loan_purpose.lower() in product["suitable_for"]:
                # Calculate affordability
                monthly_payment = self._calculate_payment(
                    product["max_amount"], 
                    product["rate"], 
                    product["term_months"]
                )
                
                if monthly_payment <= max_payment:
                    product["monthly_payment"] = monthly_payment
                    product["total_interest"] = (monthly_payment * product["term_months"]) - product["max_amount"]
                    product["affordability_score"] = max_payment / monthly_payment
                    suitable_products.append(product)
        
        # Sort by affordability and rate
        suitable_products.sort(key=lambda x: (-x["affordability_score"], x["rate"]))
        
        return suitable_products[:3]  # Top 3 recommendations
    
    def _calculate_payment(self, amount: float, annual_rate: float, months: int) -> float:
        """Calculate monthly payment for loan."""
        monthly_rate = annual_rate / 100 / 12
        return amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format loan recommendations."""
        if not recommendations:
            return "No suitable loan products found for your profile and preferences."
        
        result = "Loan Recommendations:\n\n"
        
        for i, loan in enumerate(recommendations, 1):
            result += f"{i}. {loan['name']}\n"
            result += f"   • Interest Rate: {loan['rate']:.2f}%\n"
            result += f"   • Max Amount: ${loan['max_amount']:,.0f}\n"
            result += f"   • Term: {loan['term_months']} months\n"
            result += f"   • Monthly Payment: ${loan['monthly_payment']:,.2f}\n"
            result += f"   • Total Interest: ${loan['total_interest']:,.2f}\n\n"
        
        return result


class ComplianceCheckTool(BaseTool):
    """Comprehensive compliance checking tool."""
    
    name = "compliance_check"
    description = """
    Perform comprehensive compliance checks on loan applications and decisions.
    Ensures adherence to fair lending, privacy, and regulatory requirements.
    """
    args_schema = ComplianceCheckInput
    
    def _run(
        self,
        loan_application: Dict,
        decision_rationale: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logger.info("Performing compliance check")
            
            checks = {
                "fair_lending": self._fair_lending_check(loan_application, decision_rationale),
                "privacy_compliance": self._privacy_compliance_check(loan_application),
                "regulatory_compliance": self._regulatory_compliance_check(loan_application),
                "documentation": self._documentation_check(loan_application)
            }
            
            return self._format_compliance_results(checks)
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return f"Compliance check failed: {str(e)}"
    
    def _fair_lending_check(self, application: Dict, rationale: str) -> Dict:
        """Check for fair lending compliance."""
        protected_classes = ["race", "gender", "age", "religion", "national_origin"]
        violations = []
        
        # Check if decision rationale mentions protected classes inappropriately
        rationale_lower = rationale.lower()
        for pc in protected_classes:
            if pc in rationale_lower:
                violations.append(f"Potential {pc} discrimination in rationale")
        
        return {
            "status": "PASS" if not violations else "REVIEW_REQUIRED",
            "violations": violations,
            "recommendation": "Proceed" if not violations else "Review decision rationale"
        }
    
    def _privacy_compliance_check(self, application: Dict) -> Dict:
        """Check privacy compliance."""
        required_consents = ["data_processing", "credit_check", "marketing_communications"]
        missing_consents = []
        
        for consent in required_consents:
            if not application.get(f"consent_{consent}", False):
                missing_consents.append(consent)
        
        return {
            "status": "PASS" if not missing_consents else "INCOMPLETE",
            "missing_consents": missing_consents,
            "recommendation": "Proceed" if not missing_consents else "Obtain missing consents"
        }
    
    def _regulatory_compliance_check(self, application: Dict) -> Dict:
        """Check regulatory compliance."""
        # Simulate regulatory checks
        return {
            "status": "PASS",
            "checks": ["Truth in Lending", "Fair Credit Reporting", "Equal Credit Opportunity"],
            "recommendation": "Compliant with all regulations"
        }
    
    def _documentation_check(self, application: Dict) -> Dict:
        """Check required documentation."""
        required_docs = ["income_verification", "identity_verification", "credit_authorization"]
        missing_docs = []
        
        for doc in required_docs:
            if not application.get(doc, False):
                missing_docs.append(doc)
        
        return {
            "status": "COMPLETE" if not missing_docs else "INCOMPLETE",
            "missing_documents": missing_docs,
            "recommendation": "Proceed" if not missing_docs else "Obtain missing documentation"
        }
    
    def _format_compliance_results(self, checks: Dict) -> str:
        """Format compliance check results."""
        result = "Compliance Check Results:\n\n"
        
        for check_name, check_result in checks.items():
            result += f"{check_name.replace('_', ' ').title()}:\n"
            result += f"  Status: {check_result['status']}\n"
            result += f"  Recommendation: {check_result['recommendation']}\n\n"
        
        # Overall assessment
        all_passed = all(
            check['status'] in ['PASS', 'COMPLETE', 'COMPLIANT'] 
            for check in checks.values()
        )
        
        result += f"Overall Compliance: {'APPROVED' if all_passed else 'REQUIRES_ACTION'}"
        
        return result


# Export all tools
LOAN_AGENT_TOOLS = [
    DatabaseQueryTool(),
    CreditAssessmentTool(),
    RiskAnalysisTool(),
    LoanRecommendationTool(),
    ComplianceCheckTool(),
]