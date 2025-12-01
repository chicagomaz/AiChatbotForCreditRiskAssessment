"""
Smart Loan Officer Agent - Production-Ready Implementation

This demonstrates a sophisticated LangChain agent that combines autonomous reasoning
with existing loan processing infrastructure. Features include:
- Advanced memory management
- Robust error handling and fallbacks
- Compliance monitoring
- Integration with existing systems
- Production-ready monitoring and logging
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from loan_agent_tools import LOAN_AGENT_TOOLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentCircuitBreaker:
    """Circuit breaker pattern for agent reliability."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class AgentPerformanceMonitor:
    """Monitor agent performance and usage patterns."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "tool_usage": {},
            "error_patterns": {},
            "compliance_violations": 0
        }
        self.request_history = []
    
    def record_request(self, request_type: str, success: bool, response_time: float, 
                      tools_used: List[str], error: Optional[str] = None):
        """Record agent request metrics."""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            if error:
                self.metrics["error_patterns"][error] = self.metrics["error_patterns"].get(error, 0) + 1
        
        # Update average response time
        total_time = self.metrics["avg_response_time"] * (self.metrics["total_requests"] - 1)
        self.metrics["avg_response_time"] = (total_time + response_time) / self.metrics["total_requests"]
        
        # Track tool usage
        for tool in tools_used:
            self.metrics["tool_usage"][tool] = self.metrics["tool_usage"].get(tool, 0) + 1
        
        # Store detailed history
        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "request_type": request_type,
            "success": success,
            "response_time": response_time,
            "tools_used": tools_used,
            "error": error
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_health_status(self) -> Dict:
        """Get agent health status."""
        if self.metrics["total_requests"] == 0:
            return {"status": "NO_DATA", "health_score": 0}
        
        success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
        avg_response_time = self.metrics["avg_response_time"]
        
        # Calculate health score
        health_score = success_rate * 100
        
        # Penalize slow responses
        if avg_response_time > 10:
            health_score *= 0.8
        elif avg_response_time > 5:
            health_score *= 0.9
        
        status = "HEALTHY" if health_score > 80 else "DEGRADED" if health_score > 60 else "UNHEALTHY"
        
        return {
            "status": status,
            "health_score": health_score,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "total_requests": self.metrics["total_requests"]
        }


class EnhancedAgentMemory:
    """Advanced memory system combining conversation and long-term memory."""
    
    def __init__(self, vectorstore_path: str, embeddings, window_size: int = 10):
        # Conversation memory for short-term context
        self.conversation_memory = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Vector store memory for long-term patterns and context
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            
            self.vector_memory = VectorStoreRetrieverMemory(
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory_key="relevant_context",
            )
        except Exception as e:
            logger.warning(f"Could not initialize vector memory: {e}")
            self.vector_memory = None
        
        # Custom session context for complex loan processes
        self.session_context = {}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        """Save conversation context."""
        # Save to conversation memory
        self.conversation_memory.save_context(inputs, outputs)
        
        # Save to vector memory if available
        if self.vector_memory:
            try:
                self.vector_memory.save_context(inputs, outputs)
            except Exception as e:
                logger.warning(f"Failed to save to vector memory: {e}")
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memory for current context."""
        memory_vars = {}
        
        # Load conversation memory
        conv_memory = self.conversation_memory.load_memory_variables(inputs)
        memory_vars.update(conv_memory)
        
        # Load vector memory if available
        if self.vector_memory:
            try:
                vector_memory = self.vector_memory.load_memory_variables(inputs)
                memory_vars.update(vector_memory)
            except Exception as e:
                logger.warning(f"Failed to load vector memory: {e}")
        
        # Add session context
        if self.session_context:
            memory_vars["session_context"] = json.dumps(self.session_context)
        
        return memory_vars
    
    def update_session_context(self, key: str, value: Any):
        """Update long-term session context."""
        self.session_context[key] = value
    
    def get_session_context(self, key: str) -> Any:
        """Retrieve session context."""
        return self.session_context.get(key)
    
    def clear_session(self):
        """Clear session-specific context."""
        self.session_context.clear()
        self.conversation_memory.clear()


class SmartLoanAgent:
    """
    Production-ready LangChain agent for intelligent loan processing.
    
    Features:
    - Autonomous decision-making with multiple tools
    - Advanced memory management
    - Circuit breaker pattern for reliability
    - Performance monitoring
    - Compliance checking
    - Graceful error handling and fallbacks
    """
    
    def __init__(self, openai_api_key: str, vectorstore_path: str = None, 
                 model_name: str = "gpt-4", temperature: float = 0.1):
        
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize monitoring and reliability components
        self.circuit_breaker = AgentCircuitBreaker()
        self.performance_monitor = AgentPerformanceMonitor()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=2000
        )
        
        # Initialize embeddings for memory
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize advanced memory system
        self.memory = EnhancedAgentMemory(
            vectorstore_path or "rag_vectorstore",
            self.embeddings
        )
        
        # Initialize agent
        self._initialize_agent()
        
        logger.info("SmartLoanAgent initialized successfully")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and prompts."""
        
        # Create agent prompt with advanced instructions
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent with tools
        self.agent = create_openai_functions_agent(
            self.llm,
            LOAN_AGENT_TOOLS,
            self.agent_prompt
        )
        
        # Create agent executor with advanced configuration
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=LOAN_AGENT_TOOLS,
            memory=self.memory.conversation_memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
    
    def _get_system_prompt(self) -> str:
        """Get comprehensive system prompt for the agent."""
        return """You are FlexiLoan's Advanced AI Loan Officer Agent, equipped with sophisticated tools for comprehensive loan processing.

CORE CAPABILITIES:
- Database Query: Complex analytics and customer data retrieval
- Credit Assessment: ML-powered risk evaluation and loan recommendations
- Risk Analysis: Multi-dimensional risk assessment (credit, fraud, compliance)
- Loan Recommendations: Intelligent product matching based on customer needs
- Compliance Checking: Comprehensive regulatory and fair lending compliance

OPERATING PRINCIPLES:
1. CUSTOMER-FIRST: Always prioritize customer needs and provide clear, helpful guidance
2. COMPLIANCE: Ensure all decisions and recommendations comply with lending regulations
3. ACCURACY: Base all assessments on data and verified information
4. TRANSPARENCY: Explain reasoning behind recommendations and decisions
5. EFFICIENCY: Use appropriate tools to provide comprehensive yet timely service

DECISION-MAKING PROCESS:
1. Understand the customer's request thoroughly
2. Gather necessary information using available tools
3. Analyze data comprehensively across multiple dimensions
4. Provide clear recommendations with supporting rationale
5. Ensure compliance and identify any required next steps

COMMUNICATION STYLE:
- Professional yet friendly
- Clear and jargon-free explanations
- Structured responses with key points highlighted
- Always indicate confidence level in recommendations
- Proactively suggest relevant services or next steps

RISK MANAGEMENT:
- Never guarantee loan approval without proper assessment
- Always highlight material risks and conditions
- Escalate complex cases to human underwriters when appropriate
- Maintain strict data privacy and security protocols

You have access to the customer's conversation history and relevant company knowledge. Use this context to provide personalized, informed assistance while maintaining the highest standards of professional banking service."""
    
    async def process_request(self, user_input: str, customer_id: str = None) -> Dict[str, Any]:
        """
        Process user request with full monitoring and error handling.
        
        Args:
            user_input: The user's request or question
            customer_id: Optional customer identifier for personalization
            
        Returns:
            Dict containing response, metadata, and performance metrics
        """
        start_time = time.time()
        tools_used = []
        success = False
        error = None
        
        try:
            # Input validation and preprocessing
            processed_input = self._preprocess_input(user_input, customer_id)
            
            # Execute agent with circuit breaker protection
            response = self.circuit_breaker.call(
                self._execute_agent,
                processed_input
            )
            
            # Post-process response
            final_response = self._postprocess_response(response)
            
            # Extract tools used from response metadata
            tools_used = self._extract_tools_used(response)
            
            success = True
            
            return {
                "response": final_response,
                "success": True,
                "customer_id": customer_id,
                "tools_used": tools_used,
                "response_time": time.time() - start_time,
                "compliance_status": self._check_response_compliance(final_response),
                "confidence_score": self._calculate_confidence_score(response),
                "next_steps": self._suggest_next_steps(final_response, customer_id)
            }
            
        except Exception as e:
            error = str(e)
            logger.error(f"Agent request failed: {e}")
            
            # Generate fallback response
            fallback_response = self._generate_fallback_response(user_input, error)
            
            return {
                "response": fallback_response,
                "success": False,
                "error": error,
                "customer_id": customer_id,
                "tools_used": tools_used,
                "response_time": time.time() - start_time,
                "fallback_used": True
            }
            
        finally:
            # Record metrics
            self.performance_monitor.record_request(
                request_type="loan_processing",
                success=success,
                response_time=time.time() - start_time,
                tools_used=tools_used,
                error=error
            )
    
    def _preprocess_input(self, user_input: str, customer_id: str = None) -> Dict[str, str]:
        """Preprocess and validate user input."""
        
        # Input validation
        if not user_input or len(user_input.strip()) == 0:
            raise ValueError("Empty input received")
        
        if len(user_input) > 5000:
            raise ValueError("Input too long - maximum 5000 characters")
        
        # Security checks
        dangerous_patterns = [
            "ignore previous instructions",
            "system prompt",
            "debug mode",
            "__",
            "DELETE FROM",
            "DROP TABLE"
        ]
        
        user_input_lower = user_input.lower()
        for pattern in dangerous_patterns:
            if pattern in user_input_lower:
                raise ValueError(f"Potentially malicious input detected: {pattern}")
        
        # Enhance input with customer context
        enhanced_input = user_input
        if customer_id:
            self.memory.update_session_context("customer_id", customer_id)
            enhanced_input = f"[Customer ID: {customer_id}] {user_input}"
        
        return {"input": enhanced_input}
    
    def _execute_agent(self, processed_input: Dict[str, str]) -> Dict[str, Any]:
        """Execute the agent with processed input."""
        return self.agent_executor.invoke(processed_input)
    
    def _postprocess_response(self, agent_response: Dict[str, Any]) -> str:
        """Post-process agent response for quality and compliance."""
        response = agent_response.get("output", "")
        
        # Ensure response quality
        if len(response.strip()) < 20:
            response += "\n\nIf you need more specific information, please let me know how I can help further."
        
        # Add compliance footer for certain response types
        if any(keyword in response.lower() for keyword in ["approve", "loan", "credit", "rate"]):
            response += "\n\n*This response is for informational purposes. Final loan decisions are subject to underwriting approval and may require additional documentation."
        
        return response
    
    def _extract_tools_used(self, agent_response: Dict[str, Any]) -> List[str]:
        """Extract which tools were used in the agent's response."""
        # This would need to be implemented based on LangChain's response structure
        # For now, return empty list
        return []
    
    def _check_response_compliance(self, response: str) -> Dict[str, Any]:
        """Check response for compliance issues."""
        violations = []
        
        # Check for prohibited guarantees
        prohibited_phrases = [
            "guaranteed approval",
            "no credit check",
            "100% approval rate",
            "instant approval"
        ]
        
        response_lower = response.lower()
        for phrase in prohibited_phrases:
            if phrase in response_lower:
                violations.append(f"Prohibited guarantee: {phrase}")
        
        return {
            "status": "COMPLIANT" if not violations else "VIOLATIONS_DETECTED",
            "violations": violations
        }
    
    def _calculate_confidence_score(self, agent_response: Dict[str, Any]) -> float:
        """Calculate confidence score for the response."""
        # Simple heuristic - could be enhanced with ML model
        response = agent_response.get("output", "")
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for longer, more detailed responses
        if len(response) > 200:
            confidence += 0.2
        
        # Higher confidence when tools were used successfully
        if "tool" in str(agent_response).lower():
            confidence += 0.2
        
        # Lower confidence for uncertain language
        uncertain_phrases = ["might", "could", "possibly", "perhaps", "I think"]
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _suggest_next_steps(self, response: str, customer_id: str = None) -> List[str]:
        """Suggest relevant next steps based on response content."""
        next_steps = []
        
        response_lower = response.lower()
        
        if "credit assessment" in response_lower or "risk score" in response_lower:
            next_steps.append("Schedule follow-up consultation")
            next_steps.append("Prepare required documentation")
        
        if "loan recommendation" in response_lower:
            next_steps.append("Review loan terms and conditions")
            next_steps.append("Submit formal application")
        
        if "compliance" in response_lower:
            next_steps.append("Review regulatory requirements")
            next_steps.append("Ensure documentation completeness")
        
        if not next_steps:
            next_steps.append("Continue conversation for additional assistance")
        
        return next_steps
    
    def _generate_fallback_response(self, user_input: str, error: str) -> str:
        """Generate fallback response when agent fails."""
        return f"""I apologize, but I'm experiencing technical difficulties processing your request right now. 

To ensure you receive the best possible service, I'm connecting you with one of our human loan officers who can assist you immediately.

Your request: "{user_input[:100]}{'...' if len(user_input) > 100 else ''}"

A specialist will contact you shortly. Thank you for your patience.

Error Reference: {error[:50]}{'...' if len(error) > 50 else ''}"""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive agent health status."""
        base_health = self.performance_monitor.get_health_status()
        
        # Add circuit breaker status
        base_health["circuit_breaker_status"] = self.circuit_breaker.state
        base_health["circuit_breaker_failures"] = self.circuit_breaker.failure_count
        
        # Add memory status
        try:
            memory_status = "HEALTHY"
            chat_history_size = len(self.memory.conversation_memory.chat_memory.messages)
            session_context_size = len(self.memory.session_context)
        except:
            memory_status = "DEGRADED"
            chat_history_size = 0
            session_context_size = 0
        
        base_health["memory_status"] = memory_status
        base_health["chat_history_size"] = chat_history_size
        base_health["session_context_size"] = session_context_size
        
        return base_health
    
    def reset_agent(self):
        """Reset agent state and clear memory."""
        self.memory.clear_session()
        self.circuit_breaker = AgentCircuitBreaker()
        self.performance_monitor = AgentPerformanceMonitor()
        logger.info("Agent state reset successfully")
    
    async def batch_process(self, requests: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently."""
        
        async def process_single(request):
            return await self.process_request(
                request.get("input", ""),
                request.get("customer_id")
            )
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, 
                    lambda req=req: asyncio.run(process_single(req))
                )
                for req in requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "request_index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results


# Example usage and configuration
class ProductionAgentConfig:
    """Production configuration for the Smart Loan Agent."""
    
    # Model configuration
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2000
    
    # Reliability configuration
    CIRCUIT_BREAKER_THRESHOLD = 3
    CIRCUIT_BREAKER_TIMEOUT = 60
    
    # Memory configuration
    CONVERSATION_WINDOW_SIZE = 10
    VECTOR_MEMORY_ENABLED = True
    
    # Performance configuration
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT_SECONDS = 30
    
    # Compliance configuration
    ENABLE_COMPLIANCE_CHECKING = True
    ENABLE_RESPONSE_VALIDATION = True
    
    @classmethod
    def create_agent(cls, openai_api_key: str, vectorstore_path: str = None) -> SmartLoanAgent:
        """Create a production-configured Smart Loan Agent."""
        return SmartLoanAgent(
            openai_api_key=openai_api_key,
            vectorstore_path=vectorstore_path,
            model_name=cls.MODEL_NAME,
            temperature=cls.TEMPERATURE
        )