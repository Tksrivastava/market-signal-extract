import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from core.prompts import LLMPrompts
from core.logging import LoggerFactory
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

# Initializing logger
logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)

load_dotenv(dotenv_path="./.env")
logger.info("Environment variables loaded from .env")

# Graph State
class GraphState(TypedDict):
    full_article: str
    facts: str
    classified_signals: str
    price_bias: str
    paraphrase: str

# Market Singal
class MarketSignals:
    def __init__(self, local_inference:bool=True):
        self.local_inference = local_inference
        if self.local_inference:
            self.model_name = os.getenv("LOCAL_LLM_MODEL")
            logger.info(f"Initializing MarketSignals with model: {self.model_name}")
            self.llm = ChatOllama(model=self.model_name, temperature=0)
        else:
            self.model_name = os.getenv("GROQ_LLM_MODEL")
            logger.info(f"Initializing MarketSignals with model: {self.model_name}")
            self.llm = ChatGroq(model=self.model_name, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
        logger.info("LLM loaded successfully")
        self.workflow = StateGraph(GraphState)
        self._build_workflow()
        logger.info("LangGraph worflow created")
        
    def extract_facts(self, state: GraphState):
        prompt = f"""{LLMPrompts.extract_facts}

        Article:
        {state["full_article"]}
        """
        response = self.llm.invoke(prompt)
        logger.info("Node: extract_facts - completed")
        return {"facts": response.content}
    
    def classify_signals(self, state: GraphState):
        prompt = f"""{LLMPrompts.classify_signals}
        
        Facts:
        {state["facts"]}
        """
        response = self.llm.invoke(prompt)
        logger.info("Node: classify_signals - completed")
        return {"classified_signals": response.content}
    
    def evaluate_bias(self, state: GraphState):
        prompt = f"""{LLMPrompts.evaluate_bias}
            
            Classified Signals:
            {state["classified_signals"]}
            """
        response = self.llm.invoke(prompt)
        logger.info("Node: evaluate_bias - completed")
        return {"price_bias": response.content}
    
    def paraphrase_signals(self, state: GraphState):
        prompt = f"""{LLMPrompts.paraphrase}
            
            Classified Signals:
            {state["classified_signals"]}
            Evaluated Bias:
            {state["price_bias"]}
            """
        response = self.llm.invoke(prompt)
        logger.info("Node: paraphrase - completed")
        return {"paraphrase": response.content}
    
    def _build_workflow(self):
        self.workflow.add_node("extract_facts", self.extract_facts)
        self.workflow.add_node("classify_signals", self.classify_signals)
        self.workflow.add_node("evaluate_bias", self.evaluate_bias)
        self.workflow.add_node("paraphrase_content", self.paraphrase_signals)

        self.workflow.set_entry_point("extract_facts")

        self.workflow.add_edge("extract_facts", "classify_signals")
        self.workflow.add_edge("classify_signals", "evaluate_bias")
        self.workflow.add_edge("evaluate_bias", "paraphrase_content")
        self.workflow.add_edge("paraphrase_content", END)

        self.app = self.workflow.compile()
        logger.info("Workflow compiled successfully")

    def get_signals(self, full_article: str):
        logger.info("Graph execution started")
        result = self.app.invoke({"full_article": full_article})
        logger.info("Graph execution completed")
        return result