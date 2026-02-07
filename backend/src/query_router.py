"""
LangGraph-based query router to classify whether a query needs document retrieval.
Uses an LLM to intelligently determine query intent.
"""

import json
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import requests
from eilco_prompts import get_query_router_prompt


class QueryRouterState(TypedDict):
    """State for the query router workflow"""
    query: str
    chat_history: str
    classification: Literal["conversational", "knowledge_seeking", "ambiguous"]
    reasoning: str
    needs_retrieval: bool


class QueryRouter:
    """
    LangGraph-based router that uses LLM to classify queries and determine if retrieval is needed.
    """
    
    def __init__(self, llm_api_url: str, llm_name: str):
        self.llm_api_url = llm_api_url
        self.llm_name = llm_name
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        graph = StateGraph(QueryRouterState)
        
        # Add nodes
        graph.add_node("classify", self._classify_query)
        graph.add_node("route", self._route_query)
        
        # Add edges
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "route")
        graph.add_edge("route", END)
        
        return graph.compile()
    
    def _call_llm(self, messages: list, max_tokens: int = 256) -> str:
        """Call the LLM API and return the response content"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.llm_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        
        try:
            response = requests.post(self.llm_api_url, headers=headers, json=payload, timeout=30)
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "unknown"
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return "unknown"
    
    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response, handling various formats.
        Extracts JSON even if wrapped in markdown code blocks.
        """
        import re
        
        # Try to parse as-is first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: return default with reasoning about parse failure
        return {
            "classification": "ambiguous",
            "reasoning": "Failed to parse LLM response"
        }
    
    def _classify_query(self, state: QueryRouterState) -> QueryRouterState:
        """
        Node: Classify the query intent using LLM
        Returns: conversational, knowledge_seeking, or ambiguous
        """
        query = state["query"]
        chat_history = state.get("chat_history", "")
        
        # Use personalized EILCO prompt
        classification_prompt = get_query_router_prompt(query, chat_history)
        
        response = self._call_llm([
            {"role": "system", "content": "Tu es un expert en classification d'intentions d'utilisateurs. Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans explications supplémentaires."},
            {"role": "user", "content": classification_prompt}
        ])
        
        print(f"Router LLM response: {response}")
        
        # Parse the response with robust error handling
        result = self._parse_json_response(response)
        
        classification = result.get("classification", "ambiguous")
        reasoning = result.get("reasoning", "")
        
        # Validate classification is one of the allowed values
        if classification not in ["conversational", "knowledge_seeking", "ambiguous"]:
            classification = "ambiguous"
        
        state["classification"] = classification
        state["reasoning"] = reasoning
        
        return state
    
    def _route_query(self, state: QueryRouterState) -> QueryRouterState:
        """
        Node: Determine if retrieval is needed based on classification.
        Always checks for questions, even if LLM classified as conversational.
        """
        classification = state["classification"]
        query = state["query"]
        query_lower = query.lower().strip()
        
        # French question words (mots interrogatifs)
        french_question_words = [
            "quoi", "quand", "où", "pourquoi", "comment", "quels", "quelle", "quel",
            "qui", "que", "qu'", "lequel", "laquelle", "lesquels", "lesquelles",
            "à quoi", "à qui", "de quoi", "de qui", "combien"
        ]
        
        # English question words
        english_question_words = [
            "what", "when", "where", "why", "how", "which", "who", "whom", "whose"
        ]
        
        # Check if there's a question mark
        has_question_mark = "?" in query
        
        # Check for question words (flexible matching)
        has_question_word = False
        for word in french_question_words + english_question_words:
            # Match at start, after space, or with fuzzy matching for typos
            if (query_lower.startswith(word) or 
                f" {word}" in query_lower or 
                f" {word} " in f" {query_lower} "):
                has_question_word = True
                print(f"DEBUG: Found question word '{word}' in query")
                break
        
        # Request indicators (demandes d'information)
        request_indicators = [
            "explique", "explain", "donne", "give", "dis", "tell", "montre", "show",
            "listes", "list", "énumère", "enumerate", "décris", "describe", "détailles", "detail",
            "donne-moi", "dites-moi", "telle-moi", "informe-moi"
        ]
        has_request = any(indicator in query_lower for indicator in request_indicators)
        
        print(f"DEBUG: Query='{query_lower}'")
        print(f"DEBUG: has_question_mark={has_question_mark}, has_question_word={has_question_word}, has_request={has_request}")
        
        # Override classification: if there's a question OR request, it's knowledge_seeking
        # This ALWAYS overrides conversational classification
        if has_question_mark or has_question_word or has_request:
            classification = "knowledge_seeking"
            state["reasoning"] = f"Override: question détectée (? ={has_question_mark}, word={has_question_word}, request={has_request})"
        
        # Set needs_retrieval based on final classification
        if classification == "conversational":
            needs_retrieval = False
        elif classification == "knowledge_seeking":
            needs_retrieval = True
        else:  # ambiguous - default to safe behavior (no retrieval)
            needs_retrieval = False
        
        state["needs_retrieval"] = needs_retrieval
        
        return state
    
    def route(self, query: str, chat_history: str = "") -> dict:
        """
        Main entry point: Route a query and determine if retrieval is needed.
        
        Args:
            query: The user query
            chat_history: Optional chat history context
            
        Returns:
            dict with keys:
            - needs_retrieval: bool
            - classification: str (conversational, knowledge_seeking, ambiguous)
            - reasoning: str
        """
        initial_state = QueryRouterState(
            query=query,
            chat_history=chat_history,
            classification="ambiguous",
            reasoning="",
            needs_retrieval=False
        )
        
        result = self.graph.invoke(initial_state)
        
        return {
            "needs_retrieval": result["needs_retrieval"],
            "classification": result["classification"],
            "reasoning": result["reasoning"]
        }
