import os
import json
import re
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from huggingface_hub import InferenceClient
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_API_KEY, HF_MODEL_NAME

# Initialize the LLM client
llm_client = InferenceClient(model=HF_MODEL_NAME, token=HF_API_KEY)

def query_model(prompt):
    """Query the LLM model using the prompt."""
    response = llm_client.chat_completion(messages=[{"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]

# -------- Query Analysis --------

class QueryType(str, Enum):
    INDEX_RELATED = "index_related"
    CURRENT_EVENTS = "current_events"
    OTHER = "other"

class QueryAnalysisResult(BaseModel):
    query_type: QueryType = Field(description="Categorized query type")
    reasoning: str = Field(description="Reasoning for categorization")
    related_keywords: List[str] = Field(description="Extracted keywords")

def extract_json_from_text(text):
    """Extract JSON from text that might have extra formatting."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

def analyze_query(query):
    """Analyze a query to determine its type and extract keywords."""
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
You are a query analysis system. Analyze the following query and determine its type.

Query: {query}

Categorize this query into one of the following types:
1. "index_related": The query seeks factual info that might be in a news database.
2. "current_events": The query is about recent events or breaking news.
3. "other": General knowledge or opinion-based query.

Also, extract important keywords for retrieval.

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT:
{{
    "query_type": "index_related" OR "current_events" OR "other",
    "reasoning": "Your reasoning",
    "related_keywords": ["keyword1", "keyword2", ...]
}}

DO NOT INCLUDE ANY OTHER TEXT.
"""
    )
    prompt = prompt_template.format(query=query)
    try:
        response = query_model(prompt)
        result = extract_json_from_text(response)
        if not result:
            print(f"Failed to extract JSON from response: {response[:100]}...")
            raise ValueError("Invalid JSON format")
        return QueryAnalysisResult(
            query_type=result.get("query_type", QueryType.OTHER),
            reasoning=result.get("reasoning", "No reasoning provided"),
            related_keywords=result.get("related_keywords", query.split())
        )
    except Exception as e:
        print(f"Error in query analysis: {e}")
        return QueryAnalysisResult(
            query_type=QueryType.OTHER,
            reasoning=f"Error: {str(e)}",
            related_keywords=query.split()
        )

# -------- Retrieval --------

def get_vector_store_retriever(vector_store, search_type="similarity", k=4):
    """Return a retriever from the vector store."""
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

def perform_web_search(query, num_results=5):
    """Perform a web search using DuckDuckGo."""
    search_tool = DuckDuckGoSearchRun()
    try:
        return search_tool.run(query)
    except Exception as e:
        print(f"Error performing web search: {e}")
        return f"Failed to retrieve web search results for: {query}"

def query_llm_directly(query):
    """Query the LLM directly without external context."""
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
Please answer the following question to the best of your knowledge:

Question: {query}

Answer:
"""
    )
    prompt = prompt_template.format(query=query)
    return query_model(prompt)

def retrieve_information(query, analysis_result, vector_store):
    """Retrieve information based on query analysis."""
    retrieval_results = {
        "query": query,
        "query_type": analysis_result.query_type,
        "sources": [],
        "content": ""
    }
    if analysis_result.query_type == QueryType.INDEX_RELATED:
        retriever = get_vector_store_retriever(vector_store)
        docs = retriever.get_relevant_documents(query)
        retrieval_results["sources"] = [
            {"source_type": "vector_store", "content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        retrieval_results["content"] = "\n\n".join([doc.page_content for doc in docs])
    elif analysis_result.query_type == QueryType.CURRENT_EVENTS:
        web_results = perform_web_search(query)
        retrieval_results["sources"] = [{"source_type": "web_search", "content": web_results}]
        retrieval_results["content"] = web_results
    else:
        direct_response = query_llm_directly(query)
        retrieval_results["sources"] = [{"source_type": "llm", "content": direct_response}]
        retrieval_results["content"] = direct_response
    return retrieval_results

# -------- Self-Reflection --------

class ReflectionResult(BaseModel):
    is_relevant: bool = Field(description="Is retrieved info relevant")
    has_hallucinations: bool = Field(description="Does the answer include hallucinations")
    answers_question: bool = Field(description="Does the answer address the question")
    reasoning: str = Field(description="Detailed reasoning")

def evaluate_relevance(query, retrieved_content):
    prompt_template = PromptTemplate(
        input_variables=["query", "content"],
        template="""
You are an information relevance evaluator. Determine if the retrieved content is relevant to the query.

Query: {query}

Retrieved Content:
{content}

Is this content relevant?

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT:
{{
    "is_relevant": true or false,
    "reasoning": "Your reasoning"
}}

DO NOT INCLUDE ANY OTHER TEXT.
"""
    )
    prompt = prompt_template.format(query=query, content=retrieved_content)
    try:
        response = query_model(prompt)
        result = extract_json_from_text(response)
        if result and "is_relevant" in result and "reasoning" in result:
            return result["is_relevant"], result["reasoning"]
        else:
            print("Malformed relevance response:", response[:100])
            return False, "Malformed JSON in relevance evaluation"
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
        return False, f"Error: {str(e)}"

def check_for_hallucinations(generated_answer, retrieved_content):
    prompt_template = PromptTemplate(
        input_variables=["answer", "content"],
        template="""
You are a hallucination detector. Check if the generated answer contains info not supported by the retrieved content.

Generated Answer:
{answer}

Retrieved Content:
{content}

Does the answer contain hallucinations?

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT:
{{
    "has_hallucinations": true or false,
    "reasoning": "Your reasoning"
}}

DO NOT INCLUDE ANY OTHER TEXT.
"""
    )
    prompt = prompt_template.format(answer=generated_answer, content=retrieved_content)
    try:
        response = query_model(prompt)
        result = extract_json_from_text(response)
        if result and "has_hallucinations" in result and "reasoning" in result:
            return result["has_hallucinations"], result["reasoning"]
        else:
            print("Malformed hallucination response:", response[:100])
            return True, "Malformed JSON in hallucination check"
    except Exception as e:
        print(f"Error in hallucination check: {e}")
        return True, f"Error: {str(e)}"

def verify_answers_question(query, answer):
    prompt_template = PromptTemplate(
        input_variables=["query", "answer"],
        template="""
You are an answer validator. Check if the generated answer addresses the question.

Question:
{query}

Generated Answer:
{answer}

Does the answer address the question?

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT:
{{
    "answers_question": true or false,
    "reasoning": "Your reasoning"
}}

DO NOT INCLUDE ANY OTHER TEXT.
"""
    )
    prompt = prompt_template.format(query=query, answer=answer)
    try:
        response = query_model(prompt)
        result = extract_json_from_text(response)
        if result and "answers_question" in result and "reasoning" in result:
            return result["answers_question"], result["reasoning"]
        else:
            print("Malformed answer verification response:", response[:100])
            return False, "Malformed JSON in answer verification"
    except Exception as e:
        print(f"Error in answer verification: {e}")
        return False, f"Error: {str(e)}"

def perform_self_reflection(query, retrieved_content, generated_answer):
    is_relevant, relevance_reasoning = evaluate_relevance(query, retrieved_content)
    has_hallucinations, hallucination_reasoning = check_for_hallucinations(generated_answer, retrieved_content)
    answers_question, question_reasoning = verify_answers_question(query, generated_answer)
    reflection_result = ReflectionResult(
        is_relevant=is_relevant,
        has_hallucinations=has_hallucinations,
        answers_question=answers_question,
        reasoning=f"Relevance: {relevance_reasoning}\nHallucinations: {hallucination_reasoning}\nAnswers Question: {question_reasoning}"
    )
    return reflection_result

# -------- Response Generation --------

def generate_response_with_rag(query, retrieved_content):
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""
You are a helpful assistant. Answer the following question based on the context.
If the context lacks sufficient info, state that.

Context:
{context}

Question:
{query}

Answer:
"""
    )
    prompt = prompt_template.format(query=query, context=retrieved_content)
    return query_model(prompt)

def generate_response_without_context(query):
    prompt = f"Please answer the following question to the best of your knowledge:\n\n{query}\n\nAnswer:"
    return query_model(prompt)

def integrate_self_reflection(query, retrieved_content, reflection_result):
    if not reflection_result.is_relevant:
        return generate_response_without_context(query)
    if reflection_result.has_hallucinations:
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
You are a careful assistant. Answer the question using ONLY the provided context.
If the context is insufficient, state the limitation.

Context:
{context}

Question:
{query}

Answer:
"""
        )
        prompt = prompt_template.format(query=query, context=retrieved_content)
        return query_model(prompt)
    if not reflection_result.answers_question:
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
You are a focused assistant. The previous answer did not fully address the question.
Please provide a direct answer using the context.

Context:
{context}

Question:
{query}

Answer:
"""
        )
        prompt = prompt_template.format(query=query, context=retrieved_content)
        return query_model(prompt)
    return generate_response_with_rag(query, retrieved_content)

# -------- State Graph Workflow --------

class GraphState(BaseModel):
    query: str
    query_analysis: Optional[Dict] = None
    retrieval_results: Optional[Dict] = None
    initial_response: Optional[str] = None
    reflection_result: Optional[Dict] = None
    final_response: Optional[str] = None

def query_analysis_node(state):
    query = state.dict().get("query")
    analysis_result = analyze_query(query)
    return {"query_analysis": analysis_result.dict()}

def retrieval_node(state):
    query = state["query"]
    analysis_result = QueryAnalysisResult(**state["query_analysis"])
    retrieval_results = retrieve_information(query, analysis_result, state["vector_store"])
    return {"retrieval_results": retrieval_results}

def initial_response_node(state):
    query = state["query"]
    retrieved_content = state["retrieval_results"]["content"]
    initial_response = generate_response_with_rag(query, retrieved_content)
    return {"initial_response": initial_response}

def reflection_node(state):
    query = state["query"]
    retrieved_content = state["retrieval_results"]["content"]
    initial_response = state["initial_response"]
    reflection_result = perform_self_reflection(query, retrieved_content, initial_response)
    return {"reflection_result": reflection_result.dict()}

def final_response_node(state):
    query = state["query"]
    retrieved_content = state["retrieval_results"]["content"]
    reflection_result = ReflectionResult(**state["reflection_result"])
    final_response = integrate_self_reflection(query, retrieved_content, reflection_result)
    return {"final_response": final_response}

def direct_llm_node(state):
    return {"final_response": generate_response_without_context(state.query)}

def should_use_rag(state):
    if "query_analysis" in state:
     analysis_result = QueryAnalysisResult(**state["query_analysis"])
    else:
     analysis_result = QueryAnalysisResult(
        query_type="other",
        reasoning="No analysis available",
        related_keywords=[]
    )

    if analysis_result.query_type == QueryType.OTHER:
        return "direct_llm"
    return "use_rag"

def needs_improvement(state):
    reflection_result = ReflectionResult(**state["reflection_result"])
    if reflection_result.has_hallucinations or not reflection_result.answers_question:
        return "improve"
    if not reflection_result.is_relevant:
        return "use_direct"
    return "use_initial"

def build_rag_graph(vector_store):
    workflow = StateGraph(GraphState)
    unique_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    workflow.add_node(f"{unique_prefix}_query_analysis", query_analysis_node)
    workflow.add_node(f"{unique_prefix}_retrieval", retrieval_node)
    workflow.add_node(f"{unique_prefix}_initial_response", initial_response_node)
    workflow.add_node(f"{unique_prefix}_reflection", reflection_node)
    workflow.add_node(f"{unique_prefix}_final_response", final_response_node)
    workflow.add_node(f"{unique_prefix}_direct_llm_response", direct_llm_node)

    workflow.add_conditional_edges(
        f"{unique_prefix}_query_analysis",
        should_use_rag,
        {
            "use_rag": f"{unique_prefix}_retrieval",
            "direct_llm": f"{unique_prefix}_direct_llm_response"
        }
    )
    workflow.add_edge(f"{unique_prefix}_retrieval", f"{unique_prefix}_initial_response")
    workflow.add_edge(f"{unique_prefix}_initial_response", f"{unique_prefix}_reflection")
    workflow.add_conditional_edges(
        f"{unique_prefix}_reflection",
        needs_improvement,
        {
            "improve": f"{unique_prefix}_final_response",
            "use_direct": f"{unique_prefix}_direct_llm_response",
            "use_initial": END
        }
    )
    workflow.add_edge(f"{unique_prefix}_final_response", END)
    workflow.add_edge(f"{unique_prefix}_direct_llm_response", END)
    workflow.set_entry_point(f"{unique_prefix}_query_analysis")
    return workflow.compile()

def run_rag_system(query, vector_store):
    """Run the complete RAG system with the given query."""
    graph = build_rag_graph(vector_store)
    state = {"query": query, "vector_store": vector_store}
    result = graph.invoke(state)
    return result.get("final_response", "No response generated")
