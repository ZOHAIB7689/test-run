import streamlit as st
import os
import re
from typing import Optional, Tuple
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from litellm import completion
import logging
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IslamicKnowledgeManager:
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.tools = self._create_tools()

    def _create_tools(self):
        """Define tools for the agent"""
        quran_tool = Tool(
            name="Quranic Research",
            func=self._research_quran,
            description="Finds relevant Quranic verses with explanations"
        )
        
        hadith_tool = Tool(
            name="Hadith Research",
            func=self._research_hadith,
            description="Identifies authentic hadith with proper citations"
        )
        
        return [quran_tool, hadith_tool]

    def _research_quran(self, query: str) -> str:
        """Research Quranic verses"""
        prompt = f"""
        Find Quranic verses relevant to: {query}.
        Provide Surah:Verse references and explanations.
        Use Markdown formatting with clear sections.
        """
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _research_hadith(self, query: str) -> str:
        """Research Hadith"""
        prompt = f"""
        Find authentic hadith relevant to: {query}.
        Provide proper citations.
        Use Markdown formatting with clear sections.
        """
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def execute_research(self, query: str, language: str) -> Tuple[str, Optional[str]]:
        """Execute research using LangChain agent"""
        try:
            # Combine results from both tools
            quran_result = self._research_quran(query)
            hadith_result = self._research_hadith(query)
            
            # Format results
            formatted_result = self._format_results(
                f"## Quranic References\n{quran_result}\n\n## Hadith References\n{hadith_result}",
                language
            )
            
            # Generate related questions
            related = self._generate_related_questions(query)
            
            return (formatted_result, related)
        except Exception as e:
            logger.error(f"Research Error: {str(e)}")
            return ("Error processing request", None)

    def _generate_related_questions(self, query: str) -> Optional[str]:
        """Generate related questions"""
        prompt = f"Generate 3 related Islamic questions for: {query}"
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            max_tokens=500
        )
        return response.choices[0].message.content

    def _format_results(self, results: str, language: str) -> str:
        """Format results into structured response"""
        return re.sub(r'(```arabic)(.*?)(```)', 
                    r'<div dir="rtl" class="arabic-text">\2</div>', 
                    results, flags=re.DOTALL)

# Session Management
def initialize_session():
    defaults = {
        'search_history': [],
        'language': "English",
        'current_query': "",
        'selected_model': "gemini/gemini-2.0-flash",
        'search_in_progress': False,
        'favorites': []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# UI Components
def display_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e5631, #3a7e4f); 
                border-radius: 15px; margin-bottom: 25px;">
        <h1 style="color: white; margin-bottom: 10px;">🕌 Quranic Insights Pro</h1>
        <p style="color: #e0e0e0; font-size: 16px;">Islamic knowledge using Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_search_interface():
    col1, col2 = st.columns([4, 1])
    with col1:
        query_placeholder = "e.g., Rules of fasting during travel..." if st.session_state.language == "English" else "مثال: سفر میں روزے کے احکام..."
        query = st.text_area(
            "Ask an Islamic question:" if st.session_state.language == "English" else "اسلامی سوال پوچھیں:",
            value=st.session_state.current_query,
            placeholder=query_placeholder,
            height=100
        )
        
        if st.button("🔍 Search", type="primary") and query.strip():
            st.session_state.current_query = query
            st.session_state.search_in_progress = True
            st.rerun()

def display_result(response: str, related: str):
    if st.session_state.search_in_progress:
        with st.status("Researching...", expanded=True) as status:
            st.write("🔍 Analyzing Quranic references...")
            time.sleep(1)
            st.write("📚 Verifying hadith authenticity...")
            time.sleep(1)
            status.update(label="Research Complete", state="complete")
        st.session_state.search_in_progress = False
    
    st.markdown(response, unsafe_allow_html=True)
    
    if related:
        st.markdown("### Related Questions")
        for q in related.split('\n')[:3]:
            if q.strip() and st.button(q.strip()):
                st.session_state.current_query = q.strip()
                st.rerun()

def main():
    st.set_page_config(
        page_title="Quranic Insights Pro",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    initialize_session()

    st.markdown("""
    <style>
    .arabic-text {
        font-family: 'Lateef', 'Scheherazade New';
        font-size: 24px;
        text-align: right;
        line-height: 2.5;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    display_header()
    
    with st.container():
        display_search_interface()
        
        if st.session_state.current_query and not st.session_state.search_in_progress:
            try:
                manager = IslamicKnowledgeManager(st.session_state.selected_model)
                response, related = manager.execute_research(
                    st.session_state.current_query, 
                    st.session_state.language
                )
                
                st.session_state.search_history.insert(0, {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': st.session_state.current_query,
                    'response': response
                })
                
                display_result(response, related)
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")

    with st.sidebar:
        st.radio("🌐 Language", ["English", "Urdu"], key="language")

if __name__ == "__main__":
    main()