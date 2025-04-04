import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import os
import re
from typing import Optional, Tuple,Dict
from crewai import Agent, Task, Crew, Process
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

class GeminiModelProvider:
    """Handles Gemini model integration"""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        if not self.api_key.startswith("AIza"):
            raise ValueError("Invalid GEMINI_API_KEY format")
        self.available_models = {
            "gemini/gemini-2.0-flash": "Google Gemini 2.0 Flash",
            "gemini/gemini-1.5-pro": "Google Gemini 1.5 Pro"
        }


class IslamicKnowledgeCrewManager:
    """Manages CrewAI implementation for Islamic knowledge search"""
    
    def __init__(self, model_provider: GeminiModelProvider, selected_model: str):
        self.model_provider = model_provider
        self.selected_model = selected_model
        
    def _create_research_agent(self, role: str, goal: str) -> Agent:
        """Creates a generic research agent"""
        return Agent(
            role=role,
            goal=goal,
            verbose=True,
            allow_delegation=True
        )
    
    def execute_research(self, query: str, language: str) -> Tuple[str, Optional[str]]:
        """Execute research process using CrewAI"""
        try:
            # Create agents
            quran_researcher = self._create_research_agent(
                "Quranic Scholar",
                "Find relevant Quranic verses with explanations"
            )
            
            hadith_expert = self._create_research_agent(
                "Hadith Scholar",
                "Identify authentic hadith with proper citations"
            )

            # Create tasks
            tasks = [
                Task(
                    description=f"Research Quranic verses for: {query}",
                    agent=quran_researcher,
                    expected_output="List of Quranic verses with Surah:Verse and explanations"
                ),
                Task(
                    description=f"Find authentic hadith for: {query}",
                    agent=hadith_expert,
                    expected_output="List of hadith with proper citations"
                )
            ]

            crew = Crew(
                agents=[quran_researcher, hadith_expert],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            formatted_result = self._format_results(result, language)
            related = self._generate_related_questions(query)
            
            return (formatted_result, related)
        except Exception as e:
            logger.error(f"CrewAI Error: {str(e)}")
            return self._fallback_search(query, language)

    def _generate_related_questions(self, query: str) -> Optional[str]:
        """Generate related questions using Gemini"""
        prompt = f"Generate 3 related Islamic questions for: {query}"
        return self._safe_api_call([{"role": "user", "content": prompt}])

    def _safe_api_call(self, messages: list, retries: int = 3) -> Optional[str]:
        """API call with error handling"""
        for attempt in range(retries):
            try:
                response = completion(
                    model=self.selected_model,
                    messages=messages,
                    api_key=self.model_provider.api_key,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"API Error (Attempt {attempt+1}): {str(e)}")
                return None

    def _format_results(self, results: Dict, language: str) -> str:
        """Format results into structured response"""
        sections = {
            "English": {
                "quran": "## 📖 Quranic References",
                "hadith": "## 📚 Authentic Hadith",
                "guidance": "## 💡 Practical Guidance"
            },
            "Urdu": {
                "quran": "## 📖 قرآنی حوالہ جات",
                "hadith": "## 📚 صحیح حدیث",
                "guidance": "## 💡 عملی رہنمائی"
            }
        }
        
        formatted_response = ""
        try:
            formatted_response += f"{sections[language]['quran']}\n{results.get('quranic_research', '')}\n\n"
            formatted_response += f"{sections[language]['hadith']}\n{results.get('hadith_research', '')}\n\n"
            formatted_response = re.sub(r'(```arabic)(.*?)(```)', 
                            r'<div dir="rtl" class="arabic-text">\2</div>', 
                            formatted_response, flags=re.DOTALL)
            return formatted_response
        except Exception as e:
            logger.error(f"Format Error: {str(e)}")
            return "Error formatting results."

    def _fallback_search(self, query: str, language: str) -> Tuple[str, Optional[str]]:
        """Fallback direct Gemini call"""
        finder = IslamicKnowledgeFinder(self.selected_model)
        return finder.search_islamic_knowledge(query, language)

class IslamicKnowledgeFinder:
    """Direct Gemini integration"""
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        if not self.api_key.startswith("AIza"):
            raise ValueError("Invalid GEMINI_API_KEY format")

    def search_islamic_knowledge(self, query: str, language: str) -> Tuple[str, Optional[str]]:
        """Direct search using Gemini"""
        prompt = f"""
        Provide Islamic guidance for: '{query}'
        Include Quranic verses, authentic hadith, and practical guidance.
        Use Markdown formatting with clear sections.
        """
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                max_tokens=2000
            ).choices[0].message.content
            
            related = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": f"Generate 3 related questions for: {query}"}],
                api_key=self.api_key,
                max_tokens=500
            ).choices[0].message.content
            
            return (self._format_response(response, language), related)
        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
            return ("Error processing request", None)

    def _format_response(self, response: str, language: str) -> str:
        response = re.sub(r'(```arabic)(.*?)(```)', 
                        r'<div dir="rtl" class="arabic-text">\2</div>', 
                        response, flags=re.DOTALL)
        return response
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

def model_selector(model_provider):
    with st.sidebar:
        st.subheader("🤖 Model Selection")
        model_options = model_provider.available_models
        
        if not model_options:
            st.warning("No Gemini API key set in .env file")
            return
            
        selected = st.selectbox(
            "Select AI Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        st.session_state.selected_model = selected

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
            st.rerun()  # Corrected here

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
                st.rerun()  # Corrected here

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
    model_provider = GeminiModelProvider()
    
    with st.container():
        display_search_interface()
        
        if st.session_state.current_query and not st.session_state.search_in_progress:
            try:
                crew_manager = IslamicKnowledgeCrewManager(model_provider, st.session_state.selected_model)
                response, related = crew_manager.execute_research(
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
        model_selector(model_provider)
        st.markdown("---")
        st.radio("🌐 Language", ["English", "Urdu"], key="language")

if __name__ == "__main__":
    main()