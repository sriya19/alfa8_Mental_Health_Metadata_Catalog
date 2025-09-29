# ~/Desktop/enhanced_data_qa_system.py
import json
import tempfile
import sqlite3
from pathlib import Path
import sys
from datetime import datetime
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from openai import OpenAI
import anthropic

# Make sure Python can find the analyzer
sys.path.insert(0, str(Path(__file__).parent))
from file_metadata_analyzer import FileMetadataAnalyzer

# Page configuration
st.set_page_config(
    page_title="🧠 Mental Health Data Q&A System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .qa-response {
        background: #e6f3ff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DataStore:
    """Store and manage uploaded data for querying"""
    
    def __init__(self):
        self.datasets = {}  # Store actual dataframes/content
        self.metadata = {}  # Store metadata
        
    def add_dataset(self, filename: str, data: Any, metadata: Dict[str, Any]):
        """Store dataset and its metadata"""
        self.datasets[filename] = data
        self.metadata[filename] = metadata
        
    def get_dataset(self, filename: str):
        """Retrieve a specific dataset"""
        return self.datasets.get(filename)
    
    def get_all_datasets(self):
        """Get all datasets"""
        return self.datasets
    
    def search_datasets(self, query: str) -> Dict[str, Any]:
        """Search across all datasets for relevant information"""
        results = {}
        query_lower = query.lower()
        
        for filename, data in self.datasets.items():
            file_type = self.metadata[filename].get('file_type', '')
            
            if file_type == 'csv' and isinstance(data, pd.DataFrame):
                # Search in CSV data
                relevant_info = self._search_dataframe(data, query_lower)
                if relevant_info:
                    results[filename] = relevant_info
                    
            elif file_type == 'json':
                # Search in JSON data
                relevant_info = self._search_json(data, query_lower)
                if relevant_info:
                    results[filename] = relevant_info
                    
            elif file_type == 'txt':
                # Search in text data
                relevant_info = self._search_text(data, query_lower)
                if relevant_info:
                    results[filename] = relevant_info
                    
        return results
    
    def _search_dataframe(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Search within a dataframe"""
        results = {}
        
        # Search column names
        matching_cols = [col for col in df.columns if query in str(col).lower()]
        
        # Search for numeric queries
        if any(word in query for word in ['mean', 'average', 'sum', 'total', 'count']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results['statistics'] = {
                    'means': df[numeric_cols].mean().to_dict(),
                    'sums': df[numeric_cols].sum().to_dict(),
                    'counts': df[numeric_cols].count().to_dict()
                }
        
        # Search for specific values
        for col in df.columns:
            if df[col].dtype == 'object':
                mask = df[col].astype(str).str.lower().str.contains(query, na=False)
                if mask.any():
                    results[f'matches_in_{col}'] = df[mask].head(5).to_dict('records')
        
        # Get general info
        results['shape'] = {'rows': len(df), 'columns': len(df.columns)}
        results['columns'] = list(df.columns)
        results['sample'] = df.head(3).to_dict('records')
        
        return results
    
    def _search_json(self, data: Any, query: str) -> Dict[str, Any]:
        """Search within JSON data"""
        results = {}
        
        def search_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if query in str(key).lower() or query in str(value).lower():
                        results[f"{path}.{key}" if path else key] = value
                    search_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:10]):  # Limit to first 10 items
                    if query in str(item).lower():
                        results[f"{path}[{i}]"] = item
                    search_recursive(item, f"{path}[{i}]")
        
        search_recursive(data)
        return results
    
    def _search_text(self, text: str, query: str) -> Dict[str, Any]:
        """Search within text data"""
        lines = text.split('\n')
        matching_lines = []
        
        for i, line in enumerate(lines):
            if query in line.lower():
                matching_lines.append({'line_number': i+1, 'content': line[:200]})
                if len(matching_lines) >= 10:  # Limit results
                    break
        
        return {
            'matching_lines': matching_lines,
            'total_matches': len(matching_lines),
            'document_length': len(text)
        }

class DynamicVisualizer:
    """Generate dynamic visualizations based on actual data content"""
    
    @staticmethod
    def create_visualizations(data: Any, file_type: str, filename: str) -> List[Any]:
        """Create appropriate visualizations based on data type and content"""
        visualizations = []
        
        if file_type == 'csv' and isinstance(data, pd.DataFrame):
            visualizations.extend(DynamicVisualizer._visualize_dataframe(data, filename))
        elif file_type == 'json':
            if isinstance(data, list) and all(isinstance(item, dict) for item in data[:10]):
                # Convert to dataframe if possible
                df = pd.DataFrame(data)
                visualizations.extend(DynamicVisualizer._visualize_dataframe(df, filename))
            else:
                visualizations.append(DynamicVisualizer._visualize_json_structure(data, filename))
        elif file_type == 'txt':
            visualizations.append(DynamicVisualizer._visualize_text_stats(data, filename))
            
        return visualizations
    
    @staticmethod
    def _visualize_dataframe(df: pd.DataFrame, filename: str) -> List[Any]:
        """Create visualizations for dataframe data"""
        figs = []
        
        # 1. Numeric columns distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Create histograms for numeric columns
            if len(numeric_cols) <= 4:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=numeric_cols[:4]
                )
                for i, col in enumerate(numeric_cols[:4]):
                    row = i // 2 + 1
                    col_idx = i % 2 + 1
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, nbinsx=30),
                        row=row, col=col_idx
                    )
                fig.update_layout(
                    title=f"Numeric Distributions in {filename}",
                    height=600,
                    showlegend=False
                )
                figs.append(("Numeric Distributions", fig))
            else:
                # Create box plot for many numeric columns
                fig = go.Figure()
                for col in numeric_cols[:10]:  # Limit to 10 columns
                    fig.add_trace(go.Box(y=df[col], name=col))
                fig.update_layout(
                    title=f"Numeric Column Distributions in {filename}",
                    yaxis_title="Values",
                    height=500
                )
                figs.append(("Numeric Distributions", fig))
        
        # 2. Categorical columns distribution
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols[:3]:  # Top 3 categorical columns
                value_counts = df[col].value_counts().head(10)
                if len(value_counts) > 0:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                    figs.append((f"{col} Distribution", fig))
        
        # 3. Correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title=f"Correlation Matrix - {filename}",
                height=500
            )
            figs.append(("Correlation Matrix", fig))
        
        # 4. Time series if date column exists
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='coerce')
                if df[col].dtype == 'datetime64[ns]' or col.lower() in ['date', 'time', 'timestamp']:
                    date_cols.append(col)
            except:
                pass
        
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            try:
                df_sorted = df.sort_values(date_col)
                fig = go.Figure()
                for col in numeric_cols[:3]:  # Plot up to 3 numeric columns
                    fig.add_trace(go.Scatter(
                        x=df_sorted[date_col],
                        y=df_sorted[col],
                        mode='lines',
                        name=col
                    ))
                fig.update_layout(
                    title=f"Time Series Analysis - {filename}",
                    xaxis_title=date_col,
                    yaxis_title="Values",
                    height=400
                )
                figs.append(("Time Series", fig))
            except:
                pass
        
        # 5. Missing data visualization
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title=f"Missing Data by Column - {filename}",
                labels={'x': 'Column', 'y': 'Missing Count'},
                color=missing_data.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            figs.append(("Missing Data", fig))
        
        return figs
    
    @staticmethod
    def _visualize_json_structure(data: Any, filename: str) -> Tuple[str, Any]:
        """Visualize JSON structure"""
        def get_structure(obj, level=0, max_level=3):
            if level > max_level:
                return {"type": "truncated"}
            
            if isinstance(obj, dict):
                return {
                    "type": "object",
                    "keys": list(obj.keys())[:10],  # Limit keys shown
                    "size": len(obj)
                }
            elif isinstance(obj, list):
                return {
                    "type": "array",
                    "length": len(obj),
                    "sample": get_structure(obj[0], level+1) if obj else None
                }
            else:
                return {"type": type(obj).__name__}
        
        structure = get_structure(data)
        
        # Create a text representation
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"<b>JSON Structure - {filename}</b><br><br>{json.dumps(structure, indent=2)[:500]}...",
            showarrow=False,
            font=dict(family="Courier New", size=12),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        
        return ("JSON Structure", fig)
    
    @staticmethod
    def _visualize_text_stats(text: str, filename: str) -> Tuple[str, Any]:
        """Visualize text statistics"""
        lines = text.split('\n')
        words = text.split()
        
        # Word frequency
        word_freq = {}
        for word in words:
            word_clean = word.lower().strip('.,!?";:()[]{}')
            if len(word_clean) > 3:  # Only words longer than 3 chars
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        # Top 15 words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_words:
            words_list, counts = zip(*top_words)
            fig = px.bar(
                x=counts, y=words_list,
                orientation='h',
                title=f"Top Words in {filename}",
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig.update_layout(height=400)
            return ("Word Frequency", fig)
        
        return None

class EnhancedLLM:
    """Enhanced LLM that can query actual data"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        
        if provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-opus-20240229"
        else:
            # Fallback mode without LLM
            self.client = None
            self.model = None
    
    def answer_with_data(self, question: str, data_context: Dict[str, Any], 
                        data_store: DataStore) -> Tuple[str, Dict[str, Any]]:
        """Answer questions using actual data"""
        
        # Search for relevant data
        search_results = data_store.search_datasets(question)
        
        # Build context from actual data
        context_parts = []
        for filename, results in search_results.items():
            context_parts.append(f"\nData from {filename}:")
            context_parts.append(json.dumps(results, indent=2, default=str)[:2000])
        
        context = "\n".join(context_parts)
        
        if not context:
            return "No relevant data found for your question. Please make sure you've uploaded data files.", {}
        
        # If no LLM, provide direct data results
        if not self.client:
            return self._format_direct_answer(search_results), search_results
        
        # Generate LLM response with actual data
        system_prompt = """You are a data analysis assistant. You have access to actual data 
        from uploaded files. Provide specific, data-driven answers based on the information 
        provided. Always cite specific values, statistics, and findings from the data."""
        
        user_prompt = f"""
        Question: {question}
        
        Available Data:
        {context}
        
        Please provide a comprehensive answer based on this actual data. Include:
        1. Direct answer with specific values from the data
        2. Any relevant statistics or patterns
        3. Suggestions for further analysis if applicable
        """
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                answer = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                answer = response.content[0].text
            
            return answer, search_results
            
        except Exception as e:
            return self._format_direct_answer(search_results), search_results
    
    def _format_direct_answer(self, search_results: Dict[str, Any]) -> str:
        """Format search results as a direct answer without LLM"""
        if not search_results:
            return "No data found matching your query."
        
        answer_parts = ["Based on the uploaded data:\n"]
        
        for filename, results in search_results.items():
            answer_parts.append(f"\n**From {filename}:**")
            
            if 'statistics' in results:
                stats = results['statistics']
                answer_parts.append("\nStatistics:")
                if 'means' in stats:
                    for col, mean in stats['means'].items():
                        answer_parts.append(f"- Average {col}: {mean:.2f}")
                if 'sums' in stats:
                    for col, total in stats['sums'].items():
                        answer_parts.append(f"- Total {col}: {total:.2f}")
            
            if 'shape' in results:
                answer_parts.append(f"\nDataset has {results['shape']['rows']} rows and {results['shape']['columns']} columns")
            
            if 'matching_lines' in results:
                answer_parts.append(f"\nFound {results['total_matches']} matching lines in the text")
            
            for key, value in results.items():
                if key.startswith('matches_in_'):
                    answer_parts.append(f"\nFound matches in column: {key.replace('matches_in_', '')}")
                    answer_parts.append(f"Sample: {str(value)[:200]}...")
        
        return "\n".join(answer_parts)

# Initialize session state
if 'data_store' not in st.session_state:
    st.session_state.data_store = DataStore()
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Mental Health Data Q&A System</h1>
    <p>Upload data files, get dynamic visualizations, and ask questions about your data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM setup
    llm_provider = st.selectbox(
        "LLM Provider (Optional)",
        ["none", "openai", "anthropic"],
        help="Select LLM for enhanced responses"
    )
    
    if llm_provider != "none":
        api_key = st.text_input(
            f"{llm_provider.upper()} API Key",
            type="password",
            help="Enter API key for enhanced Q&A"
        )
        
        if api_key:
            try:
                st.session_state.llm = EnhancedLLM(provider=llm_provider, api_key=api_key)
                st.success(f"✅ {llm_provider.upper()} connected")
            except Exception as e:
                st.error(f"Failed to connect: {e}")
    else:
        st.session_state.llm = EnhancedLLM(provider="none")
        st.info("ℹ️ Running in direct data mode (no LLM)")
    
    st.divider()
    
    # Data summary
    st.header("📊 Loaded Data")
    datasets = st.session_state.data_store.get_all_datasets()
    
    if datasets:
        for filename in datasets.keys():
            st.success(f"✓ {filename}")
    else:
        st.info("No data uploaded yet")

# Initialize components
analyzer = FileMetadataAnalyzer()
visualizer = DynamicVisualizer()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Visualize", "💬 Q&A Interface", "📝 History"])

with tab1:
    st.header("Upload Your Data Files")
    
    uploaded_files = st.file_uploader(
        "Choose CSV, JSON, or TXT files",
        type=["csv", "json", "txt"],
        accept_multiple_files=True,
        help="Upload files to analyze and query"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Read the file content
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                try:
                    if file_type == 'csv':
                        data = pd.read_csv(uploaded_file)
                        st.success(f"✅ Loaded {uploaded_file.name}: {len(data)} rows, {len(data.columns)} columns")
                    
                    elif file_type == 'json':
                        content = uploaded_file.read()
                        data = json.loads(content)
                        st.success(f"✅ Loaded {uploaded_file.name}: JSON data")
                    
                    elif file_type == 'txt':
                        data = uploaded_file.read().decode('utf-8')
                        st.success(f"✅ Loaded {uploaded_file.name}: {len(data)} characters")
                    
                    # Analyze metadata
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                        if file_type == 'csv':
                            data.to_csv(tmp_file.name, index=False)
                        elif file_type == 'json':
                            json.dump(data, open(tmp_file.name, 'w'))
                        else:
                            open(tmp_file.name, 'w').write(data)
                        
                        metadata = analyzer.analyze_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                    
                    # Store data
                    st.session_state.data_store.add_dataset(
                        uploaded_file.name, 
                        data, 
                        metadata
                    )
                    
                    # Display data preview
                    st.subheader(f"📋 {uploaded_file.name}")
                    
                    with st.expander("View Data Preview", expanded=True):
                        if file_type == 'csv':
                            st.dataframe(data.head(10))
                            
                            # Basic statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Rows", len(data))
                            with col2:
                                st.metric("Columns", len(data.columns))
                            with col3:
                                numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                                st.metric("Numeric Columns", numeric_cols)
                            with col4:
                                missing = data.isnull().sum().sum()
                                st.metric("Missing Values", missing)
                        
                        elif file_type == 'json':
                            st.json(data if isinstance(data, dict) else data[:3] if isinstance(data, list) else data)
                        
                        elif file_type == 'txt':
                            st.text(data[:1000] + "..." if len(data) > 1000 else data)
                    
                    # Generate and display visualizations
                    st.subheader(f"📊 Visualizations for {uploaded_file.name}")
                    
                    visualizations = visualizer.create_visualizations(data, file_type, uploaded_file.name)
                    
                    if visualizations:
                        # Create tabs for different visualizations
                        if len(visualizations) > 1:
                            viz_tabs = st.tabs([name for name, _ in visualizations])
                            for (name, fig), tab in zip(visualizations, viz_tabs):
                                if fig:
                                    with tab:
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            for name, fig in visualizations:
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

with tab2:
    st.header("Ask Questions About Your Data")
    
    datasets = st.session_state.data_store.get_all_datasets()
    
    if not datasets:
        st.warning("⚠️ Please upload data files first in the 'Upload & Visualize' tab")
    else:
        st.success(f"✅ {len(datasets)} dataset(s) loaded and ready for querying")
        
        # Display available datasets
        with st.expander("View Available Datasets"):
            for filename, data in datasets.items():
                if isinstance(data, pd.DataFrame):
                    st.write(f"**{filename}**: {len(data)} rows, columns: {', '.join(data.columns[:10])}")
                else:
                    st.write(f"**{filename}**: {type(data).__name__} data")
        
        # Q&A Form
        with st.form("qa_form"):
            question = st.text_area(
                "Your Question:",
                placeholder="Examples:\n- What is the average of all numeric columns?\n- Show me the distribution of [column name]\n- Find all rows containing 'mental health'\n- What are the top values in [column]?\n- Calculate statistics for the data",
                height=120
            )
            
            col1, col2 = st.columns([4, 1])
            with col1:
                submit = st.form_submit_button("🔍 Get Answer", use_container_width=True)
            with col2:
                clear = st.form_submit_button("🗑️ Clear History")
        
        if clear:
            st.session_state.qa_history = []
            st.rerun()
        
        if submit and question:
            with st.spinner("Analyzing your data..."):
                # Get answer
                answer, data_used = st.session_state.llm.answer_with_data(
                    question, 
                    {}, 
                    st.session_state.data_store
                )
                
                # Save to history
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data_sources': list(data_used.keys())
                })
                
                # Display answer
                st.markdown("""<div class="qa-response">""", unsafe_allow_html=True)
                st.markdown(f"**Question:** {question}")
                st.markdown("**Answer:**")
                st.write(answer)
                
                if data_used:
                    st.markdown("**Data sources used:** " + ", ".join(data_used.keys()))
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show relevant data if available
                if data_used:
                    with st.expander("View Source Data"):
                        for filename, results in data_used.items():
                            st.write(f"**{filename}:**")
                            if 'sample' in results:
                                st.write("Sample data:")
                                st.json(results['sample'])
                            if 'statistics' in results:
                                st.write("Statistics:")
                                st.json(results['statistics'])

with tab3:
    st.header("Q&A History")
    
    if st.session_state.qa_history:
        for i, item in enumerate(reversed(st.session_state.qa_history), 1):
            with st.expander(f"Q{i}: {item['question'][:100]}...", expanded=False):
                st.markdown(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                if item.get('data_sources'):
                    st.markdown(f"**Sources:** {', '.join(item['data_sources'])}")
    else:
        st.info("No Q&A history yet. Ask questions in the Q&A Interface tab!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>💡 Tip: Upload your CSV files to see automatic visualizations of your actual data!</p>
    <p>Ask specific questions like "What is the average age?" or "Show distribution of column X"</p>
</div>
""", unsafe_allow_html=True)