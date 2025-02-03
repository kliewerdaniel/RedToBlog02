
```python

#requirements.txt

streamlit==1.25.0
pandas
plotly>=5.13.0
networkx
requests
praw
python-dotenv
sqlalchemy

#.env

REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
REDDIT_USERNAME=
REDDIT_PASSWORD=

#reddit_blog_app.py

import os
import streamlit as st
import sqlite3
import json
from datetime import datetime
import pandas as pd
import networkx as nx
import praw
import requests
from dotenv import load_dotenv
from textwrap import dedent

# Load environment variables
load_dotenv()

# Database setup
def init_db():
    with sqlite3.connect("metrics.db") as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      metrics TEXT,
                      final_blog TEXT,
                      status TEXT)''')

def save_to_db(metrics, final_blog, status="complete"):
    with sqlite3.connect("metrics.db") as conn:
        conn.execute(
            "INSERT INTO results (timestamp, metrics, final_blog, status) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(metrics), final_blog, status)
        )

def fetch_history():
    with sqlite3.connect("metrics.db") as conn:
        return pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)

# Reddit integration
class RedditManager:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")
        )

    def fetch_content(self, limit=10):
        submissions = [post.title + "\n" + post.selftext for post in self.reddit.user.me().submissions.new(limit=limit)]
        comments = [comment.body for comment in self.reddit.user.me().comments.new(limit=limit)]
        return "\n\n".join(submissions + comments)

# Base agent
class BaseAgent:
    def __init__(self, model="vanilj/Phi-4:latest"):
        self.endpoint = "http://localhost:11434/api/generate"
        self.model = model

    def request_api(self, prompt):
        try:
            response = requests.post(self.endpoint, json={"model": self.model, "prompt": prompt, "stream": False})
            if response.status_code != 200:
                print(f"API request failed: {response.status_code} - {response.text}")
                return ""

            json_response = response.json()
            print(f"Full API Response: {json_response}")  # Print full response for debugging

            return json_response.get('response', json_response)  # Return full response if 'response' key is missing
        except Exception as e:
            print(f"API request error: {str(e)}")
            return ""

# Blog generator
class BlogGenerator:
    def __init__(self):
        self.agents = {
            'Expand': self.ExpandAgent(),
            'Analyze': self.AnalyzeAgent(),
            'Metric': self.MetricAgent(),
            'Final': self.FinalAgent(),
            'Format': self.FormatAgent()
        }
        self.workflow = nx.DiGraph([('Expand', 'Analyze'), ('Analyze', 'Metric'), ('Metric', 'Final'), ('Final', 'Format')])

    class ExpandAgent(BaseAgent):
        def process(self, content):
            return {"expanded": self.request_api(f"Expand: {content}")}
    
    class FormatAgent(BaseAgent): pass

    class AnalyzeAgent(BaseAgent):
        def process(self, state):
            return {"analysis": self.request_api(f"Analyze: {state.get('expanded', '')}")}

    class MetricAgent(BaseAgent):
        def process(self, state):
            raw_response = self.request_api(f"Extract Metrics: {state.get('analysis', '')}")
            if not raw_response:
                print("Error: Received empty response from API")
                return {"metrics": {}}
            try:
                return {"metrics": json.loads(raw_response)}
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Raw response: {raw_response}")
                return {"metrics": {}}


    class FormatAgent(BaseAgent):
        def process(self, state):
            blog_content = state.get('final_blog', '')
            formatting_prompt = dedent(f"""
            Transform this raw content into a properly formatted Markdown blog post. Use these guidelines:
            - Start with a # Heading
            - Use ## and ### subheadings to organize content
            - Add bullet points for lists
            - Use **bold** for key metrics
            - Include --- for section dividers
            - Maintain original insights but improve readability
            
            Content to format:
            {blog_content}
            """)
            formatted_blog = self.request_api(formatting_prompt)
            return {"final_blog": formatted_blog}

    class FinalAgent(BaseAgent):
        def process(self, state):
            return {"final_blog": self.request_api(f"Generate Blog: {state.get('metrics', '')}")}

    def run_analysis(self, content):
        state = {'raw_content': content}
        for node in nx.topological_sort(self.workflow):
            state.update(self.agents[node].process(state))
        return state

# Streamlit UI
def main():
    st.set_page_config(page_title="Reddit Content Analyzer", page_icon="📊", layout="wide")
    st.title("Reddit Content Analysis and Blog Generator")
    st.sidebar.header("Settings")
    post_limit = st.sidebar.slider("Posts to analyze", 1, 20, 5)

    init_db()
    reddit_manager = RedditManager()
    blog_generator = BlogGenerator()

    tab_analyze, tab_history = st.tabs(["New Analysis", "History"])
    
    with tab_analyze:
        if st.button("Start Analysis"):
            with st.spinner("Collecting and analyzing Reddit content..."):
                content = reddit_manager.fetch_content(post_limit)
                results = blog_generator.run_analysis(content)
                
                # Debugging print to verify UI is receiving full response
                print("Final Results:", results)
                
                save_to_db(results['metrics'], results['final_blog'])
                
                st.subheader("Analysis Metrics")
                st.json(results)  # Show full results object

                st.subheader("Detailed Metrics")
                if 'metrics' in results and isinstance(results['metrics'], dict):
                    for key, value in results['metrics'].items():
                        st.write(f"**{key}:** {value}")

                st.subheader("Generated Blog Post")
                st.markdown(results['final_blog'])

    with tab_history:
        history_df = fetch_history()
        if not history_df.empty:
            for _, row in history_df.iterrows():
                with st.expander(f"Analysis from {row['timestamp']}"):
                    st.json(json.loads(row['metrics']))
                    st.markdown(row['final_blog'])
        else:
            st.info("No previous analyses found")

if __name__ == "__main__":
    main()


```


# Comprehensive Guide to the Reddit Content Analysis System

## 1. Architecture Overview
This program combines web scraping, data analysis, and natural language processing in a Streamlit-based web interface. Key components:

- **Reddit API Integration**: Uses PRAW library for secure Reddit access
- **Data Pipeline**: Multi-stage processing workflow with specialized AI agents
- **Database**: SQLite for storing analysis history
- **LLM Integration**: Local Ollama API for content generation
- **Visualization**: Plotly and Streamlit for data presentation

## 2. Core Components Breakdown

### 2.1 Reddit Integration (RedditManager)
- Authentication via .env file credentials
- Fetches both submissions and comments
- Configurable post limit (default 10 each)
- Returns combined text content for analysis

### 2.2 Processing Workflow
Seven-stage pipeline managed through networkx DAG:
1. **Content Expansion**: Enriches raw text with context
2. **Semantic Analysis**: Identifies themes and patterns
3. **Metric Extraction**: Quantifies key insights
4. **Blog Generation**: Creates initial draft content 
5. **Formatting**: Applies Markdown styling
6. **Storage**: SQLite database persistence
7. **Visualization**: Interactive Streamlit presentation

### 2.3 AI Agent System
- BaseAgent handles Ollama API communication
- Specialized agents for each processing stage:
  - ExpandAgent: Contextual enrichment
  - AnalyzeAgent: Pattern recognition
  - MetricAgent: Data quantification
  - FinalAgent: Content generation
  - FormatAgent: Presentation styling

### 2.4 Database Structure
SQLite table schema:
- Timestamp: Analysis datetime
- Metrics: JSON-formatted insights
- Final Blog: Formatted Markdown
- Status: Process completion state

## 3. Execution Flow
1. User sets parameters via Streamlit sidebar
2. Reddit content collection through PRAW
3. Multi-stage LLM processing pipeline
4. Results storage and visualization
5. Historical data retrieval system

## 4. Alternative Use Cases

### 4.1 Personal Analytics
- Track emotional states over time
- Identify cognitive biases in writing
- Monitor personal development progress
- Analyze communication style evolution

### 4.2 Content Creation
- Automated social media post generation
- Newsletter content production
- Idea generation for creative writing
- Video script outlining

### 4.3 Community Analysis
- Subreddit trend identification
- Controversy detection in discussions
- Sentiment analysis across communities
- Network mapping of user interactions

### 4.4 Professional Applications
- Market research from niche communities
- Customer feedback analysis
- Brand perception monitoring
- Competitor strategy insights

## 5. Advanced Modifications

### 5.1 Enhanced Analysis
- Add sentiment analysis layer
- Implement topic modeling (LDA/NMF)
- Integrate personality prediction models
- Add cross-platform comparison (Twitter, HN)

### 5.2 Deployment Options
- Docker containerization
- Cloud deployment (AWS/GCP)
- Scheduled daily analysis via Cron
- Email newsletter integration

### 5.3 Security Improvements
- User authentication system
- Data encryption at rest
- API rate limiting
- Content anonymization

## 6. Setup Guide

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Ollama:
```bash
ollama pull vanilj/Phi-4:latest
```

3. Populate .env file with Reddit credentials

4. Launch application:
```bash
streamlit run reddit_blog_app.py
```

## 7. Customization Opportunities

- **Template System**: Add multiple blog format templates
- **Multi-User Support**: Implement account system
- **API Expansion**: Add Twitter/Medium integrations
- **Advanced Metrics**: 
  - Readability scores
  - Engagement predictions
  - Controversy index
  - Topic diversity metrics

## 8. Ethical Considerations

- Respect Reddit API terms of service
- Implement content anonymization
- Add opt-out mechanisms
- Include data deletion features
- Monitor for biased LLM outputs

This system provides a foundation for automated content analysis and generation, adaptable to various text-based data sources beyond Reddit. The modular architecture allows for customization while maintaining core functionality.