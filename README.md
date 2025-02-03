
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


# Reddit Content Analysis and Blog Generator

## Overview
This application automates content analysis and blog generation from Reddit posts and comments. Using a structured multi-agent workflow, it extracts key insights, performs semantic analysis, and generates structured Markdown-formatted blog posts.

## Features
- **Reddit API Integration**: Securely fetches user submissions and comments.
- **Automated Analysis Pipeline**: Multi-stage processing for semantic enrichment, metric extraction, and blog generation.
- **Local LLM Integration**: Utilizes Ollama API for AI-powered content generation.
- **Database Storage**: Saves analysis history in SQLite for future reference.
- **Interactive UI**: Built with Streamlit for an intuitive user experience.
- **Markdown Formatting**: Automatically structures output for readability and publication.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Ollama (for local LLM execution)
- Reddit API credentials (stored in `.env` file)

### Setup
1. Clone the repository:
   ```shell
   git clone https://github.com/kliewerdaniel/RedToBlog02.git
   cd RedToBlog02
   ```
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
3. Configure the Ollama model:
   ```shell
   ollama pull vanilj/Phi-4:latest
   ```
4. Set up Reddit API credentials in a `.env` file:
   ```plaintext
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   REDDIT_USERNAME=your_username
   REDDIT_PASSWORD=your_password
   ```
5. Initialize the database:
   ```shell
   python -c "import reddit_blog_app; reddit_blog_app.init_db()"
   ```
6. Run the application:
   ```shell
   streamlit run reddit_blog_app.py
   ```

## Usage
1. Open the Streamlit interface.
2. Select the number of Reddit posts to analyze.
3. Click **Start Analysis** to fetch and process content.
4. View extracted metrics and generated blog posts.
5. Access previous analyses in the **History** tab.

## Architecture
### System Components
- **RedditManager**: Handles API authentication and content retrieval.
- **BlogGenerator**: Orchestrates AI-driven analysis and blog generation.
- **AI Agents**:
  - `ExpandAgent`: Enhances raw text with contextual information.
  - `AnalyzeAgent`: Extracts semantic and psychological insights.
  - `MetricAgent`: Quantifies key metrics from the analysis.
  - `FinalAgent`: Generates structured blog content.
  - `FormatAgent`: Formats content into Markdown for readability.
- **SQLite Database**: Stores analysis results for future retrieval.
- **Streamlit UI**: Provides an interactive front-end for user interaction.

## Use Cases
### Personal Analytics
- Track sentiment and emotional trends over time.
- Identify cognitive biases in writing.
- Monitor personal development through linguistic patterns.

### Content Creation
- Generate automated blog posts from Reddit activity.
- Convert discussions into structured articles.
- Improve writing efficiency with AI-assisted summarization.

### Community Analysis
- Detect emerging topics and trends in subreddits.
- Analyze sentiment shifts in online discussions.
- Measure engagement and controversy metrics.

### Professional Applications
- Market research through subreddit analysis.
- Customer sentiment tracking for businesses.
- Competitive analysis based on Reddit discussions.

## Future Enhancements
- **Advanced NLP Features**: Sentiment analysis, topic modeling, and bias detection.
- **Cross-Platform Integration**: Support for Twitter, Hacker News, and other platforms.
- **Enhanced Database Queries**: Advanced search and filtering for historical analyses.
- **User Authentication**: Multi-user support with secure login.
- **Deployment Options**: Docker containerization and cloud hosting.


## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
For more information, visit the [GitHub Repository](https://github.com/kliewerdaniel/RedToBlog02).

