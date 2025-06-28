import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Set Streamlit page config as the very first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="ClueAI: Smart Workplace Assistant", 
    page_icon="🤖", 
    layout="wide"
)

# Make sure numpy is imported before whisper
import numpy as np
import validators
import whisper
import requests
import gspread
import pytubefix as pytube
from google.oauth2.service_account import Credentials
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain_core.documents import Document

# Configuration
sys.stdout.reconfigure(encoding='utf-8')
# Ensure FFmpeg is in the PATH
if r"C:\ffmpeg\bin" not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
    print("Added FFmpeg to PATH")
else:
    print("FFmpeg already in PATH")
DEBUG = True
SESSION_FILE = "user_sessions.json"

class SessionManager:
    """Handles user session management"""
    
    @staticmethod
    def get_default_session():
        return {
            "transcript": "",
            "summary": "",
            "tasks": "",
            "chat_history": [],
            "credentials_dict": None,
            "view_message_index": None,
            "input_processed": False,
            "processing_stats": {"started": False, "step": "", "progress": 0},
            "processing_log": [],
            "timestamp": ""
        }
    
    @staticmethod
    def load_sessions():
        try:
            if os.path.exists(SESSION_FILE):
                with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            if DEBUG: print(f"Debug: Error loading sessions: {e}")
            return {}
    
    @staticmethod
    def save_sessions(sessions):
        try:
            with open(SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            if DEBUG: print(f"Debug: Error saving sessions: {e}")

class AudioProcessor:
    """Handles audio processing and transcription"""
    
    def __init__(self):
        self.whisper_model = None
        self._check_ffmpeg()
        try:
            self.whisper_model = self._load_whisper_model()
        except Exception as e:
            st.error(f"❌ Failed to load Whisper model: {e}")
            print(f"Error loading Whisper model: {e}")
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is installed and accessible"""
        try:
            # Try to run ffmpeg to check if it's in PATH
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print("FFmpeg found and working")
            return True
        except Exception as e:
            print(f"FFmpeg check failed: {e}")
            st.error("❌ FFmpeg not found. Please make sure FFmpeg is installed and in your PATH.")
            return False
    
    def _load_whisper_model(self):
        try:
            # Make sure numpy is properly imported
            if 'numpy' not in sys.modules:
                import numpy as np
            
            # Check for GPU availability
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Load the model on the appropriate device
            print("Loading Whisper model...")
            model = whisper.load_model("base", device=device)
            print("Whisper model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
    
    def download_youtube_audio(self, url: str) -> Optional[str]:
        try:
            yt = pytube.YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            audio_path = f"temp_youtube_{datetime.now().timestamp()}.mp3"
            stream.download(filename=audio_path)
            return audio_path if os.path.exists(audio_path) else None
        except Exception as e:
            st.error(f"❌ Failed to download YouTube audio: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        if not self.whisper_model:
            raise Exception("Whisper model not loaded")
        
        # Use GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Transcribing audio using {device}...")
        start_time = datetime.now()
        
        # Set fp16=True for GPU and fp16=False for CPU
        transcription = self.whisper_model.transcribe(
            audio_path, 
            fp16=(device == "cuda")
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Transcription completed in {duration:.2f} seconds")
        
        return transcription["text"]

class TextProcessor:
    """Handles text processing, summarization, and task extraction"""
    
    def __init__(self, groq_api_key: str):
        # Use a faster model for better performance
        self.llm = ChatGroq(
            model="Gemma2-9b-it", 
            groq_api_key=groq_api_key,
            max_tokens=2048,  # Limit output tokens for faster response
            temperature=0.3   # Lower temperature for more focused outputs
        )
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self):
        return {
            "summary": PromptTemplate(
                template="""Provide a concise summary of the following text in 300 words. 
                Focus on key points, decisions made, and important information:
                
                {text}
                
                Your summary should be well-structured and easy to understand.""",
                input_variables=["text"]
            ),
            "tasks": PromptTemplate(
                template="""From the following text, extract specific, actionable tasks. 
                Each task should be concise (max 30 words) and include action, deadline, and responsible party.

                {text}

                Format each task exactly like this:
                Task: [Task description]
                Deadline: [urgent and important/not urgent but important/urgent but not important/not urgent and not important/None/specific date]
                Responsible: [Person, Group, or None]

                If no tasks are identified, return an empty string.""",
                input_variables=["text"]
            ),
            "refine": PromptTemplate(
                template="""Refine the summary or action items based on the user's request.
                
                Original transcript: {transcript}
                Current summary: {current_summary}
                Current action items: {current_tasks}
                User request: {user_request}

                Provide refined content addressing the specific request.""",
                input_variables=["transcript", "current_summary", "current_tasks", "user_request"]
            )
        }
    
    def generate_summary(self, text: str) -> str:
        print("Generating summary...")
        start_time = datetime.now()
        
        document = [Document(page_content=text)]
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=self.prompts["summary"])
        result = chain.invoke(document)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Summary generated in {duration:.2f} seconds")
        
        return result.get("output_text", "")
    
    def extract_tasks(self, text: str) -> str:
        print("Extracting tasks...")
        start_time = datetime.now()
        
        document = [Document(page_content=text)]
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=self.prompts["tasks"])
        result = chain.invoke(document)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Tasks extracted in {duration:.2f} seconds")
        
        return result.get("output_text", "")
    
    def refine_content(self, transcript: str, summary: str, tasks: str, user_request: str) -> str:
        print("Refining content...")
        start_time = datetime.now()
        
        chain = LLMChain(llm=self.llm, prompt=self.prompts["refine"])
        result = chain.invoke({
            "transcript": transcript,
            "current_summary": summary,
            "current_tasks": tasks,
            "user_request": user_request
        })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Content refined in {duration:.2f} seconds")
        
        return result.get("text", "")

class TaskParser:
    """Handles task parsing and formatting"""
    
    @staticmethod
    def parse_tasks(tasks_text: str) -> Tuple[List[Dict], str]:
        tasks_data = []
        
        if not tasks_text:
            return tasks_data, "No tasks extracted."
        
        if "**Refined Summary and Action Items:**" in tasks_text:
            return TaskParser._parse_refined_tasks(tasks_text)
        else:
            return TaskParser._parse_standard_tasks(tasks_text)
    
    @staticmethod
    def _parse_refined_tasks(tasks_text: str) -> Tuple[List[Dict], str]:
        tasks_data = []
        
        if "**Action Items:**" in tasks_text:
            action_items_section = tasks_text.split("**Action Items:**")[1].strip()
            task_blocks = re.split(r'\* \*\*Task:', action_items_section)[1:]
            
            for block in task_blocks:
                task_description = block.split("**Deadline:**")[0].strip() if "**Deadline:**" in block else block.strip()
                
                deadline = "None"
                if "**Deadline:**" in block:
                    deadline_part = block.split("**Deadline:**")[1]
                    deadline = deadline_part.split("**")[0].strip() if "**" in deadline_part else deadline_part.strip()
                
                responsible = "None"
                if "**Responsible:**" in block:
                    responsible_part = block.split("**Responsible:**")[1]
                    responsible = responsible_part.split("*")[0].strip()
                    if responsible and responsible[-1] in ['*', '\n']:
                        responsible = responsible[:-1].strip()
                
                tasks_data.append({"Task": task_description, "Deadline": deadline, "Responsible": responsible})
        
        return tasks_data, tasks_text
    
    @staticmethod
    def _parse_standard_tasks(tasks_text: str) -> Tuple[List[Dict], str]:
        tasks_data = []
        lines = tasks_text.strip().split('\n')
        current_task = {}
        
        for line in lines:
            if line.startswith("Task:"):
                if current_task:
                    tasks_data.append(current_task)
                current_task = {"Task": line[5:].strip(), "Deadline": "None", "Responsible": "None"}
            elif line.startswith("Deadline:") and current_task:
                current_task["Deadline"] = line[9:].strip()
            elif line.startswith("Responsible:") and current_task:
                current_task["Responsible"] = line[12:].strip()
        
        if current_task:
            tasks_data.append(current_task)
        
        bullet_points = ""
        for task in tasks_data:
            bullet_points += f"  *Task: {task['Task']}\n\n"
            bullet_points += f"  *Deadline: {task['Deadline']}\n\n"
            bullet_points += f"  *Responsible: {task['Responsible']}\n\n"
        
        return tasks_data, bullet_points if bullet_points else "No tasks extracted."

class GoogleSheetsIntegration:
    """Handles Google Sheets integration"""
    
    @staticmethod
    def push_tasks(tasks: str, credentials_dict: dict) -> Tuple[bool, str]:
        try:
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            client = gspread.authorize(credentials)
            sheet_id = "16TTOfGAo1o0eFNO1Ty3jq_HoNTXUIEpzZkbh-gZkR8A"
            spreadsheet = client.open_by_key(sheet_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            worksheet = spreadsheet.add_worksheet(title=f"Tasks_{timestamp}", rows=100, cols=3)
            worksheet.update("A1:C1", [["Task", "Deadline", "Responsible"]])
            
            tasks_data, _ = TaskParser.parse_tasks(tasks)
            if tasks_data:
                sheet_data = [[task["Task"], task["Deadline"], task["Responsible"]] for task in tasks_data]
                worksheet.update(f"A2:C{len(sheet_data) + 1}", sheet_data)
                return True, f"Tasks pushed to Google Sheet: Tasks_{timestamp}"
            else:
                return False, "No tasks found to push"
        except Exception as e:
            return False, f"Google API Error: {e}"

class ClueAIApp:
    """Main application class"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.audio_processor = AudioProcessor()
        self.setup_page()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = None
        if "user_sessions" not in st.session_state:
            st.session_state.user_sessions = {}
        if "current_session" not in st.session_state:
            st.session_state.current_session = self.session_manager.get_default_session()
    
    def setup_page(self):
        """Setup page configuration and styling"""
        # Custom CSS
        st.markdown("""
        <style>
            .main { background-color: #2e2e2e; color: #e0e0e0; padding: 20px; border-radius: 10px; }
            .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; }
            .stButton>button:hover { background-color: #45a049; }
            .output-box, .task-output { 
                background-color: #2e2e2e; 
                color: #e0e0e0; 
                padding: 15px; 
                border-radius: 5px; 
                border: 1px solid #555; 
                margin-top: 10px;
                max-height: 500px;
                overflow-y: auto;
            }
            /* Progress indicators */
            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
                font-weight: bold;
            }
            .status-processing {
                background-color: #2196F3;
                color: white;
            }
            .status-complete {
                background-color: #4CAF50;
                color: white;
            }
            .status-error {
                background-color: #F44336;
                color: white;
            }
            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #333;
                border-radius: 4px 4px 0 0;
                gap: 1px;
                padding: 10px 16px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4CAF50;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🔍 ClueAI: Smart Workplace Assistant")
        st.subheader("📎 Summarize Meetings, Texts, or YouTube Videos")
    
    def render_sidebar(self):
        """Render sidebar with user session and API configuration"""
        with st.sidebar:
            # User Session Management
            st.subheader("👤 User Session")
            user_id = st.text_input("Enter Your Email or Name", value="")
            
            if st.button("Load Session"):
                if user_id.strip():
                    self.load_user_session(user_id.strip())
                else:
                    st.error("Please enter a valid email or name")
            
            # API Keys & Credentials
            st.subheader("🔑 API Keys & Credentials")
            groq_api_key = st.text_input("Groq API Key", value="", type="password", 
                                       help="Enter your Groq API key from console.groq.com")
            
            google_credentials_file = st.file_uploader("Upload Google Service Account Credentials (JSON)", 
                                                     type=["json"])
            
            if google_credentials_file:
                self.load_google_credentials(google_credentials_file)
            
            # Chat History
            if st.session_state.user_id:
                self.render_chat_history()
            
            return groq_api_key
    
    def load_user_session(self, user_id: str):
        """Load user session"""
        st.session_state.user_id = user_id
        sessions = self.session_manager.load_sessions()
        
        if user_id not in sessions:
            sessions[user_id] = []
        
        st.session_state.user_sessions = sessions
        
        if sessions[user_id]:
            st.session_state.current_session = sessions[user_id][-1]
        else:
            st.session_state.current_session = self.session_manager.get_default_session()
        
        st.success(f"Session loaded for {user_id}")
    
    def load_google_credentials(self, credentials_file):
        """Load Google credentials"""
        try:
            st.session_state.current_session["credentials_dict"] = json.loads(
                credentials_file.getvalue().decode()
            )
            st.sidebar.success("✅ Credentials loaded successfully")
        except json.JSONDecodeError:
            st.sidebar.error("❌ Invalid JSON credentials file format")
        except Exception as e:
            st.sidebar.error(f"❌ Credentials processing error: {str(e)}")
    
    def render_chat_history(self):
        """Render chat history in sidebar"""
        st.subheader(f"📜 Chat History for {st.session_state.user_id}")
        sessions = st.session_state.user_sessions.get(st.session_state.user_id, [])
        
        for session_idx, session in enumerate(reversed(sessions)):
            session_time = session.get("timestamp", "Unknown")
            for msg_idx, message in enumerate(session["chat_history"]):
                preview = f"{message['content'][:50]}..." if len(message['content']) > 50 else message['content']
                
                if st.button(f"{'🤖' if message['role'] == 'assistant' else '👤'} {session_time}: {preview}", 
                           key=f"history_{session_idx}_{msg_idx}"):
                    st.session_state.current_session = session
                    st.session_state.current_session["view_message_index"] = msg_idx
    
    def render_input_section(self):
        """Render input selection section"""
        st.subheader("📥 Choose Input Type")
        input_type = st.radio("Select input method:", 
                            ("Audio File", "Text File", "YouTube Video", "Paste Text"))
        
        transcript_text = ""
        audio_file = None
        youtube_url = ""
        
        if input_type == "Audio File":
            audio_file = st.file_uploader("Upload Meeting Audio (MP3/WAV)", type=["mp3", "wav"])
            if audio_file:
                st.info("Audio file uploaded. Click 'Process Input' to transcribe and analyze.")
        
        elif input_type == "Text File":
            text_file = st.file_uploader("Upload Text File (TXT)", type=["txt"])
            if text_file:
                transcript_text = text_file.read().decode("utf-8")
        
        elif input_type == "YouTube Video":
            youtube_url = st.text_input("Enter YouTube Video URL")
            if youtube_url and not validators.url(youtube_url):
                st.error("Please enter a valid YouTube URL")
        
        elif input_type == "Paste Text":
            transcript_text = st.text_area("Paste your transcript here")
        
        return input_type, transcript_text, audio_file, youtube_url
    
    def process_input(self, input_type: str, transcript_text: str, audio_file, youtube_url: str, groq_api_key: str):
        """Process the input and generate summary and tasks"""
        try:
            # Initialize text processor
            text_processor = TextProcessor(groq_api_key)
            
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            timing_placeholder = st.empty()
            
            with st.spinner("Processing input..."):
                # Step 1: Get transcript
                if input_type in ["Audio File", "YouTube Video"]:
                    status_placeholder.info("🎤 Step 1/3: Transcribing audio using GPU acceleration...")
                    start_time = datetime.now()
                    
                    transcript_text = self.process_audio_input(input_type, audio_file, youtube_url, progress_placeholder)
                    
                    if not transcript_text:
                        return
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    timing_placeholder.success(f"✅ Transcription completed in {duration:.2f} seconds")
                
                # Step 2: Generate summary and extract tasks
                st.session_state.current_session["transcript"] = transcript_text
                
                # Generate summary
                self.update_progress(progress_placeholder, "Generating summary", 60)
                status_placeholder.info("📝 Step 2/3: Generating summary...")
                start_time = datetime.now()
                
                st.session_state.current_session["summary"] = text_processor.generate_summary(transcript_text)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                timing_placeholder.success(f"✅ Summary generated in {duration:.2f} seconds")
                
                # Extract tasks
                self.update_progress(progress_placeholder, "Extracting tasks", 80)
                status_placeholder.info("📋 Step 3/3: Extracting tasks...")
                start_time = datetime.now()
                
                st.session_state.current_session["tasks"] = text_processor.extract_tasks(transcript_text)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                timing_placeholder.success(f"✅ Tasks extracted in {duration:.2f} seconds")
                
                self.update_progress(progress_placeholder, "Complete", 100)
                status_placeholder.success("✅ Processing complete!")
                
                # Update session
                st.session_state.current_session["chat_history"].append({
                    "role": "assistant",
                    "content": f"Here's the summary of your input:\n\n{st.session_state.current_session['summary']}"
                })
                st.session_state.current_session["input_processed"] = True
                st.session_state.current_session["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                self.save_current_session()
                progress_placeholder.empty()
                
        except Exception as e:
            st.error(f"❌ Exception: {e}")
            if DEBUG: print(f"Debug: Exception occurred: {e}")
    
    def process_audio_input(self, input_type: str, audio_file, youtube_url: str, progress_placeholder) -> str:
        """Process audio input (file or YouTube)"""
        if not self.audio_processor.whisper_model:
            st.error("❌ Whisper model not loaded. Check FFmpeg installation.")
            return ""
        
        if input_type == "Audio File":
            audio_path = f"temp_audio_{datetime.now().timestamp()}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
        else:  # YouTube Video
            self.update_progress(progress_placeholder, "Downloading YouTube audio", 30)
            audio_path = self.audio_processor.download_youtube_audio(youtube_url)
            if not audio_path:
                return ""
        
        self.update_progress(progress_placeholder, "Transcribing audio", 40)
        transcript_text = self.audio_processor.transcribe_audio(audio_path)
        os.remove(audio_path)
        
        return transcript_text
    
    def update_progress(self, placeholder, step: str, progress: int):
        """Update progress display"""
        with placeholder.container():
            st.subheader("⏳ Processing Status")
            st.progress(progress / 100)
            st.info(f"Step: {step}")
    
    def render_results(self):
        """Render processing results"""
        if st.session_state.current_session["input_processed"] and st.session_state.current_session["summary"]:
            # Create tabs for better organization
            summary_tab, tasks_tab, transcript_tab = st.tabs(["📝 Summary", "📋 Tasks", "🎤 Transcript"])
            
            with summary_tab:
                st.subheader("Meeting/Text Summary")
                # Limit and clean the summary to avoid repetition issues
                summary = st.session_state.current_session['summary']
                # Remove any repetitive phrases that might be in the output
                if "Let me know if you'd like me to expand" in summary:
                    summary = summary.split("Let me know if you'd like me to expand")[0].strip()
                st.markdown(f"<div class='output-box'>{summary}</div>", unsafe_allow_html=True)
            
            with tasks_tab:
                st.subheader("Extracted Tasks")
                tasks_data, bullet_points = TaskParser.parse_tasks(st.session_state.current_session["tasks"])
                
                # Display tasks in a more structured way
                if tasks_data:
                    for i, task in enumerate(tasks_data):
                        with st.expander(f"Task {i+1}: {task['Task'][:50]}...", expanded=True):
                            st.write(f"**Task:** {task['Task']}")
                            st.write(f"**Deadline:** {task['Deadline']}")
                            st.write(f"**Responsible:** {task['Responsible']}")
                else:
                    st.info("No specific tasks were extracted from this content.")
                
                # Google Sheets integration
                self.render_sheets_integration()
            
            with transcript_tab:
                st.subheader("Original Transcript")
                st.markdown(f"<div class='output-box'><pre>{st.session_state.current_session['transcript'][:1000]}{'...' if len(st.session_state.current_session['transcript']) > 1000 else ''}</pre></div>", 
                           unsafe_allow_html=True)
                if len(st.session_state.current_session['transcript']) > 1000:
                    with st.expander("Show full transcript"):
                        st.text(st.session_state.current_session['transcript'])
    
    def render_sheets_integration(self):
        """Render Google Sheets integration buttons"""
        if st.session_state.current_session["tasks"] and st.session_state.current_session["credentials_dict"]:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Push Tasks to Google Sheets"):
                    self.push_to_sheets(st.session_state.current_session["tasks"], col2)
        elif st.session_state.current_session["tasks"]:
            st.warning("⚠️ Please upload your Google Service Account credentials to push tasks to Google Sheets")
    
    def push_to_sheets(self, tasks: str, result_col):
        """Push tasks to Google Sheets"""
        with st.spinner("Pushing tasks to Google Sheets..."):
            success, message = GoogleSheetsIntegration.push_tasks(
                tasks, st.session_state.current_session["credentials_dict"]
            )
            
            if success:
                st.session_state.current_session["chat_history"].append({
                    "role": "system", 
                    "content": f"Sheets Update: {message}"
                })
                with result_col:
                    st.success(message)
            else:
                with result_col:
                    st.error(f"Failed to update Sheets: {message}")
            
            self.save_current_session()
    
    def render_chat_section(self, groq_api_key: str):
        """Render chat section for content refinement"""
        st.subheader("💬 Chat to Refine Summary or Action Items")
        
        # Display chat history
        for message in st.session_state.current_session["chat_history"]:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.write(message["content"])
        
        # Chat input
        chat_input = st.chat_input("Enter your request to refine the summary, action items, or extract more key points...")
        
        if chat_input and st.session_state.current_session["transcript"]:
            self.process_chat_input(chat_input, groq_api_key)
        elif chat_input:
            st.warning("⚠️ Please process input first before using the chat feature.")
    
    def process_chat_input(self, chat_input: str, groq_api_key: str):
        """Process chat input for content refinement"""
        if not st.session_state.user_id:
            st.error("❌ Please enter and load a user ID to use the chat feature")
            return
        
        try:
            st.session_state.current_session["chat_history"].append({"role": "user", "content": chat_input})
            
            text_processor = TextProcessor(groq_api_key)
            
            with st.spinner("Refining content..."):
                refined_output = text_processor.refine_content(
                    st.session_state.current_session["transcript"],
                    st.session_state.current_session["summary"],
                    st.session_state.current_session["tasks"],
                    chat_input
                )
                
                # Determine if output is tasks or summary
                if "**Action Items:**" in refined_output or ("Task:" in refined_output and "Deadline:" in refined_output):
                    st.session_state.current_session["tasks"] = refined_output
                    st.session_state.current_session["chat_history"].append({
                        "role": "assistant", 
                        "content": f"Updated action items:\n\n{refined_output}"
                    })
                    
                    # Offer to push to sheets
                    if st.session_state.current_session["credentials_dict"]:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("Push Updated Tasks to Google Sheets", key="push_refined"):
                                self.push_to_sheets(refined_output, col2)
                else:
                    st.session_state.current_session["summary"] = refined_output
                    st.session_state.current_session["chat_history"].append({
                        "role": "assistant", 
                        "content": refined_output
                    })
                
                self.save_current_session()
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error refining content: {str(e)}")
            st.session_state.current_session["chat_history"].append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error while refining: {str(e)}"
            })
            self.save_current_session()
            st.rerun()
    
    def save_current_session(self):
        """Save current session to file"""
        if st.session_state.user_id:
            sessions = st.session_state.user_sessions
            if st.session_state.user_id not in sessions:
                sessions[st.session_state.user_id] = []
            
            # Update the last session or add new one
            if sessions[st.session_state.user_id]:
                sessions[st.session_state.user_id][-1] = st.session_state.current_session.copy()
            else:
                sessions[st.session_state.user_id].append(st.session_state.current_session.copy())
            
            self.session_manager.save_sessions(sessions)
            st.session_state.user_sessions = sessions
    
    def run(self):
        """Main application runner"""
        # Check if viewing a specific chat message
        if (st.session_state.current_session["view_message_index"] is not None and 
            0 <= st.session_state.current_session["view_message_index"] < len(st.session_state.current_session["chat_history"])):
            
            selected_message = st.session_state.current_session["chat_history"][st.session_state.current_session["view_message_index"]]
            st.subheader(f"📄 Viewing Chat Message from {selected_message['role'].capitalize()}")
            
            with st.chat_message(selected_message["role"], avatar="🤖" if selected_message["role"] == "assistant" else "👤"):
                st.write(selected_message["content"])
            
            if st.button("← Return to Main Interface"):
                st.session_state.current_session["view_message_index"] = None
                st.rerun()
            return
        
        # Main interface
        groq_api_key = self.render_sidebar()
        input_type, transcript_text, audio_file, youtube_url = self.render_input_section()
        
        # Process input button
        if st.button("Process Input"):
            if not groq_api_key.strip():
                st.error("❌ Please provide the Groq API Key")
            elif not self._validate_input(input_type, audio_file, transcript_text, youtube_url):
                st.error("❌ Please provide the required input")
            elif not st.session_state.user_id:
                st.error("❌ Please enter and load a user ID to start a session")
            else:
                self.process_input(input_type, transcript_text, audio_file, youtube_url, groq_api_key)
        
        # Display results
        self.render_results()
        
        # Chat section
        self.render_chat_section(groq_api_key)
    
    def _validate_input(self, input_type: str, audio_file, transcript_text: str, youtube_url: str) -> bool:
        """Validate input based on type"""
        if input_type == "Audio File" and not audio_file:
            return False
        elif input_type == "Text File" and not transcript_text:
            return False
        elif input_type == "YouTube Video" and not youtube_url:
            return False
        elif input_type == "Paste Text" and not transcript_text:
            return False
        return True

# Run the application
if __name__ == "__main__":
    app = ClueAIApp()
    app.run()