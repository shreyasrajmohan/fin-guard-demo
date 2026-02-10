import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import os
import asyncio

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Financial Tech" look
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #00d26a; /* Success Green */
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
    }
    .main-header {
        font-size: 3rem; 
        font-weight: 800;
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# API KEY (Hardcoded for demo, or use secrets)
API_KEY = "espr_lowBI3wZLVvZRRBzkwMIbiQyjF0j4URbktAq1Pmhj1M"
THREAD_FILE = "finance_thread_id.txt"

# --- 1. ROBUST DATA LOADING (From your fixed app.py) ---
@st.cache_data
def load_data():
    cleaned_dfs = []
    standard_cols = ['tran_date', 'particulars', 'debit', 'credit', 'balance']
    
    # Find all CSVs
    all_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'anomalies' not in f]
    
    if not all_files:
        return pd.DataFrame()

    for file_name in all_files:
        try:
            # Smart Header Detection
            raw_lines = pd.read_csv(file_name, nrows=30, header=None, encoding='latin1')
            header_idx = -1
            for i, row in raw_lines.iterrows():
                row_str = ' '.join(row.astype(str)).lower()
                if 'date' in row_str and ('debit' in row_str or 'withdrawal' in row_str):
                    header_idx = i
                    break
            
            if header_idx != -1:
                df = pd.read_csv(file_name, header=header_idx, encoding='latin1')
                df.columns = df.columns.str.lower().str.strip()
                
                # Normalize columns
                col_map = {}
                for c in df.columns:
                    if 'particular' in c: col_map[c] = 'particulars'
                    elif 'debit' in c or 'withdraw' in c: col_map[c] = 'debit'
                    elif 'credit' in c or 'deposit' in c: col_map[c] = 'credit'
                    elif 'date' in c and 'tran' in c: col_map[c] = 'tran_date'
                    elif 'balance' in c: col_map[c] = 'balance'
                
                df = df.rename(columns=col_map)
                
                # Ensure standard columns & clean numbers
                for col in standard_cols:
                    if col not in df.columns: df[col] = 0
                
                for col in ['debit', 'credit']:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Parse Dates (Crucial for charts)
                df['tran_date'] = pd.to_datetime(df['tran_date'], dayfirst=True, errors='coerce')
                
                df['source'] = file_name
                cleaned_dfs.append(df[standard_cols + ['source']])
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            
    if cleaned_dfs:
        full_df = pd.concat(cleaned_dfs, ignore_index=True)
        return full_df.dropna(subset=['tran_date']).sort_values('tran_date')
    return pd.DataFrame()

# --- 2. AI BACKEND CONNECTION ---
async def get_ai_response(user_msg, system_context=None):
    try:
        from backboard import BackboardClient
        client = BackboardClient(api_key=API_KEY)
        
        # Load or Create Thread ID
        thread_id = None
        if os.path.exists(THREAD_FILE):
            with open(THREAD_FILE, 'r') as f:
                thread_id = f.read().strip()
        
        if not thread_id:
            assistant = await client.create_assistant(
                name="FinGuard",
                system_prompt="You are FinGuard, a financial anomaly expert. Be concise. Remember user preferences."
            )
            thread = await client.create_thread(assistant.assistant_id)
            thread_id = str(thread.thread_id)
            with open(THREAD_FILE, 'w') as f:
                f.write(thread_id)

        # Send Message
        if system_context:
            await client.add_message(thread_id=thread_id, content=f"SYSTEM DATA:\n{system_context}", memory="Auto")
        
        response = await client.add_message(thread_id=thread_id, content=user_msg, memory="Auto")
        return response.content
    except ImportError:
        return "⚠️ Error: Backboard SDK not found. Please install it."
    except Exception as e:
        return f"⚠️ Error connecting to AI: {e}"

# --- 3. MAIN APP UI ---

# Sidebar
with st.sidebar:
    st.title("🛡️ FinGuard")
    st.write("Financial Anomaly Detection System")
    st.divider()
    contamination = st.slider("Anomaly Sensitivity", 0.01, 0.10, 0.02, help="Higher = Flag more transactions")
    st.caption("Powered by Isolation Forest & Backboard AI")

# Load Data
df = load_data()

if df.empty:
    st.warning("No data found. Please place CSV files in the folder.")
    st.stop()

# Run Anomaly Detection
model = IsolationForest(contamination=contamination, random_state=42)
df['anomaly_score'] = model.fit_predict(df[['debit', 'credit']].fillna(0))
df['status'] = df['anomaly_score'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')
anomalies = df[df['status'] == 'Suspicious']

# --- DASHBOARD HEADER ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", len(df))
with col2:
    st.metric("Total Spending", f"${df['debit'].sum():,.2f}")
with col3:
    st.metric("Anomalies Found", len(anomalies), delta="-Flagged", delta_color="inverse")

st.divider()

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📊 Visual Analysis", "💬 AI Investigation"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Spending Timeline")
        # Fancy Line Chart
        fig_line = px.line(df, x='tran_date', y='debit', title="Daily Spending Flow",
                           color_discrete_sequence=['#00d26a'])
        # Add red dots for anomalies
        fig_line.add_trace(go.Scatter(
            x=anomalies['tran_date'], y=anomalies['debit'],
            mode='markers', name='Anomaly', marker=dict(color='red', size=10)
        ))
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_b:
        st.subheader("Anomaly Distribution")
        fig_pie = px.pie(df, names='status', title="Normal vs Suspicious", 
                         color='status', color_discrete_map={'Normal':'#00d26a', 'Suspicious':'#ff4b4b'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🚨 Suspicious Transactions List")
    st.dataframe(
        anomalies[['tran_date', 'particulars', 'debit', 'source']].sort_values('debit', ascending=False),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.header("Chat with FinGuard AI")
    st.caption("The AI remembers your previous explanations (e.g., 'Netflix is approved').")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Pre-load context to AI invisibly
        context_data = anomalies.head(5).to_string()
        asyncio.run(get_ai_response("Analyze these new anomalies.", system_context=context_data))
        st.session_state.messages.append({"role": "assistant", "content": "I've analyzed your latest transactions. Several high-value items look suspicious. What would you like to know?"})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Explain the $500 transaction..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = asyncio.run(get_ai_response(prompt))
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})