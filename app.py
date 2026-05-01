import os
import json
import numpy as np
import pandas as pd
import faiss
import chromadb
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
load_dotenv()
st.set_page_config(
    page_title="Sentiment Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── Password Protection ───────────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div style='max-width:400px; margin:100px auto; text-align:center;'>
            <h2 style='color:#0f3460;'>🔍 Sentiment Intelligence Platform</h2>
            <p style='color:#888;'>Enter password to continue</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            password = st.text_input("Password", type="password")
            if st.button("Enter", use_container_width=True):
                if password == os.getenv("APP_PASSWORD", "sentimentiq"):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        st.stop()

check_password()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f2f5; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3460 0%, #16213e 100%) !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    padding: 6px 0 !important;
    font-weight: 500 !important;
}

.hero {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 60%, #1a1a2e 100%);
    padding: 64px 40px 48px;
    text-align: center;
    color: white;
}
.hero-badge {
    display: inline-block;
    background: rgba(233,69,96,0.2);
    border: 1px solid rgba(233,69,96,0.5);
    color: #ff6b8a;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0 0 12px 0;
    line-height: 1.2;
}
.hero h1 span { color: #e94560; }
.hero p {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    max-width: 560px;
    margin: 0 auto 32px;
}

.stats-bar {
    background: white;
    padding: 20px 40px;
    display: flex;
    justify-content: center;
    gap: 56px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-item { text-align: center; }
.stat-number { font-size: 1.7rem; font-weight: 800; color: #0f3460; }
.stat-label { font-size: 0.75rem; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

.chat-bubble-user {
    background: #0f3460;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0 6px auto;
    max-width: 72%;
    width: fit-content;
    font-size: 0.92rem;
    line-height: 1.5;
    float: right;
    clear: both;
}
.chat-bubble-assistant {
    background: white;
    color: #1a1a2e;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px auto 6px 0;
    max-width: 72%;
    width: fit-content;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    float: left;
    clear: both;
}
.chat-clearfix { clear: both; }
.chat-welcome {
    text-align: center;
    padding: 32px 20px;
    color: #888;
}
.chat-welcome h3 { color: #1a1a2e; font-weight: 700; margin-bottom: 6px; }

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.section-subtitle {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 24px;
}

.result-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    border-left: 4px solid #0f3460;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.result-name { font-weight: 700; color: #1a1a2e; font-size: 0.93rem; }
.score-badge {
    background: #f0f4ff;
    color: #0f3460;
    border-radius: 8px;
    padding: 4px 12px;
    font-weight: 700;
    font-size: 0.82rem;
    white-space: nowrap;
}

.badge-weak {
    background: #fff0f0; color: #c0392b;
    border: 1px solid #f5c6cb; border-radius: 20px;
    padding: 5px 13px; font-size: 0.82rem; font-weight: 600;
    margin: 3px; display: inline-block;
}
.badge-strong {
    background: #f0fff4; color: #27ae60;
    border: 1px solid #c3e6cb; border-radius: 20px;
    padding: 5px 13px; font-size: 0.82rem; font-weight: 600;
    margin: 3px; display: inline-block;
}

.team-card {
    background: white;
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 4px solid #0f3460;
}
.team-card .avatar {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #0f3460, #e94560);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin: 0 auto 12px;
    color: white; font-weight: 700;
}
.team-card h4 { font-size: 1rem; font-weight: 700; color: #1a1a2e; margin: 0 0 4px; }
.team-card .netid { font-size: 0.78rem; color: #888; margin-bottom: 10px; }

.about-hero {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border-radius: 16px;
    padding: 40px;
    color: white;
    margin-bottom: 28px;
}
.about-hero h2 { font-size: 1.8rem; font-weight: 800; margin: 0 0 10px; }
.about-hero p { color: rgba(255,255,255,0.75); font-size: 0.95rem; line-height: 1.7; margin: 0; }

.tech-badge {
    background: #f0f4ff; color: #0f3460;
    border-radius: 8px; padding: 5px 12px;
    font-size: 0.8rem; font-weight: 600;
    margin: 3px; display: inline-block;
    border: 1px solid #d0d9f0;
}

.stButton > button {
    background: #0f3460 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover { background: #e94560 !important; }

div[data-testid="stHorizontalBlock"] .stButton > button {
    background: white !important;
    color: #0f3460 !important;
    border: 1.5px solid #d0d9f0 !important;
    border-radius: 20px !important;
    padding: 6px 12px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #0f3460 !important;
    color: white !important;
    border-color: #0f3460 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    product_vector_df = pd.read_csv("models/absa/product_aspect_vectors.csv")
    product_ids = product_vector_df["product_id"].tolist()
    feature_cols = [c for c in product_vector_df.columns if c != "product_id"]
    vectors = product_vector_df[feature_cols].values.astype("float32")
    vectors_normalized = normalize(vectors).astype("float32")

    dimension = vectors_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_normalized)

    chroma_client = chromadb.PersistentClient(path="models/stage4/chroma_db")
    collection = chroma_client.get_or_create_collection(name="product_reviews")

    product_names_df = pd.read_csv("models/absa/product_names.csv")
    id_to_label = dict(zip(product_names_df["ProductId"], product_names_df["label"]))
    label_to_id = dict(zip(product_names_df["label"], product_names_df["ProductId"]))

    return (product_ids, feature_cols, vectors, vectors_normalized,
            index, collection, id_to_label, label_to_id)

(product_ids, feature_cols, vectors, vectors_normalized,
 index, collection, id_to_label, label_to_id) = load_all()

dropdown_options = [id_to_label.get(pid, pid) for pid in product_ids]

def get_id(label):
    return label_to_id.get(label, label)

def clean_name(label):
    return label.split(" — ")[-1] if " — " in label else label

# ── Agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    def search_products_by_keyword(keyword: str) -> str:
        keyword = keyword.lower().strip()
        words = keyword.split()
        matches = []
        for pid in product_ids:
            label = id_to_label.get(pid, "").lower()
            if any(word in label for word in words):
                matches.append(f"{pid} — {id_to_label.get(pid, pid)}")
        if not matches:
            return f"No products found matching '{keyword}'. Try: 'coffee', 'tea', 'dog food', 'chocolate', 'vitamin'."
        return "\n".join(matches[:10])

    def get_product_profile(product_id: str) -> str:
        product_id = product_id.strip()
        if product_id not in product_ids:
            return f"Product {product_id} not found."
        idx = product_ids.index(product_id)
        vec = vectors[idx]
        name = clean_name(id_to_label.get(product_id, product_id))
        profile = {feature_cols[i]: round(float(vec[i]), 3) for i in range(len(feature_cols))}
        profile = dict(sorted(profile.items(), key=lambda x: x[1], reverse=True))
        return f"**{name}** scores:\n{json.dumps(profile, indent=2)}"

    def similar_products(product_id: str) -> str:
        product_id = product_id.strip()
        if product_id not in product_ids:
            return f"Product {product_id} not found."
        idx = product_ids.index(product_id)
        query_vec = vectors_normalized[idx].reshape(1, -1)
        scores, indices = index.search(query_vec, 6)
        results = []
        for score, i in zip(scores[0], indices[0]):
            if i == idx:
                continue
            pid = product_ids[i]
            name = clean_name(id_to_label.get(pid, pid))
            results.append(f"**{name}** (similarity: {round(float(score), 3)})")
        return "\n".join(results[:5])

    def gap_fill_recommender(product_id: str) -> str:
        product_id = product_id.strip()
        if product_id not in product_ids:
            return f"Product {product_id} not found."
        idx = product_ids.index(product_id)
        vec = vectors[idx]
        weak_aspects = [feature_cols[i] for i, s in enumerate(vec) if s < -0.1]
        if not weak_aspects:
            name = clean_name(id_to_label.get(product_id, product_id))
            return f"**{name}** scores well on everything."
        gap_scores = []
        for i, pid in enumerate(product_ids):
            if pid == product_id:
                continue
            other_vec = vectors[i]
            gap_score = np.mean([other_vec[feature_cols.index(a)] for a in weak_aspects])
            gap_scores.append((pid, round(float(gap_score), 4)))
        gap_scores.sort(key=lambda x: x[1], reverse=True)
        result = f"Weak aspects: {', '.join(weak_aspects)}\n\nBetter alternatives:\n"
        result += "\n".join([
            f"**{clean_name(id_to_label.get(pid, pid))}** (gap score: {score})"
            for pid, score in gap_scores[:5]
        ])
        return result

    def fetch_reviews(product_id: str) -> str:
        product_id = product_id.strip()
        results = collection.query(
            query_texts=[f"review about {product_id}"],
            n_results=3,
            where={"product_id": product_id}
        )
        if not results["documents"][0]:
            return f"No reviews found for {product_id}."
        name = clean_name(id_to_label.get(product_id, product_id))
        output = f"Reviews for **{name}**:\n"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output += f"⭐ {meta['score']}/5 — {meta['summary']}: {doc[:200]}\n\n"
        return output

    tools = [
        Tool(name="search_products_by_keyword",
             func=search_products_by_keyword,
             description="Search products by keyword. Use FIRST when user describes a product type. After results, call get_product_profile on top 2-3 to compare scores, then recommend the BEST one with clear reasoning. Input: keyword like 'dog food', 'coffee'"),
        Tool(name="get_product_profile",
             func=get_product_profile,
             description="Get aspect sentiment scores for a product. Use to compare and identify strengths/weaknesses. Input: product_id"),
        Tool(name="similar_products",
             func=similar_products,
             description="Find products with similar sentiment profiles. Input: product_id"),
        Tool(name="gap_fill_recommender",
             func=gap_fill_recommender,
             description="Find products better in the weak aspects of a given product. Input: product_id"),
        Tool(name="fetch_reviews",
             func=fetch_reviews,
             description="Fetch real customer reviews for a product. Input: product_id"),
    ]

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         max_iterations=12, handle_parsing_errors=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Product Intelligence Platform")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "🔍  Product Explorer",
        "🔗  Similar Products",
        "🎯  Gap-Fill Recommender",
        "📖  About"
    ])
    

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🧠 AI-Powered · NLP · Amazon Reviews</div>
        <h1>Sentiment <span>Intelligence</span> Platform</h1>
        <p>Ask our AI anything about products. We analyze real customer reviews so you get honest, data-driven answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item"><div class="stat-number">691</div><div class="stat-label">Products</div></div>
        <div class="stat-item"><div class="stat-number">4,963</div><div class="stat-label">Reviews</div></div>
        <div class="stat-item"><div class="stat-number">76K+</div><div class="stat-label">Aspects Extracted</div></div>
        <div class="stat-item"><div class="stat-number">13</div><div class="stat-label">Sentiment Dimensions</div></div>
        <div class="stat-item"><div class="stat-number">5</div><div class="stat-label">Pipeline Stages</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:16px;'>
        <span style='font-size:1.3rem; font-weight:800; color:#1a1a2e;'>Ask our AI anything</span><br>
        <span style='color:#888; font-size:0.88rem;'>Powered by Llama 3.3 70B + real customer review data</span>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clickable suggestion pills
    pill_cols = st.columns(5)
    suggestions = ["🐕 Best dog food?", "☕ Top rated coffee", "🍫 Best chocolate", "💊 Good vitamins", "🍵 Recommend a tea"]
    for i, (col, suggestion) in enumerate(zip(pill_cols, suggestions)):
        with col:
            if st.button(suggestion, key=f"pill_{i}", use_container_width=True):
                clean_suggestion = suggestion.split(" ", 1)[1]
                st.session_state.chat_history.append({"role": "user", "content": clean_suggestion})
                with st.spinner("🤖 Analyzing reviews..."):
                    agent_executor = load_agent()
                    response = agent_executor.invoke({"input": clean_suggestion})
                    answer = response["output"]
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-welcome">
            <h3>👋 Hi! I'm your product advisor</h3>
            <p>Click a suggestion above or type your own question below.<br>
            I analyze real customer reviews to give you honest, data-driven answers.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div><div class="chat-clearfix"></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-assistant">{msg["content"]}</div><div class="chat-clearfix"></div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask me anything about products...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("🤖 Analyzing thousands of reviews..."):
            agent_executor = load_agent()
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── PRODUCT EXPLORER ──────────────────────────────────────────────────────────
elif page == "🔍  Product Explorer":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔍 Product Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Deep dive into any product\'s sentiment profile across 13 dimensions</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Search for a product", dropdown_options)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("Analyze Product", use_container_width=True)

    if analyze:
        product_id = get_id(selected)
        idx = product_ids.index(product_id)
        vec = vectors[idx]
        name = clean_name(selected)
        strong = [feature_cols[i] for i, v in enumerate(vec) if v > 0.1]
        weak = [feature_cols[i] for i, v in enumerate(vec) if v < -0.1]

        st.markdown(f"### **{name}**")
        st.markdown("---")

        chart_col, badge_col = st.columns([3, 1])
        with chart_col:
            colors = ["#27ae60" if v > 0.1 else "#c0392b" if v < -0.1 else "#bdc3c7" for v in vec]
            fig = go.Figure(go.Bar(
                x=feature_cols, y=vec,
                marker_color=colors,
                text=[round(float(v), 2) for v in vec],
                textposition="outside"
            ))
            fig.update_layout(
                yaxis_title="Sentiment Score",
                yaxis_range=[-1.3, 1.3],
                plot_bgcolor="white", paper_bgcolor="white",
                height=360, xaxis_tickangle=-30,
                font=dict(family="Inter"),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with badge_col:
            st.markdown("**✅ Strengths**")
            if strong:
                for s in strong:
                    st.markdown(f'<span class="badge-strong">✅ {s}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*None*")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**⚠️ Weaknesses**")
            if weak:
                for w in weak:
                    st.markdown(f'<span class="badge-weak">⚠️ {w}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*None*")

# ── SIMILAR PRODUCTS ──────────────────────────────────────────────────────────
elif page == "🔗  Similar Products":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔗 Similar Products</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Find products with matching sentiment profiles using FAISS cosine similarity</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected = st.selectbox("Search for a product", dropdown_options)
    with col2:
        top_k = st.slider("Results", 3, 10, 5)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        find = st.button("Find Similar", use_container_width=True)

    if find:
        product_id = get_id(selected)
        idx = product_ids.index(product_id)
        query_vec = vectors_normalized[idx].reshape(1, -1)
        scores, indices = index.search(query_vec, top_k + 1)

        results = []
        for score, i in zip(scores[0], indices[0]):
            if i == idx:
                continue
            pid = product_ids[i]
            name = clean_name(id_to_label.get(pid, pid))
            results.append({"name": name, "score": round(float(score), 4)})
        results = results[:top_k]

        col_list, col_chart = st.columns([1, 1])
        with col_list:
            for rank, r in enumerate(results, 1):
                st.markdown(f"""
                <div class="result-card">
                    <div>
                        <span style='color:#888;font-size:0.78rem;font-weight:600;'>#{rank}</span>
                        <span class="result-name" style='margin-left:8px;'>{r['name']}</span>
                    </div>
                    <div class="score-badge">⚡ {r['score']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            fig = px.bar(
                pd.DataFrame(results), x="name", y="score",
                color="score", color_continuous_scale="Blues",
                labels={"name": "", "score": "Similarity"}
            )
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_tickangle=-25, font=dict(family="Inter"),
                margin=dict(t=10, b=10), height=320,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ── GAP FILL ─────────────────────────────────────────────────────────────────
elif page == "🎯  Gap-Fill Recommender":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎯 Gap-Fill Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Unhappy with a product? Find something better in its weak areas</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select a product you're not satisfied with", dropdown_options)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        find = st.button("Find Alternatives", use_container_width=True)

    if find:
        product_id = get_id(selected)
        idx = product_ids.index(product_id)
        vec = vectors[idx]
        name = clean_name(selected)
        weak_aspects = [feature_cols[i] for i, s in enumerate(vec) if s < -0.1]

        if not weak_aspects:
            st.success(f"✅ **{name}** already scores well on all 13 aspects!")
        else:
            st.markdown(f"**{name}** needs improvement in:")
            for w in weak_aspects:
                st.markdown(f'<span class="badge-weak">⚠️ {w}</span>', unsafe_allow_html=True)

            gap_scores = []
            for i, pid in enumerate(product_ids):
                if pid == product_id:
                    continue
                other_vec = vectors[i]
                gap_score = np.mean([other_vec[feature_cols.index(a)] for a in weak_aspects])
                gap_scores.append((pid, round(float(gap_score), 4)))
            gap_scores.sort(key=lambda x: x[1], reverse=True)

            st.markdown("<br>**🏆 Top Alternatives**")
            col_list, col_chart = st.columns([1, 1])
            with col_list:
                for rank, (pid, score) in enumerate(gap_scores[:8], 1):
                    pname = clean_name(id_to_label.get(pid, pid))
                    st.markdown(f"""
                    <div class="result-card">
                        <div>
                            <span style='color:#888;font-size:0.78rem;font-weight:600;'>#{rank}</span>
                            <span class="result-name" style='margin-left:8px;'>{pname}</span>
                        </div>
                        <div class="score-badge">📈 {score}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_chart:
                top8 = [(clean_name(id_to_label.get(pid, pid)), score) for pid, score in gap_scores[:8]]
                fig = px.bar(
                    pd.DataFrame(top8, columns=["Product", "Score"]),
                    x="Product", y="Score",
                    color="Score", color_continuous_scale="Greens",
                    labels={"Product": "", "Score": "Gap Score"}
                )
                fig.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_tickangle=-25, font=dict(family="Inter"),
                    margin=dict(t=10, b=10), height=320
                )
                st.plotly_chart(fig, use_container_width=True)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif page == "📖  About":
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
        <h2>Sentiment Intelligence Platform</h2>
        <p>An end-to-end NLP pipeline that transforms raw Amazon product reviews into structured,
        actionable product intelligence. Built as part of IS567 Text Mining at the University of Illinois Urbana-Champaign.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 What it does")
        st.markdown("""
        Instead of relying on a single star rating, this platform extracts **aspect-level sentiment**
        from customer reviews — breaking down exactly what customers love or dislike about a product
        across 13 dimensions like taste, price, packaging, and nutrition.

        The result is a **13-dimensional sentiment vector** per product that powers similarity search,
        gap-fill recommendations, and AI-generated explanations.
        """)

    with col_b:
        st.markdown("### 🛠️ Tech Stack")
        techs = ["PyABSA", "DistilBERT", "Sentence-Transformers", "KMeans",
                 "FAISS", "ChromaDB", "LangChain", "Llama 3.3 70B",
                 "Streamlit", "Groq", "scikit-learn", "VADER"]
        st.markdown("<br>".join([
            "".join([f'<span class="tech-badge">{t}</span>' for t in techs[i:i+3]])
            for i in range(0, len(techs), 3)
        ]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 Team")
    t1, t2, t3, t4 = st.columns(4)

    t1, t2, t3, t4 = st.columns(4)

    team = [
        (t1, "H", "Harsh Patel", "harshnp2"),
        (t2, "A", "Aman Joshi", "abj4"),
        (t3, "S", "Shithil Shetty", "shetty7"),
        (t4, "K", "Kanishka Gupta", "kgupta17"),
    ]

    for col, initial, name, netid in team:
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div class="avatar">{initial}</div>
                <h4>{name}</h4>
                <div class="netid">{netid} · UIUC</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#888; font-size:0.85rem;'>
        IS567 — Text Mining · Spring 2026 · University of Illinois Urbana-Champaign
    </div>
    """, unsafe_allow_html=True)