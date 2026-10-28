# Nexus GraphRAG — Vercel Deployment & Resume Guide

This guide explains how to host the **Nexus GraphRAG Studio** on Vercel and showcase it on your resume and portfolio.

---

## 🚀 1. Deploying Frontend to Vercel (2-Minute Setup)

### Option A: Direct Vercel Deployment via Git (Recommended)

1. **Push your repository** to GitHub.
2. In [Vercel Dashboard](https://vercel.com/new), click **Add New Project** and select your GitHub repo.
3. In **Project Settings**:
   - **Framework Preset**: `Other`
   - **Root Directory**: `web` (or leave default with `vercel.json` in root)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
4. **Environment / API Connection**:
   - To connect to your deployed backend (e.g. on Railway, Render, or Fly.io), edit `vercel.json` rewrites or add this script in `web/index.html` right before `app.js`:
     ```html
     <script>
       // Point to your live backend API
       window.GRAPH_RAG_API_URL = "https://your-backend.railway.app";
     </script>
     ```
5. Click **Deploy**. Your Vercel frontend is live!

---

## 🌐 2. Hosting the FastAPI Backend (Free Tier Options)

Because GraphRAG connects to PostgreSQL (pgvector) and Neo4j, host the backend container on any of these platforms:

| Provider | Setup | Best For |
| :--- | :--- | :--- |
| **Railway** | Deploy via Dockerfile / GitHub repo | Easiest (free credits, zero config) |
| **Render** | Create "Web Service" from repo | 1-click Python setup |
| **Fly.io** | `fly launch` via CLI | Low latency, global edge |

**Managed Databases (Free Tiers):**
- **Neo4j AuraDB**: Free cloud instance at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/)
- **Supabase / Neon**: Free PostgreSQL with pgvector at [supabase.com](https://supabase.com) or [neon.tech](https://neon.tech)

---

## 💼 3. Adding Nexus GraphRAG to Your Resume

Here is a high-impact, ATS-optimized bullet-point section ready for your resume:

### **Enterprise GraphRAG Studio — Hybrid Vector & Knowledge Graph Engine**
*Full-Stack AI Engineer / Applied LLM System Project*

- **Architecture**: Designed and implemented an enterprise-grade hybrid retrieval-augmented generation (GraphRAG) system fusing **PostgreSQL (pgvector)** dense semantic search with **Neo4j** multi-hop Cypher entity graph traversals via **Reciprocal Rank Fusion (RRF)**.
- **Reasoning & Accuracy**: Integrated **Hypothetical Document Embeddings (HyDE)** and conversational coreference rewriting to resolve multi-turn pronoun ambiguities, increasing contextual retrieval precision.
- **Knowledge Pipeline**: Engineered an automated document ingestion pipeline supporting PDF, DOCX, and Markdown; built asynchronous chunking, Gemini 1.5 embedding, and automated entity/relationship extraction with graph garbage collection.
- **UI / UX**: Architected a production-grade, minimalist dark-mode studio featuring an interactive **Cytoscape.js** knowledge graph explorer, dynamic document scope isolation, and token-level streaming with verifiable chunk citations.
- **Tech Stack**: Python, FastAPI, PostgreSQL (`pgvector`), Neo4j (APOC, Cypher), LangChain, Google Gemini, Cytoscape.js, Modern CSS (Vercel/Linear design system).

