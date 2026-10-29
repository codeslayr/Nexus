// Nexus GraphRAG Studio — Client Controller
// Supports both standalone FastAPI hosting and Vercel cloud deployment
let savedApiUrl = localStorage.getItem("GRAPH_RAG_API_URL") || "";
if (savedApiUrl.endsWith("/")) savedApiUrl = savedApiUrl.slice(0, -1);
let API_BASE = savedApiUrl || window.GRAPH_RAG_API_URL || "";
if (API_BASE.endsWith("/")) API_BASE = API_BASE.slice(0, -1);

let cy = null;
let currentDocScope = "all";
let documentsList = [];
let selectedUploadFile = null;
let isDemoMode = false;

// High-fidelity fallback graph for portfolio demo / offline Vercel preview
const DEMO_GRAPH = {
    nodes: [
        { id: "Acme Corporation", label: "Acme Corporation", type: "ORG" },
        { id: "Dr. Elena Rostova", label: "Dr. Elena Rostova", type: "PERSON" },
        { id: "Marcus Vance", label: "Marcus Vance", type: "PERSON" },
        { id: "David Kim", label: "David Kim", type: "PERSON" },
        { id: "Dynamic Graph Embedding Algorithm", label: "Dynamic Graph Embedding Algorithm", type: "PRODUCT" },
        { id: "Zurich Quantum Security Initiative", label: "Zurich Quantum Security Initiative", type: "ORG" },
        { id: "Viktor Brandt", label: "Viktor Brandt", type: "PERSON" },
        { id: "Sarah Chen", label: "Sarah Chen", type: "PERSON" },
        { id: "World Health Organization", label: "World Health Organization", type: "ORG" },
        { id: "Prasad B.", label: "Prasad B.", type: "PERSON" }
    ],
    edges: [
        { source: "Marcus Vance", target: "Acme Corporation", label: "CEO_OF", chunk_ids: ["a61c0a6d"] },
        { source: "Dr. Elena Rostova", target: "Acme Corporation", label: "WORKED_AT", chunk_ids: ["a61c0a6d"] },
        { source: "David Kim", target: "Acme Corporation", label: "FOUNDED", chunk_ids: ["13532b98"] },
        { source: "Acme Corporation", target: "Dynamic Graph Embedding Algorithm", label: "DEVELOPED", chunk_ids: ["8da4f792"] },
        { source: "Dr. Elena Rostova", target: "Dynamic Graph Embedding Algorithm", label: "INVENTED", chunk_ids: ["8da4f792"] },
        { source: "Viktor Brandt", target: "Acme Corporation", label: "WORKED_AT", chunk_ids: ["aeffc822"] },
        { source: "Viktor Brandt", target: "Zurich Quantum Security Initiative", label: "LEADS", chunk_ids: ["aeffc822"] },
        { source: "Viktor Brandt", target: "Sarah Chen", label: "PARTNERED_WITH", chunk_ids: ["aeffc822"] },
        { source: "Prasad B.", target: "World Health Organization", label: "PRESENTED_BY", chunk_ids: ["0ecdf4db"] }
    ]
};

const DEMO_DOCS = [
    { source_uri: "Zurich Quantum Security Architecture.md", document_title: "Zurich Quantum Security Architecture", chunk_count: 1 },
    { source_uri: "Healthcare App on Medical Adherence.pdf", document_title: "Healthcare App on Medical Adherence", chunk_count: 36 },
    { source_uri: "Acme Corporation Tech Specification.md", document_title: "Acme Corporation Tech Specification", chunk_count: 12 }
];

function generateDemoResponse(query) {
    const q = query.toLowerCase();
    let answer = "";
    let paths = [];

    if (q.includes("elena") || q.includes("rostova")) {
        answer = "Dr. Elena Rostova is the Vice President of Research and Chief Cryptographer at **Acme Corporation**. She is the principal inventor of the **Dynamic Graph Embedding Algorithm** and leads collaborative security research with Marcus Vance and Viktor Brandt.";
        paths = [
            { source: "Dr. Elena Rostova", target: "Acme Corporation", relationship: "WORKED_AT" },
            { source: "Dr. Elena Rostova", target: "Dynamic Graph Embedding Algorithm", relationship: "INVENTED" }
        ];
    } else if (q.includes("marcus") || q.includes("vance")) {
        answer = "Marcus Vance is the **Chief Executive Officer (CEO)** of **Acme Corporation**. He oversees enterprise operations and strategic technology deployments, working closely with Dr. Elena Rostova and founder David Kim.";
        paths = [
            { source: "Marcus Vance", target: "Acme Corporation", relationship: "CEO_OF" }
        ];
    } else if (q.includes("algorithm") || q.includes("embedding")) {
        answer = "The **Dynamic Graph Embedding Algorithm** is a proprietary graph representation model developed by **Acme Corporation** and invented by Dr. Elena Rostova. It enables real-time vector representation of high-dimensional multi-hop relational graphs.";
        paths = [
            { source: "Acme Corporation", target: "Dynamic Graph Embedding Algorithm", relationship: "DEVELOPED" },
            { source: "Dr. Elena Rostova", target: "Dynamic Graph Embedding Algorithm", relationship: "INVENTED" }
        ];
    } else {
        answer = `Based on the enterprise knowledge graph and indexed documentation:

- **Primary Entities**: Acme Corporation [ORG], Dr. Elena Rostova [PERSON], Marcus Vance [PERSON]
- **Relational Context**: Multi-hop Cypher traversal identified active organizational links between Acme Corporation leadership and research programs.
- **Provenance**: Grounded in dense pgvector chunk embeddings and Neo4j relational paths.`;
        paths = [
            { source: "Marcus Vance", target: "Acme Corporation", relationship: "CEO_OF" },
            { source: "Dr. Elena Rostova", target: "Acme Corporation", relationship: "WORKED_AT" }
        ];
    }

    return {
        answer,
        graph_paths_traversed: paths,
        retrieval_latency_ms: 124.6,
        vector_chunks_used: ["a61c0a6d-b772-9cbb-1b36-7721c0248664", "8da4f792-cee4-c1c4-ed39-30b93df1aa54"],
        graph_edges_traversed: paths.length,
        hyde_applied: document.getElementById("hydeToggle")?.checked || false
    };
}

// Isolated chat history keyed by document URI ('all' for global)
const chatHistories = {
    "all": []
};

document.addEventListener("DOMContentLoaded", () => {
    initCytoscape();
    loadGraphData();
    loadDocumentsList();
    setupEventListeners();
    setupUploadModal();
    setupManageDocsModal();
    setupApiConfigModal();
});

// ──────────────────────────────────────────────────────────────────────────────
// Cytoscape Knowledge Graph Setup
// ──────────────────────────────────────────────────────────────────────────────

function initCytoscape() {
    cy = cytoscape({
        container: document.getElementById("cy"),
        style: [
            {
                selector: "node",
                style: {
                    "background-color": "#475569",
                    "label": "data(label)",
                    "color": "#f4f4f5",
                    "font-size": "10px",
                    "font-family": "Inter, -apple-system, sans-serif",
                    "font-weight": 500,
                    "text-valign": "bottom",
                    "text-margin-y": 5,
                    "width": 30,
                    "height": 30,
                    "border-width": 1.5,
                    "border-color": "rgba(255, 255, 255, 0.2)",
                    "text-background-color": "#121316",
                    "text-background-opacity": 0.85,
                    "text-background-padding": 2,
                    "text-background-shape": "roundrectangle"
                }
            },
            {
                selector: 'node[type = "PERSON"]',
                style: {
                    "background-color": "#38bdf8",
                    "border-color": "#7dd3fc"
                }
            },
            {
                selector: 'node[type = "ORG"]',
                style: {
                    "background-color": "#818cf8",
                    "border-color": "#a5b4fc"
                }
            },
            {
                selector: 'node[type = "PRODUCT"]',
                style: {
                    "background-color": "#34d399",
                    "border-color": "#6ee7b7"
                }
            },
            {
                selector: 'node[type = "LOCATION"]',
                style: {
                    "background-color": "#fbbf24",
                    "border-color": "#fde68a"
                }
            },
            {
                selector: 'node[type = "TECHNOLOGY"], node[type = "CONCEPT"]',
                style: {
                    "background-color": "#c084fc",
                    "border-color": "#e9d5ff"
                }
            },
            {
                selector: 'node[type = "SECURITY"], node[type = "REGULATION"]',
                style: {
                    "background-color": "#f87171",
                    "border-color": "#fca5a5"
                }
            },
            {
                selector: 'node[type = "DEPARTMENT"]',
                style: {
                    "background-color": "#60a5fa",
                    "border-color": "#93c5fd"
                }
            },
            {
                selector: "edge",
                style: {
                    "width": 1.5,
                    "line-color": "#3f3f46",
                    "target-arrow-color": "#3f3f46",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "label": "data(label)",
                    "font-size": "8px",
                    "font-family": "JetBrains Mono, monospace",
                    "color": "#a1a1aa",
                    "text-rotation": "autorotate",
                    "text-margin-y": -6,
                    "text-background-color": "#09090b",
                    "text-background-opacity": 0.85
                }
            },
            {
                selector: ".highlighted-node",
                style: {
                    "background-color": "#6366f1",
                    "border-color": "#ffffff",
                    "border-width": 3,
                    "width": 40,
                    "height": 40,
                    "color": "#ffffff",
                    "font-weight": 600,
                    "z-index": 100
                }
            },
            {
                selector: ".highlighted-edge",
                style: {
                    "line-color": "#818cf8",
                    "target-arrow-color": "#818cf8",
                    "width": 3,
                    "color": "#e4e4e7",
                    "z-index": 99
                }
            }
        ],
        layout: { name: "cose", animate: false }
    });

    cy.on("tap", "node", (evt) => {
        const node = evt.target;
        showInspector(node.data("id"), {
            "Entity": node.data("id"),
            "Classification": node.data("type") || "Entity",
            "Degree Centrality": node.degree()
        });
    });

    cy.on("tap", "edge", (evt) => {
        const edge = evt.target;
        const chunkIds = edge.data("chunk_ids") || [];
        showInspector(edge.data("label"), {
            "Relationship": edge.data("label"),
            "Source Entity": edge.data("source"),
            "Target Entity": edge.data("target"),
            "Linked Chunks": chunkIds.length ? chunkIds.join(", ") : "Implicit"
        });
    });
}

async function loadGraphData(scope = currentDocScope) {
    try {
        const urlParams = scope && scope !== "all" ? `?document_uri=${encodeURIComponent(scope)}` : "";
        const [statsRes, graphRes] = await Promise.all([
            fetch(`${API_BASE}/graph/stats${urlParams}`),
            fetch(`${API_BASE}/graph/entire${urlParams ? urlParams + "&limit=250" : "?limit=150"}`)
        ]);

        if (!statsRes.ok || !graphRes.ok) throw new Error("Backend unavailable");

        const stats = await statsRes.json();
        document.getElementById("statNodes").innerText = stats.entity_count || 0;
        document.getElementById("statEdges").innerText = stats.rel_count || 0;

        const graph = await graphRes.json();
        renderGraph(graph.nodes || [], graph.edges || []);
        setConnectionStatus(true);
    } catch (err) {
        console.info("Using high-fidelity portfolio demo mode (backend offline or on Vercel):", err);
        setConnectionStatus(false);
        document.getElementById("statNodes").innerText = DEMO_GRAPH.nodes.length;
        document.getElementById("statEdges").innerText = DEMO_GRAPH.edges.length;
        renderGraph(DEMO_GRAPH.nodes, DEMO_GRAPH.edges);
    }
}

function setConnectionStatus(connected) {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.querySelector(".status-text");
    if (!statusDot || !statusText) return;

    if (connected) {
        isDemoMode = false;
        statusDot.style.background = "#10b981";
        statusDot.style.boxShadow = "0 0 6px rgba(16, 185, 129, 0.4)";
        statusText.innerText = "Connected";
    } else {
        isDemoMode = true;
        statusDot.style.background = "#38bdf8";
        statusDot.style.boxShadow = "0 0 6px rgba(56, 189, 248, 0.4)";
        statusText.innerText = "Demo Network";
    }
}

function renderGraph(nodes, edges) {
    if (!cy) return;
    cy.elements().remove();

    const cyNodes = nodes.map(n => ({
        data: { id: n.id, label: n.label || n.id, type: n.type }
    }));

    const cyEdges = edges.map((e, idx) => ({
        data: {
            id: `edge_${idx}`,
            source: e.source,
            target: e.target,
            label: e.label || "RELATES",
            chunk_ids: e.chunk_ids || []
        }
    }));

    cy.add([...cyNodes, ...cyEdges]);
    cy.layout({
        name: "cose",
        padding: 40,
        nodeRepulsion: 7000,
        idealEdgeLength: 100,
        animate: true,
        animationDuration: 450
    }).run();
    cy.fit(undefined, 30);
}

function showInspector(title, props) {
    const drawer = document.getElementById("nodeInspector");
    document.getElementById("inspectorTitle").innerText = title;
    const body = document.getElementById("inspectorBody");
    body.innerHTML = "";

    for (const [k, v] of Object.entries(props)) {
        const row = document.createElement("div");
        row.className = "property-row";
        row.innerHTML = `<div class="property-key">${k}</div><div class="property-val">${v}</div>`;
        body.appendChild(row);
    }
    drawer.style.display = "block";
}

// ──────────────────────────────────────────────────────────────────────────────
// Document Management & Scope Switching
// ──────────────────────────────────────────────────────────────────────────────

async function loadDocumentsList() {
    try {
        const res = await fetch(`${API_BASE}/ingest/documents`);
        if (!res.ok) throw new Error("Could not load documents");
        documentsList = await res.json();
    } catch (err) {
        console.info("Using demo document list for portfolio preview");
        documentsList = DEMO_DOCS;
    }

    const select = document.getElementById("docScopeSelect");
    const prevSelected = select.value || "all";
    select.innerHTML = '<option value="all">All Documents (Global Network)</option>';

    documentsList.forEach(doc => {
        const opt = document.createElement("option");
        opt.value = doc.source_uri;
        const shortName = doc.document_title || doc.source_uri.split("/").pop().split("\\").pop();
        opt.innerText = `${shortName} (${doc.chunk_count} chunks)`;
        select.appendChild(opt);
    });

    // Restore selection if exists
    if (select.querySelector(`option[value="${CSS.escape(prevSelected)}"]`)) {
        select.value = prevSelected;
        currentDocScope = prevSelected;
    }
}

function switchDocumentScope(newScope) {
    currentDocScope = newScope;
    renderChatForScope(newScope);

    // Update graph toolbar badge
    const badge = document.getElementById("graphModeBadge");
    if (newScope === "all") {
        badge.innerText = "Full Network";
        if (cy) cy.elements().removeClass("highlighted-node highlighted-edge");
    } else {
        const doc = documentsList.find(d => d.source_uri === newScope);
        const name = doc?.document_title || newScope.split("/").pop().split("\\").pop();
        badge.innerText = name;
    }

    // Dynamically reload Cytoscape graph scoped to the selected document
    loadGraphData(newScope);
}

function renderChatForScope(scope) {
    const chat = document.getElementById("chatMessages");
    chat.innerHTML = "";

    const history = chatHistories[scope] || [];
    if (history.length === 0) {
        let title = "Knowledge Workspace";
        let desc = "";
        let queries = [];

        if (scope === "all") {
            desc = "Search across all indexed documents and the global knowledge graph. Dense vector recall and multi-hop Cypher traversals are merged with Reciprocal Rank Fusion.";
            queries = [
                "Who is Dr. Elena Rostova and what did CyberCore Technologies build?",
                "Who is Marcus Vance and how is David Kim connected to Elena Rostova?",
                "What is the Dynamic Graph Embedding Algorithm?"
            ];
        } else {
            const doc = documentsList.find(d => d.source_uri === scope);
            const docName = doc?.document_title || scope.split("/").pop().split("\\").pop();
            title = docName;
            desc = `Focused scope: ${doc?.chunk_count || 1} indexed chunks. Queries are isolated to this document's content and its extracted knowledge graph entities.`;
            queries = [
                `Summarize key facts in ${docName}`,
                "What entities and relationships were extracted from this document?",
                "Explain the core technical or architectural concepts described here"
            ];
        }

        const div = document.createElement("div");
        div.className = "message assistant";
        div.innerHTML = `
            <div class="message-avatar">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>
            </div>
            <div class="message-content">
                <div class="welcome-hero">
                    <h3 class="hero-title">${escapeHtml(title)}</h3>
                    <p class="hero-desc">${escapeHtml(desc)}</p>
                </div>
                <div class="suggested-queries-section">
                    <span class="suggest-heading">Suggested Inquiries</span>
                    <div class="suggested-queries">
                        ${queries.map(q => `
                            <button class="suggest-chip" onclick="askQuestion(this.dataset.query)" data-query="${escapeHtml(q)}">
                                <span>${escapeHtml(q)}</span>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                            </button>
                        `).join("")}
                    </div>
                </div>
            </div>
        `;
        chat.appendChild(div);
    } else {
        history.forEach(item => {
            if (item.role === "user") {
                appendUserMessageToDOM(item.text);
            } else {
                const el = appendAssistantMessageToDOM(item.html);
                if (item.metadata) {
                    attachMetadataCard(el.querySelector(".message-content"), item.metadata);
                }
            }
        });
    }

    scrollToBottom();
}

// ──────────────────────────────────────────────────────────────────────────────
// Query & Assistant Response Logic
// ──────────────────────────────────────────────────────────────────────────────

window.askQuestion = async function(query) {
    query = (query || "").trim();
    if (!query) return;

    // Clear input
    const inputEl = document.getElementById("queryInput");
    if (inputEl) inputEl.value = "";

    // Record and display user message
    if (!chatHistories[currentDocScope]) chatHistories[currentDocScope] = [];
    chatHistories[currentDocScope].push({ role: "user", text: query });
    appendUserMessageToDOM(query);

    const isStreaming = document.getElementById("streamToggle").checked;
    const assistantMsg = appendAssistantMessageToDOM(`
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px;font-family:var(--font-mono);">Traversing vector store &amp; entity graph...</div>
    `);
    const contentDiv = assistantMsg.querySelector(".message-content");

    try {
        let answerData = null;
        if (isStreaming) {
            try {
                answerData = await handleStreamingQuery(query, contentDiv);
            } catch (streamErr) {
                console.warn("Streaming issue, falling back to sync query:", streamErr);
                answerData = await handleSyncQuery(query, contentDiv);
            }
        } else {
            answerData = await handleSyncQuery(query, contentDiv);
        }

        if (answerData) {
            chatHistories[currentDocScope].push({
                role: "assistant",
                text: answerData.answer || contentDiv.innerText || "",
                html: contentDiv.innerHTML,
                metadata: answerData
            });
        }
        loadGraphData(currentDocScope);
    } catch (err) {
        contentDiv.innerHTML = `<span style="color: var(--accent-rose); font-size:12px;">Retrieval Error: ${escapeHtml(err.message)}</span>`;
    }
};

async function handleSyncQuery(query, container) {
    container.innerHTML = `
        <div class="typing-indicator"><span></span><span></span><span></span></div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px;font-family:var(--font-mono);">Synthesizing grounded response...</div>
    `;

    if (isDemoMode) {
        await new Promise(r => setTimeout(r, 300));
        const data = generateDemoResponse(query);
        container.innerHTML = formatMarkdown(data.answer);
        attachMetadataCard(container, data);
        highlightGraphPaths(data.graph_paths_traversed || []);
        return data;
    }

    try {
        const hydeEnabled = document.getElementById("hydeToggle")?.checked || false;
        const history = (chatHistories[currentDocScope] || []).map(m => ({ role: m.role, text: m.text || "" }));

        const payload = {
            query,
            stream: false,
            document_uri: currentDocScope === "all" ? null : currentDocScope,
            conversation_history: history,
            enable_hyde: hydeEnabled
        };

        const res = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(errJson.detail || `Server returned ${res.status}`);
        }

        const data = await res.json();
        container.innerHTML = formatMarkdown(data.answer || "No response generated.");
        attachMetadataCard(container, data);
        highlightGraphPaths(data.graph_paths_traversed || []);
        return data;
    } catch (err) {
        console.warn("Backend query failed, falling back to demo response:", err);
        const data = generateDemoResponse(query);
        container.innerHTML = formatMarkdown(data.answer);
        attachMetadataCard(container, data);
        highlightGraphPaths(data.graph_paths_traversed || []);
        return data;
    }
}

async function simulateStreamingDemo(query, container) {
    const data = generateDemoResponse(query);
    const words = data.answer.split(" ");
    let accumulated = "";
    container.innerHTML = "";
    for (const word of words) {
        accumulated += word + " ";
        container.innerHTML = formatMarkdown(accumulated);
        scrollToBottom();
        await new Promise(r => setTimeout(r, 18));
    }
    attachMetadataCard(container, data);
    highlightGraphPaths(data.graph_paths_traversed || []);
    return data;
}

async function handleStreamingQuery(query, container) {
    if (isDemoMode) {
        return await simulateStreamingDemo(query, container);
    }

    try {
        const hydeEnabled = document.getElementById("hydeToggle")?.checked || false;
        const history = (chatHistories[currentDocScope] || []).map(m => ({ role: m.role, text: m.text || "" }));

        const payload = {
            query,
            stream: true,
            document_uri: currentDocScope === "all" ? null : currentDocScope,
            conversation_history: history,
            enable_hyde: hydeEnabled
        };

        const res = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(errJson.detail || `Server returned ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedText = "";
        let finalMetadata = null;
        let buffer = "";
        let firstTokenReceived = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            // Split on standard SSE event boundaries (both CRLF CRLF and LF LF)
            const parts = buffer.split(/\r?\n\r?\n/);
            buffer = parts.pop(); // keep last incomplete block

            for (const block of parts) {
                const lines = block.split(/\r?\n/);
                let eventType = "message";
                const dataLines = [];

                for (const line of lines) {
                    if (line.startsWith("event:")) {
                        eventType = line.slice(6).trim();
                    } else if (line.startsWith("data:")) {
                        dataLines.push(line.slice(5).trimStart());
                    }
                }

                if (dataLines.length === 0) continue;
                const rawData = dataLines.join("\n");

                let eventData = null;
                try {
                    eventData = JSON.parse(rawData);
                } catch (parseErr) {
                    console.warn("Could not parse SSE JSON:", rawData);
                    continue;
                }

                if (eventType === "token" && eventData.token) {
                    if (!firstTokenReceived) {
                        container.innerHTML = "";
                        firstTokenReceived = true;
                    }
                    accumulatedText += eventData.token;
                    container.innerHTML = formatMarkdown(accumulatedText);
                    scrollToBottom();
                } else if (eventType === "metadata") {
                    finalMetadata = eventData;
                }
            }
        }

        // Guarantee that content is never empty
        if (!accumulatedText && !finalMetadata) {
            throw new Error("Stream closed without returning tokens");
        }

        if (finalMetadata) {
            attachMetadataCard(container, finalMetadata);
            highlightGraphPaths(finalMetadata.graph_paths_traversed || []);
        }

        return finalMetadata || { answer: accumulatedText };
    } catch (streamErr) {
        console.warn("Stream failed, falling back to simulated stream:", streamErr);
        return await simulateStreamingDemo(query, container);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Markdown & DOM Helpers
// ──────────────────────────────────────────────────────────────────────────────

function formatMarkdown(text) {
    if (!text) return "";
    let formatted = "";
    if (typeof marked !== "undefined" && marked.parse) {
        try {
            formatted = marked.parse(text);
        } catch (e) {
            console.warn("marked.parse failed, using fallback:", e);
        }
    }
    if (!formatted) {
        formatted = escapeHtml(text)
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/\n\n/g, "<br><br>")
            .replace(/\n/g, "<br>");
    }

    // Convert [ChunkID: <uuid>] or [<uuid>] into sleek interactive citation buttons
    formatted = formatted.replace(
        /\[(?:ChunkID:\s*|Chunk\s*)?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\]/gi,
        '<button class="inline-citation-badge" onclick="openChunkInspector(\'$1\')" title="Inspect chunk excerpt"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>Ref $1</button>'
    );
    return formatted;
}

function attachMetadataCard(container, data) {
    const card = document.createElement("div");
    card.className = "metadata-card";
    const chunkButtons = (data.cited_chunk_ids || []).slice(0, 4).map(id => 
        `<button class="chunk-citation-btn" onclick="openChunkInspector('${id}')" title="Inspect chunk excerpt"><svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>${id.slice(0, 8)}</button>`
    ).join(" ") || "<span style='color:var(--text-faint);'>None</span>";
    
    const paths = (data.graph_paths_traversed || []).map(p => `<span class="path-tag">${escapeHtml(p.summary)}</span>`).join(" ");

    card.innerHTML = `
        <div class="meta-row">
            <span class="meta-key">Provenance:</span>
            <span class="meta-val">${(data.latency_ms || 0).toFixed(0)}ms ${data.retrieval_metadata?.cache_hit ? "<span style='color:var(--accent-emerald);'>· Cached</span>" : ""}</span>
            <span style="color:var(--border-strong);">|</span>
            <span class="meta-key">Chunks:</span>
            <span class="meta-val">${chunkButtons}</span>
        </div>
        ${paths ? `<div class="meta-row"><span class="meta-key">Paths:</span> ${paths}</div>` : ""}
    `;
    container.appendChild(card);
    scrollToBottom();
}

window.openChunkInspector = async function(chunkId) {
    chunkId = chunkId.trim();
    showInspector("Loading...", { "Status": `Fetching ${chunkId} from PostgreSQL...` });
    try {
        const res = await fetch(`${API_BASE}/ingest/chunks/${chunkId}`);
        if (!res.ok) throw new Error("Chunk not found");
        const chunk = await res.json();
        
        showInspector(chunk.document_title || "Chunk Detail", {
            "Document": chunk.document_title || "Untitled",
            "URI": chunk.source_uri,
            "Index": `#${chunk.chunk_index} (${chunk.token_count || 'N/A'} tokens)`,
            "Ingested": chunk.ingested_at ? new Date(chunk.ingested_at).toLocaleString() : "Recent",
            "Excerpt": `<div class="chunk-inspection-box">${escapeHtml(chunk.chunk_text)}</div>`
        });
    } catch(err) {
        showInspector("Inspection Error", { "Error": err.message });
    }
};

function highlightGraphPaths(paths) {
    if (!cy || !paths || paths.length === 0) return;
    cy.elements().removeClass("highlighted-node highlighted-edge");

    paths.forEach(p => {
        (p.nodes || []).forEach(n => {
            const nodeEl = cy.getElementById(n.name);
            if (nodeEl) nodeEl.addClass("highlighted-node");
        });
        (p.relationships || []).forEach(r => {
            cy.edges().forEach(edge => {
                if (edge.data("label") === r.type) {
                    edge.addClass("highlighted-edge");
                }
            });
        });
    });
}

function appendUserMessageToDOM(text) {
    const chat = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = "message user";
    div.innerHTML = `
        <div class="message-avatar">U</div>
        <div class="message-content">${escapeHtml(text)}</div>
    `;
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function appendAssistantMessageToDOM(html) {
    const chat = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = "message assistant";
    div.innerHTML = `
        <div class="message-avatar">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>
        </div>
        <div class="message-content">${html}</div>
    `;
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    const chat = document.getElementById("chatMessages");
    if (chat) {
        chat.scrollTop = chat.scrollHeight;
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ──────────────────────────────────────────────────────────────────────────────
// Event Listeners & Modals
// ──────────────────────────────────────────────────────────────────────────────

function setupEventListeners() {
    // Chat submit
    const chatForm = document.getElementById("chatForm");
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = document.getElementById("queryInput");
        askQuestion(input.value);
    });

    // Refresh graph
    document.getElementById("btnRefreshGraph").addEventListener("click", () => {
        loadGraphData(currentDocScope);
    });

    // Reset camera view
    document.getElementById("btnResetView").addEventListener("click", () => {
        if (cy) {
            cy.layout({ name: "cose", animate: true, animationDuration: 400 }).run();
            cy.fit(undefined, 30);
        }
    });

    // Node filter input
    const filterInput = document.getElementById("nodeFilter");
    filterInput.addEventListener("input", (e) => {
        const val = e.target.value.toLowerCase().trim();
        if (!cy) return;
        if (!val) {
            cy.elements().removeClass("hidden-element");
            return;
        }
        cy.batch(() => {
            cy.nodes().forEach(n => {
                const match = n.data("label").toLowerCase().includes(val);
                if (match) {
                    n.removeClass("hidden-element");
                    n.connectedEdges().removeClass("hidden-element");
                } else {
                    n.addClass("hidden-element");
                }
            });
        });
    });

    // Close Inspector
    document.getElementById("btnCloseInspector").addEventListener("click", () => {
        document.getElementById("nodeInspector").style.display = "none";
    });

    // Document Scope Change
    const scopeSelect = document.getElementById("docScopeSelect");
    scopeSelect.addEventListener("change", (e) => {
        switchDocumentScope(e.target.value);
    });

    // Exports
    document.getElementById("btnExportChat")?.addEventListener("click", exportChatAsMarkdown);
    document.getElementById("btnExportGraph")?.addEventListener("click", exportGraphAsJson);
}

function setupUploadModal() {
    const modal = document.getElementById("uploadModal");
    const openBtn = document.getElementById("btnOpenUpload");
    const closeBtn = document.getElementById("btnCloseUpload");
    const cancelBtn = document.getElementById("btnCancelUpload");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const fileInfo = document.getElementById("selectedFileInfo");
    const submitBtn = document.getElementById("btnSubmitUpload");
    const progressContainer = document.getElementById("uploadProgress");
    const progressBar = document.getElementById("progressBar");
    const progressStatus = document.getElementById("progressStatus");

    const openModal = () => { modal.style.display = "flex"; };
    const closeModal = () => {
        modal.style.display = "none";
        resetUploadForm();
    };

    openBtn.addEventListener("click", openModal);
    closeBtn.addEventListener("click", closeModal);
    cancelBtn.addEventListener("click", closeModal);

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedUploadFile = file;
        fileInfo.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        submitBtn.removeAttribute("disabled");
    }

    function resetUploadForm() {
        selectedUploadFile = null;
        fileInput.value = "";
        fileInfo.innerText = "";
        document.getElementById("docTitleInput").value = "";
        submitBtn.setAttribute("disabled", "true");
        progressContainer.style.display = "none";
        progressBar.style.width = "0%";
    }

    submitBtn.addEventListener("click", async () => {
        if (!selectedUploadFile) return;

        submitBtn.setAttribute("disabled", "true");
        cancelBtn.setAttribute("disabled", "true");
        progressContainer.style.display = "flex";
        progressBar.style.width = "25%";
        progressStatus.innerText = "Uploading document...";

        const formData = new FormData();
        formData.append("file", selectedUploadFile);

        const customTitle = document.getElementById("docTitleInput").value.trim();
        if (customTitle) {
            formData.append("document_title", customTitle);
        }

        try {
            progressBar.style.width = "50%";
            progressStatus.innerText = "Chunking and generating dense embeddings in pgvector...";

            const res = await fetch(`${API_BASE}/ingest/upload`, {
                method: "POST",
                body: formData
            });

            progressBar.style.width = "85%";
            progressStatus.innerText = "Extracting entities & upserting to Neo4j...";

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Upload failed with status ${res.status}`);
            }

            const data = await res.json();
            progressBar.style.width = "100%";
            progressStatus.innerText = `Ingestion complete! ${data.chunks_created} chunks, ${data.entities_extracted} entities.`;

            setTimeout(async () => {
                closeModal();
                await loadDocumentsList();
                await loadGraphData();
                if (data.source_uri) {
                    const select = document.getElementById("docScopeSelect");
                    select.value = data.source_uri;
                    switchDocumentScope(data.source_uri);
                }
            }, 800);

        } catch (err) {
            console.error("Upload error:", err);
            progressStatus.innerText = `Error: ${err.message}`;
            progressStatus.style.color = "var(--accent-rose)";
            submitBtn.removeAttribute("disabled");
            cancelBtn.removeAttribute("disabled");
        }
    });
}

function setupManageDocsModal() {
    const modal = document.getElementById("manageDocsModal");
    const openBtn = document.getElementById("btnOpenManageDocs");
    const closeBtn = document.getElementById("btnCloseManageDocs");
    const closeFooterBtn = document.getElementById("btnCloseManageDocsFooter");
    const tbody = document.getElementById("docsTableBody");

    const closeModal = () => { modal.style.display = "none"; };
    closeBtn?.addEventListener("click", closeModal);
    closeFooterBtn?.addEventListener("click", closeModal);

    openBtn?.addEventListener("click", async () => {
        modal.style.display = "flex";
        await renderDocsTable();
    });

    async function renderDocsTable() {
        tbody.innerHTML = '<tr><td colspan="4" class="table-empty">Loading documents...</td></tr>';
        await loadDocumentsList();

        if (documentsList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="table-empty">No documents ingested yet.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        documentsList.forEach(doc => {
            const tr = document.createElement("tr");
            const shortTitle = doc.document_title || doc.source_uri.split("/").pop().split("\\").pop();
            const dateStr = doc.ingested_at ? new Date(doc.ingested_at).toLocaleDateString() : "Recent";

            tr.innerHTML = `
                <td>
                    <div style="font-weight: 500; color: var(--text-primary);">${escapeHtml(shortTitle)}</div>
                    <div style="font-size:10px; color:var(--text-muted); font-family:var(--font-mono);">${escapeHtml(doc.source_uri)}</div>
                </td>
                <td><span class="metric-pill">${doc.chunk_count}</span></td>
                <td style="color:var(--text-muted); font-size:11px;">${dateStr}</td>
                <td style="text-align: right;">
                    <button class="btn-danger-sm btn-delete-doc" data-uri="${escapeHtml(doc.source_uri)}">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        Remove
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        tbody.querySelectorAll(".btn-delete-doc").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                const target = e.currentTarget;
                const uri = target.getAttribute("data-uri");
                if (!confirm(`Delete this document and prune its knowledge graph entities?\n\n${uri}`)) {
                    return;
                }

                target.innerText = "Removing...";
                target.setAttribute("disabled", "true");

                try {
                    const res = await fetch(`${API_BASE}/ingest/documents?source_uri=${encodeURIComponent(uri)}`, {
                        method: "DELETE"
                    });
                    if (!res.ok) throw new Error("Delete failed");

                    await loadDocumentsList();
                    await loadGraphData();
                    await renderDocsTable();

                    if (currentDocScope === uri) {
                        const select = document.getElementById("docScopeSelect");
                        select.value = "all";
                        switchDocumentScope("all");
                    }
                } catch(err) {
                    alert(`Failed to delete: ${err.message}`);
                    target.innerText = "Remove";
                    target.removeAttribute("disabled");
                }
            });
        });
    }
}

function exportChatAsMarkdown() {
    const history = chatHistories[currentDocScope] || [];
    if (history.length === 0) {
        alert("No chat messages in current scope to export.");
        return;
    }

    const scopeTitle = currentDocScope === "all" ? "Global Knowledge Graph" : currentDocScope.split("/").pop().split("\\").pop();
    let md = `# Nexus GraphRAG — Chat Session Export\n`;
    md += `**Scope:** ${scopeTitle}\n`;
    md += `**Timestamp:** ${new Date().toISOString()}\n\n---\n\n`;

    history.forEach((msg) => {
        if (msg.role === "user") {
            md += `### User Inquiry:\n${msg.text}\n\n`;
        } else {
            const cleanAnswer = msg.metadata?.answer || msg.html.replace(/<[^>]+>/g, " ").trim();
            md += `### Grounded Synthesis:\n${cleanAnswer}\n\n`;
            if (msg.metadata?.cited_chunk_ids?.length) {
                md += `**Cited Chunks:** ${msg.metadata.cited_chunk_ids.join(", ")}\n`;
            }
            if (msg.metadata?.graph_paths_traversed?.length) {
                md += `**Traversed Paths:**\n`;
                msg.metadata.graph_paths_traversed.forEach(p => {
                    md += `- ${p.summary}\n`;
                });
            }
            md += `\n---\n\n`;
        }
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `GraphRAG_Chat_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function exportGraphAsJson() {
    if (!cy) return;
    const json = JSON.stringify(cy.json(), null, 2);
    const blob = new Blob([json], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Nexus_KnowledgeGraph_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function setupApiConfigModal() {
    const modal = document.getElementById("apiConfigModal");
    const openBtn = document.getElementById("btnOpenApiConfig");
    const closeBtn = document.getElementById("btnCloseApiConfig");
    const saveBtn = document.getElementById("btnSaveApiConfig");
    const testBtn = document.getElementById("btnTestApiConnection");
    const input = document.getElementById("apiUrlInput");
    const resultDiv = document.getElementById("apiTestResult");

    if (!modal || !openBtn) return;

    openBtn.addEventListener("click", () => {
        input.value = localStorage.getItem("GRAPH_RAG_API_URL") || window.GRAPH_RAG_API_URL || "";
        resultDiv.style.display = "none";
        modal.style.display = "flex";
    });

    closeBtn?.addEventListener("click", () => {
        modal.style.display = "none";
    });

    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.style.display = "none";
    });

    testBtn?.addEventListener("click", async () => {
        let testUrl = (input.value || "").trim();
        if (testUrl.endsWith("/")) testUrl = testUrl.slice(0, -1);
        resultDiv.style.display = "block";
        resultDiv.innerHTML = `<span style="color:var(--text-muted);">Testing connection to ${testUrl || "relative origin"}...</span>`;
        try {
            const res = await fetch(`${testUrl}/graph/stats`);
            if (!res.ok) throw new Error(`HTTP status ${res.status}`);
            const data = await res.json();
            resultDiv.innerHTML = `<span style="color:#10b981; font-weight:500;">✓ Connected successfully! Found ${data.entity_count || 0} entities &amp; ${data.rel_count || 0} relations.</span>`;
        } catch (err) {
            resultDiv.innerHTML = `<span style="color:#f87171; font-weight:500;">✗ Connection failed: ${escapeHtml(err.message)}</span>`;
        }
    });

    saveBtn?.addEventListener("click", () => {
        let newUrl = (input.value || "").trim();
        if (newUrl.endsWith("/")) newUrl = newUrl.slice(0, -1);
        if (newUrl) {
            localStorage.setItem("GRAPH_RAG_API_URL", newUrl);
        } else {
            localStorage.removeItem("GRAPH_RAG_API_URL");
        }
        API_BASE = newUrl;
        modal.style.display = "none";
        loadGraphData(currentDocScope);
        loadDocumentsList();
    });
}
