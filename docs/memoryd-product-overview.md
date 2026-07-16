# memoryd

### Product Overview for Marketing and Communications

---

## What memoryd Is

memoryd is a memory system for AI agents. It gives every agent a persistent, structured memory that grows with use, stays organized automatically, and can be shared with other people or other agents. Agents built on memoryd run locally on the user's machine, accumulate knowledge over time, and remain available across sessions, tools, and interfaces.

---

## Memory Architecture

memoryd stores, organizes, and retrieves memory in four layers working together.

### 1. Novelty Gate: Filtering at Ingestion

Every message that passes through memoryd hits the novelty gate first. The gate scores how new and meaningful a piece of information is relative to what the agent already knows. Filler, repetition, and low-signal content are dropped. Decisions, preferences, facts, and plans are saved.

This keeps memory clean and precise regardless of how much the agent is used.

### 2. Consolidation: Synthesis Over Time

Raw memories are periodically grouped by topic and distilled into structured knowledge blocks. Each consolidated memory has four fields:

- **Topic path** — a hierarchical label such as "product/roadmap/Q3" that organizes memory by domain
- **Title** — a short noun phrase naming what the memory is about
- **Body** — the key facts and context, deduped and distilled from the raw exchanges
- **Foresight** — forward-looking notes, such as "this decision will need revisiting after launch"

Consolidation is not summarization. The raw memories are preserved. The consolidated form is what gets surfaced at recall time, with the full detail available underneath.

### 3. Graph Memory: Contradiction Detection and Resolution

memoryd maintains a knowledge graph over consolidated memories. When new information conflicts with something already stored, the system surfaces the contradiction rather than silently overwriting.

The interface is similar to a git log. Each memory has a version history. When a contradiction is detected, both versions are shown with a diff view, the timestamps and sources of each version are visible, and the user can resolve it explicitly by choosing which version is current, merging them, or marking one as superseded.

This means memory stays accurate at scale. Old decisions do not silently persist when newer ones replace them. The history of what changed and when is always visible.

### 4. Hybrid Retrieval with Neural Reranking

At recall time, three search methods run in parallel:

- **Keyword search** for exact terms, names, and project codes
- **Semantic search** using vector embeddings, which finds the right memory even when the user phrases the question differently than the original text
- **Neural reranking** that scores the top candidates against the full context of the current question

Results from all three are merged and ranked. The agent surfaces the most genuinely relevant memories, not just the most lexically similar ones.

---

## The Agent Harness

The agent harness is the environment that runs agents locally on the user's machine. The goal is for users to build long-running personal agents that accumulate knowledge over time, stay available indefinitely, and can be shared with other people or connected to other agents.

Agents are persistent processes. They are not reset between sessions. They run in the background, maintain their memory continuously, and are available whenever the user wants to talk.

### Interfaces

Users can talk to their agents through three interfaces simultaneously:

**Chat** is the conversational interface. Users ask questions, share information, make requests, and the agent responds with full access to its memory.

**File system (FUSE mount)** makes the agent's memory appear as real files and folders on the computer. Users can browse, read, and edit memories directly in their file manager, the same way they would navigate documents. This also makes memories inspectable and portable without any special tooling.

**MCP protocol** is the standard AI tool interface that lets any MCP-compatible tool or service communicate with the agent's memory. Developer tools, AI coding assistants, and third-party agent frameworks can all read from and write to the agent's memory through this interface.

### Skills

Agents support skills, which are predefined capabilities the agent can invoke. Skills let the agent go beyond answering questions and actually do things: run a search, call an API, send a message, generate a document, or interact with a connected service. Users can install skills from a library or define their own.

### MCP Tool Support

Agents are MCP-native. Any MCP server can be connected to an agent, giving it access to external tools and services. Calendar access, code execution, web browsing, database queries, and any other MCP-compatible capability can be wired in and the agent will use them as needed based on what the user asks.

---

## Knowledge Bases and the /learn Command

Users can point their agent at any source of knowledge and the agent will read, process, and remember it.

The `/learn` command accepts:

- A URL pointing to a web page, article, or documentation site
- A document file (PDF, Word, text, markdown)
- A folder of files
- A plain text block pasted directly into the chat

The agent reads the source, extracts what is meaningful, and stores the learnings in memory alongside everything it has learned from direct conversations. Learnings from documents and learnings from chat are stored in the same memory space and retrieved together at recall time, so the agent can draw on both without the user having to specify which source to consult.

Over time, a knowledge base grows naturally. Users can continue adding sources whenever they encounter something relevant, and the consolidation engine organizes new learnings into the existing structure rather than creating a separate pile of documents.

Examples of what users build with this:

- An agent trained on a company's internal documentation, product specs, and meeting notes
- A research agent that has read dozens of papers, articles, and books on a domain
- A personal agent that knows the user's entire note archive, email summaries, and project history
- A team agent that accumulates learnings from every team member's contributions over time

---

## Creating an Agent

Creating an agent takes minutes.

The user gives the agent a name and a purpose. "Strategy Agent." "Research Partner." "Q4 Planning." "My Personal Assistant." From that point, the agent starts building memory immediately from the first conversation. Nothing else is required to get started.

Over time, users can:

- Feed the agent documents or links using `/learn`
- Give it explicit facts to remember ("My company's fiscal year ends in March")
- Install skills to extend what it can do
- Connect MCP tools to give it access to external services
- Promote memories to shared scope to make them visible to other agents or collaborators

### Agent Types Users Commonly Build

**Personal assistant agent** that knows the user's working style, priorities, communication preferences, and ongoing commitments.

**Project agent** dedicated to a specific initiative, containing every decision, open question, and milestone. Useful for briefing new team members instantly by sharing its memory.

**Research agent** that accumulates knowledge on a domain across papers, articles, conversations, and documents. Grows into a genuine expert over months of use.

**Team agent** shared across an organization, building a collective memory of decisions and context that every team member can access.

---

## Sharing Agents and Agent-to-Agent Communication

### Sharing With People

Every memory in memoryd has a visibility scope: private (only the owning agent sees it) or shared (accessible to other agents or people the user designates).

When a user wants to give a colleague access to their agent's knowledge, they promote the relevant memories to a shared scope and grant the colleague access to that scope. The colleague's own agent can then read those memories and answer questions grounded in them.

The colleague does not get access to the user's private conversations or private memories. They get access to a defined set of shared knowledge, which their agent surfaces through their own interface in their own context.

This is how teams build a shared knowledge base without meetings, status updates, or manual documentation. Decisions made by one team member's agent become available to every agent in the team scope automatically.

### Agent-to-Agent Communication

Agents communicate through shared memory rather than direct API calls. When two agents share a memory scope, they are both reading from and writing to the same knowledge pool. There is no coupling between the agents themselves. Either agent can be replaced with a different model or configuration and the shared memory persists unchanged.

A concrete example: a strategy agent accumulates six months of product direction, competitive positioning, and key decisions. A content agent is brought in to write marketing copy. The strategy agent's relevant memories are promoted to a shared scope. The content agent reads them and now writes copy grounded in the real strategy, without a single briefing session.

This pattern scales to large teams. Multiple agents with different specializations can all draw from the same shared knowledge base, each contributing their own learnings back to it, creating a collective intelligence that no individual agent or team member maintains alone.

---

## What Users Do With Their Agents

### Understand

Users ask their agents questions about their own history, projects, and goals and get direct answers grounded in everything the agent knows.

"What did we decide about the pricing strategy in March?"
"What are the open risks on the Phoenix project?"
"Summarize everything you know about our competitor's product direction."

### Plan

Users use their agents as thinking partners who never lose track of prior decisions. Planning sessions pick up exactly where they left off, with all prior context automatically present.

"Continue the Q4 planning we were doing last month. What were the three open questions?"
"Given everything you know about the team, what risks should I flag for the board?"

### Take Action

Users ask their agents to produce work grounded in real context. Because the agent knows the user's actual situation, what it produces is specific and useful rather than generic.

"Write a briefing on the competitive landscape using everything you know about our positioning."
"Draft a message to the client about the delay. You know the relationship history."
"Prepare a project status summary with the key milestones, current state, and open decisions."

---

## Privacy and Data Ownership

The agent runs on the user's hardware. The memory database lives on the user's machine or their own server. No data leaves the local environment unless the user explicitly shares a memory scope.

Users can delete any memory at any time. They can export the entire memory database in open formats. They control visibility at the individual memory level.

---

## Feature Summary

| Feature | Description |
|---|---|
| Novelty gate | Filters incoming content so only meaningful information is stored |
| Consolidation engine | Groups and distills memories into structured knowledge over time |
| Graph memory | Tracks contradictions with version history, visible like a git log |
| Hybrid retrieval | Keyword search, semantic search, and neural reranking run in parallel |
| Long-running local agents | Persistent processes that accumulate knowledge and never reset |
| FUSE file system | Agent memory browsable as real files and folders |
| MCP protocol | Connect any MCP-compatible tool or service to the agent |
| Skills | Predefined capabilities the agent can invoke to take action |
| /learn command | Point the agent at any URL, document, or file to build a knowledge base |
| Shared memory scopes | Share specific knowledge with colleagues or connect agents together |
| Agent-to-agent communication | Agents collaborate through shared memory pools, not direct coupling |
| Local-first privacy | All data stays on the user's hardware by default |
