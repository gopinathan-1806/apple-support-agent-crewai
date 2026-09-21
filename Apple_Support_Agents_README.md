# Apple Support Agent

An AI-powered multi-agent customer support system built with **CrewAI +
Streamlit + OpenAI + SerpApi**.

The application processes a user's query through **three specialized
agents sequentially**:

1.  **Assistant Agent** --- answers using the LLM's existing knowledge.
2.  **Web Search Assistant** --- performs live Google search through
    SerpApi and generates an answer from the retrieved web information.
3.  **Entry Agent** --- combines the query and both answers, saves the
    interaction to a timestamped `.txt` file, and returns the final
    output to the Streamlit UI.

------------------------------------------------------------------------

## Architecture

![Apple Support Agent Flow](docs/apple-support-agent-flow.png)

### End-to-End Flow

``` text
User
  |
  v
Streamlit UI
  |
  v
CrewAI Sequential Process
  |
  +----------------------------+
  |                            |
  v                            v
Agent 1                     Agent 2
Assistant                   Web Search Assistant
  |                            |
  | LLM Knowledge              | SerpApi
  |                            | Live Google Search
  v                            v
Answer 1                     Answer 2
  |                            |
  +-------------+--------------+
                |
                v
          Agent 3
        Entry Agent
                |
                +--> Save Query + Answer 1 + Answer 2
                |    to entries/*.txt
                |
                v
          Streamlit Output
```

------------------------------------------------------------------------

## Key Features

-   **Three-agent CrewAI architecture**
-   **Sequential agent execution**
-   LLM-based answer using existing model knowledge
-   Live web search using **SerpApi**
-   Web-grounded second answer with source URLs
-   Timestamped interaction files
-   Streamlit-based user interface
-   API keys loaded from environment variables
-   No API keys hardcoded in the application
-   Clear separation between LLM knowledge and live web information
-   Single `app.py` implementation

------------------------------------------------------------------------

## Agent Responsibilities

### 1. Assistant Agent

The first agent answers the user's query directly using the LLM's
existing knowledge.

**Responsibilities:** - Understand the user query - Generate an initial
answer - Do not perform web search - Clearly identify the response as
coming from the Assistant Agent

Example:

``` text
Source: Assistant Agent (LLM's existing knowledge)

The response is generated from the model's available knowledge.
```

> Because this agent does not use live search, its answer may not
> reflect the latest information.

------------------------------------------------------------------------

### 2. Web Search Assistant

The second agent performs a live web search using **SerpApi**.

**Flow:**

``` text
User Query
    |
    v
SerpApi
    |
    v
Google Search Results
    |
    v
Relevant Web Results
    |
    v
LLM Analysis
    |
    v
Answer 2 + Sources
```

The application uses:

``` text
https://serpapi.com/search.json
```

The search response is processed from the `organic_results` returned by
SerpApi.

The agent is instructed to base its response on the retrieved web
information rather than silently falling back to the LLM's existing
knowledge when live search fails.

The UI identifies the result as:

``` text
Source: Web Search Assistant (SerpApi)
Search Status: SUCCESS
```

------------------------------------------------------------------------

### 3. Entry Agent

The third agent runs after the first two agents complete.

**Responsibilities:**

-   Receive the original user query
-   Receive Answer 1
-   Receive Answer 2
-   Save the complete interaction to a `.txt` file
-   Return both answers to the Streamlit application

Example file:

``` text
entries/support_entry_20260915_143025.txt
```

Typical contents:

``` text
Apple Support Agent Entry
=========================

Timestamp:
2026-09-15 14:30:25

User Query:
What is the latest iPhone model?

Answer 1:
Source: Assistant Agent (LLM's existing knowledge)
...

Answer 2:
Source: Web Search Assistant (SerpApi)
Search Status: SUCCESS
...
```

------------------------------------------------------------------------

## CrewAI Execution Model

The application uses CrewAI's sequential process:

``` python
Crew(
    agents=self.agents,
    tasks=self.tasks,
    process=Process.sequential,
    verbose=True
)
```

Execution order:

``` text
Agent 1
  |
  v
Agent 2
  |
  v
Agent 3
```

Agent 2 receives the workflow context after Agent 1, and Agent 3 runs
after both answers are available.

------------------------------------------------------------------------

## Project Structure

``` text
apple-support-agent/
│
├── app.py
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── entries/
│   └── support_entry_YYYYMMDD_HHMMSS.txt
│
└── docs/
    └── apple-support-agent-flow.png
```

### Main Files

  File / Directory   Purpose
  ------------------ -------------------------------------------------------
  `app.py`           Complete CrewAI + Streamlit application
  `.env`             Local API credentials and configuration
  `.env.example`     Example environment-variable configuration
  `entries/`         Stores generated support interaction files
  `docs/`            Project documentation assets and architecture diagram
  `README.md`        Project documentation

------------------------------------------------------------------------

## Environment Variables

Create a `.env` file in the project root:

``` env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_openai_model
SERPAPI_API_KEY=your_serpapi_api_key
OUTPUT_DIR=entries
```

### Important

Do **not** commit `.env` to GitHub.

Recommended `.gitignore`:

``` gitignore
.env
.env.*
!.env.example

__pycache__/
*.pyc
.venv/
venv/

entries/*.txt

.DS_Store
```

Keep `.env.example` in the repository with placeholder values only.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd apple-support-agent
```

### 2. Create a virtual environment

``` bash
python3 -m venv .venv
```

Activate it:

### macOS / Linux

``` bash
source .venv/bin/activate
```

### Windows

``` powershell
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If a requirements file is not present, install the required packages:

``` bash
pip install crewai streamlit python-dotenv requests
```

### 4. Configure environment variables

Create `.env`:

``` env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_openai_model
SERPAPI_API_KEY=your_serpapi_api_key
OUTPUT_DIR=entries
```

------------------------------------------------------------------------

## Run the Application

Start Streamlit:

``` bash
streamlit run app.py
```

The application will open in the browser.

Example query:

``` text
What is the latest iPhone model?
```

Other example queries:

``` text
How do I reset my Apple ID password?

What are the key features of the latest iOS version?

How can I check my AirPods warranty?

What is the price of the MacBook Air?

How do I contact Apple Support?
```

------------------------------------------------------------------------

## Streamlit UI

The application provides:

``` text
Apple Support Agent

AI-powered multi-agent support system

Enter your query or task
[ Run Apple Support Agent ]

        ↓

Entry Agent Output

Answer 1
Source: Assistant Agent (LLM's existing knowledge)
...

Answer 2
Source: Web Search Assistant (SerpApi)
Search Status: SUCCESS
...

Entry File
entries/support_entry_YYYYMMDD_HHMMSS.txt
```

The UI intentionally presents the two answers separately so the user can
distinguish between:

-   the model's existing knowledge, and
-   information obtained through live web search.

------------------------------------------------------------------------

## Web Search and Retry Handling

The SerpApi integration includes retry handling for temporary request
failures.

Conceptually:

``` text
Web Search Request
       |
       v
   SerpApi API
       |
   +---+---+
   |       |
Success   Failure
   |       |
   v       v
Answer   Retry
           |
           v
       Retry 2
           |
           v
       Retry 3
           |
           v
       Return actual
       search status
```

The application uses a request timeout and multiple attempts before
reporting the search failure.

This avoids treating a temporary API/network issue as a successful
web-grounded response.

------------------------------------------------------------------------

## Source Transparency

The application deliberately keeps the two answer sources separate.

### Answer 1

``` text
Source: Assistant Agent (LLM's existing knowledge)
```

This indicates that the response was generated from the model's
available knowledge.

### Answer 2

``` text
Source: Web Search Assistant (SerpApi)
Search Status: SUCCESS
```

This indicates that the response was generated after retrieving live web
search results.

When available, the web-grounded response also includes source URLs from
the search results.

------------------------------------------------------------------------

## Why This Architecture?

The project demonstrates a simple but useful multi-agent pattern:

``` text
Knowledge-based Agent
        +
Web Research Agent
        +
Entry / Persistence Agent
        =
Multi-Agent Support Workflow
```

Instead of asking a single LLM to perform every responsibility, each
agent has a clearly defined role.

This makes the workflow easier to understand, extend, and debug.

------------------------------------------------------------------------

## Technologies Used

### AI / Agent Framework

-   CrewAI
-   OpenAI LLM

### Web Search

-   SerpApi
-   Google Search results through SerpApi

### Application

-   Streamlit
-   Python

### Configuration

-   Python `os`
-   `python-dotenv`

### HTTP Integration

-   Python `requests`

------------------------------------------------------------------------

## Security Considerations

API credentials are loaded from environment variables:

``` python
os.getenv("OPENAI_API_KEY")
os.getenv("SERPAPI_API_KEY")
```

They should never be written directly into `app.py`.

### Never commit:

``` text
.env
API keys
Access tokens
Passwords
Private credentials
Generated files containing sensitive customer information
```

If a credential is accidentally exposed in a Git repository,
rotate/revoke it immediately and remove it from the repository history.

------------------------------------------------------------------------

## Current Limitations

-   The first agent relies on the LLM's existing knowledge and does not
    perform live verification.
-   Web answer quality depends on SerpApi search results.
-   The application is currently designed as a
    demonstration/support-agent workflow rather than a production
    customer-support platform.
-   The generated `.txt` files are local application files and are not
    backed by a database.
-   Authentication and user-level access control are not implemented.
-   The application currently uses HTTP API communication for the
    external search service.

------------------------------------------------------------------------

## Possible Future Enhancements

``` text
Current
  |
  +--> Add conversation memory
  |
  +--> Add streaming responses
  |
  +--> Add structured source extraction
  |
  +--> Add evaluation / response quality checks
  |
  +--> Add LangSmith tracing
  |
  +--> Add persistent database storage
  |
  +--> Add authentication
  |
  +--> Containerize with Docker
  |
  +--> Deploy behind HTTPS
```

------------------------------------------------------------------------

## Example End-to-End Execution

For:

``` text
What is the latest iPhone model?
```

The workflow becomes:

``` text
User Query
    |
    v
Streamlit
    |
    v
Assistant Agent
    |
    +--> Answer 1
    |
    v
Web Search Assistant
    |
    +--> SerpApi
    |      |
    |      +--> Google Search
    |      |
    |      +--> Search Results
    |
    +--> Answer 2 + Sources
    |
    v
Entry Agent
    |
    +--> Save interaction
    |
    v
entries/support_entry_YYYYMMDD_HHMMSS.txt
    |
    v
Streamlit
    |
    +--> Answer 1
    |
    +--> Answer 2
    |
    +--> Entry File
```

------------------------------------------------------------------------

## Project Objective

This project demonstrates how **CrewAI sequential agents** can be
combined with an LLM, live web search, and a lightweight persistence
layer to create a practical AI support workflow.

The main focus is not just generating an answer, but demonstrating
**role-based agent execution, external tool integration, source
awareness, and workflow persistence** in a Streamlit application.

------------------------------------------------------------------------

## Flow

<img width="1312" height="1199" alt="Apple support agent" src="https://github.com/user-attachments/assets/3425669c-cccd-4d20-b132-78a6be293533" />

## Results

<img width="1490" height="778" alt="image" src="https://github.com/user-attachments/assets/f16e426e-05ba-45a3-bfbd-47e7d2a8e1e0" />

