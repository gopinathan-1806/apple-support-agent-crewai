import os
import json
from pathlib import Path
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "entries"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def perform_web_search(query: str) -> dict:
    """
    Perform a live Google search through SerpApi.

    Retries transient request failures/timeouts up to 3 times.
    A successful response is required before web-grounded content
    is passed to Agent 2.
    """

    if not SERPAPI_API_KEY:
        return {
            "status": "FAILED",
            "error": "SERPAPI_API_KEY is not configured.",
            "results": [],
            "attempts": 0,
        }

    max_attempts = 3
    timeout_seconds = 45
    last_error = ""

    for attempt in range(1, max_attempts + 1):

        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": SERPAPI_API_KEY,
                },
                timeout=timeout_seconds,
            )

            if response.status_code >= 400:
                return {
                    "status": "FAILED",
                    "error": (
                        f"SerpApi HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    ),
                    "results": [],
                    "attempts": attempt,
                }

            data = response.json()

            results = []

            for item in data.get("organic_results", [])[:5]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )

            if not results:
                return {
                    "status": "NO_RESULTS",
                    "error": "SerpApi returned no organic search results.",
                    "results": [],
                    "attempts": attempt,
                }

            return {
                "status": "SUCCESS",
                "error": "",
                "results": results,
                "attempts": attempt,
            }

        except requests.exceptions.Timeout:
            last_error = (
                f"Request timed out after {timeout_seconds} seconds "
                f"(attempt {attempt}/{max_attempts})."
            )

        except requests.exceptions.ConnectionError as exc:
            last_error = (
                f"Connection error (attempt {attempt}/{max_attempts}): {exc}"
            )

        except requests.RequestException as exc:
            last_error = (
                f"Request error (attempt {attempt}/{max_attempts}): {exc}"
            )

        except ValueError as exc:
            last_error = f"Invalid JSON response from SerpApi: {exc}"
            break

    return {
        "status": "FAILED",
        "error": (
            f"SerpApi search failed after {max_attempts} attempts. "
            f"{last_error}"
        ),
        "results": [],
        "attempts": max_attempts,
    }


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the live web using SerpApi.
    Returns structured search results with status, title, URL and snippet.
    """
    result = perform_web_search(query)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("save_entry_file")
def save_entry_file(query: str, answer_1: str, answer_2: str) -> str:
    """Save the customer query and both agent answers into a timestamped text file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = OUTPUT_DIR / f"support_entry_{timestamp}.txt"

    content = (
        "APPLE SUPPORT AGENT ENTRY\n"
        "============================\n\n"
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "USER QUERY\n"
        "----------\n"
        f"{query}\n\n"
        "ANSWER 1\n"
        "--------\n"
        "Source: Assistant Agent (LLM's existing knowledge)\n\n"
        f"{answer_1}\n\n"
        "ANSWER 2\n"
        "--------\n"
        "Source: Web Search Assistant (SerpApi)\n"
        "Search status is included in the answer below.\n\n"
        f"{answer_2}\n"
    )

    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def format_search_results(search_data: dict) -> str:
    """Convert SerpApi results into clean context for Agent 2."""

    status = search_data.get("status", "FAILED")
    results = search_data.get("results", [])
    error = search_data.get("error", "")

    lines = [
        f"WEB SEARCH STATUS: {status}",
        f"WEB SEARCH ATTEMPTS: {search_data.get('attempts', 0)}",
    ]

    if error:
        lines.append(f"WEB SEARCH ERROR: {error}")

    if not results:
        lines.append("WEB SEARCH RESULTS: None")
        return "\n".join(lines)

    lines.append("\nWEB SEARCH RESULTS:")

    for index, item in enumerate(results, 1):
        lines.extend(
            [
                f"{index}. {item.get('title', 'Untitled')}",
                f"URL: {item.get('url', '')}",
                f"Snippet: {item.get('snippet', '')}",
                "",
            ]
        )

    return "\n".join(lines)


def validate_environment():
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not SERPAPI_API_KEY:
        missing.append("SERPAPI_API_KEY")

    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.info(
            "Create a .env file beside app.py. "
            "Never hard-code API keys."
        )
        st.stop()


def build_crew(query: str, search_data: dict):
    """
    Build exactly three sequential CrewAI agents.

    Agent 1:
        Answers from its own LLM knowledge.

    Agent 2:
        Answers only from the live SerpApi results supplied to it.
        The web_search tool is also available to the agent for an additional
        search if required.

    Agent 3:
        Saves and returns both answers.
    """

    llm = LLM(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

    search_context = format_search_results(search_data)

    # ---------------------------------------------------------
    # AGENT 1 — Assistant
    # ---------------------------------------------------------
    assistant = Agent(
        role="Assistant",
        goal=(
            "Answer the user's query directly using your existing LLM knowledge. "
            "Do not use web search."
        ),
        backstory=(
            "You are the first-line customer support assistant. "
            "Provide a clear and useful answer using your existing knowledge. "
            "Do not claim that you searched the web."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # ---------------------------------------------------------
    # AGENT 2 — Web Search Assistant
    # ---------------------------------------------------------
    web_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Answer the user's query using live SerpApi web-search results. "
            "Do not use your own knowledge when web-search results are unavailable."
        ),
        backstory=(
            "You are a customer support web research specialist. "
            "You must ground your answer in the supplied live SerpApi results. "
            "If the search status is FAILED or NO_RESULTS, clearly report that "
            "web research was unsuccessful. Never invent web results and never "
            "pretend that an answer from your own knowledge came from the web."
        ),
        tools=[web_search],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # ---------------------------------------------------------
    # AGENT 3 — Entry Agent
    # ---------------------------------------------------------
    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Save the customer query, Answer 1 and Answer 2 into a text file "
            "and return both complete answers to the user."
        ),
        backstory=(
            "You are the final customer support record keeper. "
            "You receive the outputs from the first two agents, save the "
            "complete interaction, preserve the actual source information, "
            "and prepare the final response."
        ),
        tools=[save_entry_file],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # ---------------------------------------------------------
    # TASK 1 — Answer from Agent 1 knowledge
    # ---------------------------------------------------------
    task_1 = Task(
        description=(
            "Answer this customer query directly from your own knowledge. "
            "Do not use web search.\n\n"
            f"Customer query:\n{query}"
        ),
        expected_output=(
            "A clear answer based on the Assistant Agent's existing LLM knowledge."
        ),
        agent=assistant,
    )

    # ---------------------------------------------------------
    # TASK 2 — Answer from real SerpApi results
    # ---------------------------------------------------------
    task_2 = Task(
        description=(
            "You are the Web Search Assistant.\n\n"
            f"Customer query:\n{query}\n\n"
            "IMPORTANT RULES:\n"
            "1. The web-search status and results below were retrieved from "
            "SerpApi before this task started.\n"
            "2. Use ONLY these web-search results as the factual basis of your "
            "answer when status is SUCCESS.\n"
            "3. Do not use Answer 1 as a factual source.\n"
            "4. Do not fall back to your own knowledge.\n"
            "5. If status is FAILED or NO_RESULTS, do not invent an answer. "
            "Return the exact search failure reason supplied in the web-search context "
            "and clearly state that no web-grounded answer is available.\n"
            "6. Include the most relevant source title and URL when status is SUCCESS.\n\n"
            f"{search_context}\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "Source: Web Search Assistant (SerpApi)\n"
            "Search Status: SUCCESS or FAILED or NO_RESULTS\n"
            "Answer: <answer grounded in the search results>\n"
            "Sources:\n"
            "- <source title> — <URL>"
        ),
        expected_output=(
            "A web-grounded answer with explicit SerpApi search status and source "
            "URLs, or an explicit failure/no-results message without using model knowledge."
        ),
        agent=web_assistant,
        context=[task_1],
    )

    # ---------------------------------------------------------
    # TASK 3 — Save + return both answers
    # ---------------------------------------------------------
    task_3 = Task(
        description=(
            "You are the final Entry Agent.\n\n"
            f"Customer query:\n{query}\n\n"
            "Use the sequential context to obtain the complete outputs from "
            "Agent 1 and Agent 2.\n\n"
            "REQUIREMENTS:\n"
            "1. Identify the complete Answer 1 from the Assistant Agent.\n"
            "2. Identify the complete Answer 2 from the Web Search Assistant.\n"
            "3. MUST call save_entry_file exactly once with the original query, "
            "complete Answer 1 and complete Answer 2.\n"
            "4. Preserve the actual Search Status and source URLs from Answer 2.\n"
            "5. Do not change a FAILED or NO_RESULTS status to SUCCESS. "
            "Preserve the web-search failure reason when one is provided.\n"
            "6. Return both complete answers. Do not replace them with a summary.\n\n"
            "FINAL RESPONSE FORMAT:\n\n"
            "Answer 1\n"
            "Source: Assistant Agent (LLM's existing knowledge)\n"
            "[complete Answer 1]\n\n"
            "Answer 2\n"
            "Source: Web Search Assistant (SerpApi)\n"
            "Search Status: [actual status]\n"
            "[complete Answer 2]\n\n"
            "Entry File\n"
            "[path returned by save_entry_file]"
        ),
        expected_output=(
            "Answer 1 with its source, Answer 2 with its actual SerpApi search "
            "status and sources, plus the saved text-file path."
        ),
        agent=entry_agent,
        context=[task_1, task_2],
    )

    # Exactly 3 agents and exactly 3 sequential tasks.
    return Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[task_1, task_2, task_3],
        process=Process.sequential,
        verbose=True,
    )


# =============================================================
# STREAMLIT UI
# =============================================================

st.set_page_config(
    page_title="Apple Support Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Apple Support Agent")
st.caption("AI-powered multi-agent Apple support system")

with st.sidebar:
    st.header("System")
    st.write("**Framework:** CrewAI")
    st.write("**Process:** Sequential")
    st.write("**Agents:** 3")
    st.write("**UI:** Streamlit")
    st.write("**LLM:** OpenAI")
    st.write("**Web Search:** SerpApi")
    st.write("**Storage:** Text files")
    st.write("**Search Timeout:** 45 seconds")
    st.write("**Search Retries:** 3")

validate_environment()

query = st.text_area(
    "Enter your query or task",
    placeholder="Example: What is the latest Kubernetes version?",
    height=120,
)

if st.button(
    "Run Apple Support Agent",
    type="primary",
    use_container_width=True,
):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    query = query.strip()

    # Quick configuration validation before starting the Crew.
    if not SERPAPI_API_KEY:
        st.error(
            "SERPAPI_API_KEY is missing. Add a valid SerpApi API key to your .env file."
        )
        st.stop()

    with st.status(
        "Running three agents sequentially...",
        expanded=True,
    ) as status:

        # -----------------------------------------------------
        # Real web search happens before Agent 2.
        # This guarantees that Answer 2 has actual web data.
        # -----------------------------------------------------
        st.write("🔎 Retrieving live web results from SerpApi...")
        search_data = perform_web_search(query)

        search_status = search_data.get("status")

        if search_status == "SUCCESS":
            st.write(
                f"✅ SerpApi returned "
                f"{len(search_data.get('results', []))} web results."
            )
        else:
            st.error(
                f"SerpApi search status: {search_status}\n\n"
                f"Reason: {search_data.get('error', 'No additional details.')}"
            )

        st.write("1️⃣ Assistant — answering from its own knowledge")
        st.write("2️⃣ Web Search Assistant — answering from SerpApi results")
        st.write("3️⃣ Entry Agent — saving the interaction")

        try:
            result = build_crew(query, search_data).kickoff()

            status.update(
                label="Crew completed successfully",
                state="complete",
                expanded=False,
            )

        except Exception as exc:
            status.update(
                label="Crew execution failed",
                state="error",
                expanded=True,
            )
            st.exception(exc)
            st.stop()

    st.success("All three agents completed sequentially.")

    st.markdown("### Entry Agent Output")
    st.markdown(str(result))

    st.caption(
        "Answer 1 = Assistant Agent knowledge | "
        "Answer 2 = live SerpApi results | "
        "Entry Agent = saves query + both answers to .txt"
    )