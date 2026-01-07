import os
import re
from typing import Annotated, Dict, List

from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient
from pydantic import Field


# ---------------------------
#  Tools
# ---------------------------

@ai_function(
    name="extract_indicators",
    description=(
        "Extracts basic indicators from a log snippet: IP addresses, domains, "
        "URLs, file paths, and common status/error codes."
    ),
)
def extract_indicators(
    log_text: Annotated[
        str,
        Field(description="Raw log text provided by the user. Can be multi-line."),
    ],
) -> Dict:
    """
    Parse the provided log text and return a structured summary of indicators.
    This tool is intentionally simple and deterministic (regex-based).
    """
    text = log_text or ""

    # IPv4 (basic)
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    ips: List[str] = sorted(set(ip_re.findall(text)))

    # Domain (very lightweight heuristic; avoids matching pure IPs)
    # Matches things like example.com, api.example.co.il
    domain_re = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
    domains_all = domain_re.findall(text)
    domains = sorted({d for d in domains_all if d not in ips})

    # URLs
    url_re = re.compile(r"\bhttps?://[^\s)>\"]+\b")
    urls = sorted(set(url_re.findall(text)))

    # Unix-like paths + Windows paths (basic)
    unix_path_re = re.compile(r"(?:(?:/[^ \n\t\r]+)+)")
    win_path_re = re.compile(r"\b[A-Za-z]:\\[^\n\r\t]+\b")
    file_paths = sorted(
        set(unix_path_re.findall(text) + win_path_re.findall(text))
    )

    # HTTP status codes (common)
    status_re = re.compile(r"\b(1\d{2}|2\d{2}|3\d{2}|4\d{2}|5\d{2})\b")
    statuses = sorted(set(status_re.findall(text)))

    # Common keywords (simple signal)
    keywords = []
    for kw in ["error", "failed", "denied", "timeout", "exception", "forbidden", "unauthorized"]:
        if re.search(rf"\b{kw}\b", text, flags=re.IGNORECASE):
            keywords.append(kw)

    return {
        "ok": True,
        "counts": {
            "ips": len(ips),
            "domains": len(domains),
            "urls": len(urls),
            "file_paths": len(file_paths),
            "status_codes": len(statuses),
        },
        "ips": ips,
        "domains": domains,
        "urls": urls,
        "file_paths": file_paths,
        "status_codes": statuses,
        "keywords": keywords,
    }


# ---------------------------
#  Provider config
# ---------------------------

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL", "qwen/qwen3-32b")

if not api_key:
    raise RuntimeError(
        "API_KEY is not set. "
        "Set it in your .env file or docker compose environment."
    )

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)

# ---------------------------
#  Agent definition for DevUI
# ---------------------------

agent = ChatAgent(
    chat_client=client,
    name="log-explainer-agent",
    instructions="""
You are Log Explainer Agent.

Purpose:
- Help the user interpret short log snippets.
- You must extract structured indicators using tools (not guess).

Tools:
- extract_indicators(log_text): extract IPs, domains, URLs, file paths, status/error codes.

Rules:
1) If the user provides any log text (even one line), ALWAYS call extract_indicators first.
2) Use the tool output to produce:
   - A short plain-English explanation (2–5 sentences)
   - An "Extracted Indicators" section listing the extracted fields (IPs, domains, URLs, etc.)
3) If the user asks follow-up questions, answer using the extracted indicators. If new logs are provided, call the tool again.
4) If the user message contains no log text, ask them to paste a snippet and specify what they want to understand.
5) Always answer in English. Keep responses concise and structured.
""",
    tools=[extract_indicators],
)
