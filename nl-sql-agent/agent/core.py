"""The NL-to-SQL agent: an LLM that decides which tools to call, in a loop, until it can answer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent.sql_tools import Database, make_tools

DEFAULT_MODEL = "groq:openai/gpt-oss-120b"
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "retail.db"
MAX_STEPS = 12  # hard stop on tool-call loops

SYSTEM_PROMPT = """You are a data analyst assistant. You answer questions about a SQLite database \
by writing and running SQL with the tools you have.

How to work:
1. Call list_tables, then describe_tables for the tables you need. Never guess table or column names.
2. Write ONE SQLite SELECT query that answers the question. Use JOINs, GROUP BY and aggregates \
instead of pulling raw rows. Add ORDER BY and LIMIT for "top N" questions.
3. Call run_sql. If it returns an error, read the message, fix the query, and try again \
(at most 3 attempts).
4. Answer in 1-3 plain-language sentences using the actual numbers from the result. \
Round money to 2 decimals and say it is in INR.

Rules:
- You only have read access. Never try to insert, update, delete or change the schema.
- Revenue means SUM(order_items.quantity * order_items.unit_price). Unless the user says otherwise, \
exclude orders whose status is 'cancelled' or 'returned' from revenue and sales counts.
- Dates are stored as text in YYYY-MM-DD format; use strftime('%Y-%m', order_date) or \
strftime('%Y', order_date) to group by month or year.
- If the question cannot be answered from this data, say so plainly instead of inventing an answer.
- Do not put the SQL in your final answer; it is shown to the user separately.
"""


@dataclass
class AgentAnswer:
    question: str
    answer: str
    sql: str | None          # last query that ran successfully
    result: str | None       # output of that query
    steps: int               # number of tool calls made


def load_model(model: str | None = None) -> BaseChatModel:
    """Model string format is "<provider>:<model-name>", e.g. "groq:openai/gpt-oss-120b"."""
    return init_chat_model(model or os.getenv("MODEL", DEFAULT_MODEL), temperature=0)


class SQLAgent:
    def __init__(self, db_path: str | Path = DEFAULT_DB, model: str | BaseChatModel | None = None):
        self.db = Database(db_path)
        llm = model if isinstance(model, BaseChatModel) else load_model(model)
        self.graph = create_agent(llm, tools=make_tools(self.db), system_prompt=SYSTEM_PROMPT)

    def ask(self, question: str) -> AgentAnswer:
        self.db.query_log.clear()
        state = self.graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": MAX_STEPS * 2 + 1},
        )
        messages = state["messages"]

        # Find the last successful run_sql call and its output.
        sql, result = None, None
        call_args = {}
        steps = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    steps += 1
                    call_args[call["id"]] = call
            elif isinstance(msg, ToolMessage) and msg.name == "run_sql":
                text = str(msg.content)
                if not text.startswith(("REJECTED", "SQL ERROR")):
                    sql = call_args.get(msg.tool_call_id, {}).get("args", {}).get("query")
                    result = text

        final = messages[-1]
        answer = final.text if isinstance(final, AIMessage) else str(final.content)
        return AgentAnswer(question, answer.strip(), sql, result, steps)
