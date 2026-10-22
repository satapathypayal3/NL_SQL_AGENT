"""Checks the agent loop end to end with a scripted fake LLM, so no API key is needed.

The fake model makes a bad query first, gets an error back, fixes it, then answers.
This verifies the tool-calling wiring and the error-recovery path.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent.core import SQLAgent
from data.create_db import build
from eval.run_eval import results_match


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def test_agent_recovers_from_sql_error(tmp_path):
    db_path = build(tmp_path / "t.db")
    script = iter([
        call("list_tables", {}, 1),
        call("describe_tables", {"tables": "customers"}, 2),
        call("run_sql", {"query": "SELECT COUNT(*) FROM customer WHERE city = 'Pune'"}, 3),  # typo
        call("run_sql", {"query": "SELECT COUNT(*) FROM customers WHERE city = 'Pune'"}, 4),  # fixed
        AIMessage(content="Pune has this many customers: see result."),
    ])
    agent = SQLAgent(db_path=db_path, model=ScriptedModel(messages=script))
    ans = agent.ask("How many customers are from Pune?")

    assert ans.sql == "SELECT COUNT(*) FROM customers WHERE city = 'Pune'"
    assert ans.steps == 4
    assert "Pune" in ans.answer
    assert ans.result is not None


def test_results_match():
    assert results_match([("Pune", 41)], [(41, "pune")])            # column order, case
    assert results_match([(1999.0,)], [("x", 1999)])                  # extra column, int vs float
    assert results_match([(1,), (2,)], [(2,), (1,)])                  # row order
    assert not results_match([(1,)], [(1,), (2,)])
    assert not results_match([(100.5,)], [(100.0,)])
