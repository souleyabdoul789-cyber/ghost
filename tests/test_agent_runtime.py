import unittest

from ghost.agent import AgentRuntime, LLMJSONPlanner, ToolError, ToolRegistry
from ghost.text import Tokenizer


class AgentRuntimeTests(unittest.TestCase):
    def test_closed_tool_loop(self):
        registry = ToolRegistry()
        registry.register("lookup", lambda key: {"value": key.upper()}, "lookup test value")
        decisions = iter([
            {"type": "tool", "tool": "lookup", "arguments": {"key": "ghost"}},
            {"type": "final", "answer": "GHOST"},
        ])
        runtime = AgentRuntime(lambda task, history, tools: next(decisions), registry, max_steps=3)
        result = runtime.run("convert ghost")
        self.assertTrue(result.success)
        self.assertEqual(result.answer, "GHOST")
        self.assertEqual(len(result.steps), 2)

    def test_unknown_tool_fails_closed_and_can_finish(self):
        registry = ToolRegistry()
        decisions = iter([
            {"type": "tool", "tool": "shell", "arguments": {"cmd": "bad"}},
            {"type": "final", "answer": "Action refusée."},
        ])
        result = AgentRuntime(lambda *args: next(decisions), registry, max_steps=3).run("test")
        self.assertTrue(result.success)
        self.assertIn("refusée", result.answer)

    def test_llm_planner_bounds_prompt_and_validates_tool(self):
        tokenizer = Tokenizer()

        class FakeModel:
            max_seq_len = 512
            def __init__(self, response):
                self.response = response
                self.prompt_length = 0
            def generate(self, prompt_ids, **kwargs):
                self.prompt_length = len(prompt_ids)
                return list(prompt_ids) + tokenizer.encode(self.response)

        model = FakeModel('{"type":"tool","tool":"lookup","arguments":{"key":"ghost"}}')
        planner = LLMJSONPlanner(model, tokenizer, max_new_tokens=48)
        decision = planner("Trouve une valeur", [{"tool":"old","result":{"text":"x" * 2000}}],
                           {"lookup":"lookup a value"})
        self.assertEqual(decision["tool"], "lookup")
        self.assertLessEqual(model.prompt_length + planner.max_new_tokens, model.max_seq_len)

        invalid = FakeModel('{"type":"tool","tool":"shell","arguments":{}}')
        planner_invalid = LLMJSONPlanner(invalid, tokenizer)
        with self.assertRaises(ToolError):
            planner_invalid("test", [], {"lookup":"known"})

    def test_runtime_compacts_long_observations(self):
        registry = ToolRegistry()
        registry.register("large", lambda: {"text": "z" * 5000}, "large observation")
        calls = []
        def planner(task, history, tools):
            calls.append(history)
            if not history:
                return {"type":"tool", "tool":"large", "arguments":{}}
            return {"type":"final", "answer":"done"}
        result = AgentRuntime(planner, registry, max_steps=3, max_history_chars=1000).run("test")
        self.assertTrue(result.success)
        self.assertTrue(calls[1])
        self.assertLessEqual(len(str(calls[1])), 1200)

    def test_planner_json_extraction(self):
        self.assertEqual(LLMJSONPlanner._extract_json('prefix {"type":"final","answer":"ok"} suffix')['answer'], 'ok')
        with self.assertRaises(Exception):
            LLMJSONPlanner._extract_json("no json")


if __name__ == "__main__":
    unittest.main()
