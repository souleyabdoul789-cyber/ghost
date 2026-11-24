import hashlib
import os
import tempfile
import unittest

from ghost.agent import ResearchAgent, ToolError, ToolRegistry
from ghost.memory import GMemory
from ghost.web import WebAccessError, WebDocument, WebReader, _PageParser


class FakeReader:
    allowed_domains = frozenset({"docs.example.test"})

    def _is_allowed(self, host):
        return host == "docs.example.test" or (host or "").endswith(".docs.example.test")

    def __init__(self):
        self.pages = {
            "https://docs.example.test/start": WebDocument(
                "https://docs.example.test/start", "Start", 
                "G tensors use native C memory and automatic differentiation.",
                (("https://docs.example.test/autograd", "Autograd"),), 1.0,
                hashlib.sha256(b"start").hexdigest(), "text/html"),
            "https://docs.example.test/autograd": WebDocument(
                "https://docs.example.test/autograd", "Autograd",
                "The backward pass accumulates gradients for every trainable tensor.",
                (), 2.0, hashlib.sha256(b"autograd").hexdigest(), "text/html"),
        }

    def fetch(self, url):
        return self.pages[url]


class AgentTests(unittest.TestCase):
    def test_web_reader_requires_https_and_explicit_domain(self):
        reader = WebReader("docs.example.test", respect_robots=False)
        self.assertEqual(
            reader._validate_url("https://docs.example.test/start", resolve_dns=False),
            "https://docs.example.test/start",
        )
        with self.assertRaises(WebAccessError):
            reader._validate_url("http://docs.example.test/start", resolve_dns=False)
        with self.assertRaises(WebAccessError):
            reader._validate_url("https://other.example.test/start", resolve_dns=False)

    def test_parser_ignores_scripts_and_extracts_links(self):
        parser = _PageParser()
        parser.feed("<html><title>G docs</title><script>private()</script>"
                    "<main><h1>Native tensors</h1><p>Autograd works.</p>"
                    "<a href='/api'>API reference</a></main></html>")
        title, text, links = parser.result()
        self.assertEqual(title, "G docs")
        self.assertIn("Autograd works.", text)
        self.assertNotIn("private()", text)
        self.assertEqual(links[0][0], "/api")

    def test_bounded_research_and_memory(self):
        memory = GMemory(":memory:")
        agent = ResearchAgent(FakeReader(), memory)
        report = agent.research(
            ["https://docs.example.test/start"],
            query="native C gradients tensors",
            max_pages=3,
            max_depth=1,
        )
        self.assertEqual(len(report.pages_read), 2)
        self.assertEqual(len(report.errors), 0)
        self.assertGreaterEqual(len(report.evidence), 1)
        self.assertTrue(any("native C" in item.content for item in report.evidence))
        self.assertEqual(len(agent.answer("backward gradients")["evidence"]), 1)
        self.assertTrue(agent.remember_feedback("Use gradient clipping before AdamW."))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corpus.txt")
            self.assertEqual(agent.export_training_corpus(path), 3)
            with open(path, encoding="utf-8") as stream:
                self.assertIn("Autograd", stream.read())
        memory.close()

    def test_tool_registry_blocks_unknown_tools(self):
        registry = ToolRegistry()
        registry.register("safe", lambda value: value, "test")
        self.assertEqual(registry.call("safe", value=7), 7)
        with self.assertRaises(ToolError):
            registry.call("shell", command="not allowed")


if __name__ == "__main__":
    unittest.main()
