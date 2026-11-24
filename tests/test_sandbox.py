import unittest
from unittest.mock import patch

from ghost.sandbox import CodeSandbox, SandboxUnavailable


class SandboxTests(unittest.TestCase):
    def test_execution_fails_closed_without_bubblewrap(self):
        sandbox = CodeSandbox()
        sandbox.bwrap = None
        self.assertFalse(sandbox.available)
        with self.assertRaises(SandboxUnavailable):
            sandbox.run_python("print('must not execute')")
        with self.assertRaises(SandboxUnavailable):
            sandbox.run_cpp("int main(){return 0;}")


if __name__ == "__main__":
    unittest.main()
