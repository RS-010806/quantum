import re
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "qubolens" / "web"


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        element_id = dict(attrs).get("id")
        if element_id:
            self.ids.append(element_id)


class WebInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.html = (WEB_ROOT / "index.html").read_text()
        self.script = (WEB_ROOT / "app.js").read_text()
        self.styles = (WEB_ROOT / "styles.css").read_text()

    def test_lab_starts_clean_and_uses_compact_source_tabs(self):
        self.assertIn('id="upload-tab"', self.html)
        self.assertIn('id="sample-tab"', self.html)
        self.assertIn('id="lab-empty"', self.html)
        self.assertIn("results-panel hidden", self.html)
        self.assertIn("results-content hidden", self.html)
        self.assertIn('value="fast" checked', self.html)
        self.assertNotIn("journey-strip", self.html)
        self.assertNotIn("dataset-option", self.html)

    def test_interface_ids_are_unique_and_script_targets_exist(self):
        collector = _IdCollector()
        collector.feed(self.html)
        duplicates = [
            element_id
            for element_id, count in Counter(collector.ids).items()
            if count > 1
        ]
        self.assertEqual(duplicates, [])
        script_targets = set(
            re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', self.script)
        )
        self.assertEqual(sorted(script_targets - set(collector.ids)), [])

    def test_run_recovers_and_pointer_shadow_is_visible(self):
        self.assertIn("AbortController", self.script)
        self.assertIn("30_000", self.script)
        self.assertIn("z-index: 9999", self.styles)
        self.assertIn(".pointer-glow.interactive", self.styles)
        pointer_rule = self.styles.split(".pointer-glow {", 1)[1].split(
            ".pointer-glow.visible", 1
        )[0]
        self.assertNotIn("mix-blend-mode", pointer_rule)


if __name__ == "__main__":
    unittest.main()
