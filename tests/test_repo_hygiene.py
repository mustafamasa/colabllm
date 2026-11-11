"""Repository hygiene: license metadata, no personal deployment details, idempotent service overrides."""
import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

SKIP_DIRS = {".git", ".state", "dist", "build", "__pycache__", ".venv", "venv", "node_modules"}
LEAK_PATTERNS = [re.escape(token) for token in ("443998" + ".xyz", "/etc/" + "sing-box", "/home/" + "eric")]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS))


def iter_repo_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_file() and path.suffix not in (".pyc", ".tar", ".gz", ".whl", ".png", ".jpg"):
            yield path


def load_service(name):
    path = ROOT / "src/colabllm/services" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_") + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepoHygieneTest(unittest.TestCase):
    def test_mit_license_is_declared_and_packaged(self):
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("MIT License", license_text)
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn('license = "MIT"', pyproject)
        self.assertIn('license-files = ["LICENSE"]', pyproject)
        self.assertIn("include LICENSE", (ROOT / "MANIFEST.in").read_text())
        self.assertIn("## License", (ROOT / "README.md").read_text())

    def test_no_personal_deployment_details_in_repository(self):
        leaks = []
        for path in iter_repo_files():
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if LEAK_RE.search(line):
                    leaks.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(leaks, [], "personal deployment details found")

    def test_readme_links_are_absolute_for_pypi(self):
        readme = (ROOT / "README.md").read_text()
        relative = re.findall(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", readme)
        self.assertEqual(relative, [])

    def test_prepare_up_overrides_do_not_leak_between_calls(self):
        comfyui = load_service("comfyui")
        comfyui.prepare_up(types.SimpleNamespace(workflow=["a.json"]))
        self.assertEqual(comfyui.SERVICE.env["CT_WORKFLOW"], "a.json")
        comfyui.prepare_up(types.SimpleNamespace(workflow=None))
        self.assertNotIn("CT_WORKFLOW", comfyui.SERVICE.env)

        ollama = load_service("ollama")
        baseline = dict(ollama.SERVICE.env)
        ollama.prepare_up(types.SimpleNamespace(models=["x"]))
        self.assertEqual(ollama.SERVICE.env["CT_MODELS"], "x")
        ollama.prepare_up(types.SimpleNamespace(models=None))
        self.assertEqual(ollama.SERVICE.env, baseline)

        deepseek = load_service("deepseek-coder")
        ds_baseline = dict(deepseek.SERVICE.env)
        deepseek.prepare_up(types.SimpleNamespace(up_CT_DS_CONTEXT="524288", up_no_vision=True))
        self.assertEqual(deepseek.SERVICE.env["CT_DS_CONTEXT"], "524288")
        self.assertEqual(deepseek.SERVICE.env["CT_DS_VISION"], "0")
        deepseek.prepare_up(types.SimpleNamespace())
        self.assertEqual(deepseek.SERVICE.env, ds_baseline)

        drama3 = load_service("drama3")
        d3_baseline = dict(drama3.SERVICE.env)
        drama3.prepare_up(types.SimpleNamespace(up_CT_DRAMA3_VARIANT="int4"))
        self.assertEqual(drama3.SERVICE.env["CT_DRAMA3_VARIANT"], "int4")
        drama3.prepare_up(types.SimpleNamespace())
        self.assertEqual(drama3.SERVICE.env, d3_baseline)

        tangoflux = load_service("tangoflux")
        tf_baseline = dict(tangoflux.SERVICE.env)
        tangoflux.prepare_up(types.SimpleNamespace(up_CT_TANGOFLUX_VARIANT="base"))
        self.assertEqual(tangoflux.SERVICE.env["CT_TANGOFLUX_VARIANT"], "base")
        tangoflux.prepare_up(types.SimpleNamespace())
        self.assertEqual(tangoflux.SERVICE.env, tf_baseline)

    def test_service_default_hostnames_are_short_aliases(self):
        for name in ("qwen-flash", "deepseek-coder", "nemotron-transcribe", "indextts", "qwen3-tts", "drama3", "tangoflux"):
            alias = load_service(name).SERVICE.default_hostname
            self.assertTrue(alias and "." not in alias, (name, alias))


if __name__ == "__main__":
    unittest.main()
