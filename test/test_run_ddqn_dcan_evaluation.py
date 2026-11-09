"""Exercise saved-model launcher scope and process boundaries."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import ANY, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_ddqn_dcan_evaluation.py"
SPEC = importlib.util.spec_from_file_location("run_ddqn_dcan_evaluation", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SavedModelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.cache = (ROOT / ".cache/ddqn_dcan_launcher_tests").resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.cache)
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(self.cache))
        self.addCleanup(self.cleanup_fixture)
        self.project = Path(self.tmp.name) / "project with spaces"
        self.helper = self.project / "scripts/resume_full_evaluation.py"
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text("# fixture helper\n", encoding="utf-8")
        self.outputs = {method: self.project / relative
                        for method, relative in launcher.DEFAULT_OUTPUT_DIRS.items()}
        for output in self.outputs.values():
            self.make_inputs(output)

    def cleanup_fixture(self):
        target = Path(self.tmp.name).resolve()
        self.assertTrue(target.is_relative_to(self.cache))
        self.assertNotEqual(target, self.cache)
        self.tmp.cleanup()

    @staticmethod
    def make_inputs(output):
        output.mkdir(parents=True, exist_ok=True)
        for name in ("preflight.json", "source.zip"):
            (output / name).write_bytes(b"fixture")

    def invoke(self, arguments):
        stdout = io.StringIO()
        with patch.object(launcher.subprocess, "run") as run, contextlib.redirect_stdout(stdout):
            launcher.main(["--project-root", str(self.project), *arguments])
        return run, stdout.getvalue()

    def expected_command(self, method, output=None, *, prepare_only=False):
        command = [sys.executable, "-u", str(self.helper.resolve()),
                   "--project-root", str(self.project.resolve()),
                   "--output-dir", str((output or self.outputs[method]).resolve()), "--arms", method]
        return command + (["--prepare-only"] if prepare_only else [])

    def test_zero_arguments_uses_script_project_and_default_sequence(self):
        self.assertEqual(launcher.DEFAULT_PROJECT_ROOT, ROOT)
        unrelated = Path(self.tmp.name) / "unrelated working directory"
        unrelated.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(unrelated)
            with patch.object(launcher, "DEFAULT_PROJECT_ROOT", self.project), \
                    patch.object(launcher.subprocess, "run") as run, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                launcher.main([])
        finally:
            os.chdir(previous)
        self.assertEqual(run.call_count, 2)
        for call, method in zip(run.call_args_list, ("rule_ddqn", "dcan")):
            self.assertEqual(call.args, (self.expected_command(method),))
            self.assertEqual(call.kwargs, {"cwd": self.project.resolve(), "check": True,
                                           "stdout": stdout, "stderr": sys.stderr})
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines, [f"[{index}/2] {method} evaluation: {self.outputs[method] / 'evaluation'}"
                                 for index, method in enumerate(("rule_ddqn", "dcan"), 1)])

    def test_selected_method_ignores_absent_unselected_inputs(self):
        for name in ("preflight.json", "source.zip"):
            (self.outputs["rule_ddqn"] / name).unlink()
        self.outputs["rule_ddqn"].rmdir()
        run, _ = self.invoke(["--methods", "dcan"])
        run.assert_called_once_with(self.expected_command("dcan"), cwd=self.project.resolve(), check=True,
                                    stdout=ANY, stderr=sys.stderr)

    def test_explicit_method_order_is_preserved(self):
        run, _ = self.invoke(["--methods", "dcan", "rule_ddqn"])
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ["dcan", "rule_ddqn"])

    def test_relative_and_absolute_output_paths_with_spaces(self):
        relative = Path("saved results/rule model")
        absolute = Path(self.tmp.name) / "DCAN model with spaces"
        self.make_inputs(self.project / relative)
        self.make_inputs(absolute)
        run, _ = self.invoke(["--rule-ddqn-output-dir", str(relative),
                              "--dcan-output-dir", str(absolute)])
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [self.expected_command("rule_ddqn", self.project / relative),
                          self.expected_command("dcan", absolute)])

    def test_prepare_only_is_forwarded_to_each_helper(self):
        run, _ = self.invoke(["--prepare-only"])
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [self.expected_command(method, prepare_only=True)
                          for method in ("rule_ddqn", "dcan")])

    def test_missing_selected_inputs_prevents_all_subprocesses(self):
        for method in self.outputs:
            for name in ("preflight.json", "source.zip"):
                with self.subTest(method=method, name=name):
                    missing = self.outputs[method] / name
                    missing.unlink()
                    with patch.object(launcher.subprocess, "run") as run, \
                            contextlib.redirect_stderr(io.StringIO()) as stderr, \
                            self.assertRaises(SystemExit) as error:
                        launcher.main(["--project-root", str(self.project)])
                    self.assertEqual(error.exception.code, 2)
                    run.assert_not_called()
                    self.assertIn(str(missing), stderr.getvalue())
                    missing.write_bytes(b"fixture")

    def test_missing_helper_prevents_all_subprocesses(self):
        self.helper.unlink()
        with patch.object(launcher.subprocess, "run") as run, \
                contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            launcher.main(["--project-root", str(self.project)])
        run.assert_not_called()
        self.assertIn(str(self.helper), stderr.getvalue())

    def test_duplicate_methods_are_rejected(self):
        with patch.object(launcher.subprocess, "run") as run, \
                contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            launcher.main(["--project-root", str(self.project), "--methods", "dcan", "dcan"])
        run.assert_not_called()
        self.assertIn("must be unique", stderr.getvalue())

    def test_first_helper_failure_stops_the_second_method(self):
        failure = subprocess.CalledProcessError(9, ["fixture helper"])
        with patch.object(launcher.subprocess, "run", side_effect=failure) as run, \
                contextlib.redirect_stdout(io.StringIO()) as stdout, self.assertRaises(subprocess.CalledProcessError):
            launcher.main(["--project-root", str(self.project)])
        run.assert_called_once_with(self.expected_command("rule_ddqn"), cwd=self.project.resolve(), check=True,
                                    stdout=stdout, stderr=sys.stderr)

    def test_real_cli_forwards_paths_and_inherits_helper_output(self):
        capture = self.project / "captured arguments.jsonl"
        self.helper.write_text(
            "import json\nfrom pathlib import Path\nimport sys\n"
            "with (Path(__file__).parents[1] / 'captured arguments.jsonl').open('a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'args': sys.argv[1:], 'cwd': str(Path.cwd())}) + '\\n')\n"
            "method = sys.argv[sys.argv.index('--arms') + 1]\n"
            "print('fixture helper output: ' + method, flush=True)\n"
            "print('fixture helper diagnostic: ' + method, file=sys.stderr, flush=True)\n",
            encoding="utf-8")
        completed = subprocess.run([sys.executable, str(SCRIPT), "--project-root", self.project.name,
                                    "--prepare-only"], cwd=Path(self.tmp.name),
                                   capture_output=True, text=True, check=True, timeout=20)
        records = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["args"] for record in records],
                         [self.expected_command(method, prepare_only=True)[3:]
                          for method in ("rule_ddqn", "dcan")])
        self.assertTrue(all(Path(record["cwd"]) == self.project for record in records))
        self.assertIn("fixture helper output: rule_ddqn", completed.stdout)
        self.assertIn("fixture helper output: dcan", completed.stdout)
        self.assertIn("fixture helper diagnostic: rule_ddqn", completed.stderr)
        self.assertIn("fixture helper diagnostic: dcan", completed.stderr)

    def test_real_helper_forwards_verifier_and_evaluator_output(self):
        (self.project / "data").mkdir()
        frozen = self.project / "frozen source with spaces"
        scripts = frozen / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "__init__.py").write_text("", encoding="utf-8")
        (frozen / "experiment_identity.json").write_text(
            json.dumps({"configs": {"full": {"device": "cuda"}}}), encoding="utf-8")
        (scripts / "experiment_suite.py").write_text(
            "import sys\nmethod = sys.argv[sys.argv.index('--arms') + 1]\n"
            "print('evaluator output: ' + method, flush=True)\n"
            "print('evaluator diagnostic: ' + method, file=sys.stderr, flush=True)\n",
            encoding="utf-8")
        verification = ("import sys\nprint('verifier output', flush=True)\n"
                        "print('verifier diagnostic', file=sys.stderr, flush=True)\n")
        self.helper.write_text(
            "import importlib.util\nfrom pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('real_resume_helper', {str(ROOT / 'scripts/resume_full_evaluation.py')!r})\n"
            "helper = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(helper)\n"
            f"helper.prepare = lambda project, output, arms=None: (Path({str(frozen)!r}), 'fixture-identity', [11])\n"
            f"helper._VERIFY_RUNTIME = {verification!r}\nhelper.main()\n",
            encoding="utf-8")
        completed = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(self.project)],
                                   cwd=Path(self.tmp.name), capture_output=True, text=True,
                                   check=True, timeout=20)
        self.assertEqual(completed.stdout.count("verifier output"), 2)
        self.assertEqual(completed.stderr.count("verifier diagnostic"), 2)
        for method in ("rule_ddqn", "dcan"):
            self.assertIn("evaluator output: " + method, completed.stdout)
            self.assertIn("evaluator diagnostic: " + method, completed.stderr)


if __name__ == "__main__":
    unittest.main()
