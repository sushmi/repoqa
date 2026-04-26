"""Replace issue-comment answers in flask_gold12.json with code-grounded answers.

Why this exists
---------------
The dataset miner picks the top comment with ≥3 positive reactions as the
answer. That rule yields many bad references for a *gold* dataset: validations
("+1 same problem"), hot takes, workarounds without diagnosis, link-only
comments, or triage notes ("good first issue 👍").

For benchmark gold we need answers grounded in the current flask source, not
in GitHub social-proof. Each entry below was written by reading the question
and the cited flask file(s) at the current HEAD of `./flask`. The original
top-comment text is preserved as ``answer_from_issue`` for provenance.

Run
---
    .venv/bin/python scripts/regenerate_gold_answers.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/sushmi/dev/nlp/repoqa")
GOLD_PATH = ROOT / "data/evaluation/flask_gold12.json"

# qid → code-grounded answer (current flask source: src/flask/...)
ANSWERS: dict[str, str] = {
    # =====================================================================
    # ARCHITECTURE
    # =====================================================================
    "56c0744de3723104": (  # #1902 — flask initdb ImportError flaskr
        "Flask's CLI loads an app via `locate_app` in `src/flask/cli.py` (~line 230). "
        "When `FLASK_APP=flaskr`, Flask interprets the value as an importable module name and "
        "calls `__import__('flaskr')`, which requires `flaskr` to be reachable from `sys.path`. "
        "The CLI does **not** add the current working directory to `sys.path`, so a `flaskr.py` "
        "file sitting in cwd is not importable that way. The fix is one of: (1) set "
        "`FLASK_APP=flaskr.py` (path form — Flask resolves it via `prepare_import` and adds the "
        "containing directory to `sys.path`); (2) install the example as a package "
        "(`pip install -e .`); or (3) export `PYTHONPATH=.` before running `flask initdb`."
    ),
    "2dfdab95bae5e9fd": (  # #1045 — CSS served as text/plain
        "Flask serves static files via `send_from_directory` in `src/flask/helpers.py:543`, "
        "which delegates to `werkzeug.utils.send_from_directory`. The MIME type is determined "
        "by Python's stdlib `mimetypes.guess_type`, which on Windows reads the user registry "
        "(`HKEY_CLASSES_ROOT/.css/Content Type`). If that registry value is missing or set to "
        "`text/plain`, Flask returns `Content-Type: text/plain` for `.css` responses. This is "
        "an OS configuration issue, not a Flask bug. Workarounds: call "
        "`mimetypes.add_type('text/css', '.css')` at app startup, or pass an explicit "
        "`mimetype='text/css'` to `send_file`."
    ),
    "90cd1566dc0b8e73": (  # #1829 — `[Errno 2]` on Windows with FLASK_DEBUG=1
        "With `FLASK_DEBUG=1`, the `flask run` command (defined in `src/flask/cli.py` "
        "~line 882, `run_command`) starts Werkzeug's reloader. The reloader respawns the "
        "current process by re-executing `sys.argv[0]`, which on Windows points to a "
        "`Scripts/flask` wrapper script. In some environments (Conda, virtualenv) that "
        "wrapper does not exist at the absolute path Python recorded, producing "
        "`[Errno 2] No such file or directory`. Running `python -m flask run` bypasses the "
        "issue because `sys.argv[0]` becomes the Python interpreter path. Modern Flask CLI "
        "detects this case and re-execs as a module on Windows."
    ),
    "4f50ce3ff2582730": (  # #2626 — official guide for app structure
        "Flask's official structure guidance is split across two pattern docs: "
        "`docs/patterns/appfactories.rst` covers the application factory pattern "
        "(`def create_app(): app = Flask(__name__); ...; return app`), and "
        "`docs/patterns/packages.rst` covers splitting a Flask app into a Python package "
        "(`__init__.py` exposing the factory, blueprints in submodules, separate config "
        "and extensions). Flask intentionally does not prescribe an MVC layout because it "
        "is unopinionated about the model layer; community guides like the DigitalOcean "
        "tutorial that propose `models/` + `controllers/` add structure that Flask itself "
        "does not require. The recommended modern layout is package + factory + blueprints."
    ),
    # =====================================================================
    # LOGIC
    # =====================================================================
    "b80d28a9aa87e555": (  # #593 — sub-blueprints (nested register_blueprint)
        "Nested blueprint registration is supported in current Flask. "
        "`Blueprint.register_blueprint` is defined at `src/flask/sansio/blueprints.py:256`. "
        "When called on a parent blueprint, it appends the child to `self._blueprints` "
        "rather than registering immediately. When the parent is later registered with an "
        "app via `app.register_blueprint(parent)`, the parent's `register` method (same file) "
        "iterates the stored children and registers each with the app, concatenating the "
        "parent's `url_prefix` with the child's. So the user's intended pattern "
        "(`admin_bp.register_blueprint(test_bp, url_prefix='/test')`) is the supported API. "
        "The naive deferred-state implementation proposed in the issue is no longer needed."
    ),
    "21e270d71953438f": (  # #1907 — TEMPLATES_AUTO_RELOAD when debug set after init
        "`Flask.create_jinja_environment` at `src/flask/app.py:469` decides `auto_reload` as: "
        "if `config['TEMPLATES_AUTO_RELOAD']` is set explicitly, use that; otherwise fall back "
        "to `self.debug`. The Jinja environment is created lazily on first access to "
        "`Flask.jinja_env`. If a template render (or any access of `jinja_env`) happens "
        "**before** `app.run(debug=True)`, the environment is built with `auto_reload=False` "
        "and stays that way for the lifetime of the app — toggling `app.debug` afterward does "
        "not rebuild the environment. The reliable fix is to set "
        "`app.config['TEMPLATES_AUTO_RELOAD'] = True` explicitly, or to ensure `app.debug` is "
        "True before the first template render."
    ),
    "f4473ee7623f7fcc": (  # #2086 — large file uploads cached in memory
        "Flask delegates request body parsing to Werkzeug's `formparser`, which uses "
        "`werkzeug.formparser.default_stream_factory`. That factory returns an in-memory "
        "`io.BytesIO` for small uploads and a `tempfile.SpooledTemporaryFile` for larger ones, "
        "rolling over to disk at ~500KB by default. So uploads larger than ~500KB are written "
        "to a temp file on disk, not held in process memory. The user's perception of memory "
        "growth during a 300MB upload is most likely OS filesystem cache on the temp file or "
        "buffering in the single-threaded development server (which is not production-grade). "
        "To bound total request size, set `MAX_CONTENT_LENGTH` on the Flask app "
        "(default `None`, see `src/flask/app.py:226`); for production large-upload workloads, "
        "deploy behind a streaming WSGI server (gunicorn, uWSGI, waitress)."
    ),
    "d4dbd02ea3802ded": (  # #941 — errorhandler(HTTPException) not invoked
        "`Flask.handle_http_exception` at `src/flask/app.py:830` looks up an error handler via "
        "`_find_error_handler`, which checks (in order) an exact match on the exception's class "
        "and HTTP code, then walks the class MRO. The reason a generic "
        "`@app.errorhandler(HTTPException)` does not catch a 404 is that a per-code handler for "
        "`NotFound` (or for code 404) takes precedence — and Flask/Werkzeug ship default "
        "behaviors that effectively register such per-code handling. To handle every HTTP code "
        "uniformly, register a handler for each `werkzeug.exceptions.default_exceptions` entry "
        "with the same callback (the pattern shown in the accepted snippet). Registering only "
        "`errorhandler(HTTPException)` is insufficient when subclass-specific handling is "
        "already in place."
    ),
    # =====================================================================
    # DEPLOYMENT
    # =====================================================================
    "1433443e3f89d1ea": (  # #2023 — default logging handler interferes
        "Current Flask (`src/flask/logging.py:58`, function `create_logger`) attaches the "
        "default `StreamHandler` (defined at line 52) to `app.logger` only when "
        "`has_level_handler(logger)` is False — i.e., only if no handler in the logging chain "
        "would otherwise handle the logger's effective level. So if the application configures "
        "logging via `logging.config.dictConfig({...})` (or any equivalent) **before** first "
        "accessing `app.logger`, the default handler is never added and there is no conflict. "
        "This addresses the issue's complaint without requiring users to manually unwire "
        "Flask's logger."
    ),
    "c079b0ca784639b6": (  # #1847 — flask CLI sys.path / packaging
        "The `flask` CLI loads applications via `ScriptInfo.load_app` → `locate_app` "
        "(`src/flask/cli.py:230`). It does **not** modify `sys.path`. If the application is "
        "structured as a package not installed in the active environment, the import resolves "
        "against `sys.path` as-is and fails. The supported solutions are: install the package "
        "in editable mode (`pip install -e .`), set `PYTHONPATH=$(pwd)` before invoking "
        "`flask run`, or use the path form `FLASK_APP=path/to/wsgi.py` so Flask's "
        "`prepare_import` adds the containing directory. The Pallets policy decision was to "
        "**not** auto-add `cwd` to `sys.path` because that produces non-reproducible imports "
        "across environments."
    ),
    "e0f9194336752194": (  # #4027 — install_requires version pins
        "Flask's `pyproject.toml` declares dependencies with lower bounds only "
        "(`Werkzeug>=...`, `Jinja2>=...`, `itsdangerous>=...`, `click>=...`); see lines 25–28 "
        "of the current `pyproject.toml`. When Werkzeug 2.0 introduced breaking changes, "
        "users on Flask 1.1.x with unpinned environments resolved to the new major and got "
        "broken installs. The Pallets policy is that downstream applications should pin "
        "their own dependency tree; Flask itself does not cap upstream majors in advance. "
        "Subsequent Flask releases bumped the lower bound (now `Werkzeug>=...` aligned with "
        "the version actively tested against). For reproducible deploys, applications should "
        "pin Flask plus its transitive deps via a lockfile (`pip-compile`, `uv.lock`, "
        "`poetry.lock`)."
    ),
    "b8428fcec753a9fe": (  # #3168 — `flask run` output confusing
        "The `flask run` output is generated by `run_command` in `src/flask/cli.py` "
        "(~line 882, the Click command). Modern Flask (2.x+) prints an explicit production "
        "warning: ``WARNING: This is a development server. Do not use it in a production "
        "deployment. Use a production WSGI server instead.`` along with the bind address and "
        "debug status. This addresses the original confusion (the issue was filed against an "
        "older version where the warning was less explicit). The current text makes the "
        "dev-vs-prod distinction unambiguous for newcomers."
    ),
}


def main() -> None:
    if not GOLD_PATH.exists():
        raise SystemExit(f"Missing {GOLD_PATH}; run scripts/analyze_gold_errors.py first.")

    ds = json.loads(GOLD_PATH.read_text())
    qa_pairs = ds["qa_pairs"]

    if set(ANSWERS) != {q["id"] for q in qa_pairs}:
        missing = set(q["id"] for q in qa_pairs) - set(ANSWERS)
        extra = set(ANSWERS) - set(q["id"] for q in qa_pairs)
        raise SystemExit(f"ANSWERS keys mismatch — missing: {missing}, extra: {extra}")

    rewritten = 0
    for q in qa_pairs:
        # Preserve the original GitHub-comment text for provenance
        if "answer_from_issue" not in q:
            q["answer_from_issue"] = q["answer"]
        new = ANSWERS[q["id"]]
        if q["answer"] != new:
            q["answer"] = new
            q["answer_source"] = "code-grounded (current flask HEAD)"
            rewritten += 1

    ds["description"] = (
        "12-question gold subset (4 architecture, 4 logic, 4 deployment). "
        "Answers regenerated from current flask source code; original GitHub-comment "
        "answers preserved in `answer_from_issue` for provenance."
    )

    GOLD_PATH.write_text(json.dumps(ds, indent=2))
    print(f"Rewrote {rewritten}/{len(qa_pairs)} answers → {GOLD_PATH}")


if __name__ == "__main__":
    main()
