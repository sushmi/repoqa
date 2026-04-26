"""Qualitative semantic check on a 12-summary stratified sample.

For each sampled summary, we record:
  - the existing summary from `summaries.json` (column: summary_existing)
  - a fresh summary derived by reading the source chunks (column: summary_derived)
  - a semantic verdict (column: semantic_verdict ∈ {correct, partial, wrong, plausible})
  - notes explaining the verdict (column: semantic_notes)

These are merged into the deterministic CSV from `validate_summary_levels.py`
to produce `summary_level_validation_qualitative.csv`. Sampled rows get all
columns filled; unsampled rows get blanks in the qualitative columns.

Run
---
    .venv/bin/python scripts/qualitative_summary_check.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/Users/sushmi/dev/nlp/repoqa")
SUMMARIES_JSON = ROOT / "data/flask/summaries.json"
CSV_IN  = ROOT / "data/flask/summary_level_validation.csv"
CSV_OUT = ROOT / "data/flask/summary_level_validation_qualitative.csv"

# (level, path) → (derived_summary, verdict, notes)
# verdicts: "correct" | "partial" | "wrong" | "plausible"
SAMPLE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("project", "."): (
        "The flask repository is the source for the Flask web framework. "
        "The package itself lives in `src/flask/` (210 chunks across `app.py`, "
        "`cli.py`, `helpers.py`, `blueprints.py`, `logging.py`, `templating.py`, "
        "`sessions.py`, `wrappers.py`, `views.py`, `ctx.py`, `globals.py`, "
        "the `sansio/` core, and the `json/` subpackage). `docs/` (328 chunks) "
        "holds Sphinx docs including `patterns/` how-tos and `tutorial/`. "
        "`tests/` (489 chunks) holds the pytest suite plus fixture apps under "
        "`tests/test_apps/`. `examples/` (90 chunks) ships demo apps "
        "(tutorial, javascript, celery). Top-level files: `pyproject.toml` "
        "(flit_core build backend, ruff, pytest config), `README.md`, "
        "`CHANGES.rst`, `LICENSE.txt`, `.pre-commit-config.yaml`, "
        "`.readthedocs.yaml`.",
        "partial",
        "Existing summary OMITS `src/flask/` (the actual package!) and "
        "`examples/` directories — major gap for a project overview. Also "
        "calls the build system 'a poetry environment' but the actual "
        "build-system is `flit_core` (verified in pyproject.toml). High-level "
        "framing of Flask as a lightweight WSGI framework is correct.",
    ),

    ("directory", "tests/test_apps/blueprintapp/apps/admin"): (
        "Test fixture for an `admin` Flask blueprint. Contains `__init__.py` "
        "defining a `Blueprint('admin', ...)` with sample routes, plus "
        "`templates/` and `static/` subdirectories used to verify blueprint "
        "template/static-folder resolution.",
        "plausible",
        "Existing summary names `index()` and `index2()` route handlers and "
        "the `/admin` URL prefix — these specific claims weren't independently "
        "verified against `__init__.py` content but are typical for such "
        "fixtures. High-level role (test fixture for admin blueprint) is correct.",
    ),

    ("directory", "docs/patterns"): (
        "Sphinx pattern docs covering: app structure (`appfactories.rst`, "
        "`packages.rst`, `singlepageapplications.rst`); database integrations "
        "(`sqlalchemy.rst`, `sqlite3.rst`, `mongoengine.rst`); forms "
        "(`wtforms.rst`); concurrency/streaming (`celery.rst`, "
        "`lazyloading.rst`, `streaming.rst`, `deferredcallbacks.rst`); "
        "request handling (`fileuploads.rst`, `methodoverrides.rst`, "
        "`urlprocessors.rst`, `viewdecorators.rst`, `requestchecksum.rst`); "
        "frontend (`javascript.rst`, `jquery.rst`); presentation "
        "(`flashing.rst`, `templateinheritance.rst`, `favicon.rst`, "
        "`subclassing.rst`); deployment (`appdispatch.rst`); plus `caching.rst` "
        "and the `index.rst` table of contents.",
        "partial",
        "Existing summary correctly names ~6 files (index, appfactories, "
        "celery, wtforms, mongoengine, sqlite3) but claims the directory "
        "covers 'method overrides' — methodoverrides.rst exists but isn't "
        "in the cited list, and the summary structure suggests it's "
        "summarizing the wrong set. Also vague phrases like 'best practices' "
        "and 'integration strategies' don't add specific information.",
    ),

    ("directory", "."): (
        "Root directory of the flask repository. Holds project metadata "
        "and config files: `pyproject.toml` (project metadata, dependencies, "
        "ruff/pytest/flit config), `README.md`, `CHANGES.rst` (release "
        "history), `LICENSE.txt` (BSD-3-Clause), `.pre-commit-config.yaml` "
        "(ruff and other lint hooks), `.readthedocs.yaml` (Sphinx build "
        "config for RTD).",
        "correct",
        "Existing summary names the same 6 files and correctly describes "
        "their roles (build config, lint hooks, docs, history, license). "
        "Treating it as a 'directory' summary at the root level is reasonable "
        "even though the actual flask package lives in `src/flask/`.",
    ),

    ("directory", "tests/templates/nested"): (
        "Test fixture directory for templates engine's nested-folder "
        "resolution. Contains `nested.txt` with the literal string "
        "\"I'm nested\" used by tests verifying that Flask correctly resolves "
        "templates in subdirectories.",
        "correct",
        "Existing summary correctly identifies `nested.txt` as the lone file "
        "with placeholder content used to test nested-template handling. "
        "Concise and accurate.",
    ),

    ("directory", "tests/test_apps/blueprintapp/apps/admin/static"): (
        "Static-files test fixture for the `admin` blueprint. Contains "
        "`test.txt` (literal content: \"Admin File\") used to verify that "
        "Flask serves blueprint-scoped static files at the expected URL.",
        "correct",
        "Existing summary identifies test.txt and its role as a static-content "
        "verification fixture. Slightly verbose for a 2-word file but the "
        "headline is correct.",
    ),

    ("directory", "tests/test_apps/subdomaintestmodule/static"): (
        "Static-files test fixture for a subdomain-routed test module. "
        "Contains `hello.txt` with the string \"Hello Subdomain\" used to "
        "verify static-file serving when subdomain matching is enabled.",
        "correct",
        "Existing summary correctly names hello.txt and its content. "
        "Brief and accurate.",
    ),

    ("file", "tests/test_reqctx.py"): (
        "Pytest module for Flask's request-context machinery and session "
        "interface. Tests cover: `teardown_request` callbacks "
        "(`test_teardown_on_pop`, `test_teardown_with_previous_exception`, "
        "`test_teardown_with_handled_exception`); explicit context binding "
        "(`test_proper_test_request_context`, `test_context_binding`, "
        "`test_manual_context_binding`); greenlet context copying "
        "(`TestGreenletContextCopying.test_greenlet_context_copying`, "
        "skipped if greenlet not installed); and session-interface failures "
        "(`test_session_error_pops_context`, `test_session_dynamic_cookie_name` "
        "using a custom `SecureCookieSessionInterface`). Imports flask, "
        "pytest, and optionally greenlet.",
        "correct",
        "Existing summary names test_teardown_on_pop, test_context_binding, "
        "test_greenlet_context_copying — all real, verified in source. Misses "
        "the session-interface tests (test_session_error_pops_context, "
        "test_session_dynamic_cookie_name) but doesn't fabricate. Also misses "
        "self-mention of file basename (`test_reqctx.py`).",
    ),

    ("file", "tests/test_testing.py"): (
        "Pytest module for Flask's test client and CLI runner. Covers: "
        "EnvironBuilder behavior (`test_environ_defaults`, `test_path_is_url`, "
        "`test_environbuilder_json_dumps`, `test_specify_url_scheme`); "
        "subdomain matching (`test_blueprint_with_subdomain`, "
        "`test_subdomain`, `test_nosubdomain`); test_client session and "
        "cookie management (`test_session_transactions` and variants, "
        "`test_redirect_session`, `test_test_client_context_binding`); JSON "
        "request/response (`test_json_request_and_response`, "
        "`test_client_json_no_app_context`); CLI runner "
        "(`test_cli_runner_class`, `test_cli_invoke`, `test_cli_custom_obj` "
        "using `FlaskCliRunner` and `ScriptInfo`).",
        "correct",
        "Existing summary correctly names `FlaskCliRunner`, `EnvironBuilder`, "
        "and broad areas (request contexts, sessions, client testing, CLI). "
        "Captures the breadth without fabricating. Misses self-mention of "
        "file basename.",
    ),

    ("file", "tests/test_apps/blueprintapp/apps/admin/static/test.txt"): (
        "Test fixture file containing the literal text \"Admin File\". Used "
        "by blueprint tests to verify that Flask serves static files from a "
        "blueprint's `static_folder`.",
        "partial",
        "Existing summary is technically not wrong but disproportionately "
        "long (130+ tokens of meta-description for 2 words of content). "
        "It includes filler like 'no dependencies or patterns explicitly "
        "coded' and 'no side effects or configuration are specified within "
        "the file itself' which are tautologies for a static .txt fixture. "
        "Bloat is the issue, not factual error.",
    ),

    ("file", "examples/javascript/pyproject.toml"): (
        "Project metadata for the `js_example` demo (version 1.1.0). "
        "Demonstrates Flask + AJAX. Build backend: `flit_core<4`. Dependencies: "
        "`flask`. Optional `test` extra: `pytest`. Configures pytest "
        "(`testpaths=['tests']`, warnings-as-errors), coverage (branch=true, "
        "sources=js_example+tests), and ruff (`src=['src']`). Marked "
        "`Private :: Do Not Upload` (not for PyPI).",
        "correct",
        "Existing summary correctly names: js_example, Flask backend, AJAX "
        "purpose, flask + pytest deps, coverage.py, Ruff, pytest, Flit, "
        "private marker. All verified against the file content. Slightly "
        "imprecise — says 'Flit' when the build backend is `flit_core` — but "
        "the distinction is marginal in a project summary.",
    ),

    ("file", "tests/static/config.json"): (
        "Two-key JSON test fixture: `TEST_KEY=\"foo\"` and `SECRET_KEY=\"config\"`. "
        "Used by tests that exercise `Flask.config.from_file` / `from_json` "
        "to verify config loading from a JSON source.",
        "correct",
        "Existing summary correctly names both keys and their values. The "
        "speculation that values 'may be used for authentication or "
        "identification purposes' is mild over-reach — `TEST_KEY=\"foo\"` "
        "is just a probe, not an auth token — but it doesn't fabricate.",
    ),
}


def main() -> None:
    if not CSV_IN.exists():
        raise SystemExit(
            f"Missing {CSV_IN}. Run scripts/validate_summary_levels.py first."
        )

    summaries = json.loads(SUMMARIES_JSON.read_text())
    by_key = {(s["level"], s["path"]): s for s in summaries}

    rows_in = list(csv.DictReader(CSV_IN.open()))
    fieldnames = list(rows_in[0].keys()) + [
        "summary_existing", "summary_derived", "semantic_verdict", "semantic_notes"
    ]

    sampled = 0
    verdict_counts: dict[str, int] = {"correct": 0, "partial": 0, "wrong": 0, "plausible": 0, "": 0}
    out_rows: list[dict] = []
    for r in rows_in:
        key = (r["level"], r["path"])
        if key in SAMPLE:
            existing = by_key[key]["content"]
            derived, verdict, notes = SAMPLE[key]
            r = {
                **r,
                "summary_existing": existing,
                "summary_derived":  derived,
                "semantic_verdict": verdict,
                "semantic_notes":   notes,
            }
            sampled += 1
            verdict_counts[verdict] += 1
        else:
            r = {**r, "summary_existing": "", "summary_derived": "",
                 "semantic_verdict": "", "semantic_notes": ""}
            verdict_counts[""] += 1
        out_rows.append(r)

    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows ({sampled} with qualitative columns) → {CSV_OUT}")
    print()
    print("=== Semantic verdict distribution (sampled) ===")
    for v in ("correct", "partial", "wrong", "plausible"):
        if verdict_counts[v]:
            print(f"  {v:9s}: {verdict_counts[v]}")

    # Brief readout for sampled rows
    print()
    print("=== Sampled rows ===")
    for r in out_rows:
        if r["semantic_verdict"]:
            print(f"  [{r['level']:9s}] {r['path']:55s}  "
                  f"deterministic_pass={r['pass']}  semantic={r['semantic_verdict']}")


if __name__ == "__main__":
    main()
