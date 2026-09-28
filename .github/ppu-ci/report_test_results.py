#!/usr/bin/env python3
"""
Generate GitHub Step Summary from JUnit XML test results.

Supports two mutually-exclusive modes:

Mode A — single-board report (--xml):
    python report_test_results.py --xml <path> [--title T] [--run-number N]
        [--branch B] [--job-status S] [--duration D] [--pods P]
        [--docker-image IMG] [--env-info JSON]

    Prints: Title → 环境 table → single-row 概览 table → Failed Test Details
    → collapsible All Test Cases.  Exit code 1 if any failures, 0 otherwise.

Mode B — combined multi-board report (--result-root):
    python report_test_results.py --result-root <dir> [--title T]
        [--run-number N] [--branch B]

    Auto-discovers board subdirs under <dir>, prints: Title → combined 概览
    table (one row per board) → per-board sections (env + failures +
    collapsible full list).  Exit code is always 0.

Output is Markdown printed to stdout, intended to be appended to
$GITHUB_STEP_SUMMARY.
"""

import argparse
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

# GitHub Step Summary hard limit is 1024 KB; we truncate at 900 KB to leave
# room for the warning footer and any UTF-8 expansion.
_MAX_SUMMARY_BYTES = 900 * 1024


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_xml(xml_path):
    """Parse a JUnit XML and return (testcases_list, stats_dict) or None."""
    if not os.path.isfile(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    total = fail = err = skip = 0
    time_s = 0.0
    for ts in root.findall(".//testsuite"):
        total += int(ts.get("tests", 0))
        fail += int(ts.get("failures", 0))
        err += int(ts.get("errors", 0))
        skip += int(ts.get("skipped", 0))
        try:
            time_s += float(ts.get("time", 0))
        except (TypeError, ValueError):
            pass

    passed = total - fail - err - skip
    stats = {
        "total": total,
        "passed": passed,
        "failed": fail + err,
        "skipped": skip,
        "time": round(time_s, 1),
    }

    testcases = []
    for tc in root.findall(".//testcase"):
        f = tc.find("failure")
        e = tc.find("error")
        s = tc.find("skipped")
        msg = ""
        full_msg = ""
        if f is not None:
            status = "fail"
            raw = f.text or f.get("message", "") or ""
            msg = raw[:200]
            full_msg = raw
        elif e is not None:
            status = "error"
            raw = e.text or e.get("message", "") or ""
            msg = raw[:200]
            full_msg = raw
        elif s is not None:
            status = "skip"
        else:
            status = "pass"
        testcases.append(
            {
                "name": tc.get("name", "unknown"),
                "classname": tc.get("classname", ""),
                "time": tc.get("time", "?"),
                "status": status,
                "message": msg,
                "full_message": full_msg,
            }
        )
    return testcases, stats


def _escape(text, max_len=120):
    """Escape a message for use inside a Markdown table cell."""
    return text.replace("\n", " ").replace("|", "\\|")[:max_len]


def _status_icon(status):
    if status == "fail":
        return "❌"
    if status == "error":
        return "⚠️"
    if status == "skip":
        return "⏭️"
    return "✅"


def _module_key(classname):
    """Extract a module-level grouping key from a JUnit classname.

    Typical classname forms in triton JUnit XML:
      - ``test_core``
      - ``python.test.unit.language.test_core``
      - ``python.test.unit.language.test_core.SomeTestClass``

    Strategy: split on ``'.'``, drop trailing segments whose first character
    is uppercase (those are class names), then return the remaining dotted
    path.  If nothing remains, fall back to the full *classname*.
    """
    if not classname:
        return "<unknown>"
    parts = classname.split(".")
    # Strip trailing ClassName-like segments (start with uppercase)
    while len(parts) > 1 and parts[-1][:1].isupper():
        parts.pop()
    return ".".join(parts) if parts else classname


def _aggregate_by_file(testcases):
    """Aggregate testcases by module (file) key.

    Returns an *OrderedDict*-style list of ``(key, stats_dict)`` tuples
    sorted with failures first, then alphabetically.
    """
    agg = {}  # key -> {pass, fail, skip, total, time}
    for tc in testcases:
        key = _module_key(tc.get("classname", ""))
        if key not in agg:
            agg[key] = {"pass": 0, "fail": 0, "skip": 0, "total": 0, "time": 0.0}
        bucket = agg[key]
        bucket["total"] += 1
        if tc["status"] in ("fail", "error"):
            bucket["fail"] += 1
        elif tc["status"] == "pass":
            bucket["pass"] += 1
        else:
            bucket["skip"] += 1
        try:
            bucket["time"] += float(tc.get("time", 0))
        except (TypeError, ValueError):
            pass

    # Sort: failures first, then alphabetical
    return sorted(agg.items(), key=lambda kv: (kv[1]["fail"] == 0, kv[0]))


def _print_file_summary_table(testcases):
    """Print a collapsible per-file summary table."""
    aggregated = _aggregate_by_file(testcases)
    print("<details>")
    print(f"<summary>📋 Test Summary by File ({len(testcases)} tests, {len(aggregated)} files)</summary>")
    print()
    print("| File | ✅ Pass | ❌ Fail | ⏭️ Skip | Total | Time |")
    print("|------|---------|---------|---------|-------|------|")
    for file_key, stats in aggregated:
        fail_cell = f"**{stats['fail']}**" if stats["fail"] > 0 else str(stats["fail"])
        print(f"| `{file_key}` | {stats['pass']} | {fail_cell} | {stats['skip']} | {stats['total']} | {stats['time']:.1f}s |")
    print()
    print("</details>")
    print()


# ---------------------------------------------------------------------------
# Mode A — single-board report  (--xml)
# ---------------------------------------------------------------------------

def _report_single_board(args):
    """Generate Markdown report from a single JUnit XML.  Returns 0/1."""
    # Title
    title = args.title
    if args.run_number:
        title += f" #{args.run_number}"
    if args.branch:
        title += f" @ {args.branch}"
    print(f"# {title}")
    print()

    # Environment info
    env_info = {}
    if args.env_info and os.path.isfile(args.env_info):
        try:
            with open(args.env_info) as f:
                env_info = json.load(f)
        except Exception:
            pass

    if env_info or args.docker_image:
        print("## 环境")
        print()
        print("| Item | Value |")
        print("|------|-------|")
        if args.docker_image:
            print(f"| Docker Image | `{args.docker_image}` |")
        for key, val in env_info.items():
            print(f"| {key} | `{val}` |")
        print()

    # Check XML file
    if not os.path.isfile(args.xml):
        print("## 概览")
        print()
        if args.job_status:
            print(f"- **Status**: {args.job_status}")
        if args.duration:
            print(f"- **Duration**: {args.duration}s")
        if args.pods:
            print(f"- **Pods**: {args.pods}")
        print()
        print("> ⚠️ Test result XML not found — check pod logs in step output above.")
        return 1

    # Parse XML (reuse shared helper)
    parsed = _parse_xml(args.xml)
    if parsed is None:
        print("## 概览")
        print()
        print("> ⚠️ Test result XML could not be parsed.")
        return 1

    testcases, stats = parsed

    # Overview table — single-row format identical to original
    print("## 概览")
    print()
    print("| Total Cases | ✅ Passed | ❌ Failed | ⏭️ Skipped | ⏱️ Run Time |")
    print("|-------------|----------|----------|-----------|------------|")
    print(f"| {stats['total']} | {stats['passed']} | {stats['failed']} | {stats['skipped']} | {stats['time']}s |")
    print()

    # Failed test details
    failures = [tc for tc in testcases if tc["status"] in ("fail", "error")]

    print("## Failed Test Details")
    print()
    if failures:
        print("| Test | Time | Error |")
        print("|------|------|-------|")
        for tc in failures:
            print(f"| {tc['name']} | {tc['time']}s | {_escape(tc['message'])} |")
        print()
        for tc in failures:
            full = tc.get("full_message", "")
            if full:
                print("<details>")
                print(f"<summary>🔍 {tc['name']}</summary>")
                print()
                print("```")
                if len(full) > 2000:
                    print(full[:2000])
                    print(f"... (truncated, {len(full)} chars total)")
                else:
                    print(full)
                print("```")
                print()
                print("</details>")
                print()
    else:
        print("🎉 All tests passed — no failures to display.")
    print()

    # Per-file collapsible summary replaces the old NAS-only pointer.
    _print_file_summary_table(testcases)

    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# Mode B — combined multi-board report  (--result-root)
# ---------------------------------------------------------------------------

def _parse_boards_arg(raw):
    """Parse --boards value: JSON array string or comma-separated list."""
    if not raw or not raw.strip():
        return []
    raw = raw.strip()
    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = [str(b).strip().strip('"').strip("'") for b in parsed]
            return [b for b in result if b]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fall back to comma-separated
    result = [b.strip().strip('"').strip("'") for b in raw.split(",")]
    return [b for b in result if b]


def _report_combined(args):
    """Generate combined multi-board Markdown report.  Always returns 0."""
    root_dir = args.result_root
    expected_boards = _parse_boards_arg(args.boards)

    # Title
    title = args.title
    if args.run_number:
        title += f" #{args.run_number}"
    if args.branch:
        title += f" @ {args.branch}"
    print(f"# {title}")
    print()

    # Discover boards from filesystem
    discovered = []
    if os.path.isdir(root_dir):
        discovered = sorted(
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

    # Build union: expected (order preserved) + any discovered-but-unexpected
    if expected_boards:
        seen = set(expected_boards)
        boards = list(expected_boards)
        for d in discovered:
            if d not in seen:
                boards.append(d)
                seen.add(d)
    else:
        boards = discovered

    # Edge case: nothing at all
    if not boards:
        print(f"> ⚠️ No board results found under `{root_dir}` — all jobs may have failed during setup. Check the per-board job logs.")
        return 0

    # Pre-parse all boards
    board_data = {}  # board -> (testcases, stats) | None
    for board in boards:
        xml_path = os.path.join(root_dir, board, "test_result.xml")
        board_data[board] = _parse_xml(xml_path)

    # Status banner
    boards_with_results = sum(1 for b in boards if board_data[b] is not None)
    total_boards = len(boards)
    if boards_with_results == 0:
        print("> \U0001f534 **No test results were produced by any board — all jobs likely failed during environment setup.**")
        print()
    elif boards_with_results < total_boards:
        print(f"> ⚠️ **Partial results: {boards_with_results} of {total_boards} boards produced test results.**")
        print()

    # Combined overview table
    print("## 概览")
    print()
    print("| Board | Total | ✅ Passed | ❌ Failed | ⏭️ Skipped | ⏱️ Run Time |")
    print("|-------|-------|----------|----------|-----------|------------|")
    for board in boards:
        parsed = board_data[board]
        if parsed is None:
            print(f"| {board} | ⚠️ No results (setup failed?) | | | | |")
        else:
            _, stats = parsed
            print(
                f"| {board} | {stats['total']} | {stats['passed']} "
                f"| {stats['failed']} | {stats['skipped']} | {stats['time']}s |"
            )
    print()

    # Per-board detail sections
    for board in boards:
        print(f"## {board}")
        print()

        # Environment info
        env_path = os.path.join(root_dir, board, "env_info.json")
        if os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    env_info = json.load(f)
                if env_info:
                    print("| Item | Value |")
                    print("|------|-------|")
                    for key, val in env_info.items():
                        print(f"| {key} | `{val}` |")
                    print()
            except Exception:
                pass

        parsed = board_data[board]
        if parsed is None:
            print("> ⚠️ No test results were produced for this board. The job likely failed during environment setup (before tests ran). Check the pod logs in the per-board job output above.")
            print()
            continue

        testcases, stats = parsed

        # Failed test details
        failures = [tc for tc in testcases if tc["status"] in ("fail", "error")]
        if failures:
            print("| Test | Time | Error |")
            print("|------|------|-------|")
            for tc in failures:
                print(
                    f"| {tc['name']} | {tc['time']}s "
                    f"| {_escape(tc['message'])} |"
                )
            print()
            for tc in failures:
                full = tc.get("full_message", "")
                if full:
                    print("<details>")
                    print(f"<summary>🔍 {tc['name']}</summary>")
                    print()
                    print("```")
                    if len(full) > 2000:
                        print(full[:2000])
                        print(f"... (truncated, {len(full)} chars total)")
                    else:
                        print(full)
                    print("```")
                    print()
                    print("</details>")
                    print()
        else:
            print("🎉 All tests passed — no failures to display.")
        print()

        # Per-file collapsible summary replaces the old NAS-only pointer.
        _print_file_summary_table(testcases)

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test result summary for GitHub Actions (single-board or combined)"
    )
    # Mode A args
    parser.add_argument("--xml", default=None, help="[Mode A] Path to JUnit XML result file")
    parser.add_argument("--job-status", default="", help="[Mode A] Job status from ppu-scheduler-action")
    parser.add_argument("--duration", default="", help="[Mode A] Job duration in seconds")
    parser.add_argument("--pods", default="", help="[Mode A] Pod names")
    parser.add_argument("--docker-image", default="", help="[Mode A] Docker image used")
    parser.add_argument("--env-info", default="", help="[Mode A] Path to env info JSON file")
    # Mode B args
    parser.add_argument("--result-root", default=None, help="[Mode B] NAS directory containing per-board subdirectories")
    parser.add_argument("--boards", default="", help="[Mode B] Expected board list: JSON array or comma-separated")
    # Shared args
    parser.add_argument("--title", default="Triton PPU Test Results", help="Report title")
    parser.add_argument("--run-number", default="", help="CI run number")
    parser.add_argument("--branch", default="", help="Branch name")
    return parser.parse_args()


def _truncate_if_needed(text):
    """Ensure *text* fits within the GitHub Step Summary size budget.

    If the UTF-8 encoded size exceeds ``_MAX_SUMMARY_BYTES``, the output is
    truncated at a line boundary and a warning footer is appended.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_SUMMARY_BYTES:
        return text
    # Find the last newline within the budget (keep whole lines)
    cut = encoded[:_MAX_SUMMARY_BYTES].rfind(b"\n")
    if cut < 0:
        cut = _MAX_SUMMARY_BYTES
    truncated = encoded[:cut].decode("utf-8", errors="ignore")
    truncated += (
        "\n\n> ⚠️ Report truncated to fit GitHub Step Summary size limit (1024KB).\n"
    )
    return truncated


def main():
    args = parse_args()

    # Capture all print output so we can enforce the size limit before writing.
    buf = io.StringIO()
    _real_stdout = sys.stdout
    sys.stdout = buf

    exit_code = 0
    try:
        if args.result_root is not None:
            # Mode B — combined multi-board
            try:
                _report_combined(args)
            except Exception as exc:
                print(f"> ⚠️ Report generation error: {exc}", file=sys.stderr)
        elif args.xml is not None:
            # Mode A — single-board
            exit_code = _report_single_board(args)
        else:
            sys.stdout = _real_stdout
            print("Error: specify either --xml (single-board) or --result-root (combined).", file=sys.stderr)
            sys.exit(2)
    finally:
        sys.stdout = _real_stdout

    # Write (possibly truncated) report to real stdout
    report = _truncate_if_needed(buf.getvalue())
    _real_stdout.write(report)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
