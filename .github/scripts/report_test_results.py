#!/usr/bin/env python3
"""
Generate GitHub Step Summary from JUnit XML test results.

Usage:
    python report_test_results.py --xml <path> [options]

Output is Markdown printed to stdout, intended to be appended to $GITHUB_STEP_SUMMARY.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description="Generate test result summary for GitHub Actions")
    parser.add_argument("--xml", required=True, help="Path to JUnit XML result file")
    parser.add_argument("--title", default="Triton PPU Test Results", help="Report title")
    parser.add_argument("--run-number", default="", help="CI run number")
    parser.add_argument("--branch", default="", help="Branch name")
    parser.add_argument("--job-status", default="", help="Job status from ppu-scheduler-action")
    parser.add_argument("--duration", default="", help="Job duration in seconds")
    parser.add_argument("--pods", default="", help="Pod names")
    parser.add_argument("--docker-image", default="", help="Docker image used")
    parser.add_argument("--env-info", default="", help="Path to env info JSON file (sdk/torch versions)")
    return parser.parse_args()


def generate_report(args):
    """Generate Markdown report from JUnit XML."""
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
        import json
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

    # Parse XML
    tree = ET.parse(args.xml)
    root = tree.getroot()

    # Overview table
    for ts in root.findall(".//testsuite"):
        total = int(ts.get("tests", 0))
        fail = int(ts.get("failures", 0))
        err = int(ts.get("errors", 0))
        skip = int(ts.get("skipped", 0))
        passed = total - fail - err - skip
        time_s = ts.get("time", "?")

        print("## 概览")
        print()
        print("| Total Cases | ✅ Passed | ❌ Failed | ⏭️ Skipped | ⏱️ Run Time |")
        print("|-------------|----------|----------|-----------|------------|")
        print(f"| {total} | {passed} | {fail + err} | {skip} | {time_s}s |")
        print()

    # Failed test details
    failures = []
    for tc in root.findall(".//testcase"):
        f = tc.find("failure")
        e = tc.find("error")
        if f is not None or e is not None:
            msg = (f.text if f is not None else e.text) or ""
            failures.append((tc.get("name", "?"), tc.get("time", "?"), msg[:200]))

    print("## Failed Test Details")
    print()
    if failures:
        print("| Test | Time | Error |")
        print("|------|------|-------|")
        for name, t, msg in failures:
            msg_oneline = msg.replace("\n", " ").replace("|", "\\|")[:120]
            print(f"| {name} | {t}s | {msg_oneline} |")
    else:
        print("🎉 All tests passed — no failures to display.")
    print()

    # Full test list (collapsible)
    testcases = root.findall(".//testcase")
    if testcases:
        print("<details>")
        print(f"<summary>📋 All Test Cases ({len(testcases)} tests)</summary>")
        print()
        print("| Status | Test | Time |")
        print("|--------|------|------|")
        for tc in testcases:
            f = tc.find("failure")
            e = tc.find("error")
            if f is not None:
                icon = "❌"
            elif e is not None:
                icon = "⚠️"
            else:
                icon = "✅"
            print(f'| {icon} | {tc.get("name", "unknown")} | {tc.get("time", "?")}s |')
        print()
        print("</details>")

    return 0 if not failures else 1


def main():
    args = parse_args()
    sys.exit(generate_report(args))


if __name__ == "__main__":
    main()
