#!/usr/bin/env python3

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path

TESTS_FILE = Path(__file__).parent / "ctd_tests.json"

def authenticate(jwt_path=None):
    if jwt_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = jwt_path
        print("[INFO] Using JWT for authentication.")
    else:
        print("[INFO] No JWT supplied. Falling back to 'gcloud auth login'...")
        try:
            subprocess.run(["gcloud", "auth", "login"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] gcloud login failed: {e}")
            sys.exit(1)

def load_tests():
    if not TESTS_FILE.exists():
        print(f"[ERROR] Test file {TESTS_FILE} not found.")
        sys.exit(1)
    with open(TESTS_FILE, "r") as f:
        return json.load(f)

def list_tests(tests):
    print("\nAvailable CTD Tests:\n")
    for key, val in tests.items():
        print(f"- {key}: {val['description']}")
    print()

def run_test(name, command):
    print(f"\n[RUNNING] {name}")
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        print("[STDOUT]:")
        print(result.stdout)
        print("[STDERR]:")
        print(result.stderr)
        if result.returncode != 0:
            print(f"[ERROR] Exit code: {result.returncode}")
    except Exception as e:
        print(f"[ERROR] Failed to run test '{name}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Run GCP Container Threat Detection Tests (x86 only)")
    parser.add_argument('--jwt', help="Path to service account JSON (optional)")
    parser.add_argument('--all', action='store_true', help="Run all available tests")
    parser.add_argument('--tests', nargs='+', help="Specify test keys to run (use --list to view options)")
    parser.add_argument('--list', action='store_true', help="List available test names")

    args = parser.parse_args()

    tests = load_tests()

    if args.list:
        list_tests(tests)
        sys.exit(0)

    authenticate(args.jwt)

    if args.all:
        for name, data in tests.items():
            run_test(name, data['command'])
    elif args.tests:
        for name in args.tests:
            if name not in tests:
                print(f"[WARNING] Test '{name}' not found. Skipping.")
                continue
            run_test(name, tests[name]['command'])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
