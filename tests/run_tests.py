#!/usr/bin/env python3
"""
Comprehensive test runner script for all trading algorithm modules.
Run this from the project root directory with your virtual environment activated.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():  # pylint: disable=too-many-branches,too-many-statements
    """Run all unit tests for the trading algorithm."""

    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: You don't appear to be in a virtual environment.")
        print("Make sure to activate your venv first: source venv/bin/activate")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    # Get the tests directory
    tests_dir = Path(__file__).parent

    # Find all test files
    test_files = list(tests_dir.glob('test_*.py'))

    if not test_files:
        print("No test files found!")
        return False

    print("🧪 Running Trading Algorithm Unit Tests")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    # Run each test file
    for test_file in sorted(test_files):
        module_name = test_file.stem.replace(
            'test_', '').replace('_', ' ').title()
        print(f"\n📋 Running {module_name} tests...")
        print("-" * 40)

        try:
            # Run the test file
            result = subprocess.run([
                sys.executable, str(test_file)
            ], capture_output=True, text=True, check=False, cwd=tests_dir)

            # Parse output to count tests
            output_lines = result.stdout.split('\n')
            test_summary = [
                line for line in output_lines if 'Ran' in line and 'test' in line]

            if test_summary:
                print(test_summary[0])

            if result.stdout:
                # Print abbreviated output
                lines = result.stdout.split('\n')
                for line in lines:
                    if any(keyword in line for keyword in ['OK', 'FAILED', 'ERROR', 'Ran', '===', '---']):
                        print(line)

            if result.stderr and 'import' not in result.stderr.lower():
                print("STDERR:", result.stderr)

            if result.returncode == 0:
                print(f"✅ {module_name} tests passed!")
                passed_tests += 1
            else:
                print(f"❌ {module_name} tests failed!")
                failed_tests.append(module_name)

            total_tests += 1

        except (subprocess.SubprocessError, OSError) as e:
            print(f"❌ Error running {module_name} tests: {e}")
            failed_tests.append(module_name)
            total_tests += 1

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total test modules: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\n❌ Failed modules: {', '.join(failed_tests)}")
        print("\n📝 Note: Import errors are expected until all dependencies are properly configured.")
        print("   To fix import errors:")
        print("   1. Make sure you're in the project root directory")
        print("   2. Activate your virtual environment: source venv/bin/activate")
        print("   3. Install requirements: pip install -r requirements.txt")
        print("   4. Set up your environment variables (.env file)")
    else:
        print("\n🎉 All tests passed!")

    return len(failed_tests) == 0


def run_single_test(test_name):
    """Run a specific test module."""
    tests_dir = Path(__file__).parent
    test_file = tests_dir / f"test_{test_name}.py"

    if not test_file.exists():
        print(f"Test file test_{test_name}.py not found!")
        return False

    print(f"Running {test_name} tests...")

    try:
        result = subprocess.run([
            sys.executable, str(test_file)
        ], check=False, cwd=tests_dir)

        return result.returncode == 0

    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error running test: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test
        test_to_run = sys.argv[1]
        sys.exit(0 if run_single_test(test_to_run) else 1)
    else:
        # Run all tests
        sys.exit(0 if run_tests() else 1)
