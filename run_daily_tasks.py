#!/usr/bin/env python3
"""
Master script to run all daily KIP maintenance tasks in sequence.
This script runs cleanup, auto-refresh (3x), description backfill, categorization, and daily digest.

Usage:
    python run_daily_tasks.py [--all] [--cleanup] [--refresh] [--digest]

Options:
    --all           Run all tasks (default if no options specified)
    --cleanup       Run cleanup only
    --refresh       Run auto-refresh only
    --categorize    Run categorization only
    --digest        Run daily digest only
    --dry-run       Run cleanup in dry-run mode
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_file = PROJECT_ROOT / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        # Verify it actually loaded something
        test_var = os.getenv('SUPABASE_DB_URL')
        if test_var:
            pass  # Successfully loaded
    else:
        print(f"⚠ Warning: .env file not found at {env_file}")
        print(f"   Create one with your environment variables")
except ImportError:
    print("=" * 70)
    print("ERROR: python-dotenv is not installed")
    print("=" * 70)
    print()
    print("This package is required to load environment variables from .env file")
    print()
    print("Install it with:")
    print("  pip install python-dotenv")
    print()
    print("Or install all requirements:")
    print("  pip install -r requirements.txt")
    print()
    sys.exit(1)


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}")
    print(f"{message}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def run_script(script_name, description, env_vars=None):
    """
    Run a Python script and capture output.

    Args:
        script_name: Name of the script to run (relative to scripts/ directory)
        description: Human-readable description
        env_vars: Dictionary of additional environment variables

    Returns:
        True if successful, False otherwise
    """
    print_header(f"{description}")
    print_info(f"Starting: {script_name}")
    print_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Find the script
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        # Try without scripts/ prefix
        script_path = PROJECT_ROOT / script_name
        if not script_path.exists():
            print_error(f"Script not found: {script_name}")
            return False

    # Prepare environment
    env = os.environ.copy()

    # CRITICAL: Add project root to PYTHONPATH so scripts can import from project
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{PROJECT_ROOT}{os.pathsep}{current_pythonpath}"
    else:
        env['PYTHONPATH'] = str(PROJECT_ROOT)

    if env_vars:
        env.update(env_vars)

    # Run the script
    try:
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)  # Run from project root
        )

        elapsed_time = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(f"{Colors.WARNING}Warnings/Errors:{Colors.ENDC}")
            print(result.stderr)

        # Check success
        if result.returncode == 0:
            print_success(f"Completed successfully in {elapsed_time:.1f} seconds")
            return True
        else:
            print_error(f"Failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print_error(f"Exception occurred: {e}")
        return False


def run_cleanup(dry_run=False):
    """Run the cleanup script"""
    env_vars = {
        'CLEANUP_DRY_RUN': 'true' if dry_run else 'false',
        'CLEANUP_BATCH_SIZE': '1000',
        'CLEANUP_OLD_SOLICITATIONS_DAYS': '30'
    }
    return run_script(
        "cleanup_expired_solicitations.py",
        "CLEANUP: Removing Expired Solicitations",
        env_vars
    )


def run_auto_refresh():
    """Run the auto-refresh script"""
    env_vars = {
        'DAYS_BACK': '1',
        'PAGE_SIZE': '200',
        'MAX_RECORDS': '2000'
    }
    return run_script(
        "auto_refresh.py",
        "AUTO REFRESH: Fetching New Solicitations from SAM.gov",
        env_vars
    )


def run_backfill():
    """Run the description backfill script"""
    env_vars = {
        'BACKFILL_BATCH': '50',
        'BACKFILL_MAX': '500',
        'BACKFILL_DELAY_SEC': '0.5'
    }
    return run_script(
        "backfill_descriptions.py",
        "BACKFILL: Fetching Missing Descriptions",
        env_vars,
        unbuffered=True
    )

def run_categorize():
    """Run the categorization script"""
    env_vars = {
        'CATEGORIZE_BATCH_SIZE': '100',
        'CATEGORIZE_MAX_TOTAL': '1000'
    }
    return run_script(
        "categorize_solicitations.py",
        "CATEGORIZE: AI Classification of Solicitations (parts/services/research)",
        env_vars,
        unbuffered=True
    )


def run_daily_digest():
    """Run the daily digest script"""
    env_vars = {
        'DIGEST_MAX_RESULTS': '5',
        'DIGEST_MIN_SCORE': '60',
        'DIGEST_PREFILTER_CANDIDATES': '25'
    }
    return run_script(
        "daily_digest.py",
        "DAILY DIGEST: Sending Email Summaries (Research Only)",
        env_vars
    )


def check_environment():
    """Check that required environment variables are set"""
    print_header("Environment Check")

    required = {
        'SUPABASE_DB_URL': 'Database connection',
        'SAM_KEYS': 'SAM.gov API keys',
        'OPENAI_API_KEY': 'OpenAI API key',
    }

    optional = {
        'GMAIL_EMAIL': 'Gmail for sending digests',
        'GMAIL_PASSWORD': 'Gmail app password',
    }

    missing = []
    missing_optional = []

    for var, desc in required.items():
        if os.getenv(var):
            print_success(f"{var} - {desc}")
        else:
            print_error(f"{var} - {desc} (MISSING)")
            missing.append(var)

    for var, desc in optional.items():
        if os.getenv(var):
            print_success(f"{var} - {desc}")
        else:
            print_info(f"{var} - {desc} (optional, not set)")
            missing_optional.append(var)

    if missing:
        print_error(f"\nMissing required environment variables: {', '.join(missing)}")
        print_info("Please set these in your .env file or system environment")
        return False

    if missing_optional:
        print_info(f"\nNote: Daily digest will be skipped (missing {', '.join(missing_optional)})")

    print_success("\nEnvironment check passed!")
    return True


def main():
    """Main execution function"""
    # Parse arguments
    args = sys.argv[1:]

    run_all = '--all' in args or len([a for a in args if a.startswith('--') and a != '--dry-run']) == 0
    run_cleanup_flag = '--cleanup' in args or run_all
    run_refresh_flag = '--refresh' in args or run_all
    run_categorize_flag = '--categorize' in args or run_all
    run_digest_flag = '--digest' in args or run_all
    dry_run = '--dry-run' in args

    # Print banner
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                   KIP DAILY MAINTENANCE TASKS                     ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Track results
    results = {}
    start_time = time.time()

    # Run tasks in sequence
    if run_cleanup_flag:
        results['cleanup'] = run_cleanup(dry_run)
        time.sleep(2)  # Brief pause between tasks

    if run_refresh_flag:
        # Step 1: Run auto-refresh to get new solicitations
        results['auto_refresh'] = run_auto_refresh()
        time.sleep(2)

        # Step 2: Run backfill after refresh to fetch missing descriptions
        results['backfill'] = run_backfill()
        time.sleep(2)

    if run_digest_flag:
        # Only run digest if we have Gmail credentials
        if os.getenv('GMAIL_EMAIL') and os.getenv('GMAIL_PASSWORD'):
            results['daily_digest'] = run_daily_digest()
        else:
            print_info("Skipping daily digest (Gmail credentials not configured)")
            results['daily_digest'] = None

    # Print summary
    total_time = time.time() - start_time
    print_header("SUMMARY")

    # Task order for display
    task_order = ['cleanup', 'auto_refresh', 'backfill', 'daily_digest']

    for task in task_order:
        if task in results:
            success = results[task]
            if success is None:
                print_info(f"{task}: Skipped")
            elif success:
                print_success(f"{task}: Success")
            else:
                print_error(f"{task}: Failed")

    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Exit code
    if any(v is False for v in results.values()):
        print_error("\n⚠ Some tasks failed. Check logs above for details.")
        sys.exit(1)
    else:
        print_success("\n✓ All tasks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)