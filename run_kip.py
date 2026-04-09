#!/usr/bin/env python3
"""
KIP Pipeline Master Script
==========================
Runs all daily KIP maintenance tasks in order:
  1. run_cleanup.py - Remove expired solicitations
  2. run_auto_refresh.py - Fetch new solicitations from SAM.gov
  3. run_backfill_descriptions.py - Backfill descriptions for today's solicitations
  4. run_daily_digest.py - Send email digests

Usage:
    python run_kip_pipeline.py           # Run all tasks
    python run_kip_pipeline.py --dry-run # Run cleanup in dry-run mode
    python run_kip_pipeline.py --skip-digest  # Skip email digest
    python run_kip_pipeline.py --only cleanup,refresh  # Run specific tasks only

Environment Variables:
    SUPABASE_DB_URL - Database connection string (required)
    SAM_KEYS - SAM.gov API keys (required for refresh/backfill)
    OPENAI_API_KEY - OpenAI API key (required for digest)
    GMAIL_EMAIL - Gmail address for sending digests
    GMAIL_PASSWORD - Gmail app password
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.ENDC}")


def load_env():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        env_file = SCRIPT_DIR / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            print_success(f"Loaded environment from {env_file}")
            return True
        else:
            print_warning(f".env file not found at {env_file}")
            return False
    except ImportError:
        print_warning("python-dotenv not installed. Using system environment variables.")
        return False


def check_environment():
    """Verify required environment variables are set"""
    print_header("Environment Check")

    required = {
        'SUPABASE_DB_URL': 'Database connection',
        'SAM_KEYS': 'SAM.gov API keys',
    }

    optional = {
        'OPENAI_API_KEY': 'OpenAI API key (for digest)',
        'GMAIL_EMAIL': 'Gmail for sending digests',
        'GMAIL_PASSWORD': 'Gmail app password',
    }

    missing_required = []
    missing_optional = []

    for var, desc in required.items():
        if os.getenv(var):
            print_success(f"{var}: {desc}")
        else:
            print_error(f"{var}: {desc} (MISSING)")
            missing_required.append(var)

    for var, desc in optional.items():
        if os.getenv(var):
            print_success(f"{var}: {desc}")
        else:
            print_info(f"{var}: {desc} (not set)")
            missing_optional.append(var)

    if missing_required:
        print_error(f"\nMissing required: {', '.join(missing_required)}")
        return False

    return True


def run_script(script_name: str, description: str, extra_args: list = None) -> bool:
    """
    Run a Python script and return success/failure.

    Args:
        script_name: Name of the script file (e.g., 'run_cleanup.py')
        description: Human-readable description for logging
        extra_args: Additional command-line arguments

    Returns:
        True if successful, False otherwise
    """
    print_header(description)

    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False

    print_info(f"Running: {script_path}")
    print_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        start_time = time.time()

        # Run with real-time output
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=os.environ.copy()
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print_success(f"Completed in {elapsed:.1f} seconds")
            return True
        else:
            print_error(f"Failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print_error(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run KIP daily pipeline tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_kip_pipeline.py                    # Run all tasks
    python run_kip_pipeline.py --dry-run          # Cleanup in dry-run mode
    python run_kip_pipeline.py --skip-digest      # Skip email digest
    python run_kip_pipeline.py --only refresh,backfill  # Specific tasks only
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Run cleanup in dry-run mode (no actual deletions)')
    parser.add_argument('--skip-digest', action='store_true',
                        help='Skip the daily digest email step')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated list of tasks to run: cleanup,refresh,backfill,digest')

    args = parser.parse_args()

    # Parse --only argument
    tasks_to_run = None
    if args.only:
        tasks_to_run = [t.strip().lower() for t in args.only.split(',')]
        valid_tasks = {'cleanup', 'refresh', 'backfill', 'digest'}
        invalid = set(tasks_to_run) - valid_tasks
        if invalid:
            print_error(f"Invalid tasks: {invalid}. Valid: {valid_tasks}")
            sys.exit(1)

    # Print banner
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║               KIP PIPELINE - Daily Maintenance Tasks                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print_warning("DRY RUN MODE: Cleanup will not delete anything")
    if args.skip_digest:
        print_info("Digest step will be skipped")
    if tasks_to_run:
        print_info(f"Running only: {', '.join(tasks_to_run)}")
    print()

    # Load environment
    load_env()

    # Check environment
    if not check_environment():
        print_error("\nEnvironment check failed. Please set required variables.")
        sys.exit(1)

    # Track results
    results = {}
    start_time = time.time()

    # Define task order
    tasks = [
        ('cleanup', 'run_cleanup.py', 'Step 1: Cleanup Expired Solicitations'),
        ('refresh', 'run_auto_refresh.py', 'Step 2: Fetch New Solicitations from SAM.gov'),
        ('backfill', 'run_backfill_descriptions.py', 'Step 3: Backfill Descriptions (Today Only)'),
        ('digest', 'run_daily_digest.py', 'Step 4: Send Daily Email Digest'),
    ]

    for task_id, script, description in tasks:
        # Check if we should run this task
        if tasks_to_run and task_id not in tasks_to_run:
            print_info(f"Skipping {task_id} (not in --only list)")
            results[task_id] = None
            continue

        if task_id == 'digest' and args.skip_digest:
            print_info("Skipping digest (--skip-digest)")
            results[task_id] = None
            continue

        if task_id == 'digest':
            # Check if we have Gmail credentials
            if not (os.getenv('GMAIL_EMAIL') and os.getenv('GMAIL_PASSWORD')):
                print_warning("Skipping digest (Gmail credentials not configured)")
                results[task_id] = None
                continue

        # Build extra args
        extra_args = []
        if task_id == 'cleanup' and args.dry_run:
            extra_args.append('--dry-run')

        # Run the task
        success = run_script(script, description, extra_args if extra_args else None)
        results[task_id] = success

        # Small delay between tasks
        if success:
            time.sleep(2)

    # Print summary
    total_time = time.time() - start_time

    print_header("PIPELINE SUMMARY")

    for task_id, _, description in tasks:
        short_desc = description.split(': ')[1] if ': ' in description else description
        result = results.get(task_id)
        if result is None:
            print_info(f"{task_id.capitalize()}: Skipped")
        elif result:
            print_success(f"{task_id.capitalize()}: Success")
        else:
            print_error(f"{task_id.capitalize()}: Failed")

    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print_error(f"\nSome tasks failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print_success("\nAll tasks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)