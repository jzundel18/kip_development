#!/bin/bash
################################################################################
# Setup cron jobs for KIP Daily Tasks (macOS/Linux)
################################################################################

set -e

echo ""
echo "========================================================================"
echo "  KIP Daily Tasks - Cron Setup (macOS/Linux)"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Path to Python (find the correct one)
PYTHON_PATH=$(which python3 || which python)

if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: Python not found in PATH"
    exit 1
fi

# Path to the main script
TASK_SCRIPT="$PROJECT_ROOT/run_daily_tasks.py"

# Check if script exists
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "ERROR: Script not found at $TASK_SCRIPT"
    echo "Please run this script from your project directory"
    exit 1
fi

echo "Project directory: $PROJECT_ROOT"
echo "Python path: $PYTHON_PATH"
echo "Task script: $TASK_SCRIPT"
echo ""

# Ensure script is executable
chmod +x "$TASK_SCRIPT"

# Create a temporary file for the new crontab
TEMP_CRON=$(mktemp)

# Export current crontab (if any)
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Remove any existing KIP entries
sed -i.bak '/KIP Daily Tasks/d' "$TEMP_CRON" 2>/dev/null || \
    sed -i '' '/KIP Daily Tasks/d' "$TEMP_CRON" 2>/dev/null || true

# Add header comment
echo "" >> "$TEMP_CRON"
echo "# ============================================================" >> "$TEMP_CRON"
echo "# KIP Daily Tasks - Auto-generated scheduled jobs" >> "$TEMP_CRON"
echo "# DO NOT EDIT MANUALLY - Use setup_cron.sh to modify" >> "$TEMP_CRON"
echo "# ============================================================" >> "$TEMP_CRON"

# Add cron jobs (MST times)
# Note: These times are based on system local time
# Adjust if your system is not set to Mountain Time

# Set PYTHONPATH to include project root for imports
export_path="export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""

# Morning tasks (8:00 AM) - Runs cleanup, refresh, and digest
echo "0 8 * * * $export_path && cd \"$PROJECT_ROOT\" && \"$PYTHON_PATH\" \"$TASK_SCRIPT\" --all >> \"$PROJECT_ROOT/logs/morning_tasks.log\" 2>&1  # KIP Daily Tasks - Morning" >> "$TEMP_CRON"

# Noon refresh (12:00 PM) - Refresh only
echo "0 12 * * * $export_path && cd \"$PROJECT_ROOT\" && \"$PYTHON_PATH\" \"$TASK_SCRIPT\" --refresh >> \"$PROJECT_ROOT/logs/noon_refresh.log\" 2>&1  # KIP Daily Tasks - Noon" >> "$TEMP_CRON"

# Afternoon refresh (4:00 PM) - Refresh only
echo "0 16 * * * $export_path && cd \"$PROJECT_ROOT\" && \"$PYTHON_PATH\" \"$TASK_SCRIPT\" --refresh >> \"$PROJECT_ROOT/logs/afternoon_refresh.log\" 2>&1  # KIP Daily Tasks - Afternoon" >> "$TEMP_CRON"

echo "# ============================================================" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Install the new crontab
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

echo "========================================================================"
echo "Cron Jobs Installed Successfully!"
echo "========================================================================"
echo ""
echo "The following jobs have been scheduled:"
echo "  1. Morning Tasks    - 8:00 AM  (cleanup + refresh + digest)"
echo "  2. Noon Refresh     - 12:00 PM (refresh only)"
echo "  3. Afternoon Refresh - 4:00 PM (refresh only)"
echo ""
echo "Logs will be saved to: $PROJECT_ROOT/logs/"
echo ""
echo "To view your crontab:"
echo "  crontab -l"
echo ""
echo "To test manually:"
echo "  python3 run_daily_tasks.py --all"
echo ""
echo "To remove cron jobs:"
echo "  ./remove_cron.sh"
echo ""
echo "IMPORTANT: Make sure your .env file is set up with all required"
echo "           environment variables (SUPABASE_DB_URL, SAM_KEYS, etc.)"
echo ""

# Display the relevant cron entries
echo "Installed cron entries:"
echo "------------------------------------------------------------------------"
crontab -l | grep "KIP Daily Tasks"
echo "------------------------------------------------------------------------"
echo ""

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "WARNING: .env file not found!"
    echo "         Create one with your environment variables before tasks run"
    echo ""
fi

echo "Setup complete!"