#!/bin/bash
################################################################################
# Remove cron jobs for KIP Daily Tasks
################################################################################

set -e

echo ""
echo "========================================================================"
echo "  KIP Daily Tasks - Remove Cron Jobs"
echo "========================================================================"
echo ""

# Create a temporary file for the new crontab
TEMP_CRON=$(mktemp)

# Export current crontab (if any)
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Remove KIP entries
sed -i.bak '/KIP Daily Tasks/d' "$TEMP_CRON" 2>/dev/null || \
    sed -i '' '/KIP Daily Tasks/d' "$TEMP_CRON" 2>/dev/null || true

# Also remove the header/footer comments
sed -i.bak '/# ============================================================/d' "$TEMP_CRON" 2>/dev/null || \
    sed -i '' '/# ============================================================/d' "$TEMP_CRON" 2>/dev/null || true

sed -i.bak '/# DO NOT EDIT MANUALLY/d' "$TEMP_CRON" 2>/dev/null || \
    sed -i '' '/# DO NOT EDIT MANUALLY/d' "$TEMP_CRON" 2>/dev/null || true

# Install the cleaned crontab
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

echo "========================================================================"
echo "Cron Jobs Removed"
echo "========================================================================"
echo ""
echo "All KIP scheduled tasks have been removed from crontab"
echo ""
echo "To verify:"
echo "  crontab -l"
echo ""