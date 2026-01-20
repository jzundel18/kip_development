#!/usr/bin/env python3
"""
Task Statistics Generator
Reads a CSV file with Summary, Status, and Assignee columns and outputs
task counts per assignee and per status category.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def read_csv_file(filepath):
    """Read CSV file and extract relevant columns."""
    tasks = []

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            required_cols = ['Summary', 'Status', 'Assignee']
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"Error: CSV must contain {required_cols} columns")
                print(f"\nAvailable columns in your CSV: {reader.fieldnames}")
                sys.exit(1)

            for row in reader:
                tasks.append({
                    'summary': row['Summary'],
                    'status': row['Status'],
                    'assignee': row['Assignee']
                })

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return tasks


def calculate_stats(tasks):
    """Calculate task statistics by assignee and status."""
    # Count tasks per assignee per status
    assignee_status_counts = defaultdict(lambda: defaultdict(int))
    # Count total tasks per status (across all assignees)
    status_totals = defaultdict(int)

    for task in tasks:
        assignee = task['assignee'].strip() if task['assignee'] else 'Unassigned'
        status = task['status'].strip() if task['status'] else 'No Status'

        assignee_status_counts[assignee][status] += 1
        status_totals[status] += 1

    return assignee_status_counts, status_totals


def generate_stats_report(assignee_status_counts, status_totals, total_tasks):
    """Generate formatted statistics report."""
    lines = []
    lines.append("=" * 60)
    lines.append("TASK STATISTICS")
    lines.append("=" * 60)
    lines.append(f"\nTotal Tasks: {total_tasks}")
    lines.append(f"Total Assignees: {len(assignee_status_counts)}")
    lines.append("")

    # Overall status breakdown
    lines.append("-" * 60)
    lines.append("OVERALL STATUS BREAKDOWN")
    lines.append("-" * 60)
    for status in sorted(status_totals.keys()):
        count = status_totals[status]
        pct = (count / total_tasks * 100) if total_tasks > 0 else 0
        lines.append(f"  {status}: {count} ({pct:.1f}%)")
    lines.append("")

    # Per-assignee breakdown
    lines.append("-" * 60)
    lines.append("TASKS BY ASSIGNEE")
    lines.append("-" * 60)

    for assignee in sorted(assignee_status_counts.keys()):
        statuses = assignee_status_counts[assignee]
        total = sum(statuses.values())
        lines.append(f"\n{assignee}: {total} total")

        for status in sorted(statuses.keys()):
            count = statuses[status]
            lines.append(f"    {status}: {count}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF STATISTICS")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python task_stats.py <csv_file>")
        sys.exit(1)

    csv_filepath = sys.argv[1]

    print(f"Reading CSV file: {csv_filepath}")
    tasks = read_csv_file(csv_filepath)
    total_tasks = len(tasks)
    print(f"Found {total_tasks} tasks")

    assignee_status_counts, status_totals = calculate_stats(tasks)

    report = generate_stats_report(assignee_status_counts, status_totals, total_tasks)

    # Print to console
    print("\n" + report)

    # Also save to file
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    input_filename = Path(csv_filepath).stem
    output_filepath = output_dir / f"{input_filename}_stats.txt"

    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nStats saved to: {output_filepath}")


if __name__ == "__main__":
    main()