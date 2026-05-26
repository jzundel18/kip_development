#!/usr/bin/env python3
"""
Task Statistics Generator
Reads a CSV file with Summary, Status, Assignee, and Priority columns and outputs
task counts per assignee broken down by priority (High vs Medium).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


# Statuses that count for each metric
DONE_STATUSES = {'Done'}
ACTIVE_STATUSES = {'Done', 'To Do', 'In Progress'}
# Total sprint = everything (Done, To Do, In Progress, In Review, On Hold, etc.)


def normalize_priority(priority):
    """Normalize priority into High or Medium buckets."""
    p = priority.strip().lower() if priority else ''
    if p in ('highest', 'high'):
        return 'High'
    elif p in ('medium',):
        return 'Medium'
    else:
        return None  # Skip other priorities


def read_csv_file(filepath):
    """Read CSV file and extract relevant columns."""
    tasks = []

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            required_cols = ['Summary', 'Status', 'Assignee', 'Priority']
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"Error: CSV must contain {required_cols} columns")
                print(f"\nAvailable columns in your CSV: {reader.fieldnames}")
                sys.exit(1)

            for row in reader:
                tasks.append({
                    'summary': row['Summary'],
                    'status': row['Status'],
                    'assignee': row['Assignee'],
                    'priority': row['Priority']
                })

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return tasks


def calculate_stats(tasks):
    """Calculate task statistics by assignee, priority, and status."""
    # Structure: assignee -> priority_bucket -> { done, active, total }
    stats = defaultdict(lambda: {
        'High': {'done': 0, 'active': 0, 'total': 0},
        'Medium': {'done': 0, 'active': 0, 'total': 0},
    })

    # Also track overall status totals
    status_totals = defaultdict(int)

    for task in tasks:
        assignee = task['assignee'].strip() if task['assignee'] else 'Unassigned'
        status = task['status'].strip() if task['status'] else 'No Status'
        priority_bucket = normalize_priority(task['priority'])

        status_totals[status] += 1

        if priority_bucket is None:
            continue  # Skip tasks that aren't High/Highest/Medium

        # Always count toward total sprint
        stats[assignee][priority_bucket]['total'] += 1

        # Count active (Done + To Do + In Progress)
        if status in ACTIVE_STATUSES:
            stats[assignee][priority_bucket]['active'] += 1

        # Count done
        if status in DONE_STATUSES:
            stats[assignee][priority_bucket]['done'] += 1

    return stats, status_totals


def generate_stats_report(stats, status_totals, total_tasks):
    """Generate formatted statistics report."""
    lines = []
    lines.append("=" * 60)
    lines.append("TASK STATISTICS")
    lines.append("=" * 60)
    lines.append(f"\nTotal Tasks: {total_tasks}")
    lines.append(f"Total Assignees: {len(stats)}")
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

    # Per-assignee priority breakdown
    lines.append("-" * 60)
    lines.append("TASKS BY ASSIGNEE (Priority Breakdown)")
    lines.append("-" * 60)

    for assignee in sorted(stats.keys()):
        lines.append(f"\n{assignee}:")

        for priority in ['High', 'Medium']:
            d = stats[assignee][priority]
            lines.append(f"  [{priority} Priority]")
            lines.append(f"    Done:                          {d['done']}")
            lines.append(f"    Done + To Do + In Progress:     {d['active']}")
            lines.append(f"    Total Sprint:                  {d['total']}")

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

    stats, status_totals = calculate_stats(tasks)

    report = generate_stats_report(stats, status_totals, total_tasks)

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