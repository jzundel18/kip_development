#!/usr/bin/env python3
"""
Task Report Generator
Processes a CSV file with Summary, Status, and Assignee columns and creates
a formatted text report organized by assignee and status priority.
"""

import csv
import sys
import os
from collections import defaultdict
from pathlib import Path


def read_csv_file(filepath):
    """Read CSV file and extract relevant columns."""
    tasks = []

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            # Check if required columns exist
            required_cols = ['Summary', 'Status', 'Assignee', 'Parent summary']
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"Error: CSV must contain {required_cols} columns")
                print(f"\nAvailable columns in your CSV: {reader.fieldnames}")
                sys.exit(1)

            for row in reader:
                tasks.append({
                    'summary': row['Summary'],
                    'status': row['Status'],
                    'assignee': row['Assignee'],
                    'parent_summary': row['Parent summary']
                })

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return tasks


def organize_tasks_by_assignee(tasks):
    """Organize tasks by assignee and status."""
    organized = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        assignee = task['assignee'].strip() if task['assignee'] else 'Unassigned'
        status = task['status'].strip() if task['status'] else 'No Status'
        parent = task['parent_summary'].strip() if task['parent_summary'] else ''

        # Create task entry with parent summary
        task_entry = {
            'summary': task['summary'],
            'parent_summary': parent
        }

        # Keep the original status category
        organized[assignee][status].append(task_entry)

    return organized


def generate_report(organized_tasks):
    """Generate formatted text report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TASK REPORT BY ASSIGNEE")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Sort assignees alphabetically
    for assignee in sorted(organized_tasks.keys()):
        statuses = organized_tasks[assignee]

        report_lines.append("-" * 80)
        report_lines.append(f"ASSIGNEE: {assignee}")
        report_lines.append("-" * 80)
        report_lines.append("")

        # Sort statuses alphabetically and display each
        for status in sorted(statuses.keys()):
            tasks = statuses[status]
            if tasks:
                report_lines.append(f"● {status.upper()}:")
                for i, task in enumerate(tasks, 1):
                    if task['parent_summary']:
                        report_lines.append(f"  {i}. ({task['parent_summary']}) {task['summary']}")
                    else:
                        report_lines.append(f"  {i}. {task['summary']}")
                report_lines.append("")

        # If assignee has no tasks in any category
        if not statuses:
            report_lines.append("  No tasks assigned")
            report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python task_report.py <csv_file>")
        sys.exit(1)

    csv_filepath = sys.argv[1]

    # Read and process CSV
    print(f"Reading CSV file: {csv_filepath}")
    tasks = read_csv_file(csv_filepath)
    print(f"Found {len(tasks)} tasks")

    # Organize tasks
    organized = organize_tasks_by_assignee(tasks)

    # Generate report
    report = generate_report(organized)

    # Create outputs directory if it doesn't exist
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Generate output filename
    input_filename = Path(csv_filepath).stem
    output_filepath = output_dir / f"{input_filename}_report.txt"

    # Write report to file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report generated: {output_filepath}")
    print(f"Organized tasks for {len(organized)} assignee(s)")


if __name__ == "__main__":
    main()