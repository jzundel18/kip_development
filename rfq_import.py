#!/usr/bin/env python3
"""
RFQ Import Script
-----------------
Parses RFQ Excel files and imports them into Supabase.
Supports multiple RFQ formats:
  - Original format: Header info embedded in file, columns REQ/LI/PART NO./etc.
  - LDW format: RFQ number in filename, columns WBS/Req #/P/N/etc.

Tables created:
- rfqs: Main RFQ metadata (rfq_number, due_date, contact info, etc.)
- rfq_line_items: Individual line items from each RFQ

Usage:
    python rfq_import.py <path_to_rfq_excel_file>
    python rfq_import.py --create-tables  # Just create the tables
    python rfq_import.py --parse-only <file>  # Parse without importing
"""

import os
import sys
import re
from datetime import datetime
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.environ.get("SUPABASE_DB_URL")

# SQL to create tables (run this in Supabase SQL editor)
CREATE_TABLES_SQL = """
-- RFQs table: stores main RFQ metadata
CREATE TABLE IF NOT EXISTS rfqs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    rfq_number TEXT NOT NULL UNIQUE,
    due_date TIMESTAMPTZ,
    contact_name TEXT,
    contact_phone TEXT,
    contact_email TEXT,
    source_filename TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- RFQ Line Items table: stores individual items from each RFQ
CREATE TABLE IF NOT EXISTS rfq_line_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    rfq_id UUID REFERENCES rfqs(id) ON DELETE CASCADE,
    requisition_number TEXT,
    line_item TEXT,
    part_number TEXT NOT NULL,
    description TEXT,
    unit_of_measure TEXT,
    quantity INTEGER,
    required_delivery_date DATE,
    location TEXT,
    qa_requirements TEXT,
    naics_code TEXT,
    wbs_code TEXT,
    unit_cost DECIMAL(12, 4),
    total_cost DECIMAL(12, 4),
    lead_time_comments TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_rfq_line_items_rfq_id ON rfq_line_items(rfq_id);
CREATE INDEX IF NOT EXISTS idx_rfq_line_items_part_number ON rfq_line_items(part_number);
CREATE INDEX IF NOT EXISTS idx_rfqs_rfq_number ON rfqs(rfq_number);
CREATE INDEX IF NOT EXISTS idx_rfqs_due_date ON rfqs(due_date);

-- Enable RLS (Row Level Security) - adjust as needed
ALTER TABLE rfqs ENABLE ROW LEVEL SECURITY;
ALTER TABLE rfq_line_items ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated access (adjust based on your needs)
CREATE POLICY "Allow all access to rfqs" ON rfqs FOR ALL USING (true);
CREATE POLICY "Allow all access to rfq_line_items" ON rfq_line_items FOR ALL USING (true);
"""


def get_db_engine():
    """Initialize and return SQLAlchemy engine."""
    if not DATABASE_URL:
        raise ValueError(
            "Missing database credentials. Set SUPABASE_DB_URL environment variable."
        )
    return create_engine(DATABASE_URL)


def detect_format(df: pd.DataFrame, filename: str) -> str:
    """
    Detect which RFQ format the file is in.

    Returns:
        'original' - Original format with embedded header info
        'ldw' - LDW format with RFQ number in filename
    """
    # Check for LDW format indicators
    # Look for WBS column header anywhere in first 20 rows
    for idx in range(min(20, len(df))):
        row = df.iloc[idx]
        row_values = [str(v).strip().upper() if pd.notna(v) else "" for v in row]
        if "WBS" in row_values and "P/N" in row_values:
            return "ldw"

    # Check for original format indicators (REQ and PART NO. columns)
    for idx in range(min(20, len(df))):
        row = df.iloc[idx]
        row_values = [str(v).strip().upper() if pd.notna(v) else "" for v in row]
        if "REQ" in row_values and "PART NO." in row_values:
            return "original"

    # Default based on filename pattern
    if re.search(r'RFQ[_\s-]?\d+-\d+-LDW', filename, re.IGNORECASE):
        return "ldw"

    return "original"


def extract_rfq_number_from_filename(filename: str) -> Optional[str]:
    """Extract RFQ number from filename like 'RFQ 25-1201-LDW SPREADSHEET.xlsx'"""
    # Try pattern like "RFQ 25-1201-LDW" or "RFQ_25-1201-LDW"
    match = re.search(r'RFQ[_\s-]?(\d+-\d+-[A-Z]+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try pattern like "RFQ-25-1201"
    match = re.search(r'RFQ[_\s-]?(\d+-\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def parse_rfq_header_original(df: pd.DataFrame) -> dict:
    """Extract RFQ header information from the original Excel format."""
    header_info = {
        "rfq_number": None,
        "due_date": None,
        "contact_name": None,
        "contact_phone": None,
        "contact_email": None,
    }

    # Search through the first 15 rows for header information
    for idx in range(min(15, len(df))):
        row = df.iloc[idx]

        for col_idx, cell in enumerate(row):
            if pd.isna(cell):
                continue
            cell_str = str(cell).strip()

            # Look for RFQ number
            if cell_str == "RFQ:":
                # RFQ number should be in the next column with data
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        header_info["rfq_number"] = str(row.iloc[next_col]).strip()
                        break

            # Look for due date
            if "QUOTES ARE DUE BACK" in cell_str.upper():
                # Due date info might be in column 3
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        due_str = str(row.iloc[next_col]).strip()
                        # Try to parse "11/24/2025 at 1000 EST" format
                        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', due_str)
                        if match:
                            try:
                                header_info["due_date"] = datetime.strptime(match.group(1), "%m/%d/%Y")
                            except ValueError:
                                pass
                        break

            # Look for contact name (Attn:)
            if cell_str == "Attn:":
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        header_info["contact_name"] = str(row.iloc[next_col]).strip()
                        break

            # Look for phone
            if cell_str == "Phone:":
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        header_info["contact_phone"] = str(row.iloc[next_col]).strip()
                        break

            # Look for email
            if cell_str == "Email:":
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        header_info["contact_email"] = str(row.iloc[next_col]).strip()
                        break

    return header_info


def parse_rfq_header_ldw(df: pd.DataFrame, filename: str) -> dict:
    """Extract RFQ header information from LDW format (RFQ number from filename)."""
    header_info = {
        "rfq_number": extract_rfq_number_from_filename(filename),
        "due_date": None,
        "contact_name": None,
        "contact_phone": None,
        "contact_email": None,
    }

    # LDW format might have some header info in the first rows - try to find it
    for idx in range(min(15, len(df))):
        row = df.iloc[idx]
        for col_idx, cell in enumerate(row):
            if pd.isna(cell):
                continue
            cell_str = str(cell).strip()

            # Look for due date patterns
            if "DUE" in cell_str.upper() and "DATE" in cell_str.upper():
                for next_col in range(col_idx + 1, len(row)):
                    if pd.notna(row.iloc[next_col]):
                        due_str = str(row.iloc[next_col]).strip()
                        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', due_str)
                        if match:
                            try:
                                header_info["due_date"] = datetime.strptime(match.group(1), "%m/%d/%Y")
                            except ValueError:
                                pass
                        break

            # Look for contact/email patterns
            if "@" in cell_str and "." in cell_str:
                header_info["contact_email"] = cell_str

    return header_info


def find_header_row_original(df: pd.DataFrame) -> Optional[int]:
    """Find the row index containing the column headers for original format."""
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_values = [str(v).strip().upper() if pd.notna(v) else "" for v in row]
        if "REQ" in row_values and "PART NO." in row_values:
            return idx
    return None


def find_header_row_ldw(df: pd.DataFrame) -> Optional[int]:
    """Find the row index containing the column headers for LDW format."""
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_values = [str(v).strip().upper() if pd.notna(v) else "" for v in row]
        if "WBS" in row_values and "P/N" in row_values:
            return idx
    return None


def parse_rfq_line_items_original(df: pd.DataFrame, header_row: int) -> list[dict]:
    """Extract line items from original format RFQ."""
    line_items = []

    # Get column mapping from header row
    headers = df.iloc[header_row]
    col_map = {}
    for col_idx, header in enumerate(headers):
        if pd.notna(header):
            header_str = str(header).strip().upper()
            col_map[header_str] = col_idx

    # Parse each line item row
    for idx in range(header_row + 1, len(df)):
        row = df.iloc[idx]

        # Check if this is a valid line item row (has REQ number)
        req_col = col_map.get("REQ")
        if req_col is None or pd.isna(row.iloc[req_col]):
            continue

        req_value = str(row.iloc[req_col]).strip()
        if not req_value or not req_value.replace(" ", ""):
            continue

        line_item = {
            "requisition_number": req_value,
            "line_item": None,
            "part_number": None,
            "description": None,
            "unit_of_measure": None,
            "quantity": None,
            "required_delivery_date": None,
            "location": None,
            "qa_requirements": None,
            "naics_code": None,
            "wbs_code": None,
        }

        # Map columns to line item fields
        field_mapping = {
            "LI": "line_item",
            "PART NO.": "part_number",
            "DESCRIPTION": "description",
            "UOM": "unit_of_measure",
            "QTY": "quantity",
            "RDD": "required_delivery_date",
            "LOC": "location",
            "QA*": "qa_requirements",
            "NAICS": "naics_code",
        }

        for header_name, field_name in field_mapping.items():
            col_idx = col_map.get(header_name)
            if col_idx is not None and pd.notna(row.iloc[col_idx]):
                value = row.iloc[col_idx]

                if field_name == "quantity":
                    try:
                        line_item[field_name] = int(float(value))
                    except (ValueError, TypeError):
                        line_item[field_name] = None
                elif field_name == "required_delivery_date":
                    if isinstance(value, datetime):
                        line_item[field_name] = value.strftime("%Y-%m-%d")
                    elif isinstance(value, str):
                        try:
                            parsed = pd.to_datetime(value)
                            line_item[field_name] = parsed.strftime("%Y-%m-%d")
                        except:
                            line_item[field_name] = None
                else:
                    line_item[field_name] = str(value).strip()

        if line_item["part_number"]:
            line_items.append(line_item)

    return line_items


def parse_rfq_line_items_ldw(df: pd.DataFrame, header_row: int) -> list[dict]:
    """Extract line items from LDW format RFQ."""
    line_items = []

    # Get column mapping from header row
    headers = df.iloc[header_row]
    col_map = {}
    for col_idx, header in enumerate(headers):
        if pd.notna(header):
            header_str = str(header).strip().upper()
            col_map[header_str] = col_idx

    # Parse each line item row
    for idx in range(header_row + 1, len(df)):
        row = df.iloc[idx]

        # Check if this is a valid line item row (has WBS or P/N)
        pn_col = col_map.get("P/N")
        wbs_col = col_map.get("WBS")

        # Skip if no part number
        if pn_col is None or pd.isna(row.iloc[pn_col]):
            continue

        pn_value = str(row.iloc[pn_col]).strip()
        if not pn_value:
            continue

        line_item = {
            "requisition_number": None,
            "line_item": None,
            "part_number": pn_value,
            "description": None,
            "unit_of_measure": None,
            "quantity": None,
            "required_delivery_date": None,
            "location": None,
            "qa_requirements": None,
            "naics_code": None,
            "wbs_code": None,
        }

        # Map LDW columns to line item fields
        field_mapping = {
            "WBS": "wbs_code",
            "REQ #": "requisition_number",
            "REQ": "requisition_number",
            "P/N": "part_number",
            "DESCRIPTION": "description",
            "UOM": "unit_of_measure",
            "QTY": "quantity",
            "LOC": "location",
            "QA": "qa_requirements",
        }

        for header_name, field_name in field_mapping.items():
            col_idx = col_map.get(header_name)
            if col_idx is not None and pd.notna(row.iloc[col_idx]):
                value = row.iloc[col_idx]

                if field_name == "quantity":
                    try:
                        line_item[field_name] = int(float(value))
                    except (ValueError, TypeError):
                        line_item[field_name] = None
                elif field_name == "part_number":
                    # Already set above
                    pass
                else:
                    line_item[field_name] = str(value).strip()

        if line_item["part_number"]:
            line_items.append(line_item)

    return line_items


def import_rfq(filepath: str, engine) -> dict:
    """
    Import an RFQ Excel file into the database.
    Automatically detects format (original vs LDW).

    Returns dict with rfq_id, rfq_number, and count of line items imported.
    """
    filename = os.path.basename(filepath)

    # Read Excel file
    df = pd.read_excel(filepath, header=None)

    # Detect format
    format_type = detect_format(df, filename)
    print(f"Detected format: {format_type}")

    # Parse based on format
    if format_type == "ldw":
        header_info = parse_rfq_header_ldw(df, filename)
        header_row = find_header_row_ldw(df)
        if header_row is None:
            raise ValueError("Could not find line items header row (looking for WBS/P/N columns)")
        line_items = parse_rfq_line_items_ldw(df, header_row)
    else:
        header_info = parse_rfq_header_original(df)
        header_row = find_header_row_original(df)
        if header_row is None:
            raise ValueError("Could not find line items header row (looking for REQ/PART NO. columns)")
        line_items = parse_rfq_line_items_original(df, header_row)

    if not header_info["rfq_number"]:
        raise ValueError("Could not find RFQ number in the file or filename")

    print(f"Parsed RFQ: {header_info['rfq_number']}")
    print(f"  Due Date: {header_info['due_date']}")
    print(f"  Contact: {header_info['contact_name']}")
    print(f"  Header row found at: {header_row}")
    print(f"  Found {len(line_items)} line items")

    with engine.connect() as conn:
        # Check if RFQ already exists
        result = conn.execute(
            text("SELECT id FROM rfqs WHERE rfq_number = :rfq_number"),
            {"rfq_number": header_info["rfq_number"]}
        )
        existing = result.fetchone()

        if existing:
            rfq_id = existing[0]
            # Update existing RFQ
            conn.execute(
                text("""
                    UPDATE rfqs SET 
                        due_date = :due_date,
                        contact_name = :contact_name,
                        contact_phone = :contact_phone,
                        contact_email = :contact_email,
                        source_filename = :source_filename,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {
                    "id": rfq_id,
                    "due_date": header_info["due_date"],
                    "contact_name": header_info["contact_name"],
                    "contact_phone": header_info["contact_phone"],
                    "contact_email": header_info["contact_email"],
                    "source_filename": filename,
                }
            )
            # Delete existing line items to replace
            conn.execute(
                text("DELETE FROM rfq_line_items WHERE rfq_id = :rfq_id"),
                {"rfq_id": rfq_id}
            )
            print(f"  Updated existing RFQ (ID: {rfq_id})")
        else:
            # Insert new RFQ
            result = conn.execute(
                text("""
                    INSERT INTO rfqs (rfq_number, due_date, contact_name, contact_phone, contact_email, source_filename)
                    VALUES (:rfq_number, :due_date, :contact_name, :contact_phone, :contact_email, :source_filename)
                    RETURNING id
                """),
                {
                    "rfq_number": header_info["rfq_number"],
                    "due_date": header_info["due_date"],
                    "contact_name": header_info["contact_name"],
                    "contact_phone": header_info["contact_phone"],
                    "contact_email": header_info["contact_email"],
                    "source_filename": filename,
                }
            )
            rfq_id = result.fetchone()[0]
            print(f"  Created new RFQ (ID: {rfq_id})")

        # Insert line items
        for item in line_items:
            conn.execute(
                text("""
                    INSERT INTO rfq_line_items 
                    (rfq_id, requisition_number, line_item, part_number, description, 
                     unit_of_measure, quantity, required_delivery_date, location, 
                     qa_requirements, naics_code, wbs_code)
                    VALUES 
                    (:rfq_id, :requisition_number, :line_item, :part_number, :description,
                     :unit_of_measure, :quantity, :required_delivery_date, :location,
                     :qa_requirements, :naics_code, :wbs_code)
                """),
                {
                    "rfq_id": rfq_id,
                    "requisition_number": item["requisition_number"],
                    "line_item": item["line_item"],
                    "part_number": item["part_number"],
                    "description": item["description"],
                    "unit_of_measure": item["unit_of_measure"],
                    "quantity": item["quantity"],
                    "required_delivery_date": item["required_delivery_date"],
                    "location": item["location"],
                    "qa_requirements": item["qa_requirements"],
                    "naics_code": item["naics_code"],
                    "wbs_code": item["wbs_code"],
                }
            )

        conn.commit()
        print(f"  Inserted {len(line_items)} line items")

    return {
        "rfq_id": rfq_id,
        "rfq_number": header_info["rfq_number"],
        "line_items_count": len(line_items),
        "format": format_type,
    }


def print_create_tables_sql():
    """Print the SQL to create the necessary tables."""
    print("=" * 60)
    print("Run this SQL in your Supabase SQL Editor to create tables:")
    print("=" * 60)
    print(CREATE_TABLES_SQL)
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rfq_import.py <path_to_rfq_excel_file>")
        print("  python rfq_import.py --create-tables  # Print SQL to create tables")
        print("  python rfq_import.py --parse-only <file>  # Parse without importing")
        print("")
        print("Supported formats:")
        print("  - Original: Header info embedded, columns REQ/LI/PART NO./DESCRIPTION/etc.")
        print("  - LDW: RFQ number in filename, columns WBS/Req #/P/N/Description/etc.")
        sys.exit(1)

    if sys.argv[1] == "--create-tables":
        print_create_tables_sql()
        sys.exit(0)

    if sys.argv[1] == "--parse-only":
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            sys.exit(1)
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        filename = os.path.basename(filepath)
        df = pd.read_excel(filepath, header=None)

        format_type = detect_format(df, filename)
        print(f"\n=== Detected Format: {format_type} ===")

        if format_type == "ldw":
            header_info = parse_rfq_header_ldw(df, filename)
            header_row = find_header_row_ldw(df)
            line_items = parse_rfq_line_items_ldw(df, header_row) if header_row else []
        else:
            header_info = parse_rfq_header_original(df)
            header_row = find_header_row_original(df)
            line_items = parse_rfq_line_items_original(df, header_row) if header_row else []

        print(f"\n=== RFQ Header ===")
        for key, value in header_info.items():
            print(f"  {key}: {value}")

        print(f"\n=== Header Row: {header_row} ===")

        print(f"\n=== Line Items ({len(line_items)}) ===")
        for i, item in enumerate(line_items[:10]):  # Show first 10
            desc = item['description'][:50] if item['description'] else 'N/A'
            wbs = f" [WBS: {item['wbs_code']}]" if item.get('wbs_code') else ""
            print(f"  {i + 1}. {item['part_number']}: {desc}...{wbs}")

        if len(line_items) > 10:
            print(f"  ... and {len(line_items) - 10} more items")

        sys.exit(0)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    try:
        engine = get_db_engine()
        result = import_rfq(filepath, engine)
        print(
            f"\nSuccess! Imported RFQ {result['rfq_number']} ({result['format']} format) with {result['line_items_count']} line items")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()