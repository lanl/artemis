#!/usr/bin/env python3
"""
Script to format .in input files and generate git-compatible diffs.

Formatting rules:
1. Within each section (between < > markers), align all = signs
2. Within each section, align all # comment markers
3. One space between key = value
4. One space after commas in comma-separated values
5. One space after leading # in comments
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple
import subprocess


def parse_line(line: str) -> Tuple[str, str, str, str, bool]:
    """
    Parse a line into components: (indent, key, value, comment, has_continuation)
    Returns empty strings for non-parameter lines.
    """
    stripped = line.strip()
    
    # Check if it's a parameter line (has =)
    if '=' in line and not stripped.startswith('#') and not stripped.startswith('<'):
        indent = line[:len(line) - len(line.lstrip())]
        
        # Split on first =
        parts = line.split('=', 1)
        key = parts[0].strip()
        value_part = parts[1] if len(parts) > 1 else ''
        
        # Extract comment
        comment = ''
        value_clean = value_part
        if '#' in value_part:
            comment_pos = value_part.find('#')
            comment = value_part[comment_pos:].strip()
            value_clean = value_part[:comment_pos]
        
        # Check for continuation
        has_continuation = value_clean.rstrip().endswith('&')
        if has_continuation:
            value_clean = value_clean.rstrip()[:-1]
        
        value_clean = value_clean.strip()
        
        return (indent, key, value_clean, comment, has_continuation)
    
    return ('', '', '', '', False)


def format_block_lines(lines: List[str]) -> List[str]:
    """
    Format a block of parameter lines with aligned = and # signs.
    """
    if not lines:
        return lines
    
    # Parse all lines in the block
    parsed = []
    for line in lines:
        indent, key, value, comment, has_cont = parse_line(line)
        if key:  # Only parameter lines
            parsed.append((indent, key, value, comment, has_cont, line))
    
    if not parsed:
        return lines
    
    # Find the longest key to determine = alignment
    max_key_len = max(len(key) for _, key, _, _, _, _ in parsed)
    
    # Find the longest "key = value" part (with key padding) to determine comment alignment
    max_kv_len = 0
    for indent, key, value, comment, has_cont, _ in parsed:
        # Include the key padding in the calculation
        kv_str = key + ' ' * (max_key_len - len(key)) + ' = ' + value
        if has_cont:
            kv_str += '  &'
        max_kv_len = max(max_kv_len, len(kv_str))
    
    # Reconstruct formatted lines
    formatted = []
    for indent, key, value, comment, has_cont, original in parsed:
        # Format: key = value
        new_line = indent + key + ' ' * (max_key_len - len(key)) + ' = ' + value
        
        # Add continuation marker if present
        if has_cont:
            new_line += '  &'
        
        # Add comment aligned
        if comment:
            # Ensure comment starts with "# " (one space after #)
            comment_text = comment.lstrip('#').strip()
            comment_formatted = '# ' + comment_text
            
            # Calculate current length for alignment
            current_len = len(new_line)
            # Align to at least 2 spaces after the longest key=value
            target_col = max_kv_len + 2
            if current_len < target_col:
                new_line += ' ' * (target_col - current_len)
            else:
                new_line += '  '
            new_line += comment_formatted
        
        formatted.append(new_line + '\n')
    
    return formatted


def format_continuation_line(line: str) -> str:
    """Format a continuation line (no = sign, just values)."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # Extract comment
    comment = ''
    value_part = stripped
    if '#' in stripped:
        comment_pos = stripped.find('#')
        comment = stripped[comment_pos:].strip()
        value_part = stripped[:comment_pos]
    
    # Check for continuation
    has_continuation = value_part.rstrip().endswith('&')
    if has_continuation:
        value_part = value_part.rstrip()[:-1]
    
    value_part = value_part.strip()
    
    # Normalize comma-separated values (one space after comma)
    if ',' in value_part:
        values = [v.strip() for v in value_part.split(',')]
        value_part = ', '.join(values)
    
    # Reconstruct
    new_line = indent + value_part
    if has_continuation:
        new_line += '  &'
    if comment:
        comment_text = comment.lstrip('#').strip()
        new_line += '  # ' + comment_text
    
    return new_line + '\n'


def format_file_content(content: str) -> str:
    """
    Format the entire file content.
    """
    lines = content.split('\n')
    formatted_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Pass through: empty lines, comments, section headers
        if not stripped or stripped.startswith('#') or stripped.startswith('<'):
            formatted_lines.append(line + '\n' if i < len(lines) - 1 or line else line)
            i += 1
            continue
        
        # Check if this is a parameter line or continuation line
        if '=' in line:
            # Start of a parameter block - collect all consecutive parameter lines
            block_start = i
            block_lines = []
            
            while i < len(lines):
                current = lines[i]
                current_stripped = current.strip()
                
                # Stop at empty line, comment, or section header
                if not current_stripped or current_stripped.startswith('#') or current_stripped.startswith('<'):
                    break
                
                # Stop if this is a continuation line (no =)
                if '=' not in current:
                    break
                
                block_lines.append(current)
                i += 1
            
            # Format the block
            formatted_block = format_block_lines(block_lines)
            formatted_lines.extend(formatted_block)
        else:
            # This is a continuation line (no =)
            formatted_lines.append(format_continuation_line(line))
            i += 1
    
    # Join and handle final newline properly
    result = ''.join(formatted_lines)
    if content and not content.endswith('\n') and result.endswith('\n'):
        result = result[:-1]
    
    return result


def generate_diff(filepath: Path, original: str, formatted: str) -> str:
    """
    Generate a unified diff between original and formatted content.
    """
    import difflib
    
    original_lines = original.splitlines(keepends=True)
    formatted_lines = formatted.splitlines(keepends=True)
    
    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        original_lines,
        formatted_lines,
        fromfile=f'a/{filepath.relative_to(Path.cwd())}',
        tofile=f'b/{filepath.relative_to(Path.cwd())}',
        lineterm='\n'
    ))
    
    return ''.join(diff_lines)


def main():
    """Main function to process all .in files."""
    import os
    
    inputs_dir = Path('/home/adam/artemis/inputs')
    
    # Find all .in files
    in_files = sorted(inputs_dir.rglob('*.in'))
    
    # Check if output is being redirected
    is_redirected = not sys.stdout.isatty()
    
    if not is_redirected:
        print(f"Checking {len(in_files)} .in files for formatting issues...\n", file=sys.stderr)
    
    files_with_issues = []
    all_diffs = []
    
    for filepath in in_files:
        with open(filepath, 'r') as f:
            original_content = f.read()
        
        formatted_content = format_file_content(original_content)
        
        # Check if formatting changed anything
        if original_content != formatted_content:
            files_with_issues.append(filepath)
            
            # Generate diff
            diff = generate_diff(filepath, original_content, formatted_content)
            if diff:
                all_diffs.append(diff)
    
    # Output all diffs
    for diff in all_diffs:
        print(diff, end='')
    
    if files_with_issues:
        if not is_redirected:
            print(f"\nFound formatting issues in {len(files_with_issues)} file(s):", file=sys.stderr)
            for f in files_with_issues:
                print(f"  - {f.relative_to(inputs_dir.parent)}", file=sys.stderr)
            print("\nTo apply these changes, you can:", file=sys.stderr)
            print("  1. Save the diff output to a file: ./format_input_files.py > format.patch", file=sys.stderr)
            print("  2. Apply with git: git apply format.patch", file=sys.stderr)
            print("  3. Or run with --fix to apply changes directly", file=sys.stderr)
        sys.exit(1)
    else:
        if not is_redirected:
            print("All input files are properly formatted! ✓", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    if '--fix' in sys.argv:
        # Apply formatting directly
        inputs_dir = Path('/home/adam/artemis/inputs')
        in_files = sorted(inputs_dir.rglob('*.in'))
        
        fixed_count = 0
        for filepath in in_files:
            with open(filepath, 'r') as f:
                original_content = f.read()
            
            formatted_content = format_file_content(original_content)
            
            if original_content != formatted_content:
                with open(filepath, 'w') as f:
                    f.write(formatted_content)
                print(f"Fixed: {filepath.relative_to(inputs_dir.parent)}")
                fixed_count += 1
        
        if fixed_count:
            print(f"\nFixed {fixed_count} file(s)")
        else:
            print("All files are already properly formatted! ✓")
    else:
        main()
