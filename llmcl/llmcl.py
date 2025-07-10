#!/usr/bin/env python3
"""
LLM Clue Extractor (LLMCL)
Main script for extracting ASP clues from natural language puzzle descriptions
using Large Language Models.
"""

import argparse
import sys
import subprocess
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from llm_connector import LLMConnector
from asp_generator import ASPGenerator


def load_puzzle_description(file_path: str) -> str:
    """Load puzzle description from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def save_asp_facts(facts: str, output_path: str) -> None:
    """Save generated ASP facts to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(facts)
        print(f"ASP facts saved to: {output_path}")
    except Exception as e:
        print(f"Error saving ASP facts to '{output_path}': {e}")
        sys.exit(1)


def find_clingo() -> Optional[str]:
    """Find clingo executable in system PATH."""
    try:
        result = subprocess.run(['which', 'clingo'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def validate_asp_syntax(asp_file: str, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Validate ASP syntax using basic validation (clingo validation during solving)."""
    return validate_asp_syntax_basic(asp_file, verbose)


def validate_asp_syntax_basic(asp_file: str, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Basic ASP syntax validation (fallback when clingo not available)."""
    errors = []
    
    try:
        with open(asp_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('%'):
            continue
        
        # Skip obvious non-ASP lines (markdown, etc.)
        if line.startswith('```') or line.strip() in ['```asp', '```']:
            continue
        
        # Check if line ends with period
        if not line.endswith('.'):
            errors.append(f"Line {line_num}: Missing period at end: {line}")
            continue
        
        # Remove the period for further checks
        line_content = line[:-1]
        
        # Check for basic ASP fact structure
        if not re.match(r'^[a-z][a-zA-Z0-9_]*\([^)]*\)$', line_content):
            # Allow some ASP constructs like choice rules, constraints, etc.
            if not any(pattern in line_content for pattern in ['{', '}', ':-', ':~', '#']):
                errors.append(f"Line {line_num}: Invalid ASP syntax: {line}")
    
    return len(errors) == 0, errors


def solve_puzzle(asp_file: str, encoding_file: str, verbose: bool = False) -> Tuple[bool, str, List[str]]:
    """Solve the puzzle using clingo, getting multiple solutions if available."""
    clingo_path = find_clingo()
    if not clingo_path:
        return False, "clingo not found. Please install clingo ASP solver.", []
    
    # Build clingo command
    cmd = [clingo_path]
    
    # Add encoding file if it exists
    if os.path.exists(encoding_file):
        cmd.append(encoding_file)
    elif verbose:
        print(f"⚠️ Encoding file not found: {encoding_file}")
    
    # Add ASP fact file
    cmd.append(asp_file)
    
    # Get up to 10 solutions
    cmd.extend(['-n', '10'])
    
    if verbose:
        print(f"🔄 Running clingo: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check if clingo found a solution by parsing output, not just exit code
        if "SATISFIABLE" in result.stdout:
            solution_facts = parse_clingo_solution(result.stdout)
            return True, result.stdout, solution_facts
        elif "UNSATISFIABLE" in result.stdout:
            return False, "UNSATISFIABLE: No solution found.", []
        elif result.returncode != 0:
            return False, f"clingo error (code {result.returncode}):\n{result.stderr}", []
        else:
            # No clear SAT/UNSAT result
            return False, f"clingo completed but no clear result:\n{result.stdout}", []
            
    except subprocess.TimeoutExpired:
        return False, "Timeout: clingo took too long to solve.", []
    except Exception as e:
        return False, f"Error running clingo: {e}", []


def parse_clingo_solution(clingo_output: str) -> List[str]:
    """Parse clingo output to extract multiple solution facts."""
    all_solutions = []
    
    lines = clingo_output.split('\n')
    current_solution = []
    in_answer = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Answer:'):
            # Save previous solution if exists
            if current_solution:
                all_solutions.append(current_solution)
                current_solution = []
            in_answer = True
            continue
        
        if in_answer and line:
            if line.startswith('SATISFIABLE') or line.startswith('UNSATISFIABLE') or line.startswith('Models'):
                break
            
            # Extract facts from the answer line
            facts = line.split()
            current_solution.extend(facts)
    
    # Add the last solution
    if current_solution:
        all_solutions.append(current_solution)
    
    # For backward compatibility, return the first solution as a flat list
    return all_solutions[0] if all_solutions else []


def parse_all_clingo_solutions(clingo_output: str) -> List[List[str]]:
    """Parse clingo output to extract all solution facts."""
    all_solutions = []
    
    lines = clingo_output.split('\n')
    current_solution = []
    in_answer = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Answer:'):
            # Save previous solution if exists
            if current_solution:
                all_solutions.append(current_solution)
                current_solution = []
            in_answer = True
            continue
        
        if in_answer and line:
            if line.startswith('SATISFIABLE') or line.startswith('UNSATISFIABLE') or line.startswith('Models'):
                break
            
            # Extract facts from the answer line
            facts = line.split()
            current_solution.extend(facts)
    
    # Add the last solution
    if current_solution:
        all_solutions.append(current_solution)
    
    return all_solutions


def format_solution(solution_facts: List[str]) -> str:
    """Format solution facts in a readable way."""
    if not solution_facts:
        return "No solution facts found."
    
    # Group facts by predicate
    fact_groups = {}
    
    for fact in solution_facts:
        if '(' in fact:
            predicate = fact.split('(')[0]
            if predicate not in fact_groups:
                fact_groups[predicate] = []
            fact_groups[predicate].append(fact)
    
    # Format output
    output = []
    for predicate, facts in sorted(fact_groups.items()):
        output.append(f"\n{predicate.upper()}:")
        for fact in sorted(facts):
            output.append(f"  {fact}")
    
    return '\n'.join(output)


def format_all_solutions(clingo_output: str) -> str:
    """Format all solutions from clingo output in a readable way."""
    all_solutions = parse_all_clingo_solutions(clingo_output)
    
    if not all_solutions:
        return "No solutions found."
    
    output = []
    
    # Extract number of models from clingo output
    lines = clingo_output.split('\n')
    models_info = ""
    for line in lines:
        if line.strip().startswith('Models'):
            models_info = line.strip()
            break
    
    output.append(f"🎯 FOUND {len(all_solutions)} SOLUTION(S)")
    if models_info:
        output.append(f"📊 {models_info}")
    output.append("=" * 60)
    
    for i, solution in enumerate(all_solutions, 1):
        output.append(f"\n🔸 SOLUTION {i}:")
        output.append("-" * 40)
        
        # Group facts by predicate for readability
        fact_groups = {}
        for fact in solution:
            if '(' in fact and not fact.startswith('clue(') and not fact.startswith('object(') and not fact.startswith('target('):
                predicate = fact.split('(')[0]
                if predicate not in fact_groups:
                    fact_groups[predicate] = []
                fact_groups[predicate].append(fact)
        
        # Display match facts organized by position/house
        if 'match' in fact_groups:
            output.append("\n📋 ASSIGNMENTS:")
            
            # Parse all match facts and organize by position
            position_assignments = {}
            
            for fact in fact_groups['match']:
                if fact.startswith('match('):
                    parts = fact[6:-1].split(',')  # Remove match( and )
                    if len(parts) == 4:
                        attr1, val1, attr2, val2 = parts
                        
                        # Find position-based matches
                        if attr1.endswith('position'):
                            position = val1.replace('position_', '')
                            if position not in position_assignments:
                                position_assignments[position] = {}
                            position_assignments[position][attr2] = val2
                        elif attr2.endswith('position'):
                            position = val2.replace('position_', '')
                            if position not in position_assignments:
                                position_assignments[position] = {}
                            position_assignments[position][attr1] = val1
            
            # Display each position with all its attributes
            for position in sorted(position_assignments.keys(), key=int):
                output.append(f"\n  Position {position}:")
                attrs = position_assignments[position]
                
                # Sort attributes for consistent display
                for attr_name in sorted(attrs.keys()):
                    value = attrs[attr_name]
                    clean_attr = attr_name.replace('_', ' ').title()
                    clean_value = value.replace('_', ' ').title()
                    output.append(f"    {clean_attr}: {clean_value}")
                
            output.append("")
        
        # Display input facts organized by attribute
        if 'input' in fact_groups:
            output.append("\n📝 ATTRIBUTE VALUES:")
            
            # Group input facts by attribute
            attr_values = {}
            for fact in fact_groups['input']:
                if fact.startswith('input('):
                    parts = fact[6:-1].split(',')  # Remove input( and )
                    if len(parts) == 3:
                        attr, pos, val = parts
                        if attr not in attr_values:
                            attr_values[attr] = []
                        attr_values[attr].append((int(pos), val))
            
            # Display each attribute nicely
            for attr, values in sorted(attr_values.items()):
                values.sort()  # Sort by position
                output.append(f"  {attr.replace('_', ' ').title()}:")
                for pos, val in values:
                    clean_val = val.replace('_', ' ').replace('n', '').title()
                    output.append(f"    {pos}. {clean_val}")
                output.append("")
    
    return '\n'.join(output)


def display_puzzle_description(puzzle_description: str, verbose: bool = False) -> None:
    """Display the puzzle description in a nicely formatted way."""
    try:
        # Try to parse as JSON
        puzzle_data = json.loads(puzzle_description)
        
        print("\n" + "="*70)
        print("🧩 LOGIC PUZZLE")
        print("="*70)
        
        # Display description
        description = puzzle_data.get('description', '')
        if description:
            print("\n📝 DESCRIPTION:")
            print("-" * 50)
            # Wrap text nicely
            words = description.split()
            lines = []
            current_line = []
            line_length = 0
            
            for word in words:
                if line_length + len(word) + 1 > 65:  # 65 char limit
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        line_length = len(word)
                    else:
                        lines.append(word)
                        line_length = 0
                else:
                    current_line.append(word)
                    line_length += len(word) + (1 if current_line else 0)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                print(f"  {line}")
        
        # Display clues
        clues = puzzle_data.get('clues', [])
        if clues:
            print(f"\n🔍 CLUES ({len(clues)} total):")
            print("-" * 50)
            for clue in clues:
                clue_id = clue.get('id', '?')
                clue_text = clue.get('text', '')
                print(f"  {clue_id}) {clue_text}")
        
        print("="*70)
        
        if verbose:
            print(f"📊 Puzzle stats: {len(puzzle_description)} characters, {len(clues)} clues")
        
    except json.JSONDecodeError:
        # Fallback for non-JSON format
        if verbose:
            print(f"📄 Raw puzzle description ({len(puzzle_description)} characters):")
            print("-" * 50)
            print(puzzle_description[:200] + ("..." if len(puzzle_description) > 200 else ""))
            print("-" * 50)
        else:
            print("📄 Loaded puzzle description (plain text format)")
    
    print()  # Extra space before starting extraction


def main():
    parser = argparse.ArgumentParser(
        description="Extract ASP clues from natural language puzzle descriptions using LLMs",
        epilog="💡 Tips: Use --force-derive for advanced reasoning | For web interface: python3 web_app.py"
    )
    parser.add_argument(
        "input",
        help="Input file containing the puzzle description in natural language"
    )
    parser.add_argument(
        "-o", "--output",
        default="puzzle_generated.lp",
        help="Output file for generated ASP facts (default: puzzle_generated.lp)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="LLM model to use. OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo, etc. | Anthropic: claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc. (default: gpt-4)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for the chosen model provider (if not set in environment: OPENAI_API_KEY for OpenAI models, ANTHROPIC_API_KEY for Anthropic models)"
    )
    parser.add_argument(
        "-e", "--encoding",
        default="../encodings/logic-puzzles.lp",
        help="Path to ASP encoding file (default: ../encodings/logic-puzzles.lp)"
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Generate ASP facts only, don't solve the puzzle"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive mode with confirmation steps"
    )
    parser.add_argument(
        "--force-derive",
        action="store_true",
        help="Use deep reasoning to derive all missing values instead of using placeholders"
    )
    
    args = parser.parse_args()
    
    # Load puzzle description
    if args.verbose:
        print(f"Loading puzzle description from: {args.input}")
    
    puzzle_description = load_puzzle_description(args.input)
    
    # Display the puzzle nicely formatted
    display_puzzle_description(puzzle_description, args.verbose)
    
    # Initialize LLM connector
    if args.verbose:
        print(f"Initializing LLM connector with model: {args.model}")
        if "claude" in args.model.lower():
            print("💡 Using Anthropic Claude model. Make sure ANTHROPIC_API_KEY is set.")
        elif "gpt" in args.model.lower():
            print("💡 Using OpenAI GPT model. Make sure OPENAI_API_KEY is set.")
    
    try:
        llm = LLMConnector(
            model=args.model,
            api_key=args.api_key,
            verbose=args.verbose,
            interactive=args.interactive
        )
    except Exception as e:
        print(f"Error initializing LLM connector: {e}")
        if "ANTHROPIC_API_KEY" in str(e):
            print("\n💡 For Anthropic models, get your API key from: https://console.anthropic.com/")
            print("   Then set it: export ANTHROPIC_API_KEY='your-key-here'")
        elif "OPENAI_API_KEY" in str(e):
            print("\n💡 For OpenAI models, get your API key from: https://platform.openai.com/api-keys")
            print("   Then set it: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Extract structured information from puzzle
    if args.verbose:
        print("Extracting structured information from puzzle description...")
    
    try:
        structured_info = llm.extract_puzzle_structure(puzzle_description, force_derive=args.force_derive)
        
        if args.verbose:
            print("Extracted structure:")
            print(f"  - Attributes: {list(structured_info.get('attributes', {}).keys())}")
            print(f"  - Number of clues: {len(structured_info.get('clues', []))}")
    
    except Exception as e:
        print(f"Error extracting puzzle structure: {e}")
        sys.exit(1)
    
    # Generate ASP facts
    if args.verbose:
        print("Generating ASP facts...")
    
    try:
        asp_generator = ASPGenerator(verbose=args.verbose)
        asp_facts = asp_generator.generate_facts(structured_info)
        
        if args.verbose:
            print(f"Generated {len(asp_facts.split('.'))} ASP facts")
    
    except Exception as e:
        print(f"Error generating ASP facts: {e}")
        sys.exit(1)
    
    # Save results
    save_asp_facts(asp_facts, args.output)
    
    if args.verbose:
        print("\nGenerated ASP facts:")
        print("-" * 50)
        print(asp_facts)
        print("-" * 50)
    
    print(f"✅ Successfully generated ASP facts from puzzle description!")
    print(f"📄 Input: {args.input}")
    print(f"📄 Output: {args.output}")
    
    # Always show complete ASP when using force-derive
    if args.force_derive:
        print("\n📋 COMPLETE ASP PROGRAM (Force-Derive Mode):")
        print("=" * 60)
        print(asp_facts)
        print("=" * 60)
    
    # Stop here if user doesn't want to solve
    if args.no_solve:
        return
    
    # Validate ASP syntax
    if args.verbose:
        print("\n🔄 Validating ASP syntax...")
    
    is_valid, syntax_errors = validate_asp_syntax(args.output, args.verbose)
    
    if not is_valid:
        print("\n❌ ASP syntax validation failed:")
        for error in syntax_errors:
            print(f"  {error}")
        
        print(f"\n🔄 Sending {len(syntax_errors)} syntax errors to LLM for correction...")
        
        # Read the current ASP code
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                current_asp_code = f.read()
        except Exception as e:
            print(f"❌ Error reading ASP file for correction: {e}")
            sys.exit(1)
        
        # Send to LLM for syntax correction
        try:
            print("\n📋 CURRENT ASP CODE WITH ERRORS:")
            print("-" * 60)
            print(current_asp_code)
            print("-" * 60)
            
            corrected_asp_code = llm.fix_asp_syntax(current_asp_code, syntax_errors)
            
            print("\n📋 LLM-CORRECTED ASP CODE:")
            print("-" * 60)
            print(corrected_asp_code)
            print("-" * 60)
            
            # Save the corrected version
            backup_file = args.output + ".backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(current_asp_code)
            print(f"💾 Original saved as backup: {backup_file}")
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(corrected_asp_code)
            print(f"💾 Corrected ASP saved to: {args.output}")
            
            # Re-validate the corrected ASP
            print("\n🔄 Re-validating corrected ASP syntax...")
            is_valid_corrected, remaining_errors = validate_asp_syntax(args.output, args.verbose)
            
            if is_valid_corrected:
                print("✅ LLM syntax correction successful! ASP is now valid.")
            else:
                print("❌ LLM correction incomplete. Remaining errors:")
                for error in remaining_errors:
                    print(f"  {error}")
                print("\nManual intervention may be required.")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error during LLM syntax correction: {e}")
            sys.exit(1)
    
    elif args.verbose:
        print("✅ ASP syntax validation passed")
    
    # Solve the puzzle
    if args.verbose:
        print("\n🔄 Solving puzzle with clingo...")
    
    success, raw_output, solution_facts = solve_puzzle(args.output, args.encoding, args.verbose)
    
    # Check if clingo failed due to syntax errors and try LLM correction
    if not success and ("syntax error" in raw_output or "error:" in raw_output):
        print("\n❌ Clingo syntax errors detected:")
        print(raw_output)
        
        # Extract clingo error messages
        clingo_errors = []
        for line in raw_output.split('\n'):
            if 'error:' in line.lower():
                clingo_errors.append(line.strip())
        
        if clingo_errors:
            print(f"\n🔄 Sending {len(clingo_errors)} clingo errors to LLM for correction...")
            
            # Read current ASP code
            try:
                with open(args.output, 'r', encoding='utf-8') as f:
                    current_asp_code = f.read()
            except Exception as e:
                print(f"❌ Error reading ASP file: {e}")
                sys.exit(1)
            
            # Send to LLM for correction
            try:
                print("\n📋 CURRENT ASP CODE WITH CLINGO ERRORS:")
                print("-" * 60)
                print(current_asp_code)
                print("-" * 60)
                
                corrected_asp_code = llm.fix_asp_syntax(current_asp_code, clingo_errors)
                
                print("\n📋 LLM-CORRECTED ASP CODE:")
                print("-" * 60)
                print(corrected_asp_code)
                print("-" * 60)
                
                # Save corrected version
                backup_file = args.output + ".clingo_backup"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(current_asp_code)
                print(f"💾 Original saved as backup: {backup_file}")
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(corrected_asp_code)
                print(f"💾 Corrected ASP saved to: {args.output}")
                
                # Try solving again
                print("\n🔄 Retrying solve with corrected ASP...")
                success, raw_output, solution_facts = solve_puzzle(args.output, args.encoding, args.verbose)
                
            except Exception as e:
                print(f"❌ Error during LLM clingo correction: {e}")
    
    if success:
        print("\n🎉 PUZZLE SOLVED!")
        
        # Format and display all solutions nicely
        formatted_solutions = format_all_solutions(raw_output)
        print(formatted_solutions)
        
        if args.verbose and raw_output:
            print("\n🔍 Raw clingo output:")
            print("-" * 50)
            print(raw_output)
    else:
        print(f"\n❌ Solving failed: {raw_output}")
        if "not found" in raw_output:
            print("\n💡 Install clingo ASP solver:")
            print("  - Ubuntu/Debian: sudo apt install clingo")
            print("  - macOS: brew install clingo") 
            print("  - Or visit: https://potassco.org/clingo/")
        sys.exit(1)


if __name__ == "__main__":
    main() 