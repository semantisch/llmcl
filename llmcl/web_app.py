#!/usr/bin/env python3
"""
LLMCL Web Interface
Flask web application for running logic puzzle extraction through a browser interface.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS

from llm_connector import LLMConnector
from asp_generator import ASPGenerator
from llmcl import validate_asp_syntax, solve_puzzle, format_all_solutions

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'llmcl-dev-key-change-in-production')
CORS(app)

# Sample puzzle for pre-filling the form
SAMPLE_PUZZLE = {
    "description": "Four different companies will be visited during a business tour. Each company has a different name, was founded in a different year, specializes in a different technology, and will be visited in a different order during the tour.",
    "clues": [
        {"id": 1, "text": "The four companies are TechCorp Alpha, the one specializing in AI development, the one founded in 1985, and the one that will be visited 3rd."},
        {"id": 2, "text": "Of Beta Solutions and the company founded in 1990, one specializes in cloud computing and the other will be visited 2nd."},
        {"id": 3, "text": "Gamma Industries was not founded between 1988 and 1992."},
        {"id": 4, "text": "The first company to be visited specializes in a technology with a single word name."},
        {"id": 5, "text": "The company founded in 1995 will be visited either immediately before or immediately after the company specializing in AI development."},
        {"id": 6, "text": "Of the company specializing in robotics and the company known as Delta Dynamics, one was founded in 1995 and the other is the first destination of the tour."},
        {"id": 7, "text": "The last company on the tour was founded 5 years after the first company on the tour."}
    ]
}

@app.route('/')
def index():
    """Main page with puzzle input form."""
    return render_template('index.html', sample_puzzle=json.dumps(SAMPLE_PUZZLE, indent=2))

@app.route('/process', methods=['POST'])
def process_puzzle():
    """Process the puzzle and return results."""
    try:
        # Get form data
        puzzle_text = request.form.get('puzzle_text', '').strip()
        model = request.form.get('model', 'gpt-4')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        force_derive = request.form.get('force_derive') == 'on'
        interactive = False  # Always non-interactive for web interface
        
        if not puzzle_text:
            return jsonify({'error': 'Please provide a puzzle description'}), 400
        
        # Parse puzzle (try JSON first, then treat as plain text)
        try:
            puzzle_data = json.loads(puzzle_text)
        except json.JSONDecodeError:
            puzzle_data = {
                "description": puzzle_text,
                "clues": []
            }
        
        # Initialize LLM connector
        try:
            llm = LLMConnector(
                model=model,
                api_key=api_key,
                verbose=verbose,
                interactive=interactive
            )
        except Exception as e:
            return jsonify({'error': f'Failed to initialize LLM: {str(e)}'}), 400
        
        # Extract structured information
        try:
            structured_info = llm.extract_puzzle_structure(puzzle_text, force_derive=force_derive)
        except Exception as e:
            return jsonify({'error': f'Failed to extract puzzle structure: {str(e)}'}), 500
        
        # Generate ASP facts
        try:
            asp_generator = ASPGenerator(verbose=verbose)
            asp_facts = asp_generator.generate_facts(structured_info)
        except Exception as e:
            return jsonify({'error': f'Failed to generate ASP facts: {str(e)}'}), 500
        
        # Validate ASP syntax
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
            temp_file.write(asp_facts)
            temp_file_path = temp_file.name
        
        try:
            is_valid, syntax_errors = validate_asp_syntax(temp_file_path, verbose)
            
            # If syntax errors, try LLM correction
            if not is_valid and syntax_errors:
                try:
                    corrected_asp = llm.fix_asp_syntax(asp_facts, syntax_errors)
                    # Write corrected version
                    with open(temp_file_path, 'w') as f:
                        f.write(corrected_asp)
                    
                    # Re-validate
                    is_valid_corrected, remaining_errors = validate_asp_syntax(temp_file_path, verbose)
                    
                    if is_valid_corrected:
                        asp_facts = corrected_asp
                        syntax_errors = []
                    else:
                        syntax_errors = remaining_errors
                        
                except Exception as e:
                    syntax_errors.append(f"LLM correction failed: {str(e)}")
            
            # Try to solve the puzzle
            solution_success = False
            solution_output = ""
            solution_facts = []
            
            if is_valid or not syntax_errors:
                try:
                    # Look for encoding file
                    encoding_file = "encodings/logic-puzzles.lp"
                    if not os.path.exists(encoding_file):
                        encoding_file = "../encodings/logic-puzzles.lp"
                    
                    solution_success, solution_output, solution_facts = solve_puzzle(
                        temp_file_path, encoding_file, verbose
                    )
                    
                    # If clingo syntax errors, try LLM correction
                    if not solution_success and ("syntax error" in solution_output or "error:" in solution_output):
                        clingo_errors = []
                        for line in solution_output.split('\n'):
                            if 'error:' in line.lower():
                                clingo_errors.append(line.strip())
                        
                        if clingo_errors:
                            try:
                                corrected_asp = llm.fix_asp_syntax(asp_facts, clingo_errors)
                                with open(temp_file_path, 'w') as f:
                                    f.write(corrected_asp)
                                
                                # Retry solving
                                solution_success, solution_output, solution_facts = solve_puzzle(
                                    temp_file_path, encoding_file, verbose
                                )
                                
                                if solution_success:
                                    asp_facts = corrected_asp
                                    
                            except Exception as e:
                                solution_output += f"\nLLM correction failed: {str(e)}"
                    
                except Exception as e:
                    solution_output = f"Solving failed: {str(e)}"
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        # Format results
        formatted_solutions = ""
        if solution_success and solution_output:
            try:
                formatted_solutions = format_all_solutions(solution_output)
            except Exception as e:
                formatted_solutions = f"Error formatting solutions: {str(e)}"
        
        result = {
            'success': True,
            'structured_info': structured_info,
            'asp_facts': asp_facts,
            'syntax_valid': is_valid and not syntax_errors,
            'syntax_errors': syntax_errors,
            'solution_success': solution_success,
            'solution_output': solution_output,
            'solution_facts': solution_facts,
            'formatted_solutions': formatted_solutions,
            'model_used': model,
            'provider': llm.provider
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'LLMCL Web Interface'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🌐 Starting LLMCL Web Interface...")
    print(f"   URL: http://localhost:{port}")
    print(f"   Debug mode: {debug}")
    print()
    print("💡 Make sure your API keys are set:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print()
    
    app.run(host='0.0.0.0', port=port, debug=debug) 