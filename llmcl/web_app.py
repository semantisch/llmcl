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

# Default API keys (for development)
DEFAULT_OPENAI_KEY = "your-openai-api-key-here"
DEFAULT_ANTHROPIC_KEY = "your-anthropic-api-key-here"

# Sample puzzle for pre-filling the form
SAMPLE_PUZZLE = {
    "description": "There are five houses, each with a resident, a type of house, a potion, a helper, and a music preference. Determine who keeps the strength potion and who is helped by the troll.",
    "clues": [
        {"id": 1, "text": "There are five houses."},
        {"id": 2, "text": "The Hobbit lives in the forest house."},
        {"id": 3, "text": "The Elf is helped by the pixie."},
        {"id": 4, "text": "The invisibility potion is kept in the valley house."},
        {"id": 5, "text": "The Dunedain owns the velocity potion."},
        {"id": 6, "text": "The valley is immediately to the right of the cave house."},
        {"id": 7, "text": "The punk music fan is helped by the minotaur."},
        {"id": 8, "text": "Folk music is listened to in the castle."},
        {"id": 9, "text": "The flying potion is kept in the middle house."},
        {"id": 10, "text": "The wizard lives in the first house."},
        {"id": 11, "text": "The one who listens to indie music lives in the house next to the one who's helped by the fairy."},
        {"id": 12, "text": "Folk music is listened to in the house next to the house of the one helped by the goblin."},
        {"id": 13, "text": "The opera music fan keeps the clairvoyance potion."},
        {"id": 14, "text": "The dwarf is a fan of metal."},
        {"id": 15, "text": "The wizard lives next to the lighthouse."}
    ]
}

@app.route('/')
def index():
    """Main page with step-by-step puzzle solver."""
    return render_template('index.html', sample_puzzle=json.dumps(SAMPLE_PUZZLE, indent=2))

@app.route('/step1_extract_structure', methods=['POST'])
def step1_extract_structure():
    """Step 1: Extract puzzle structure from natural language."""
    try:
        # Get form data
        puzzle_text = request.form.get('puzzle_text', '').strip()
        model = request.form.get('model', 'gpt-4')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        force_derive = request.form.get('force_derive') == 'on'
        
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
        
        # Initialize LLM connector with default keys if needed
        if not api_key:
            if model.startswith('gpt-') or model.startswith('o'):
                api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
            elif model.startswith('claude-'):
                api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
        
        try:
            llm = LLMConnector(
                model=model,
                api_key=api_key,
                verbose=verbose,
                interactive=False
            )
        except Exception as e:
            return jsonify({'error': f'Failed to initialize LLM: {str(e)}'}), 400
        
        # Extract structured information
        try:
            structured_info = llm.extract_puzzle_structure(puzzle_text, force_derive=force_derive)
        except Exception as e:
            return jsonify({'error': f'Failed to extract puzzle structure: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'structured_info': structured_info,
            'model_used': model,
            'provider': llm.provider
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/step2_generate_asp', methods=['POST'])
def step2_generate_asp():
    """Step 2: Generate ASP facts from structured information."""
    try:
        # Get the structured info (either original or user-edited)
        structured_info_str = request.form.get('structured_info', '').strip()
        verbose = request.form.get('verbose') == 'on'
        
        if not structured_info_str:
            return jsonify({'error': 'Please provide structured information'}), 400
        
        # Parse structured info
        try:
            structured_info = json.loads(structured_info_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in structured info: {str(e)}'}), 400
        
        # Generate ASP facts
        try:
            asp_generator = ASPGenerator(verbose=verbose)
            asp_facts = asp_generator.generate_facts(structured_info)
        except Exception as e:
            return jsonify({'error': f'Failed to generate ASP facts: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'asp_facts': asp_facts
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/step3_validate_asp', methods=['POST'])
def step3_validate_asp():
    """Step 3: Validate ASP syntax and optionally fix with LLM."""
    try:
        # Get the ASP facts (either original or user-edited)
        asp_facts = request.form.get('asp_facts', '').strip()
        model = request.form.get('model', 'gpt-4')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        auto_fix = request.form.get('auto_fix') == 'on'
        
        if not asp_facts:
            return jsonify({'error': 'Please provide ASP facts'}), 400
        
        # Write ASP to temp file for validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
            temp_file.write(asp_facts)
            temp_file_path = temp_file.name
        
        try:
            # Validate syntax
            is_valid, syntax_errors = validate_asp_syntax(temp_file_path, verbose)
            
            corrected_asp = asp_facts
            correction_attempted = False
            
            # If syntax errors and auto-fix enabled, try LLM correction
            if not is_valid and syntax_errors and auto_fix:
                # Get API key with defaults
                if not api_key:
                    if model.startswith('gpt-') or model.startswith('o'):
                        api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
                    elif model.startswith('claude-'):
                        api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
                
                try:
                    llm = LLMConnector(
                        model=model,
                        api_key=api_key,
                        verbose=verbose,
                        interactive=False
                    )
                    
                    corrected_asp = llm.fix_asp_syntax(asp_facts, syntax_errors)
                    correction_attempted = True
                    
                    # Write corrected version and re-validate
                    with open(temp_file_path, 'w') as f:
                        f.write(corrected_asp)
                    
                    is_valid_corrected, remaining_errors = validate_asp_syntax(temp_file_path, verbose)
                    
                    if is_valid_corrected:
                        syntax_errors = []
                        is_valid = True
                    else:
                        syntax_errors = remaining_errors
                        
                except Exception as e:
                    syntax_errors.append(f"LLM correction failed: {str(e)}")
            
            return jsonify({
                'success': True,
                'asp_facts': corrected_asp,
                'is_valid': is_valid,
                'syntax_errors': syntax_errors,
                'correction_attempted': correction_attempted
            })
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/step4_solve_puzzle', methods=['POST'])
def step4_solve_puzzle():
    """Step 4: Solve the puzzle using clingo."""
    try:
        # Get the ASP facts (either original or user-edited)
        asp_facts = request.form.get('asp_facts', '').strip()
        model = request.form.get('model', 'gpt-4')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        auto_fix = request.form.get('auto_fix') == 'on'
        
        if not asp_facts:
            return jsonify({'error': 'Please provide ASP facts'}), 400
        
        # Write ASP to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
            temp_file.write(asp_facts)
            temp_file_path = temp_file.name
        
        try:
            # Look for encoding file
            encoding_file = "encodings/logic-puzzles.lp"
            if not os.path.exists(encoding_file):
                encoding_file = "../encodings/logic-puzzles.lp"
            
            if not os.path.exists(encoding_file):
                return jsonify({'error': 'Encoding file not found. Please ensure encodings/logic-puzzles.lp exists.'}), 400
            
            # Try to solve
            solution_success, solution_output, solution_facts = solve_puzzle(
                temp_file_path, encoding_file, verbose
            )
            
            corrected_asp = asp_facts
            correction_attempted = False
            
            # If solving failed with syntax errors and auto-fix enabled, try LLM correction
            if not solution_success and auto_fix and ("syntax error" in solution_output or "error:" in solution_output):
                clingo_errors = []
                for line in solution_output.split('\n'):
                    if 'error:' in line.lower():
                        clingo_errors.append(line.strip())
                
                if clingo_errors:
                    # Get API key with defaults
                    if not api_key:
                        if model.startswith('gpt-') or model.startswith('o'):
                            api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
                        elif model.startswith('claude-'):
                            api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
                    
                    try:
                        llm = LLMConnector(
                            model=model,
                            api_key=api_key,
                            verbose=verbose,
                            interactive=False
                        )
                        
                        corrected_asp = llm.fix_asp_syntax(asp_facts, clingo_errors)
                        correction_attempted = True
                        
                        # Write corrected version and retry solving
                        with open(temp_file_path, 'w') as f:
                            f.write(corrected_asp)
                        
                        solution_success, solution_output, solution_facts = solve_puzzle(
                            temp_file_path, encoding_file, verbose
                        )
                        
                    except Exception as e:
                        solution_output += f"\nLLM correction failed: {str(e)}"
            
            # Format solutions if successful
            formatted_solutions = ""
            if solution_success and solution_output:
                try:
                    formatted_solutions = format_all_solutions(solution_output)
                except Exception as e:
                    formatted_solutions = f"Error formatting solutions: {str(e)}"
            
            return jsonify({
                'success': True,
                'asp_facts': corrected_asp,
                'solution_success': solution_success,
                'solution_output': solution_output,
                'solution_facts': solution_facts,
                'formatted_solutions': formatted_solutions,
                'correction_attempted': correction_attempted
            })
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

# Legacy endpoint for backward compatibility
@app.route('/process', methods=['POST'])
def process_puzzle():
    """Legacy endpoint - runs all steps in sequence (for backward compatibility)."""
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
        
        # Initialize LLM connector with default keys if needed
        if not api_key:
            if model.startswith('gpt-') or model.startswith('o'):
                api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
            elif model.startswith('claude-'):
                api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
        
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

@app.route('/granular_step1_extract_inputs', methods=['POST'])
def granular_step1_extract_inputs():
    """Step 1: Extract just the inputs (entities, attributes) and return as ASP facts."""
    try:
        puzzle_text = request.form.get('puzzle_text', '').strip()
        model = request.form.get('model', 'gpt-4o')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        force_derive = request.form.get('force_derive') == 'on'
        
        if not puzzle_text:
            return jsonify({'error': 'Please provide a puzzle description'}), 400
        
        # Initialize LLM connector with default keys if needed
        if not api_key:
            if model.startswith('gpt-') or model.startswith('o'):
                api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
            elif model.startswith('claude-'):
                api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
        
        try:
            llm = LLMConnector(
                model=model,
                api_key=api_key,
                verbose=verbose,
                interactive=False
            )
        except Exception as e:
            return jsonify({'error': f'Failed to initialize LLM: {str(e)}'}), 400
        
        # Extract just the inputs (entities, attributes)
        try:
            inputs_info = llm.extract_inputs_only(puzzle_text, force_derive=force_derive)
        except Exception as e:
            return jsonify({'error': f'Failed to extract inputs: {str(e)}'}), 500
        
        # Generate ASP input facts
        try:
            asp_generator = ASPGenerator(verbose=verbose)
            input_asp_facts = asp_generator.generate_input_facts(inputs_info)
        except Exception as e:
            return jsonify({'error': f'Failed to generate ASP input facts: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'input_asp_facts': input_asp_facts,
            'inputs_info': inputs_info,  # Keep for later use
            'model_used': model,
            'provider': llm.provider
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/granular_step2_process_clue', methods=['POST'])
def granular_step2_process_clue():
    """Step 2: Process a single clue and return as ASP facts."""
    try:
        puzzle_text = request.form.get('puzzle_text', '').strip()
        inputs_info_str = request.form.get('inputs_info', '').strip()
        clue_text = request.form.get('clue_text', '').strip()
        clue_id = request.form.get('clue_id', '').strip()
        model = request.form.get('model', 'gpt-4o')
        api_key = request.form.get('api_key', '').strip() or None
        verbose = request.form.get('verbose') == 'on'
        
        if not all([puzzle_text, inputs_info_str, clue_text]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Parse inputs info
        try:
            inputs_info = json.loads(inputs_info_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in inputs info: {str(e)}'}), 400
        
        # Initialize LLM connector with default keys if needed
        if not api_key:
            if model.startswith('gpt-') or model.startswith('o'):
                api_key = os.environ.get('OPENAI_API_KEY', DEFAULT_OPENAI_KEY)
            elif model.startswith('claude-'):
                api_key = os.environ.get('ANTHROPIC_API_KEY', DEFAULT_ANTHROPIC_KEY)
        
        try:
            llm = LLMConnector(
                model=model,
                api_key=api_key,
                verbose=verbose,
                interactive=False
            )
        except Exception as e:
            return jsonify({'error': f'Failed to initialize LLM: {str(e)}'}), 400
        
        # Process single clue
        try:
            clue_extraction = llm.process_single_clue(puzzle_text, inputs_info, clue_text, clue_id)
        except Exception as e:
            return jsonify({'error': f'Failed to process clue: {str(e)}'}), 500
        
        # Generate ASP facts for this clue
        try:
            asp_generator = ASPGenerator(verbose=verbose)
            clue_asp_facts = asp_generator.generate_clue_facts(clue_extraction, clue_id)
        except Exception as e:
            return jsonify({'error': f'Failed to generate ASP clue facts: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'clue_asp_facts': clue_asp_facts,
            'clue_extraction': clue_extraction,  # Keep for later use
            'clue_id': clue_id,
            'clue_text': clue_text
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/granular_step3_generate_asp', methods=['POST'])
def granular_step3_generate_asp():
    """Step 3: Generate ASP facts from inputs and processed clues."""
    try:
        inputs_info_str = request.form.get('inputs_info', '').strip()
        clues_extractions_str = request.form.get('clues_extractions', '').strip()
        verbose = request.form.get('verbose') == 'on'
        
        if not all([inputs_info_str, clues_extractions_str]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Parse data
        try:
            inputs_info = json.loads(inputs_info_str)
            clues_extractions = json.loads(clues_extractions_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON data: {str(e)}'}), 400
        
        # Combine inputs and clues into structured format
        structured_info = {
            **inputs_info,
            'clues': clues_extractions
        }
        
        # Generate ASP facts
        try:
            asp_generator = ASPGenerator(verbose=verbose)
            asp_facts = asp_generator.generate_facts(structured_info)
        except Exception as e:
            return jsonify({'error': f'Failed to generate ASP facts: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'asp_facts': asp_facts
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/get_encoding_content', methods=['GET'])
def get_encoding_content():
    """Get the content of the encoding file."""
    try:
        # Look for encoding file
        encoding_file = "encodings/logic-puzzles.lp"
        if not os.path.exists(encoding_file):
            encoding_file = "../encodings/logic-puzzles.lp"
        
        if not os.path.exists(encoding_file):
            return jsonify({'error': 'Encoding file not found'}), 404
        
        with open(encoding_file, 'r') as f:
            encoding_content = f.read()
        
        return jsonify({
            'success': True,
            'encoding_content': encoding_content,
            'encoding_file': encoding_file
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to read encoding file: {str(e)}'
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