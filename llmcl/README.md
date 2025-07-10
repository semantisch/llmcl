# LLMCL - LLM-powered Logic Puzzle Solver

**LLMCL** (Large Language Model Controlled Language) is an advanced system that automatically converts natural language logic puzzles into Answer Set Programming (ASP) code and solves them using state-of-the-art LLMs.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-orange.svg)
![Anthropic](https://img.shields.io/badge/LLM-Anthropic-purple.svg)

## Features

- **Multi-Phase Extraction**: Intelligent attribute extraction → constraint processing → clue analysis
- (**Web Interface**: Modern Bootstrap UI with real-time solving)
- **Multi-Provider Support**: Different LLMs
- **Interactive Mode**: Step-by-step confirmation and retry logic
- **Automatic ASP Generation**: Direct conversion to clingo-compatible code
- **Syntax Correction**: Automatic error detection and LLM-powered fixes
- (**Force-Derive Mode**: Advanced reasoning with 3x sampling and intelligent selection)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/semantisch/llmcl.git
cd llmcl

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose one)
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Basic Usage

```bash
# Solve a puzzle with default settings
python llmcl.py sample_puzzle.txt

# Use force-derive mode for better accuracy
python llmcl.py sample_puzzle.txt --force-derive

# Interactive mode with step-by-step confirmation
python llmcl.py sample_puzzle.txt --interactive --verbose

# Use a specific model
python llmcl.py sample_puzzle.txt --model gpt-4o --force-derive
```

### Web Interface

```bash
# Start the web server
python web_app.py

# Open http://localhost:5000 in your browser
```

## Usage Guide

### Command Line Options

```
python llmcl.py [INPUT_FILE] [OPTIONS]

Required:
  INPUT_FILE              Path to puzzle description file (JSON or plain text)

Options:
  -o, --output FILE       Output ASP file (default: puzzle_generated.lp)
  --model MODEL           LLM model to use (default: gpt-4o)
  --api-key KEY           Override environment API key
  -e, --encoding FILE     ASP encoding file (default: encodings/logic-puzzles.lp)
  --no-solve             Generate ASP only, don't solve
  -v, --verbose          Enable detailed output
  -i, --interactive      Interactive mode with confirmations
  --force-derive         Use deep reasoning to derive exact values
  -h, --help             Show help message
```

### Supported Models

Specify any OpenAI/Antrhopic model.

### Input Formats

**JSON Format:**
```json
{
  "description": "Four wineries will be visited during a wine tour...",
  "clues": [
    {"id": 1, "text": "The four wineries are Vindictive Vines, the one who won an award for their Shiraz..."},
    {"id": 2, "text": "Of Beta Solutions and the company founded in 1990..."}
  ]
}
```

## Advanced Features

### Force-Derive Mode

The **force-derive** mode uses advanced multi-sampling to resolve placeholder values:

```bash
python llmcl.py puzzle.txt --force-derive --verbose
```

**How it works:**
1. **Triple Sampling**: Generates 3 different derivation attempts
2. **Intelligent Evaluation**: LLM evaluates all attempts for quality
3. **Uniqueness Validation**: Ensures no duplicate values within attributes
4. **Best Selection**: Automatically chooses the highest-quality derivation

### Interactive Mode

Step-by-step confirmation with retry logic:

```bash
python llmcl.py puzzle.txt --interactive
```

**Features:**
- **Phase-by-phase confirmation**: Review attributes, constraints, and clues
- **Retry logic**: Regenerate any phase if unsatisfied
- **ASP preview**: See generated code before proceeding
- **Multiple attempts**: Up to 3 retries per phase

### Web Interface

Modern web UI with real-time processing:

- **Bootstrap Design**: Clean, responsive interface
- **Real-time Solving**: See results as they're generated
- **Mobile Friendly**: Works on all devices
- **All Options**: Access to all CLI features via web forms

## Examples

### Example 1: Winery Tour Puzzle

**Input:** `sample_puzzle.txt`
```json
{
  "description": "Four different wineries will be visited during a wine tour...",
  "clues": [
    {"id": 1, "text": "The four wineries are Vindictive Vines, the one who won an award for their Shiraz, the one established in 1969, and the one that will be visited 3rd."}
  ]
}
```

**Command:**
```bash
python llmcl.py sample_puzzle.txt --force-derive --verbose
```

**Output:**
```asp
input(winery_name,1,vindictive_vines).
input(winery_name,2,chateau_cork).
input(winery_name,3,boozy_bottling).
input(winery_name,4,goodness_grapecious).

input(visit_order,1,first).
input(visit_order,2,second).
input(visit_order,3,third).
input(visit_order,4,fourth).

clue(c11a,diff).
object(c11a,1,winery_name,vindictive_vines).
object(c11a,2,award_winning_wine,shiraz).
object(c11a,3,established_year,n1969).
object(c11a,4,visit_order,third).
```

### Example 2: Comedy Club Puzzle

**Input:** `sample_puzzle_cheese.txt`
```json
{
  "description": "The Cheesy Chuckle Club is planning its next stand up comedy night...",
  "clues": [
    {"id": 1, "text": "Abigail Later has a busy schedule so the club agreed to her request for the slot immediately after Larry Laughs."}
  ]
}
```

**Command:**
```bash
python llmcl.py sample_puzzle_cheese.txt --model claude-3-5-sonnet-20241022 --interactive
```

## Technical Details

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Puzzle  │ -> │   LLM Connector  │ -> │  ASP Generator  │
│   (JSON/Text)   │    │  (Multi-phase)   │    │   (Facts/Rules) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                |
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Force-Derive   │ -> │   Clingo Solver │
                       │  (3x Sampling)   │    │    (Solutions)  │
                       └──────────────────┘    └─────────────────┘
```

### Processing Phases

1. **Phase 1**: Extract attributes and values (with placeholders if needed)
2. **Phase 2**: Either derive exact values (force-derive) or create constraint rules
3. **Phase 3**: Extract clue facts and relationships
4. **Phase 4**: Generate complete ASP program
5. **Phase 5**: Solve with clingo and format results

### Uniqueness Enforcement

LLMCL enforces strict uniqueness constraints:

- **Automatic Detection**: Scans for duplicate values in attributes
- **Prevention**: Multi-layer prompt engineering prevents duplicates
- **Validation**: Post-generation checks with warnings
- **Correction**: Re-sampling if duplicates detected

## Dependencies

- **Python 3.8+**
- **OpenAI API** (for GPT models)
- **Anthropic API** (for Claude models)
- **Flask** (for web interface)
- **Clingo** (ASP solver) - install separately

### Installing Clingo

**Ubuntu/Debian:**
```bash
sudo apt install clingo
```

**macOS:**
```bash
brew install clingo
```

**From source:**
Visit [https://potassco.org/clingo/](https://potassco.org/clingo/)

## Web Interface Features

- **Rich Text Editor**: Syntax highlighting for puzzle input
- **Advanced Options**: Access to all CLI features
- **Tabbed Results**: Organized display of ASP facts, solutions, and structure
- **Real-time Processing**: Live updates during extraction and solving
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Error Handling**: Clear error messages and recovery suggestions

## Contributing

Contributions welcome! Please feel free to submit pull requests, report bugs, or suggest features.

### Development Setup

```bash
git clone https://github.com/semantisch/llmcl.git
cd llmcl
pip install -r requirements.txt

# Run tests
python llmcl.py sample_puzzle.txt --verbose
python web_app.py  # Test web interface
```

## Acknowledgments

- Built for the BilAI Summer School 2025 Project
- Uses ASP encodings for logic puzzle solving
- Powered by language models
- Thanks to the clingo ASP solver team

---

**START SOLVING, NAUGHTY MODEL!**

```bash
python llmcl.py sample_puzzle.txt --model gpt-4o --force-derive --interactive --verbose
```
