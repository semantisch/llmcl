# LLMCL Web Interface 🌐

A modern Flask web interface for the Logic Puzzle Clue Extractor (LLMCL), making it easy to extract ASP facts from natural language puzzle descriptions through your browser.

## Features ✨

- **User-Friendly Interface**: Clean, responsive web UI built with Bootstrap
- **Model Selection**: Choose between OpenAI GPT and Anthropic Claude models
- **Real-Time Processing**: See extraction progress with loading indicators
- **Multiple Result Views**: Tabbed interface showing ASP facts, solutions, and extracted structure
- **Error Handling**: Automatic syntax correction using LLM feedback
- **Pre-loaded Example**: Sample puzzle ready to test immediately
- **Force-Derive Mode**: Advanced reasoning to derive exact missing values instead of placeholders

## Quick Start 🚀

### 1. Install Dependencies
```bash
cd llmcl
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
# For OpenAI models
export OPENAI_API_KEY='your-openai-api-key'

# For Anthropic models
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

### 3. Start the Web Server
```bash
python3 web_app.py
```

### 4. Open Your Browser
Navigate to: **http://localhost:5000**

## Usage Guide 📖

### Input Methods

1. **JSON Format** (Recommended):
   ```json
   {
     "description": "Four different wineries...",
     "clues": [
       {"id": 1, "text": "The four wineries are..."},
       {"id": 2, "text": "Of Chateau Cork and..."}
     ]
   }
   ```

2. **Plain Text Format**:
   ```
   Four different wineries will be visited on a tour.
   
   1) The four wineries are Vindictive Vines...
   2) Of Chateau Cork and the winery...
   ```

### Model Selection

| Provider | Model | Speed | Quality | Best For |
|----------|-------|-------|---------|----------|
| OpenAI | `gpt-4` | Medium | Excellent | Complex puzzles |
| OpenAI | `gpt-3.5-turbo` | Fast | Good | Simple puzzles |
| Anthropic | `claude-3-5-sonnet-20241022` | Medium | Excellent | Reasoning tasks |
| Anthropic | `claude-3-haiku-20240307` | Fast | Good | Quick processing |

### Result Tabs

1. **ASP Facts**: Generated Answer Set Programming facts
2. **Solution**: Puzzle solution if successfully solved
3. **Structure**: Raw extracted structure from LLM

### Force-Derive Mode

Enable **Force-Derive Mode** for advanced reasoning that derives exact missing values:

**Standard Mode**:
```asp
input(established_year,4,unknown_1).
input(award_winning_wine,4,derived_1).
```

**Force-Derive Mode**:
```asp
input(established_year,4,1983).
input(award_winning_wine,4,shiraz).
```

**When to Use**:
- Complex puzzles with mathematical relationships
- Puzzles requiring calculated values (like "5 years after")
- When you want complete ASP facts without placeholders

**Benefits**:
- More solvable puzzles (no placeholder constraints needed)
- Clearer ASP output
- Better puzzle validation

## Configuration ⚙️

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Claude models |
| `FLASK_SECRET_KEY` | Flask session secret | Optional |
| `FLASK_DEBUG` | Enable debug mode | Optional |
| `PORT` | Server port (default: 5000) | Optional |

### Running in Production

```bash
# Set production environment
export FLASK_DEBUG=False
export FLASK_SECRET_KEY='your-secure-secret-key'
export PORT=8080

# Run with gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 web_app:app
```

## API Endpoints 🔗

### `GET /`
Main web interface

### `POST /process`
Process puzzle and return results

**Form Parameters:**
- `puzzle_text`: Puzzle description (JSON or plain text)
- `model`: LLM model name
- `api_key`: Optional API key override
- `verbose`: Enable verbose output

**Response:**
```json
{
  "success": true,
  "structured_info": {...},
  "asp_facts": "input(...)",
  "syntax_valid": true,
  "syntax_errors": [],
  "solution_success": true,
  "solution_output": "SATISFIABLE...",
  "solution_facts": ["assign(...)"],
  "model_used": "gpt-4",
  "provider": "openai"
}
```

### `GET /health`
Health check endpoint

## Screenshots 📸

### Main Interface
```
┌─────────────────────────────────────────────────────────────┐
│ 🧩 LLMCL - Logic Puzzle Clue Extractor                     │
├─────────────────┬───────────────────────────────────────────┤
│ 📝 Puzzle Input │ 📊 Results                                │
│                 │                                           │
│ [Puzzle Text]   │ ✅ Syntax OK  🎉 Solved  🤖 OpenAI-GPT4  │
│                 │                                           │
│ 🤖 Model: GPT-4 │ [ASP Facts] [Solution] [Structure]        │
│ 🔑 API Key      │                                           │
│ ☑ Verbose       │ Generated ASP facts:                      │
│                 │ input(winery,1,vindictive_vines).        │
│ 🚀 Extract      │ clue(c11a,diff).                         │
│                 │ ...                                       │
└─────────────────┴───────────────────────────────────────────┘
```

## Troubleshooting 🔧

### Common Issues

**"Failed to initialize LLM"**
- Check your API key is set correctly
- Verify the model name is valid
- Ensure you have an active API subscription

**"Network error"**
- Check your internet connection
- Verify the Flask server is running
- Look for firewall blocking port 5000

**"Syntax errors"**
- The system automatically attempts LLM correction
- Check the "ASP Facts" tab for details
- Manual intervention may be needed for complex errors

**"Could not solve puzzle"**
- Ensure clingo is installed: `brew install clingo` (macOS)
- Check if encoding file exists: `encodings/logic-puzzles.lp`
- Verify ASP facts are syntactically correct

### Debug Mode

```bash
export FLASK_DEBUG=True
python3 web_app.py
```

Check browser developer console and Flask logs for detailed error information.

## Development 👨‍💻

### File Structure
```
llmcl/
├── web_app.py              # Main Flask application
├── templates/
│   └── index.html          # Web interface template
├── static/
│   └── css/
│       └── style.css       # Custom styles
├── llm_connector.py        # LLM integration
├── asp_generator.py        # ASP facts generation
└── llmcl.py               # CLI utilities
```

### Adding New Features

1. **New Routes**: Add to `web_app.py`
2. **Frontend**: Modify `templates/index.html`
3. **Styling**: Update `static/css/style.css`
4. **Backend Logic**: Extend existing modules

### Testing

```bash
# Start development server
FLASK_DEBUG=True python3 web_app.py

# Test with curl
curl -X POST http://localhost:5000/process \
  -F "puzzle_text=Test puzzle" \
  -F "model=gpt-4"
```

## Deployment Options 🚀

### Local Development
```bash
python3 web_app.py
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python3", "web_app.py"]
```

### Cloud Platforms
- **Heroku**: Use `Procfile` with `web: python3 web_app.py`
- **Railway**: Auto-detects Flask apps
- **Render**: Deploy from GitHub repository

## License & Support 📄

This web interface is part of the LLMCL project. For issues and feature requests, please check the main project repository.

**Happy puzzle solving! 🧩✨** 