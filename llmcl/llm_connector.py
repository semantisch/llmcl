"""
LLM Connector Module
Handles communication with OpenAI API for extracting structured information
from natural language puzzle descriptions.
"""

import os
import json
import sys
import time
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMConnector:
    """Connector for OpenAI API to extract puzzle structure from natural language."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, verbose: bool = False, interactive: bool = False):
        """
        Initialize the LLM connector.
        
        Args:
            model: Model name (OpenAI: gpt-4, gpt-3.5-turbo, etc. | Anthropic: claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.)
            api_key: API key (if None, uses OPENAI_API_KEY or ANTHROPIC_API_KEY env var)
            verbose: Enable verbose logging
            interactive: Enable interactive mode with confirmation steps
        """
        self.model = model
        self.verbose = verbose
        self.interactive = interactive
        
        # Determine which provider to use based on model name
        self.provider = self._detect_provider(model)
        
        if self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError(
                    "Anthropic library not available. Install with: pip install anthropic>=0.8.0"
                )
            
            # Set up Anthropic client
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable "
                    "or pass it as --api-key argument."
                )
            
            self.client = anthropic.Anthropic(api_key=api_key)
            
        else:  # OpenAI
            # Set up OpenAI client
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                    "or pass it as --api-key argument."
                )
            
            self.client = OpenAI(api_key=api_key)
        
        if self.verbose:
            print(f"✅ LLM Connector initialized with {self.provider} model: {self.model}")
            if self.interactive:
                print("🔄 Interactive mode enabled - will wait for confirmation at each step")
            else:
                print("⚡ Non-interactive mode - will process automatically")
    
    def _detect_provider(self, model: str) -> str:
        """
        Detect which API provider to use based on model name.
        
        Args:
            model: Model name
            
        Returns:
            Provider name: "openai" or "anthropic"
        """
        # Anthropic model patterns
        anthropic_patterns = [
            "claude",
            "anthropic",
        ]
        
        model_lower = model.lower()
        
        for pattern in anthropic_patterns:
            if pattern in model_lower:
                return "anthropic"
        
        # Default to OpenAI
        return "openai"
    
    def _make_api_call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """
        Make an API call to the configured LLM provider with retry logic.
        
        Args:
            messages: List of message dictionaries for the chat completion
            max_retries: Maximum number of retries on failure
            
        Returns:
            Response content from the API
        """
        if self.provider == "anthropic":
            return self._make_anthropic_api_call(messages, max_retries)
        else:
            return self._make_openai_api_call(messages, max_retries)
    
    def _make_openai_api_call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Make an API call to OpenAI with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"🔄 Making OpenAI API call (attempt {attempt + 1}/{max_retries})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent outputs
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                
                if self.verbose:
                    print(f"✅ OpenAI API call successful. Response length: {len(content)} characters")
                
                return content
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    if self.verbose:
                        print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠️ OpenAI API call failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def _make_anthropic_api_call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Make an API call to Anthropic with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"🔄 Making Anthropic API call (attempt {attempt + 1}/{max_retries})...")
                
                # Convert OpenAI-style messages to Anthropic format
                system_message = ""
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    elif msg["role"] == "user":
                        user_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        user_messages.append({"role": "assistant", "content": msg["content"]})
                
                # Make Anthropic API call
                response = self.client.messages.create(
                    model=self.model,
                    system=system_message,
                    messages=user_messages,
                    temperature=0.1,
                    max_tokens=2000
                )
                
                content = response.content[0].text
                
                if self.verbose:
                    print(f"✅ Anthropic API call successful. Response length: {len(content)} characters")
                
                return content
                
            except Exception as e:
                # Handle Anthropic-specific rate limiting and errors
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    if self.verbose:
                        print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠️ Anthropic API call failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def _show_and_confirm(self, data: Dict[str, Any], title: str, show_asp: bool = False) -> bool:
        """
        Show extraction result and wait for user confirmation in interactive mode.
        
        Args:
            data: Data to display
            title: Title for the display section
            show_asp: Whether to also show ASP format (for Phase 1)
            
        Returns:
            True if user confirms, False otherwise
        """
        if not self.interactive:
            return True
        
        print(f"\n{'='*60}")
        print(f"📋 {title.upper()}")
        print("="*60)
        print("JSON FORMAT:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if show_asp:
            print("\nASP FORMAT:")
            if 'attributes' in data:
                # Phase 1: Show input facts
                asp_facts = self._generate_asp_preview(data)
                print(asp_facts)
            elif 'constraints' in data or 'value_definitions' in data:
                # Phase 2: Show constraint rules
                asp_constraints = self._generate_constraint_preview(data)
                print(asp_constraints)
            elif 'clues' in data or 'type' in data:
                # Phase 3: Show clue facts (handle both multi-clue and single-clue format)
                asp_clues = self._generate_clue_preview(data)
                print(asp_clues)
        
        print("="*60)
        
        while True:
            response = input("\n✨ Continue with this result? [y/n/r] (y=yes, n=no/exit, r=retry): ").lower().strip()
            if response in ['y', 'yes', '']:
                return True
            elif response in ['n', 'no']:
                print("❌ Stopping extraction process.")
                sys.exit(0)
            elif response in ['r', 'retry']:
                return False
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'r' (retry)")
    
    def _generate_asp_preview(self, inputs_data: Dict[str, Any]) -> str:
        """Generate ASP format preview for input attributes."""
        facts = []
        attributes = inputs_data.get('attributes', {})
        primary_attribute = inputs_data.get('primary_attribute', '')
        
        # Generate input facts
        for attr_name, values in attributes.items():
            for i, value in enumerate(values, 1):
                facts.append(f"input({attr_name},{i},{value}).")
        
        facts.append("")  # Empty line
        facts.append(f"index({primary_attribute}).")
        
        return '\n'.join(facts)
    
    def _generate_constraint_preview(self, constraint_data: Dict[str, Any]) -> str:
        """Generate ASP format preview for constraint rules."""
        rules = []
        
        # Add comment header
        rules.append("% Constraint rules for placeholders")
        
        # Generate constraint rules for derived values
        constraints = constraint_data.get('constraints', [])
        if constraints:
            rules.append("")
            rules.append("% Constraint rules for derived values")
            for constraint in constraints:
                placeholder = constraint.get('placeholder', '')
                description = constraint.get('description', '')
                asp_rule = constraint.get('asp_rule', '')
                
                if description:
                    rules.append(f"% {placeholder}: {description}")
                
                if asp_rule:
                    rules.append(f"{asp_rule}")
                
                rules.append("")  # Empty line between constraints
        
        # Generate value definitions for unknowns
        value_definitions = constraint_data.get('value_definitions', [])
        if value_definitions:
            rules.append("% Value definitions for unknowns")
            for defn in value_definitions:
                placeholder = defn.get('placeholder', '')
                description = defn.get('description', '')
                possible_values = defn.get('possible_values', [])
                
                if description:
                    rules.append(f"% {placeholder}: {description}")
                
                # Show possible values as comments for now
                if possible_values:
                    values_str = ', '.join(possible_values)
                    rules.append(f"% Possible values: {values_str}")
                
                rules.append("")  # Empty line between definitions
        
        return '\n'.join(rules)
    
    def _generate_clue_preview(self, clue_data: Dict[str, Any]) -> str:
        """Generate ASP format preview for clue extraction results."""
        facts = []
        
        # Handle both new multi-clue format and old single-clue format
        if 'clues' in clue_data:
            # New multi-clue format
            sub_clues = clue_data['clues']
            clue_id = "N"  # Placeholder since we don't know the actual ID yet
            
            for i, sub_clue in enumerate(sub_clues):
                sub_id = sub_clue.get('sub_id', f'{i+1}')
                sub_description = sub_clue.get('description', '')
                
                # Generate unique ASP clue ID (clingo-compatible, no underscores)
                if sub_id:
                    sanitized_sub_id = sub_id.replace('_', 'c')
                    asp_clue_id = f"c{clue_id}{sanitized_sub_id}"
                else:
                    asp_clue_id = f"c{clue_id}s{i+1}"
                
                # Add comment for sub-clue
                if sub_description:
                    facts.append(f"% Sub-clue {asp_clue_id}: {sub_description}")
                
                # Generate ASP facts for this sub-clue
                self._generate_single_clue_preview(facts, asp_clue_id, sub_clue)
                facts.append("")  # Empty line between sub-clues
        else:
            # Old single-clue format
            clue_id = "N"  # Placeholder
            description = clue_data.get('description', '')
            
            if description:
                facts.append(f"% Clue {clue_id}: {description}")
            
            self._generate_single_clue_preview(facts, clue_id, clue_data)
        
        return '\n'.join(facts)
    
    def _generate_single_clue_preview(self, facts: list, clue_id: str, clue_data: Dict[str, Any]) -> None:
        """Generate ASP facts preview for a single clue (helper method)."""
        import re
        
        def sanitize_value(value: str) -> str:
            """Sanitize a value for use in ASP facts (local copy of ASPGenerator method)."""
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).lower())
            sanitized = re.sub(r'_+', '_', sanitized).strip('_')
            if sanitized and sanitized[0].isdigit():
                sanitized = 'n' + sanitized
            return sanitized or 'unknown'
        
        clue_type = clue_data.get('type', 'same')
        
        # Generate clue/2 fact
        facts.append(f"clue({clue_id},{clue_type}).")
        
        # Generate object/4 facts
        objects = clue_data.get('objects', [])
        for obj in objects:
            position = obj.get('position', 1)
            attribute = sanitize_value(obj.get('attribute', ''))
            value = sanitize_value(obj.get('value', ''))
            
            if attribute and value:
                facts.append(f"object({clue_id},{position},{attribute},{value}).")
        
        # Generate target/2 fact if needed
        target_attr = clue_data.get('target_attribute')
        if target_attr and clue_type in ['less', 'next']:
            sanitized_target = sanitize_value(target_attr)
            facts.append(f"target({clue_id},{sanitized_target}).")
    
    def _interactive_extraction(self, extraction_func, *args, title: str, max_retries: int = 3):
        """
        Run extraction with interactive confirmation and retry logic.
        
        Args:
            extraction_func: Function to call for extraction
            *args: Arguments to pass to extraction_func
            title: Title for display
            max_retries: Maximum number of retries
            
        Returns:
            Extraction result
        """
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"\n🔄 Retry attempt {attempt + 1}/{max_retries}")
                
                result = extraction_func(*args)
                
                # Show ASP format for Phase 1 (attributes), Phase 2 (constraints), and Phase 3 (clues)
                show_asp = ("Phase 1" in title or "Attributes" in title or 
                           "Phase 2" in title or "Constraint" in title or
                           "Clue" in title or "CLUE" in title)
                if self._show_and_confirm(result, title, show_asp=show_asp):
                    return result
                else:
                    if attempt == max_retries - 1:
                        print(f"❌ Maximum retries ({max_retries}) reached for {title}")
                        sys.exit(1)
                    continue
                    
            except Exception as e:
                print(f"❌ Error in {title}: {e}")
                if attempt == max_retries - 1:
                    raise e
                if self.interactive:
                    retry = input(f"Retry {title}? [y/n]: ").lower().strip()
                    if retry not in ['y', 'yes', '']:
                        raise e
        
        raise Exception(f"Failed to complete {title} after {max_retries} attempts")
    
    def _extract_puzzle_inputs_internal(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1: Extract all attributes and their possible values from puzzle.
        
        Args:
            puzzle_data: Dictionary with 'description' and 'clues'
            
        Returns:
            Dictionary with attributes and primary_attribute
        """
        system_prompt = """You are an expert at analyzing logic puzzles and extracting entities.

Your task is to identify all the ATTRIBUTES (categories) and their possible VALUES from a logic puzzle.

Based on the puzzle description and clues, extract:
1. All attributes (categories/properties of entities)
2. All possible values for each attribute
3. The primary attribute (usually the one that represents position/order)

Return a JSON object with this structure:
{
  "attributes": {
    "attribute_name": ["value1", "value2", "value3", ...],
    "another_attribute": ["val_a", "val_b", "val_c", ...],
    ...
  },
  "primary_attribute": "attribute_name"
}

Rules:
- Extract ALL concrete entities mentioned explicitly in the puzzle
- Ensure each attribute has the same number of values
- For missing/unknown values, use standardized placeholders: "unknown_1", "unknown_2", etc.
- For derived values (defined by constraints), use: "derived_1", "derived_2", etc.
- Choose primary attribute that represents the main ordering/matching dimension
- Use lowercase with underscores for attribute names (e.g., "entity_type", "position")
- Use lowercase with underscores for values (e.g., "entity_a", "first_place")

PLACEHOLDER GUIDELINES:
- "unknown_X": For values that exist but aren't specified (e.g., missing years)
- "derived_X": For values defined by constraints (e.g., "property with specific constraint")"""

        description = puzzle_data.get('description', '')
        clues_text = '\n'.join([f"{clue['id']}) {clue['text']}" for clue in puzzle_data.get('clues', [])])
        
        user_prompt = f"""Analyze this logic puzzle and extract all attributes and values:

DESCRIPTION:
{description}

CLUES:
{clues_text}

Extract all attributes and their possible values. Return as JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response, "input extraction")
    
    def _extract_clue_facts_internal(self, clue: Dict[str, Any], attributes: Dict[str, List[str]], 
                          primary_attribute: str, constraint_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Phase 2: Extract ASP facts for a single clue.
        
        Args:
            clue: Single clue dictionary with 'id' and 'text'
            attributes: Available attributes and their values
            primary_attribute: The primary attribute for indexing
            
        Returns:
            Dictionary with clue type, objects, and target_attribute
        """
        # Generate ASP format for reference
        asp_facts = self._generate_asp_preview({'attributes': attributes, 'primary_attribute': primary_attribute})
        
        # Add constraint rules context if available
        constraint_context = ""
        if constraint_rules:
            constraint_context = "\nPLACEHOLDER DEFINITIONS:\n"
            for constraint in constraint_rules.get('constraints', []):
                placeholder = constraint.get('placeholder', '')
                description = constraint.get('description', '')
                constraint_context += f"- {placeholder}: {description}\n"
            
            for defn in constraint_rules.get('value_definitions', []):
                placeholder = defn.get('placeholder', '')
                description = defn.get('description', '')
                constraint_context += f"- {placeholder}: {description}\n"
        
        system_prompt = f"""You are an expert at converting logic puzzle clues to ASP format.

IMPORTANT: Complex clues may require MULTIPLE ASP clue facts. Decompose the logic completely.

CLUE TYPES:
- same: Attributes belong together in the solution
- diff: Attributes cannot belong together  
- less: Ordering constraint (smaller position → smaller target value)
- next: Adjacent/consecutive constraint (consecutive positions → consecutive target values)

COMPLEX CLUE DECOMPOSITION:
Identify clue patterns and decompose appropriately:

PATTERN 1 - "The N items are A, B, C, D" (listing distinct entities):
→ clue(N,diff): All items are different from each other
→ object(N,1,attr1,A), object(N,2,attr2,B), object(N,3,attr3,C), object(N,4,attr4,D)

PATTERN 2 - "Of A and B, one has X and the other has Y" (mutual exclusion with properties):
→ clue(N,diff): A and B are different entities
→ clue(N+1,same): Either A has X or B has X (disjunction)
→ clue(N+2,same): Either A has Y or B has Y (disjunction)

PATTERN 3 - "X was not established between Y and Z" (exclusion range):
→ clue(N,diff): X ≠ year_Y
→ clue(N+1,diff): X ≠ year_Y+1
→ clue(N+2,diff): X ≠ year_Y+2
→ ... (for each year in range)

PATTERN 4 - "X is immediately before OR immediately after Y" (bidirectional adjacency):
→ clue(N,next): X and Y are adjacent in ordering
→ object(N,1,attr1,X), object(N,2,attr2,Y)
→ target(N,ordering_attribute)

PATTERN 5 - "X is immediately before Y" (directional adjacency):
→ clue(N,next): X is immediately before Y  
→ object(N,1,attr1,X), object(N,2,attr2,Y)
→ target(N,ordering_attribute)

CRITICAL: 
- For "same" clues with disjunction: Put BOTH options at position 1, and the shared property at position 2
- For "diff" clues: Put different entities at different positions
- For "next" clues: SINGLE clue handles "before OR after" automatically - don't split into multiple clues
- For "less" clues: Handle ordering constraints where smaller position → smaller target value
- Don't add unnecessary exclusivity rules - stick to the explicit logic
- "OR" in adjacency contexts (before OR after) = single next clue, not multiple clues

AVAILABLE ATTRIBUTES (JSON):
{json.dumps(attributes, indent=2)}

AVAILABLE ATTRIBUTES (ASP FORMAT):
{asp_facts}{constraint_context}

PRIMARY ATTRIBUTE: {primary_attribute}

Return JSON with this structure for MULTIPLE clues:
{{
  "clues": [
    {{
      "sub_id": "2a",
      "type": "diff", 
      "description": "Entity A and Entity B are different",
      "objects": [
        {{"position": 1, "attribute": "type_1", "value": "entity_a"}},
        {{"position": 2, "attribute": "type_2", "value": "entity_b"}}
      ]
    }},
    {{
      "sub_id": "2b",
      "type": "same",
      "description": "Either Entity A or Entity B has Property X", 
      "objects": [
        {{"position": 1, "attribute": "type_1", "value": "entity_a"}},
        {{"position": 1, "attribute": "type_2", "value": "entity_b"}},
        {{"position": 2, "attribute": "property", "value": "property_x"}}
      ]
    }},
    {{
      "sub_id": "2c",
      "type": "same",
      "description": "Either Entity A or Entity B has Property Y",
      "objects": [
        {{"position": 1, "attribute": "type_1", "value": "entity_a"}},
        {{"position": 1, "attribute": "type_2", "value": "entity_b"}},
        {{"position": 2, "attribute": "property", "value": "property_y"}}
      ]
    }},
    {{
      "sub_id": "2d",
      "type": "same",
      "description": "Entity A has Property Y",
      "objects": [
        {{"position": 1, "attribute": "type_1", "value": "entity_a"}},
        {{"position": 2, "attribute": "property", "value": "property_y"}}
      ]
    }},
    {{
      "sub_id": "5a",
      "type": "next",
      "description": "Entity A and Entity B are adjacent in ordering",
      "objects": [
        {{"position": 1, "attribute": "type_1", "value": "entity_a"}},
        {{"position": 2, "attribute": "type_2", "value": "entity_b"}}
      ],
      "target_attribute": "ordering_attribute"
    }}
  ]
}}

CRITICAL RULES:
- Decompose complex logic into multiple simple ASP clues
- Use sub_id like "2a", "2b", "2c" for clues derived from the same source  
- Each ASP clue should express ONE logical relationship
- Use ONLY exact attribute names and values from the ASP format above

POSITION RULES FOR ASP CLUES:
- For "diff" clues: Put different entities at different positions (1, 2, 3, 4...)
- For "same" clues with disjunction: Put ALL disjunction options at position 1, shared property at position 2
- For "same" clues with conjunction: Put all properties that belong together at position 1

EXAMPLES:
- "A, B, C are all different" → diff clue: A at pos 1, B at pos 2, C at pos 3
- "Either A or B has property X" → same clue: A and B at pos 1, X at pos 2  
- "A has both X and Y" → same clue: A, X, and Y all at pos 1

- Include "description" to explain each sub-clue's purpose"""

        user_prompt = f"""Convert this clue to ASP format:

CLUE {clue['id']}: {clue['text']}

Extract the clue type, objects, and target attribute. Return as JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response, f"clue {clue['id']} extraction")
    
    def extract_constraint_rules(self, puzzle_data: Dict[str, Any], attributes: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Phase 3: Extract ASP constraint rules for unknown/derived values.
        
        Args:
            puzzle_data: Original puzzle data
            attributes: Extracted attributes with placeholders
            
        Returns:
            Dictionary with constraint rules and value definitions
        """
        # Find all placeholders
        placeholders = []
        for attr_name, values in attributes.items():
            for value in values:
                if value.startswith('unknown_') or value.startswith('derived_'):
                    placeholders.append({'attribute': attr_name, 'value': value})
        
        if not placeholders:
            return {'constraints': [], 'value_definitions': []}
        
        description = puzzle_data.get('description', '')
        clues_text = '\n'.join([f"{clue['id']}) {clue['text']}" for clue in puzzle_data.get('clues', [])])
        placeholders_info = '\n'.join([f"- {p['attribute']}: {p['value']}" for p in placeholders])
        
        system_prompt = f"""You are an expert at creating ASP constraint rules for logic puzzles.

CRITICAL: Each placeholder corresponds to exactly ONE position in ONE attribute. Do not create multiple rules for the same attribute.

PLACEHOLDERS TO PROCESS:
{placeholders_info}

PLACEHOLDER TYPES:
- "unknown_X": Values that exist but aren't explicitly named (provide possible values)
- "derived_X": Values defined by constraints from the puzzle (create constraint rules)

REASONING PROCESS FOR DERIVED VALUES:
1. Identify the constraint from puzzle clues
2. Analyze which existing values satisfy/violate the constraint  
3. Create COMPLETE rules that either:
   a) Positively define what the placeholder IS (if only one option remains)
   b) Eliminate ALL invalid options (if multiple options remain)

EXAMPLES:
- If constraint is "single word property" and options are [value_a, value_b_c, value_d_e_f]:
  * value_a = single word ✓
  * value_b_c = multiple words ✗  
  * value_d_e_f = multiple words ✗
  * Result: derived_1 = value_a.

- If constraint is "not in range X-Y" and options are [val1, val2, val3, val4]:
  * val1 = not in range ✓
  * val2 = in range ✗
  * val3 = in range ✗  
  * val4 = not in range ✓
  * Result: :- derived_1 = val2. and :- derived_1 = val3.

Return JSON with this structure:
{{
  "constraints": [
    {{
      "placeholder": "derived_1",
      "attribute": "property_type",
      "description": "Property with single word constraint", 
      "reasoning": "value_a=1 word✓, value_b_c=2 words✗, value_d_e=2 words✗",
      "asp_rule": "derived_1 = value_a."
    }}
  ],
  "value_definitions": [
    {{
      "placeholder": "unknown_1", 
      "attribute": "numeric_property",
      "description": "Missing numeric value with constraints",
      "possible_values": ["val1", "val2"]
    }}
  ]
}}

RULES:
1. Process ONLY the placeholders listed above
2. Include "reasoning" field showing your logical analysis
3. Create COMPLETE constraints - don't leave partial logic
4. Use positive definitions (X = value) when only one option remains
5. Use negative constraints (:- X = value) only when multiple options remain
6. Analyze ALL available values against the constraint"""
        
        user_prompt = f"""Create ASP constraint rules for these placeholders:

PLACEHOLDERS FOUND:
{placeholders_info}

PUZZLE DESCRIPTION:
{description}

CLUES:
{clues_text}

Generate ASP rules and value definitions for these placeholders."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response, "constraint rules extraction")
    
    def extract_puzzle_inputs(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public interface for Phase 1: Extract attributes and values with optional interaction.
        """
        if self.interactive:
            return self._interactive_extraction(
                self._extract_puzzle_inputs_internal, puzzle_data,
                title="Phase 1: Attributes and Values"
            )
        else:
            return self._extract_puzzle_inputs_internal(puzzle_data)
    
    def extract_clue_facts(self, clue: Dict[str, Any], attributes: Dict[str, List[str]], 
                          primary_attribute: str, constraint_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Public interface for Phase 3: Extract clue facts with optional interaction.
        """
        if self.interactive:
            return self._interactive_extraction(
                self._extract_clue_facts_internal, clue, attributes, primary_attribute, constraint_rules,
                title=f"Clue {clue.get('id', '?')}: {clue.get('text', '')}"
            )
        else:
            return self._extract_clue_facts_internal(clue, attributes, primary_attribute, constraint_rules)
    
    def _parse_json_response(self, response: str, context: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with error handling."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            if self.verbose:
                print(f"✅ Successfully parsed {context}")
            
            return result
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON parsing failed for {context}: {e}")
                print("Raw response:")
                print(response)
            raise ValueError(f"Failed to parse JSON response for {context}: {e}")
    
    def fix_asp_syntax(self, asp_code: str, syntax_errors: List[str]) -> str:
        """
        Fix ASP syntax errors using LLM.
        
        Args:
            asp_code: The ASP code with syntax errors
            syntax_errors: List of syntax error messages
            
        Returns:
            Corrected ASP code
        """
        error_list = '\n'.join([f"- {error}" for error in syntax_errors])
        
        system_prompt = """You are an expert in Answer Set Programming (ASP) syntax.

Your task is to fix syntax errors in ASP code. ASP facts must follow these rules:

VALID ASP SYNTAX:
- Facts end with a period: predicate(arg1,arg2,arg3).
- Predicates start with lowercase letter: input(attr,1,value).
- Arguments are separated by commas: object(1,2,attr,value).
- No spaces around commas: clue(1,same). NOT clue(1, same).
- Comments start with %: % This is a comment
- Constraint rules use :- syntax: :- condition.
- Choice rules use {} syntax: {choice(X)} :- condition.

COMMON ERRORS TO FIX:
- Missing periods at end of facts
- Invalid predicate names (must start with lowercase)
- Malformed constraint syntax like "derived_1 = value." → should be ":- not derived_1 = value." or equivalent
- Invalid assignment syntax → convert to proper ASP constraints
- CLINGO ANONYMOUS VARIABLE ERRORS: Variables with underscores like "1_1a" cause "unexpected <ANONYMOUS>" errors
  * Fix: Replace underscores in predicate names with letters (1_1a → c11a, 2_2b → c22b)
- CLINGO PARSING ERRORS: Any syntax that clingo cannot parse
  * Common issue: Invalid identifier names, malformed predicates, incorrect operators

FIX STRATEGY:
1. Identify each syntax error
2. Convert invalid syntax to valid ASP
3. Preserve the logical meaning
4. Keep all comments and structure intact

Return ONLY the corrected ASP code, maintaining the original structure and comments.

IMPORTANT: Do NOT wrap your response in markdown code blocks (```). Return raw ASP code only."""

        user_prompt = f"""Fix the syntax errors in this ASP code:

SYNTAX ERRORS DETECTED:
{error_list}

ASP CODE TO FIX:
{asp_code}

Please fix all syntax errors and return the corrected ASP code."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print("🔄 Sending ASP code to LLM for syntax correction...")
            print(f"  - Found {len(syntax_errors)} syntax errors")
            print("  - Errors to fix:")
            for error in syntax_errors:
                print(f"    • {error}")
        
        response = self._make_api_call(messages)
        
        if self.verbose:
            print("✅ LLM syntax correction completed")
        
        # Clean up the response - remove markdown code blocks if present
        cleaned_response = response.strip()
        
        # Remove opening code block markers
        if cleaned_response.startswith('```asp'):
            cleaned_response = cleaned_response[6:].strip()
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:].strip()
        
        # Remove closing code block markers
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3].strip()
        
        if self.verbose:
            print("🧹 Cleaned markdown formatting from LLM response")
        
        return cleaned_response
    
    def derive_missing_values(self, puzzle_data: Dict[str, Any], attributes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Use deep reasoning to derive all missing values instead of using placeholders.
        Samples multiple derivations and chooses the best one.
        
        Args:
            puzzle_data: Original puzzle data
            attributes: Extracted attributes with placeholders
            
        Returns:
            Updated attributes with derived values replacing placeholders
        """
        # Find all placeholders
        placeholders = []
        for attr_name, values in attributes.items():
            for i, value in enumerate(values):
                if value.startswith('unknown_') or value.startswith('derived_'):
                    placeholders.append({
                        'attribute': attr_name, 
                        'value': value, 
                        'position': i + 1
                    })
        
        if not placeholders:
            return attributes
        
        description = puzzle_data.get('description', '')
        clues_text = '\n'.join([f"{clue['id']}) {clue['text']}" for clue in puzzle_data.get('clues', [])])
        placeholders_info = '\n'.join([f"- {p['attribute']} position {p['position']}: {p['value']}" for p in placeholders])
        
        # Generate current attribute state for context
        current_attributes = []
        for attr_name, values in attributes.items():
            current_attributes.append(f"{attr_name}: {values}")
        attributes_context = '\n'.join(current_attributes)
        
        system_prompt = f"""You are an expert logic puzzle solver. Your task is to use deep reasoning to derive the EXACT missing values for ALL placeholders.

CRITICAL: You MUST resolve EVERY SINGLE placeholder. No placeholder should remain unresolved.

🚨 ABSOLUTE UNIQUENESS RULE: NEVER, EVER assign a value that already exists in the same attribute. This is the #1 rule - violating it is a critical error.

CURRENT ATTRIBUTES:
{attributes_context}

PLACEHOLDERS TO RESOLVE (ALL OF THEM):
{placeholders_info}

MANDATORY REQUIREMENT: You must provide a derived_value for EVERY placeholder listed above. Do not skip any.

🔍 UNIQUENESS VALIDATION REQUIRED: For each derived_value you assign:
1. Look at the current values in that attribute
2. Verify your new value is NOT in that list
3. If it would create a duplicate → STOP and choose a different value
4. Only assign values that are 100% unique within that attribute

REASONING APPROACH:
1. Analyze all constraints from the puzzle clues
2. Set up logical relationships and equations
3. Solve step-by-step to find exact values
4. Consider values that might need to be derived (like calculating values, inferring missing options)
5. For constraint-based placeholders (derived_X), analyze which existing values satisfy the constraints
6. **VALIDATE UNIQUENESS: Check that your derived value is NOT already used in that attribute**
7. If your calculated value creates a duplicate, re-examine the constraints or generate an alternative
8. Ensure all constraints are satisfied AND all values are unique

CONSTRAINT ANALYSIS FOR DERIVED PLACEHOLDERS:
- Look for clue patterns like "single word", "not between X and Y", "has property Z"
- Check each existing value against the constraint
- Eliminate values that don't satisfy the constraint
- **ALSO eliminate any values already used in the same attribute (uniqueness constraint)**
- Choose the remaining valid value or generate a new one if needed

EXAMPLE REASONING:
- If "last item is 5 years after first item" and options are [year1, year2], but first=year2 would make last=year3
- Even though year3 isn't mentioned, it's the logically derived value
- Generate new values when logic demands them

- If constraint is "single word property" and existing values are ["value_a", "value_b_c", "value_d_e", "derived_1"]
- value_a = 1 word ✓ but already used ✗, value_b_c = 2 words ✗, value_d_e = 2 words ✗
- Need single word value NOT already in list → derived_1 = "value_x" (new single-word value)

🚫 FORBIDDEN EXAMPLES (DO NOT DO THIS):
- If attribute has [value_a, value_b, value_c, placeholder] and you assign placeholder = value_a → WRONG! Duplicate!
- If calculation gives "placeholder = existing_value" → WRONG! Must find different solution!
- Assigning placeholder = any value already in that attribute's list → CRITICAL ERROR!

✅ CORRECT EXAMPLES:
- If attribute has [value_a, value_b, value_c, placeholder] → assign placeholder = value_d (new unique value)
- If calculation creates duplicate → generate alternative value that satisfies constraints but is unique

Return JSON with this structure:
{{
  "derived_values": [
    {{
      "placeholder": "unknown_1",
      "attribute": "numeric_attr",
      "position": 4,
      "derived_value": "value_z",
      "reasoning": "If first entity has value_x, then last entity (position 4) must have value_z based on constraint"
    }},
    {{
      "placeholder": "derived_1", 
      "attribute": "property_type",
      "position": 4,
      "derived_value": "value_x",
      "reasoning": "Step 1: Constraint requires single-word value. Step 2: Uniqueness check - existing values [value_a, value_b_c, value_d_e] - all checked. Step 3: Generated value_x (verified unique, not in existing list)."
    }}
  ]
}}

CRITICAL RULES:
1. **🚨 UNIQUENESS IS RULE #1: Never assign a value that already exists in the same attribute**
2. Derive EXACT values, not just constraints  
3. Generate new values if logic requires them (like calculated values)
4. Show complete step-by-step reasoning including uniqueness validation
5. Ensure all puzzle constraints are satisfied
6. **MANDATORY: Replace ALL placeholders with concrete values - every single one!**
7. **FINAL CHECK: Before submitting, verify no attribute has duplicate values**
8. If you can't determine exact value from constraints, generate a logical unique value

"""

        user_prompt = f"""Derive the exact missing values for these placeholders using deep logical reasoning:

PUZZLE DESCRIPTION:
{description}

CLUES:
{clues_text}

CURRENT ATTRIBUTES WITH PLACEHOLDERS:
{attributes_context}

Use step-by-step logical reasoning to find the exact values for each placeholder. Generate new values if the logic requires them.

🚨 CRITICAL: Before assigning any value, check that it doesn't already exist in that attribute. Duplicates are absolutely forbidden and indicate a critical error in your reasoning. You naughty model."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print("🧠 Using deep reasoning to derive missing values...")
            print(f"  - Found {len(placeholders)} placeholders to resolve")
            print("  - Sampling 3 derivation attempts...")
        
        # Sample 3 different derivation attempts
        derivation_attempts = []
        for attempt in range(3):
            if self.verbose:
                print(f"    • Attempt {attempt + 1}/3...")
            
            try:
                response = self._make_api_call(messages)
                result = self._parse_json_response(response, f"derivation attempt {attempt + 1}")
                
                # Apply derived values to get complete attributes
                test_attributes = attributes.copy()
                for attr_name in test_attributes:
                    test_attributes[attr_name] = test_attributes[attr_name].copy()
                
                for derived in result.get('derived_values', []):
                    placeholder = derived.get('placeholder')
                    attribute = derived.get('attribute')
                    position = derived.get('position', 1)
                    derived_value = derived.get('derived_value')
                    
                    if attribute in test_attributes and derived_value:
                        if 1 <= position <= len(test_attributes[attribute]):
                            test_attributes[attribute][position - 1] = derived_value
                
                derivation_attempts.append({
                    'attempt_id': attempt + 1,
                    'raw_result': result,
                    'applied_attributes': test_attributes,
                    'success': True
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"      ❌ Attempt {attempt + 1} failed: {e}")
                derivation_attempts.append({
                    'attempt_id': attempt + 1,
                    'raw_result': None,
                    'applied_attributes': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Filter successful attempts
        successful_attempts = [a for a in derivation_attempts if a['success']]
        
        if not successful_attempts:
            if self.verbose:
                print("❌ All derivation attempts failed")
            return attributes
        
        if len(successful_attempts) == 1:
            if self.verbose:
                print("  - Only 1 successful attempt, using it directly")
            chosen_attempt = successful_attempts[0]
        else:
            if self.verbose:
                print(f"  - {len(successful_attempts)} successful attempts, choosing best one...")
            
            # Choose the best derivation
            chosen_attempt = self._choose_best_derivation(puzzle_data, attributes, successful_attempts, placeholders)
        
        # Apply the chosen derivation
        if chosen_attempt and chosen_attempt['success']:
            chosen_result = chosen_attempt['raw_result']
            updated_attributes = attributes.copy()
            for attr_name in updated_attributes:
                updated_attributes[attr_name] = updated_attributes[attr_name].copy()
            
            if self.verbose:
                print(f"  - Applying chosen derivation (attempt {chosen_attempt['attempt_id']}):")
            
            for derived in chosen_result.get('derived_values', []):
                placeholder = derived.get('placeholder')
                attribute = derived.get('attribute')
                position = derived.get('position', 1)
                derived_value = derived.get('derived_value')
                reasoning = derived.get('reasoning', '')
                
                if attribute in updated_attributes and derived_value:
                    if 1 <= position <= len(updated_attributes[attribute]):
                        old_value = updated_attributes[attribute][position - 1]
                        updated_attributes[attribute][position - 1] = derived_value
                        
                        if self.verbose:
                            print(f"    ✅ {attribute}[{position}]: {old_value} → {derived_value}")
                            print(f"       Reasoning: {reasoning}")
            
            return updated_attributes
        else:
            if self.verbose:
                print("❌ No valid derivation could be applied")
            return attributes
    
    def _interactive_derive_values(self, puzzle_data: Dict[str, Any], attributes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Interactive wrapper for derive_missing_values with confirmation.
        
        Args:
            puzzle_data: Original puzzle data
            attributes: Extracted attributes with placeholders
            
        Returns:
            Updated attributes with derived values
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"\n🔄 Retry attempt {attempt + 1}/{max_retries}")
                
                # Get derived attributes
                derived_attributes = self.derive_missing_values(puzzle_data, attributes)
                
                # Format for display - show the changes made
                changes_made = []
                for attr_name, old_values in attributes.items():
                    new_values = derived_attributes.get(attr_name, old_values)
                    for i, (old_val, new_val) in enumerate(zip(old_values, new_values)):
                        if old_val != new_val:
                            changes_made.append({
                                'attribute': attr_name,
                                'position': i + 1,
                                'old_value': old_val,
                                'new_value': new_val
                            })
                
                display_data = {
                    'original_attributes': attributes,
                    'derived_attributes': derived_attributes,
                    'changes_made': changes_made
                }
                
                if self._show_and_confirm(display_data, "Phase 2: Force-Derive Missing Values", show_asp=False):
                    return derived_attributes
                else:
                    if attempt == max_retries - 1:
                        print(f"❌ Maximum retries ({max_retries}) reached for force-derive")
                        sys.exit(1)
                    continue
                    
            except Exception as e:
                print(f"❌ Error in force-derive: {e}")
                if attempt == max_retries - 1:
                    raise e
                if self.interactive:
                    retry = input(f"Retry force-derive? [y/n]: ").lower().strip()
                    if retry not in ['y', 'yes', '']:
                        raise e
        
        raise Exception(f"Failed to complete force-derive after {max_retries} attempts")
    
    def _choose_best_derivation(self, puzzle_data: Dict[str, Any], original_attributes: Dict[str, List[str]], 
                              successful_attempts: List[Dict], placeholders: List[Dict]) -> Dict:
        """
        Choose the best derivation from multiple attempts using LLM evaluation.
        
        Args:
            puzzle_data: Original puzzle data
            original_attributes: Original attributes with placeholders
            successful_attempts: List of successful derivation attempts
            placeholders: List of placeholders that need to be resolved
            
        Returns:
            The best derivation attempt
        """
        # Format attempts for comparison
        attempts_summary = []
        for attempt in successful_attempts:
            attempt_id = attempt['attempt_id']
            derived_values = attempt['raw_result'].get('derived_values', [])
            applied_attrs = attempt['applied_attributes']
            
            # Check for uniqueness violations
            uniqueness_violations = []
            for attr_name, values in applied_attrs.items():
                unique_values = set(values)
                if len(unique_values) != len(values):
                    # Find duplicates
                    seen = set()
                    duplicates = []
                    for val in values:
                        if val in seen:
                            duplicates.append(val)
                        seen.add(val)
                    uniqueness_violations.append(f"{attr_name}: {duplicates}")
            
            # Check placeholder resolution
            unresolved_placeholders = []
            for attr_name, values in applied_attrs.items():
                for i, value in enumerate(values):
                    if value.startswith('unknown_') or value.startswith('derived_'):
                        unresolved_placeholders.append(f"{attr_name}[{i+1}]: {value}")
            
            attempts_summary.append({
                'attempt_id': attempt_id,
                'derived_values': derived_values,
                'uniqueness_violations': uniqueness_violations,
                'unresolved_placeholders': unresolved_placeholders,
                'num_resolved': len(derived_values)
            })
        
        # Prepare evaluation prompt
        description = puzzle_data.get('description', '')
        clues_text = '\n'.join([f"{clue['id']}) {clue['text']}" for clue in puzzle_data.get('clues', [])])
        placeholders_info = '\n'.join([f"- {p['attribute']} position {p['position']}: {p['value']}" for p in placeholders])
        
        system_prompt = """You are an expert evaluator of logic puzzle derivations. Your task is to choose the best derivation attempt from multiple options.

EVALUATION CRITERIA (in order of importance):
1. **UNIQUENESS COMPLIANCE**: No duplicate values within any attribute (CRITICAL)
2. **COMPLETENESS**: All placeholders resolved with concrete values
3. **LOGICAL CONSISTENCY**: Derived values satisfy puzzle constraints
4. **REASONING QUALITY**: Clear, step-by-step logical reasoning

SCORING:
- Uniqueness violations = AUTOMATIC DISQUALIFICATION
- Unresolved placeholders = Major penalty
- Poor reasoning = Minor penalty
- Good constraint satisfaction = Bonus points

Return JSON with this structure:
{
  "chosen_attempt": 2,
  "reasoning": "Attempt 2 is best because: (1) No uniqueness violations (attempts 1&3 have duplicates), (2) All placeholders resolved, (3) Values satisfy puzzle constraints X, Y, Z, (4) Clear logical reasoning for each derivation.",
  "evaluation_summary": {
    "attempt_1": "DISQUALIFIED: Duplicate values in attribute_x",
    "attempt_2": "EXCELLENT: Perfect uniqueness, complete resolution, sound logic", 
    "attempt_3": "POOR: Unresolved placeholders remain"
  }
}"""

        attempts_text = []
        for summary in attempts_summary:
            attempt_text = f"""ATTEMPT {summary['attempt_id']}:
Derived Values: {summary['num_resolved']} total
{json.dumps(summary['derived_values'], indent=2)}

Uniqueness Check: {"✅ PASS" if not summary['uniqueness_violations'] else "❌ VIOLATIONS: " + str(summary['uniqueness_violations'])}
Completeness Check: {"✅ COMPLETE" if not summary['unresolved_placeholders'] else "❌ INCOMPLETE: " + str(summary['unresolved_placeholders'])}"""
            attempts_text.append(attempt_text)
        
        user_prompt = f"""Evaluate these derivation attempts and choose the best one:

PUZZLE DESCRIPTION:
{description}

CLUES:
{clues_text}

PLACEHOLDERS TO RESOLVE:
{placeholders_info}

DERIVATION ATTEMPTS:
{chr(10).join(attempts_text)}

Choose the best attempt and explain your reasoning."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print("    🔍 Evaluating derivation attempts...")
        
        try:
            response = self._make_api_call(messages)
            evaluation = self._parse_json_response(response, "derivation evaluation")
            
            chosen_id = evaluation.get('chosen_attempt', 1)
            reasoning = evaluation.get('reasoning', 'No reasoning provided')
            evaluation_summary = evaluation.get('evaluation_summary', {})
            
            if self.verbose:
                print(f"    🏆 Chosen: Attempt {chosen_id}")
                print(f"    📝 Reasoning: {reasoning}")
                for attempt_key, summary in evaluation_summary.items():
                    print(f"    - {attempt_key}: {summary}")
            
            # Find and return the chosen attempt
            for attempt in successful_attempts:
                if attempt['attempt_id'] == chosen_id:
                    return attempt
            
            # Fallback to first attempt if chosen_id not found
            if self.verbose:
                print("    ⚠️ Chosen attempt ID not found, using first successful attempt")
            return successful_attempts[0]
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Evaluation failed: {e}, using first successful attempt")
            return successful_attempts[0]
    
    def extract_puzzle_structure(self, puzzle_description: str, force_derive: bool = False) -> Dict[str, Any]:
        """
        Legacy method - now uses two-phase approach internally.
        
        Args:
            puzzle_description: Natural language description or JSON string
            
        Returns:
            Dictionary containing structured puzzle information
        """
        # Try to parse as JSON first
        try:
            puzzle_data = json.loads(puzzle_description)
        except json.JSONDecodeError:
            # Fallback: treat as plain text description
            puzzle_data = {
                "description": puzzle_description,
                "clues": []
            }
        
        if self.verbose:
            print("🔄 Phase 1: Extracting attributes and values...")
        
        # Phase 1: Extract inputs (with interactive confirmation)
        inputs_result = self.extract_puzzle_inputs(puzzle_data)
        
        attributes = inputs_result.get('attributes', {})
        primary_attribute = inputs_result.get('primary_attribute', list(attributes.keys())[0] if attributes else 'position')
        
        if self.verbose:
            print(f"  - Found {len(attributes)} attributes")
            print(f"  - Primary attribute: {primary_attribute}")
        
        # Phase 2: Handle placeholders - either derive exact values or create constraint rules
        constraint_rules = {'constraints': [], 'value_definitions': []}
        
        # Check if we have any placeholders
        has_placeholders = any(
            any(value.startswith('unknown_') or value.startswith('derived_') 
                for value in values)
            for values in attributes.values()
        )
        
        if has_placeholders:
            if force_derive:
                # Phase 2A: Force-derive mode - derive exact values instead of constraint rules
                if self.verbose:
                    print("\n🧠 Phase 2: Force-deriving exact values...")
                
                if self.interactive:
                    attributes = self._interactive_derive_values(puzzle_data, attributes)
                else:
                    attributes = self.derive_missing_values(puzzle_data, attributes)
                
                if self.verbose:
                    derived_count = sum(1 for values in attributes.values() 
                                      for value in values 
                                      if not (value.startswith('unknown_') or value.startswith('derived_')))
                    total_count = sum(len(values) for values in attributes.values())
                    print(f"  - Derived {derived_count}/{total_count} values exactly")
                
                # Check if any placeholders remain unresolved
                remaining_placeholders = []
                for attr_name, values in attributes.items():
                    for i, value in enumerate(values):
                        if value.startswith('unknown_') or value.startswith('derived_'):
                            remaining_placeholders.append(f"{attr_name}[{i+1}]: {value}")
                
                if remaining_placeholders:
                    print("⚠️ WARNING: Some placeholders remain unresolved after force-derive:")
                    for placeholder in remaining_placeholders:
                        print(f"  - {placeholder}")
                    print("  This may indicate the LLM couldn't determine exact values for all constraints.")
            else:
                # Phase 2B: Standard mode - create constraint rules for placeholders
                if self.verbose:
                    print("\n🔄 Phase 2: Processing placeholders and constraints...")
                
                if self.interactive:
                    constraint_rules = self._interactive_extraction(
                        self.extract_constraint_rules, puzzle_data, attributes,
                        title="Phase 2: Constraint Rules for Placeholders"
                    )
                else:
                    constraint_rules = self.extract_constraint_rules(puzzle_data, attributes)
                
                if self.verbose:
                    constraints_count = len(constraint_rules.get('constraints', []))
                    definitions_count = len(constraint_rules.get('value_definitions', []))
                    print(f"  - Generated {constraints_count} constraint rules")
                    print(f"  - Generated {definitions_count} value definitions")
        
        if self.verbose:
            print("\n🔄 Phase 3: Processing clues...")
        
        # Phase 3: Extract clues (with interactive confirmation for each clue)
        # Now the LLM knows what all placeholders mean from Phase 2
        clues = []
        for clue in puzzle_data.get('clues', []):
            if self.verbose:
                print(f"  - Processing clue {clue['id']}...")
            
            clue_result = self.extract_clue_facts(clue, attributes, primary_attribute, constraint_rules)
            
            clue_result['id'] = clue['id']
            clue_result['description'] = clue['text']
            clues.append(clue_result)
        
        if self.verbose:
            print(f"  - Processed {len(clues)} clues")
        
        return {
            'attributes': attributes,
            'primary_attribute': primary_attribute,
            'clues': clues,
            'constraint_rules': constraint_rules
        }
    
    def refine_clue_extraction(self, puzzle_description: str, initial_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine clue extraction by asking for more specific analysis.
        
        Args:
            puzzle_description: Original puzzle description
            initial_structure: Initially extracted structure
            
        Returns:
            Refined structure with better clue analysis
        """
        system_prompt = """You are refining the clue extraction for a logic puzzle.

Review the initial extraction and improve the clue analysis by:
1. Ensuring all clues from the text are captured
2. Correctly identifying clue types (same, diff, less, next)
3. Properly structuring object relationships
4. Adding any missing clues or fixing incorrectly parsed ones

Return the refined JSON structure."""

        user_prompt = f"""Original puzzle:
{puzzle_description}

Initial extraction:
{json.dumps(initial_structure, indent=2)}

Please review and refine this extraction, ensuring all clues are properly captured and typed."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._make_api_call(messages)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            refined_structure = json.loads(json_str)
            
            if self.verbose:
                print("✅ Successfully refined clue extraction")
            
            return refined_structure
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON parsing failed for {context}: {e}")
                print("Raw response:")
                print(response)
            raise ValueError(f"Failed to parse JSON response for {context}: {e}") 
