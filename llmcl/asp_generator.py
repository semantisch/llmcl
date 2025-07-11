"""
ASP Generator Module
Converts structured puzzle information extracted by LLM into ASP facts
compatible with the existing logic puzzle format.
"""

from typing import Dict, List, Any, Set
import re


class ASPGenerator:
    """Generates ASP facts from structured puzzle information."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the ASP generator.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.clue_counter = 1
    
    def _sanitize_value(self, value: str) -> str:
        """
        Sanitize a value for use in ASP facts.
        
        Args:
            value: Raw value string
            
        Returns:
            Sanitized value suitable for ASP
        """
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).lower())
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it starts with a letter (ASP requirement)
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n' + sanitized
        
        return sanitized or 'unknown'
    
    def _generate_input_facts(self, attributes: Dict[str, List[str]]) -> str:
        """
        Generate input/3 facts for all attributes and their values.
        
        Args:
            attributes: Dictionary mapping attribute names to value lists
            
        Returns:
            String containing input facts
        """
        facts = []
        
        for attr_name, values in attributes.items():
            sanitized_attr = self._sanitize_value(attr_name)
            
            for i, value in enumerate(values, 1):
                sanitized_value = self._sanitize_value(value)
                facts.append(f"input({sanitized_attr},{i},{sanitized_value}).")
        
        return '\n'.join(facts)
    
    def _generate_index_fact(self, primary_attribute: str) -> str:
        """
        Generate index/1 fact for the primary attribute.
        
        Args:
            primary_attribute: Name of the primary attribute
            
        Returns:
            String containing index fact
        """
        sanitized_attr = self._sanitize_value(primary_attribute)
        return f"index({sanitized_attr})."
    
    def _generate_clue_facts(self, clues: List[Dict[str, Any]]) -> str:
        """
        Generate clue/2, object/4, and target/2 facts from clue information.
        Now supports multi-clue format where one natural language clue can generate multiple ASP clues.
        
        Args:
            clues: List of clue dictionaries (may contain sub-clues)
            
        Returns:
            String containing all clue-related facts
        """
        facts = []
        
        for clue in clues:
            original_id = clue.get('id', self.clue_counter)
            description = clue.get('description', '')
            
            # Check if this clue uses the new multi-clue format
            if 'clues' in clue:
                # New multi-clue format
                sub_clues = clue['clues']
                
                # Add comment for the original clue
                if description:
                    facts.append(f"% {original_id}) {description}")
                
                # Process each sub-clue
                for sub_clue in sub_clues:
                    sub_id = sub_clue.get('sub_id', '')
                    sub_description = sub_clue.get('description', '')
                    
                    # Generate unique ASP clue ID (clingo-compatible, no underscores)
                    if sub_id:
                        # Convert "2a" to "c2a", "2_2a" to "c2c2a", etc.
                        sanitized_sub_id = sub_id.replace('_', 'c')
                        asp_clue_id = f"c{original_id}{sanitized_sub_id}"
                    else:
                        asp_clue_id = f"c{original_id}s{len(facts) + 1}"
                    
                    # Add comment for sub-clue
                    if sub_description:
                        facts.append(f"% Sub-clue {asp_clue_id}: {sub_description}")
                    
                    # Generate ASP facts for this sub-clue
                    self._generate_single_clue_facts(facts, asp_clue_id, sub_clue)
                
            else:
                # Old single-clue format - backward compatibility
                if description:
                    facts.append(f"% {original_id}) {description}")
                
                self._generate_single_clue_facts(facts, original_id, clue)
            
            self.clue_counter = max(self.clue_counter, original_id) + 1
        
        return '\n'.join(facts)
    
    def _generate_single_clue_facts(self, facts: List[str], clue_id: str, clue_data: Dict[str, Any]) -> None:
        """
        Generate ASP facts for a single clue (helper method).
        
        Args:
            facts: List to append facts to
            clue_id: ID for this clue
            clue_data: Dictionary containing clue data
        """
        clue_type = clue_data.get('type', 'same')
        
        # Generate clue/2 fact
        facts.append(f"clue({clue_id},{clue_type}).")
        
        # Generate object/4 facts
        objects = clue_data.get('objects', [])
        for obj in objects:
            position = obj.get('position', 1)
            attribute = self._sanitize_value(obj.get('attribute', ''))
            value = self._sanitize_value(obj.get('value', ''))
            
            if attribute and value:
                facts.append(f"object({clue_id},{position},{attribute},{value}).")
        
        # Generate target/2 fact if needed
        target_attr = clue_data.get('target_attribute')
        if target_attr and clue_type in ['less', 'next']:
            sanitized_target = self._sanitize_value(target_attr)
            facts.append(f"target({clue_id},{sanitized_target}).")
    
    def _generate_constraint_rules(self, constraint_rules: Dict[str, Any]) -> str:
        """
        Generate ASP constraint rules for placeholders.
        
        Args:
            constraint_rules: Dictionary with constraints and value_definitions
            
        Returns:
            String containing constraint rules and value definitions
        """
        if not constraint_rules:
            return ""
        
        rules = []
        
        # Add comment header
        rules.append("% Constraint rules for placeholders")
        
        # Generate value definitions for unknowns
        value_definitions = constraint_rules.get('value_definitions', [])
        if value_definitions:
            rules.append("")
            rules.append("% Value definitions for unknowns")
            for defn in value_definitions:
                placeholder = defn.get('placeholder', '')
                attribute = defn.get('attribute', '')
                description = defn.get('description', '')
                possible_values = defn.get('possible_values', [])
                
                if description:
                    rules.append(f"% {description}")
                
                # Generate input facts for possible values
                for value in possible_values:
                    sanitized_value = self._sanitize_value(value)
                    # Find next available number for this attribute
                    rules.append(f"% input({attribute},N,{sanitized_value}) :- ... (define N)")
        
        # Generate constraint rules for derived values
        constraints = constraint_rules.get('constraints', [])
        if constraints:
            rules.append("")
            rules.append("% Constraint rules for derived values")
            for constraint in constraints:
                placeholder = constraint.get('placeholder', '')
                description = constraint.get('description', '')
                asp_rule = constraint.get('asp_rule', '')
                
                if description:
                    rules.append(f"% {description}")
                
                if asp_rule:
                    rules.append(f"{asp_rule}")
        
        return '\n'.join(rules)
    
    def _add_comments(self, structured_info: Dict[str, Any]) -> str:
        """
        Generate helpful comments for the ASP file.
        
        Args:
            structured_info: Structured puzzle information
            
        Returns:
            String containing comments
        """
        comments = []
        comments.append("% Logic puzzle generated by LLMCL")
        comments.append("% Auto-generated from natural language description")
        comments.append("")
        
        # Add clue descriptions as comments
        clues = structured_info.get('clues', [])
        if clues:
            comments.append("% Clues:")
            for clue in clues:
                clue_id = clue.get('id', 'N/A')
                description = clue.get('description', 'No description')
                comments.append(f"% {clue_id}) {description}")
            comments.append("")
        
        return '\n'.join(comments)
    
    def _validate_structure(self, structured_info: Dict[str, Any]) -> None:
        """
        Validate the structured information before generating ASP facts.
        
        Args:
            structured_info: Structured puzzle information
            
        Raises:
            ValueError: If the structure is invalid
        """
        if not isinstance(structured_info, dict):
            raise ValueError("Structured info must be a dictionary")
        
        # Check for required keys
        if 'attributes' not in structured_info:
            raise ValueError("Missing 'attributes' key in structured info")
        
        attributes = structured_info['attributes']
        if not isinstance(attributes, dict) or not attributes:
            raise ValueError("'attributes' must be a non-empty dictionary")
        
        # Validate that all attributes have values
        for attr_name, values in attributes.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Attribute '{attr_name}' must have a non-empty list of values")
        
        # Check for primary attribute
        primary_attr = structured_info.get('primary_attribute')
        if not primary_attr:
            # Try to guess primary attribute
            for attr_name in attributes.keys():
                if any(keyword in attr_name.lower() for keyword in ['house', 'position', 'place', 'order']):
                    structured_info['primary_attribute'] = attr_name
                    if self.verbose:
                        print(f"⚠️ Guessed primary attribute: {attr_name}")
                    break
            else:
                # Use first attribute as fallback
                structured_info['primary_attribute'] = list(attributes.keys())[0]
                if self.verbose:
                    print(f"⚠️ Using first attribute as primary: {structured_info['primary_attribute']}")
        
        # Validate clues
        clues = structured_info.get('clues', [])
        if not isinstance(clues, list):
            raise ValueError("'clues' must be a list")
        
        for i, clue in enumerate(clues):
            if not isinstance(clue, dict):
                raise ValueError(f"Clue {i} must be a dictionary")
            
            if 'type' not in clue:
                clue['type'] = 'same'  # Default type
            
            if clue['type'] not in ['same', 'diff', 'less', 'next']:
                if self.verbose:
                    print(f"⚠️ Unknown clue type '{clue['type']}' in clue {i}, defaulting to 'same'")
                clue['type'] = 'same'
    
    def generate_facts(self, structured_info: Dict[str, Any]) -> str:
        """
        Generate complete ASP facts from structured puzzle information.
        
        Args:
            structured_info: Dictionary containing structured puzzle information
            
        Returns:
            String containing all ASP facts
        """
        if self.verbose:
            print("🔄 Validating structured information...")
        
        self._validate_structure(structured_info)
        
        if self.verbose:
            print("✅ Structure validation passed")
            print("🔄 Generating ASP facts...")
        
        # Generate all components
        comments = self._add_comments(structured_info)
        input_facts = self._generate_input_facts(structured_info['attributes'])
        index_fact = self._generate_index_fact(structured_info['primary_attribute'])
        clue_facts = self._generate_clue_facts(structured_info.get('clues', []))
        constraint_rules = self._generate_constraint_rules(structured_info.get('constraint_rules', {}))
        
        # Combine all parts
        asp_facts = []
        
        if comments:
            asp_facts.append(comments)
        
        if input_facts:
            asp_facts.append(input_facts)
        
        asp_facts.append("")  # Empty line
        asp_facts.append(index_fact)
        asp_facts.append("")  # Empty line
        
        if clue_facts:
            asp_facts.append(clue_facts)
        
        if constraint_rules:
            asp_facts.append("")  # Empty line
            asp_facts.append(constraint_rules)
        
        result = '\n'.join(asp_facts)
        
        if self.verbose:
            print("✅ ASP facts generated successfully")
            fact_count = len([line for line in result.split('\n') if line.strip() and not line.strip().startswith('%')])
            print(f"  - Generated {fact_count} ASP facts")
        
        return result
    
    def generate_input_facts(self, inputs_info: Dict[str, Any]) -> str:
        """
        Generate ASP input facts from inputs information only.
        
        Args:
            inputs_info: Dictionary with attributes and primary_attribute
            
        Returns:
            String containing input facts and index fact
        """
        attributes = inputs_info.get('attributes', {})
        primary_attribute = inputs_info.get('primary_attribute', '')
        
        # Generate input facts
        input_facts = self._generate_input_facts(attributes)
        index_fact = self._generate_index_fact(primary_attribute)
        
        # Combine with comments
        asp_facts = []
        asp_facts.append("% Input facts (entities and attributes)")
        asp_facts.append(input_facts)
        asp_facts.append("")
        asp_facts.append("% Primary attribute for ordering")
        asp_facts.append(index_fact)
        
        return '\n'.join(asp_facts)
    
    def generate_clue_facts(self, clue_extraction: Dict[str, Any], clue_id: str) -> str:
        """
        Generate ASP facts for a single clue extraction.
        
        Args:
            clue_extraction: Dictionary containing clue extraction results
            clue_id: ID of the clue
            
        Returns:
            String containing ASP facts for this clue
        """
        facts = []
        
        # Add comment with original clue text if available
        clue_text = clue_extraction.get('description', '')
        if clue_text:
            facts.append(f"% Clue {clue_id}: {clue_text}")
        
        # Handle both multi-clue and single-clue formats
        if 'clues' in clue_extraction:
            # Multi-clue format
            sub_clues = clue_extraction['clues']
            
            for sub_clue in sub_clues:
                sub_id = sub_clue.get('sub_id', '')
                sub_description = sub_clue.get('description', '')
                
                # Generate unique ASP clue ID
                if sub_id:
                    sanitized_sub_id = sub_id.replace('_', 'c')
                    asp_clue_id = f"c{clue_id}{sanitized_sub_id}"
                else:
                    asp_clue_id = f"c{clue_id}s{len(facts) + 1}"
                
                # Add comment for sub-clue
                if sub_description:
                    facts.append(f"% Sub-clue {asp_clue_id}: {sub_description}")
                
                # Generate ASP facts for this sub-clue
                self._generate_single_clue_facts(facts, asp_clue_id, sub_clue)
                facts.append("")  # Empty line between sub-clues
        else:
            # Single-clue format - use the clue_id directly
            self._generate_single_clue_facts(facts, f"c{clue_id}", clue_extraction)
        
        return '\n'.join(facts)
    
    def validate_asp_syntax(self, asp_facts: str) -> bool:
        """
        Basic validation of ASP syntax.
        
        Args:
            asp_facts: String containing ASP facts
            
        Returns:
            True if syntax appears valid, False otherwise
        """
        lines = asp_facts.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
            
            # Check if line ends with period
            if not line.endswith('.'):
                if self.verbose:
                    print(f"⚠️ Line {line_num} doesn't end with period: {line}")
                return False
            
            # Check for basic ASP fact structure
            if not re.match(r'^[a-z][a-zA-Z0-9_]*\([^)]*\)\.$', line):
                if self.verbose:
                    print(f"⚠️ Line {line_num} has invalid ASP syntax: {line}")
                return False
        
        return True 