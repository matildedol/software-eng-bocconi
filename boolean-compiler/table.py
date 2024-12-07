"""
codice finale, con parser di assignemnt_new, preso da table e table3 ma aggiunti gli errori e cambiato altro
"""

    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. Preliminaries and functions definitions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import re
import sys

## 0.1 Cleaning the input file

def clean_input_file(input_file):
    with open(input_file, 'r') as infile:
        content=infile.read()
    pattern = pattern = r'(?<!\n)(?=\b(var|show|show_ones)\b|\s*\b\w+\b\s*=)'
    #r'(?<!\n)(?=\b(var|show|show_ones)\b|\s*\S+\s*=)'
    cleaned_content = re.sub(pattern, '\n', content)
    lines = cleaned_content.splitlines()

    cleaned_lines = []
    current_line = ''

    for line in lines:
        line = line.split('#', 1)[0].strip()  # Remove comments and strip
        if not line:  # If the line is empty or a comment, skip it
            continue
        elif line.startswith(('var', 'show')) or ('=' in line):  # Start of a new instruction
            if current_line:  # If there's an existing line, add it to cleaned_lines
                cleaned_lines.append(current_line.strip())
                current_line = ''
            current_line = line  # Start a new current_line
        else:
            current_line += ' ' + line  # Continue the current instruction

    if current_line:
        cleaned_lines.append(current_line.strip())  # Add the last instruction

    return cleaned_lines


## 0.2 Classes for boolean expression evaluation (using AST)

class AST_node:
    def evaluate(self, values, memo):
        raise NotImplementedError

class var_node(AST_node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, values, memo, assignments):
        if isinstance(self.value, str):
            if self.value in memo:
                return memo[self.value]
            if self.value in values:
                result = values[self.value]
            elif self.value in assignments:
                result = assignments[self.value].evaluate(values, memo, assignments)
            else:
                result = 0  # Default to 0 if not found
            memo[self.value] = result
            return result
        else:
            return self.value

class not_node(AST_node):
    def __init__(self, operand):
        self.operand = operand

    def evaluate(self, values, memo, assignments):
        key = f'not_{id(self)}'
        if key in memo:
            return memo[key]
        result = not (self.operand.evaluate(values, memo, assignments)) and 1
        memo[key] = result
        return result

class conjdisj_node(AST_node):
    def __init__(self, operator, children):
        self.operator = operator
        self.children = children

    def evaluate(self, values, memo, assignments):
        key = f'{self.operator}_{id(self)}'
        if key in memo:
            return memo[key]
        if self.operator == 'and':
            result = all(child.evaluate(values, memo, assignments) for child in self.children)
        elif self.operator == 'or':
            result = any(child.evaluate(values, memo, assignments) for child in self.children)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
        memo[key] = result
        return result
    
#questa parte praticamente sostituisce eval()

## 0.3 Boolean expression evaluator

class bool_evaluator:
    def __init__(self):
        self.variables = []
        self.assignments = {}
        self.eval_sequence = []

    def tokenize(self, expression):
        return re.findall(r'\bTrue\b|\bFalse\b|not|and|or|[a-zA-Z_]\w*|\(|\)', expression)

    def parse_expression(self, tokens):
        return self._parse_expr(tokens)

    def _parse_expr(self, tokens):
        def parse_primary():
            if not tokens:
                raise ValueError("Unexpected end of expression")
            token = tokens.pop(0)
            if token in ('True', 'False'):
                return var_node(token == 'True')
            elif token == '(':
                node = parse_expression()
                if not tokens or tokens.pop(0) != ')':
                    raise ValueError("Missing closing parenthesis")
                return node
            elif token == 'not':
                return not_node(parse_primary())
            else:
                return var_node(token)

        def parse_and():
            nodes = [parse_primary()]
            while tokens and tokens[0] == 'and':
                tokens.pop(0)
                nodes.append(parse_primary())
            return nodes[0] if len(nodes) == 1 else conjdisj_node('and', nodes)

        def parse_expression():
            nodes = [parse_and()]
            while tokens and tokens[0] == 'or':
                tokens.pop(0)
                nodes.append(parse_and())
            return nodes[0] if len(nodes) == 1 else conjdisj_node('or', nodes)

        return parse_expression()

    def evaluate_assignment(self, var_name, values, memo):
            if var_name not in self.assignments:
                return values.get(var_name, 0)
            return int(self.assignments[var_name].evaluate(values, memo, self.assignments))

    
    def is_conjunction(self, node):
        return isinstance(node, conjdisj_node) and node.operator == 'and'

    def create_truth_table(self, target_vars, show_ones_only=False):
        declared_vars = self.variables
        num_vars = len(declared_vars)
        num_combinations = 1 << num_vars

        header = declared_vars + target_vars
        table = [header]

        for i in range(num_combinations):
            values = {}
            memo = {}
            for j, var in enumerate(declared_vars):
                values[var] = (i >> (num_vars - j - 1)) & 1

            # Evaluate all assigned variables
            for var in self.eval_sequence:
                values[var] = self.evaluate_assignment(var, values, memo)

            row = [values[var] for var in header]
            if not show_ones_only or any(values[var] for var in target_vars):
                table.append(row)

        return table

    def print_truth_table(self, table):
        declared_vars = self.variables
        target_vars = [var for var in table[0] if var not in declared_vars]

        # Calculate the maximum width for each column
        max_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

        # Print header
        header_parts = [' '.join(var.rjust(max_widths[i]) for i, var in enumerate(declared_vars))]
        target_parts = [' '.join(var.rjust(max_widths[i + len(declared_vars)]) for i, var in enumerate(target_vars))]
        print("# " + header_parts[0] + "   " + target_parts[0])

        # Print rows
        for row in table[1:]:
            row_parts = [str(item).rjust(max_widths[i]) for i, item in enumerate(row)]
            declared_values = ' '.join(row_parts[:len(declared_vars)])
            target_values = ' '.join(row_parts[len(declared_vars):])
            print(f"  {declared_values}   {target_values}")

    def show(self, target_vars):
        table = self.create_truth_table(target_vars)
        self.print_truth_table(table)

    def show_ones(self, target_vars):
        table = self.create_truth_table(target_vars, show_ones_only=True)
        self.print_truth_table(table)


## 0.4 Recursive descent parser to check the BNF grammar for expressions

class expr_parser:
    def __init__(self, expression):
        self.expression = expression
        self.tokens = re.findall(r'[()]|\band\b|\bor\b|\bnot\b|\bTrue\b|\bFalse\b|\w+', expression)
        self.current_token_index = 0

    def current_token(self):
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]  
        else:
            return None

    def next_token(self):
        self.current_token_index += 1

    def parse(self):
        is_valid = self.parse_expr()
        return is_valid and self.current_token() is None

    # <expr> ::= <negation> | <conjunction> | <disjunction> | <paren-expr>
    def parse_expr(self):
        saved_index = self.current_token_index
        if self.parse_negation():
            return True
        self.current_token_index = saved_index

        if self.parse_conjunction():
            return True
        self.current_token_index = saved_index

        if self.parse_disjunction():
            return True
        self.current_token_index = saved_index

        if self.parse_paren_expr():
            return True
        self.current_token_index = saved_index

        return False

    # <negation> ::= "not" <paren-expr>
    def parse_negation(self):
        if self.current_token() == 'not':
            self.next_token()  # Consume 'not'
            return self.parse_paren_expr()
        else:
            return False

    # <conjunction> ::= <paren-expr> "and" <paren-expr> | <paren-expr> "and" <conjunction>
    def parse_conjunction(self):
        if not self.parse_paren_expr():   
            return False
        if self.current_token() == 'and':
            self.next_token()  
            if self.parse_paren_expr(): 
                while self.current_token() == 'and': 
                    self.next_token() 
                    if not self.parse_paren_expr():
                        return False
                return True
            elif self.parse_conjunction():  
                return True
            else:
                return False
        else:
            return False

    # <disjunction> ::= <paren-expr> "or" <paren-expr> | <paren-expr> "or" <disjunction>
    def parse_disjunction(self):
        if not self.parse_paren_expr():
            return False
        if self.current_token() == 'or':
            self.next_token() 
            if self.parse_paren_expr():
                while self.current_token() == 'or':
                    self.next_token() 
                    if not self.parse_paren_expr():
                        return False
                return True
            elif self.parse_disjunction():
                return True
            else:
                return False
        else:
            return False

    # <paren-expr> ::= <element> | "(" <expr> ")"
    def parse_paren_expr(self):
        if self.current_token() == '(':
            self.next_token() 
            if self.parse_expr():
                if self.current_token() == ')':
                    self.next_token() 
                    return True
                else:
                    return False  
            else:
                return False  
        else:
            return self.parse_element()

    # <element> ::= "True" | "False" | <identifier>
    def parse_element(self):
        if self.current_token() in ['True', 'False'] or (
            re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.current_token()) and 
            self.current_token() not in ['and', 'not', 'or']
        ):
            self.next_token()  
            return True
        else:
            return False
    
    def check_balanced_parentheses(self, expression):
        stack = []
        for char in expression:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0
        

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Opening and checking input file (in general)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.exit('Please implement this program using: python3 table.py input.txt')

    input_file = sys.argv[1]
    if not input_file.endswith('.txt'):
        sys.exit('Warning! Invalid input format.')

    lines = clean_input_file(input_file)

    for line in lines:
        if not (line.startswith(('var', 'show')) or ('=' in line and not line.startswith('=') and not line.endswith('='))):
            sys.exit(f'Warning! Invalid line:\n{line}\nYour input file should only contain lines declaring variables, assignments, or "show" instructions, and empty or comments lines.')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Collecting variables and checking variable declaration lines 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    evaluator = bool_evaluator()
    bool_operators = ['and', 'not', 'or', 'True', 'False', '(',')']

    for line in lines:
        line = line.strip()

        if line.startswith('var'):
            line= line[3:].strip()
            if not line.endswith(';'):
                sys.exit(f'Warning! Invalid variable declaration statement:\n{line}\nPlease end the variable declaration statement with a semicolon (;).')
            line = line.rstrip(';').strip()
            if '=' in line:
                sys.exit(f'Warning! Invalid variable declaration statement:\n{line}\nVariable declarations should not contain an assignment operator (=).')
            variables_temp = line.split()
            for var in variables_temp:
                if var in bool_operators:
                    sys.exit(f'Warning! Invalid variable identifier: "{var}".\nThe keywords "and", "not", "or", "True" and "False" cannot be used as identifiers.')
                elif not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', var):
                    sys.exit(f'Warning! Invalid variable identifier: "{var}".\nIdentifiers should only be composed by letters, digits and underscores, they should begin with a letter or an underscore, and they should be separated only by a blank space.')
                if var in evaluator.variables:
                    sys.exit(f'Warning! Variable "{var}" has already been declared.')
                evaluator.variables.append(var)
                if len(evaluator.variables) > 64:
                    sys.exit(f'Warning! This program supports at most 64 variables.')
                
                
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Collecting assignments and checking assignment lines (BNF grammar)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif '=' in line and not line.startswith('show'):
            line = line.strip()
            if not line.endswith(';'):
                sys.exit(f'Warning! Invalid assignment statement:\n{line}\nPlease end assignments with a semicolon (;).')
            line = line.rstrip(';').strip()
            if line.count('=') != 1:
                raise Exception(f'Warning! Invalid assignment statement:\n{line}\nPlease use the syntax "identifier = expression" for your assignments.')
            var_name, expr = map(str.strip, line.split('='))
            if var_name in evaluator.variables or var_name in evaluator.assignments:
                raise Exception(f'Warning! Identifier "{var_name}" is already declared.')
            elif var_name in bool_operators:
                sys.exit(f'Warning! Invalid variable identifier: "{var_name}".\nThe keywords "and", "not", "or", "True" and "False" cannot be used as identifiers.')
            elif not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', var_name):
                sys.exit(f'Warning! Invalid variable identifier: "{var_name}".\nIdentifiers should only be composed by letters, digits and underscores, they should begin with a letter or an underscore, and they should be separated only by a blank space.')
            expression_tokens = evaluator.tokenize(expr)
            for i in expression_tokens:
                i = i.strip()
                if i != '':
                    if i not in bool_operators and i not in evaluator.variables and i not in evaluator.assignments.keys():
                        sys.exit(f'Warning! Invalid token "{i}" in expression:\n{var_name} = {expr}\nPlease use only declared identifiers and the operators "not", "and", "or", "(", ")" in the expressions.')

                parser = expr_parser(expr) #check compliance with BNF grammar
            if not parser.parse():
                sys.exit(f'Warning! The following expression does not comply with the required grammar:\n{var_name} = {expr}')

            evaluator.assignments[var_name] = evaluator.parse_expression(expression_tokens) #qui secondo me ci andrebbe solo aggiunta l'expression parts[1] però lui ci mette parse  
            evaluator.eval_sequence.append(var_name)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Handling show instructions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif line.startswith('show'):
            line = line.strip()
            if not line.endswith(';'):
                sys.exit(f'Warning! Invalid "show" instruction:\n{line}\nPlease end "show" instructions with a semicolon (;).')
            line = line.rstrip(';').strip()
            if '=' in line:
                sys.exit(f'Warning! Invalid "show" instruction:\n{line}\n"show" instructions should not contain an assignment operator (=).')
            elif not (line.startswith('show ') or line.startswith('show_ones')):
                sys.exit(f'Warning! Invalid "show" instruction:\n{line}\nThe keyword should be either "show" or "show_ones".')
            
            elif line.startswith('show '):
                vars_to_show = line[4:].strip().split()
                for var in vars_to_show:
                    if var not in evaluator.assignments:
                        sys.exit(f'Warning! You are trying to evaluate "{var}" that has not been defined.')
                evaluator.show(vars_to_show)
            elif line.startswith('show_ones'):
                vars_to_show = line[9:].strip().split()
                for var in vars_to_show:
                    if var not in evaluator.assignments:
                        sys.exit(f'Warning! You are trying to evaluate "{var}" that has not been defined.')
                evaluator.show_ones(vars_to_show)
    
if __name__ == "__main__":
    main()







