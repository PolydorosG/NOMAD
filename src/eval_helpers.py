import re
from collections import defaultdict
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def normalize_name(name: str, normalize: bool = True) -> str:
    """
    Normalize UML element names if `normalize=True`.
    - lowercase
    - lemmatize (handles plurals)
    - strip whitespace
    """
    name = name.strip()
    if normalize:
        name = lemmatizer.lemmatize(name.lower())
    return name

def preprocess_plantuml(uml_code: str) -> str:
    """
    Preprocess PlantUML code:
    - Remove single-line comments starting with '
    - Remove multi-line comments between /' and '/
    - Remove @startuml and @enduml
    - Strip empty lines
    """
    # Keep only between @startuml ... @enduml (if present)
    start_idx = uml_code.lower().find("@startuml")
    end_idx = uml_code.lower().rfind("@enduml")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        uml_code = uml_code[start_idx:end_idx + len("@enduml")]

    # Remove multi-line comments /' ... '/
    uml_code = re.sub(r"/'[\s\S]*?'/", "", uml_code)

    # Remove single-line comments starting with '
    uml_code = re.sub(r"^\s*'.*$", "", uml_code, flags=re.MULTILINE)

    # Remove @startuml / @enduml
    uml_code = re.sub(r'@startuml|@enduml', '', uml_code, flags=re.IGNORECASE)

    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in uml_code.splitlines() if line.strip()]

    return "\n".join(lines)

def extract_class_name(segment):
    multiplicity_pattern = re.compile(r'^"?\s*(\d+\.\.\d+|\d+\.\.\*|\*|\d+)\s*"?$')
    # Find all quoted or unquoted words
    tokens = re.findall(r'"[^"]+"|\S+', segment)
    # Remove multiplicities
    cleaned_tokens = [
        token.strip('"') for token in tokens
        if not multiplicity_pattern.match(token.strip('"'))
    ]
    # Return the last valid class name from left side, or first from right
    return cleaned_tokens[-1] if cleaned_tokens else ''

def parse_relationships(relationships: str, normalize=True):
    """
    Parses PlantUML relationships, including association classes as dotted associations.
    Normalizes direction for directed relationships.
    Sorts undirected '--' relationships alphabetically.
    Returns list of tuples: (classA, classB, rel_type, assoc_class)
      - assoc_class is None for normal relationships
    """
    parsed = []

    # Canonical relationship mapping
    canonical_rel = {
        '--|>': ('--|>', True), '<|--': ('--|>', False),
        '-->': ('-->', True), '<--': ('-->', False),
        '*--': ('*--', True), '--*': ('*--', False),
        'o--': ('o--', True), '--o': ('o--', False),
        '--': ('--', True),
        '..': ('..', True),  # dotted associations (association classes)
    }

    for line in relationships.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove labels after ':'
        line = line.split(':', 1)[0].strip()

        # Check for simplified association class syntax: (A, B) .. C
        assoc_match = re.match(r'\(([^)]+)\)\s*\.\.\s*(\w+)', line)
        if assoc_match:
            endpoints = [e.strip() for e in assoc_match.group(1).split(',')]
            if len(endpoints) == 2:
                classA, classB = endpoints
                assoc_class = assoc_match.group(2)
                parsed.append((classA, classB, '..', assoc_class))
                continue

        # Otherwise, parse normal relationships
        for rel_variant, (canon_type, left_first) in canonical_rel.items():
            if rel_variant in line:
                left, right = line.split(rel_variant)
                left_name = normalize_name(extract_class_name(left), normalize)
                right_name = normalize_name(extract_class_name(right), normalize)

                if canon_type == '--':
                    left_name, right_name = sorted([left_name, right_name])
                elif not left_first:
                    left_name, right_name = right_name, left_name

                parsed.append((left_name, right_name, canon_type, None))
                break

    return parsed


def split_plantuml(uml_code: str):
    """
    Splits PlantUML code into classes and relationships.
    - Association class lines like (A, B) .. C are treated as relationships only.
    """
    class_lines = []
    relationship_lines = []

    rel_pattern = re.compile(r"--|o--|\*--|--\|>|-->|<--|<\|--")
    assoc_class_pattern = re.compile(r'\([^)]+\)\s*\.\.\s*\w+')

    for line in uml_code.splitlines():
        line = line.strip()
        if not line:
            continue

        # Regular class declarations
        if line.startswith("class ") or line.startswith("abstract ") or  line.startswith("enum ") or "{" in line:
            class_lines.append(line)
        # Association class line -> keep only in relationships
        elif assoc_class_pattern.match(line):
            relationship_lines.append(line)
        # Regular relationships
        elif rel_pattern.search(line):
            relationship_lines.append(line)
        # Lines inside a class block
        elif class_lines and line and not line.startswith("@"):
            class_lines.append(line)

    return "\n".join(class_lines), "\n".join(relationship_lines)


def extract_classes(class_str, normalize: bool = True):
    class_names = []
    for line in class_str.splitlines():
        match = re.match(r"(?:abstract\s+|enum\s+|class\s+)(\w+)", line)
        if match:
            class_names.append(normalize_name(match.group(1), normalize))
    return set(class_names)



def extract_attributes(class_str, normalize: bool = True):
    current_class = None
    is_enum = False
    class_attrs = defaultdict(set)

    for line in class_str.splitlines():
        line = line.strip()

        # Match class/interface/enum declaration
        class_match = re.match(r"(?:abstract\s+)?(enum\s+|class\s+)(\w+)", line)
        if class_match:
            current_class = normalize_name(class_match.group(2), normalize)
            is_enum = class_match.group(1) == "enum"
            continue

        if not current_class or not line or line.startswith("@") or line.startswith("}"):
            continue

        # For enums: every non-brace, non-annotation line is a value
        if is_enum:
            attr_name = line.split()[0]  # just take the word
        else:
            # Regular class attribute (e.g., id: String)
            if ":" in line:
                attr_name = line.split(":")[0].strip()
            else:
                attr_name = line.split()[0]

        # Clean modifiers (+, -, #, ~)
        if attr_name and attr_name[0] in ['+', '-', '#', '~']:
            attr_name = attr_name[1:].strip()

        if attr_name:
            attr_name = normalize_name(attr_name, normalize)
            class_attrs[current_class].add(attr_name)
    return dict(class_attrs)



def evaluate_elements(golden, generated):
    """
    Generic evaluation for sets (class names or attributes).
    Returns precision, recall, f1, and extras/missing.
    """
    tp = golden & generated
    fp = generated - golden  # extra
    fn = golden - generated  # missing

    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else 0.0
    recall    = len(tp) / (len(tp) + len(fn)) if tp or fn else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': sorted(tp),
        'false_positives': sorted(fp),
        'false_negatives': sorted(fn),
    }


def evaluate_relationships(golden_rels, generated_rels, strict=True):
    """
    Evaluate relationships and return extra/missing ones.
    """
    if strict:
        golden_set = set(golden_rels)
        generated_set = set(generated_rels)

        tp = golden_set & generated_set
        fp = generated_set - golden_set
        fn = golden_set - generated_set

    else:
        # Relaxed: ignore direction and type (but include assoc_class)
        def unordered_pair(rel):
            classA, classB, *_ = rel
            return tuple(sorted([classA, classB]))

        golden_counter = Counter(unordered_pair(r) for r in golden_rels)
        generated_counter = Counter(unordered_pair(r) for r in generated_rels)

        tp, fp, fn = set(), set(), set()
        for pair in golden_counter:
            common = min(golden_counter[pair], generated_counter.get(pair, 0))
            if common:
                tp.add(pair)
            if golden_counter[pair] > generated_counter.get(pair, 0):
                fn.add(pair)
        for pair in generated_counter:
            if generated_counter[pair] > golden_counter.get(pair, 0):
                fp.add(pair)

    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else 0.0
    recall    = len(tp) / (len(tp) + len(fn)) if tp or fn else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': sorted(tp),
        'false_positives': sorted(fp),
        'false_negatives': sorted(fn),
    }


def evaluate_attributes(golden_attrs_dict, gen_attrs_dict, common_classes):
    """
    Evaluate attributes only for the common classes.
    Ignores types – matches only attribute names.
    """
    golden_attrs = set()
    generated_attrs = set()

    for cls in common_classes:
        golden_attrs.update({(cls, name) for name in golden_attrs_dict.get(cls, set())})
        generated_attrs.update({(cls, name) for name in gen_attrs_dict.get(cls, set())})

    return evaluate_elements(golden_attrs, generated_attrs)


def evaluate_uml(golden_uml, generated_uml, normalize: bool = True):
    """
    Full evaluation pipeline.
    """
    golden_uml = preprocess_plantuml(golden_uml)
    generated_uml = preprocess_plantuml(generated_uml)

    # Split class and relationship sections
    golden_classes_str, golden_rels_str = split_plantuml(golden_uml)
    gen_classes_str, gen_rels_str = split_plantuml(generated_uml)

    # Class and attribute extraction
    golden_classes = extract_classes(golden_classes_str, normalize)
    generated_classes = extract_classes(gen_classes_str, normalize)

    golden_attrs_dict = extract_attributes(golden_classes_str, normalize)
    gen_attrs_dict = extract_attributes(gen_classes_str, normalize)

    # Relationship parsing
    golden_rels = parse_relationships(golden_rels_str, normalize)
    generated_rels = parse_relationships(gen_rels_str, normalize)

    # Compute class intersection
    common_classes = golden_classes & generated_classes

    return {
        'classes': evaluate_elements(golden_classes, generated_classes),
        'attributes': evaluate_attributes(golden_attrs_dict, gen_attrs_dict, common_classes),
        'relationships_strict': evaluate_relationships(golden_rels, generated_rels, strict=True),
        'relationships_relaxed': evaluate_relationships(golden_rels, generated_rels, strict=False),
    }

