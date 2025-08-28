#!/usr/bin/env python3
"""
Create enhanced dataset combining GitHub issues + vLLM codebase knowledge.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    if file_path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
    return examples

def enhance_issue_with_code_context(issue_example: Dict, code_examples: List[Dict]) -> Dict:
    """Enhance GitHub issue example with relevant code context."""
    
    # Look for relevant code examples based on keywords
    issue_text = issue_example['instruction'].lower()
    
    relevant_code = []
    
    # Match based on keywords
    for code_ex in code_examples:
        code_text = (code_ex['instruction'] + ' ' + code_ex['output']).lower()
        
        # Find overlapping concepts
        if any(keyword in code_text for keyword in ['error', 'exception', 'cuda', 'memory', 'model', 'config']):
            if any(word in issue_text for word in code_text.split()[:5]):  # First 5 words for relevance
                relevant_code.append(code_ex)
    
    # If we found relevant code, enhance the response
    if relevant_code:
        # Pick the most relevant one
        best_match = relevant_code[0]
        
        enhanced_output = issue_example['output']
        
        # Add code context to the response
        code_context = f"\n\nBased on vLLM's implementation in {best_match['metadata'].get('file', 'source code')}:\n{best_match['output'][:200]}..."
        
        enhanced_example = issue_example.copy()
        enhanced_example['output'] = enhanced_output + code_context
        enhanced_example['type'] = 'enhanced_' + enhanced_example['type']
        enhanced_example['metadata']['enhanced_with_code'] = True
        enhanced_example['metadata']['code_source'] = best_match['metadata']
        
        return enhanced_example
    
    return issue_example

def create_cross_reference_examples(issue_examples: List[Dict], code_examples: List[Dict]) -> List[Dict]:
    """Create new examples that cross-reference issues and code."""
    cross_ref_examples = []
    
    # Group code examples by type
    code_by_type = {}
    for code_ex in code_examples:
        code_type = code_ex['type']
        if code_type not in code_by_type:
            code_by_type[code_type] = []
        code_by_type[code_type].append(code_ex)
    
    # Create issue-to-code mapping examples
    for issue_ex in issue_examples[:20]:  # Limit to first 20 for now
        instruction = issue_ex['instruction']
        
        # Create "implementation" question
        if "how to fix" in instruction.lower():
            # Convert error question to implementation question
            impl_question = instruction.replace("How to fix this vLLM error:", "What vLLM code handles:")
            impl_question = impl_question.replace("?", " implementation?")
            
            # Find relevant code
            relevant_funcs = [ex for ex in code_by_type.get('code_function', []) 
                            if any(word in ex['output'].lower() for word in ['error', 'exception', 'handle'])]
            
            if relevant_funcs:
                func_ex = random.choice(relevant_funcs)
                cross_ref_examples.append({
                    "instruction": impl_question,
                    "input": f"Implementation context from {func_ex['metadata']['file']}",
                    "output": f"In vLLM's implementation, this is handled by the {func_ex['metadata']['function']} function:\n\n{func_ex['output']}",
                    "type": "issue_to_code_mapping",
                    "metadata": {
                        "source": "cross_reference",
                        "original_issue": issue_ex['metadata'],
                        "code_reference": func_ex['metadata']
                    }
                })
    
    # Create code-to-usage examples
    for code_ex in code_by_type.get('code_function', [])[:15]:
        if 'init' in code_ex['metadata']['function'].lower() or 'create' in code_ex['metadata']['function'].lower():
            cross_ref_examples.append({
                "instruction": f"When would I use {code_ex['metadata']['function']} in vLLM?",
                "input": f"Function: {code_ex['metadata']['function']}",
                "output": f"The {code_ex['metadata']['function']} function is used in vLLM for: {code_ex['output']}\n\nThis is commonly needed when setting up or configuring vLLM components.",
                "type": "code_usage_guidance",
                "metadata": {
                    "source": "code_to_usage",
                    "code_reference": code_ex['metadata']
                }
            })
    
    return cross_ref_examples

def main():
    print("🔗 Creating Enhanced Code-Aware Dataset")
    print("=" * 50)
    
    # Load existing datasets
    issue_file = Path("data/training_datasets/period_2/improved_qa_examples.jsonl")
    code_file = Path("data/codebase/vllm_code_examples.jsonl")
    
    issue_examples = load_jsonl(issue_file)
    code_examples = load_jsonl(code_file)
    
    print(f"📊 Loaded {len(issue_examples)} GitHub issue examples")
    print(f"💻 Loaded {len(code_examples)} code examples")
    
    # Create enhanced dataset
    enhanced_examples = []
    
    # 1. Include original improved issues
    enhanced_examples.extend(issue_examples)
    print(f"✅ Added {len(issue_examples)} original issue examples")
    
    # 2. Include selected code examples (most valuable ones)
    valuable_code = [ex for ex in code_examples if ex['type'] in ['code_function', 'code_class', 'code_error_handling']]
    enhanced_examples.extend(valuable_code[:100])  # Top 100 code examples
    print(f"✅ Added {len(valuable_code[:100])} valuable code examples")
    
    # 3. Enhance some issues with code context
    enhanced_issues = []
    for issue_ex in issue_examples[:50]:  # Enhance first 50 issues
        enhanced = enhance_issue_with_code_context(issue_ex, code_examples)
        if enhanced != issue_ex:  # Only add if actually enhanced
            enhanced_issues.append(enhanced)
    
    enhanced_examples.extend(enhanced_issues)
    print(f"✅ Added {len(enhanced_issues)} code-enhanced issue examples")
    
    # 4. Create cross-reference examples
    cross_ref = create_cross_reference_examples(issue_examples, code_examples)
    enhanced_examples.extend(cross_ref)
    print(f"✅ Added {len(cross_ref)} cross-reference examples")
    
    # Shuffle for better training
    random.shuffle(enhanced_examples)
    
    # Save enhanced dataset
    output_file = Path("data/training_datasets/period_2/code_aware_dataset.jsonl")
    with open(output_file, 'w') as f:
        for example in enhanced_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n🎯 Enhanced Dataset Created!")
    print(f"📈 Total examples: {len(enhanced_examples)}")
    print(f"💾 Saved to: {output_file}")
    
    # Show breakdown
    type_counts = {}
    for ex in enhanced_examples:
        ex_type = ex['type']
        type_counts[ex_type] = type_counts.get(ex_type, 0) + 1
    
    print(f"\n📊 Dataset Breakdown:")
    for ex_type, count in sorted(type_counts.items()):
        print(f"  {ex_type}: {count}")
    
    # Show sample enhanced examples
    print(f"\n📝 Sample Enhanced Examples:")
    enhanced_only = [ex for ex in enhanced_examples if 'enhanced_' in ex['type'] or 'mapping' in ex['type']]
    for i, ex in enumerate(enhanced_only[:2], 1):
        print(f"\n{i}. Type: {ex['type']}")
        print(f"   Q: {ex['instruction']}")
        print(f"   A: {ex['output'][:150]}...")

if __name__ == "__main__":
    main()