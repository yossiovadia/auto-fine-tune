#!/usr/bin/env python3
"""
Improve the vLLM dataset by extracting better quality Q&A pairs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def extract_meaningful_solutions(issue: Dict) -> List[str]:
    """Extract meaningful solutions from issue comments."""
    solutions = []
    
    # Look for solutions in comments
    for comment in issue.get('comments', []):
        comment_body = comment.get('body', '').strip()
        
        # Skip short comments
        if len(comment_body) < 50:
            continue
            
        # Look for solution indicators
        solution_indicators = [
            r'(?i)(fix|solution|solve|resolve)',
            r'(?i)(try|use|add|set)',
            r'(?i)(workaround|temporary fix)',
            r'(?i)(here\'s how|you can|you should)',
            r'(?i)(pip install|docker|command)',
            r'```'  # Code blocks often contain solutions
        ]
        
        has_solution_indicator = any(re.search(pattern, comment_body) for pattern in solution_indicators)
        
        if has_solution_indicator and len(comment_body) > 100:
            # Clean the solution
            solution = clean_text(comment_body)
            if len(solution) > 50:
                solutions.append(solution)
    
    # Also check the issue body for solutions (sometimes the issue itself contains the solution)
    issue_body = issue.get('body', '')
    if 'solution' in issue_body.lower() or 'fix' in issue_body.lower():
        if len(issue_body) > 100:
            solutions.append(clean_text(issue_body))
    
    return solutions

def clean_text(text: str) -> str:
    """Clean text for better quality."""
    # Remove GitHub-specific markdown
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\d+', '', text)  # Remove issue references
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove inline code formatting
    
    # Clean excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_better_qa_pairs(data: Dict) -> List[Dict]:
    """Extract better quality Q&A pairs from vLLM issues."""
    qa_pairs = []
    
    for issue in data.get('issues', []):
        title = issue.get('title', '').strip()
        if len(title) < 10:
            continue
            
        # Skip issues that are just announcements or discussions
        skip_keywords = ['announcement', 'discussion', 'question', 'rfc']
        if any(keyword in title.lower() for keyword in skip_keywords):
            continue
            
        # Extract meaningful solutions
        solutions = extract_meaningful_solutions(issue)
        
        if solutions:
            # Create questions from the title
            question = format_question(title)
            
            for solution in solutions:
                if len(solution) > 50:  # Ensure substantial content
                    qa_pairs.append({
                        'instruction': question,
                        'input': '',
                        'output': solution[:500],  # Limit length but keep substantial
                        'type': 'troubleshooting',
                        'metadata': {
                            'issue_number': issue.get('number'),
                            'quality_score': calculate_quality_score(question, solution)
                        }
                    })
    
    # Sort by quality score and return best ones
    qa_pairs.sort(key=lambda x: x['metadata']['quality_score'], reverse=True)
    return qa_pairs

def format_question(title: str) -> str:
    """Format issue title into a proper question."""
    title = title.strip()
    
    # If it's already a question, return as-is
    if title.endswith('?'):
        return title
    
    # Add question format based on content
    if any(word in title.lower() for word in ['error', 'exception', 'fail', 'crash']):
        return f"How to fix this vLLM error: {title}?"
    elif any(word in title.lower() for word in ['support', 'compatible', 'run', 'use']):
        return f"How to {title.lower()}?"
    elif any(word in title.lower() for word in ['performance', 'optimize', 'slow']):
        return f"How to optimize: {title}?"
    else:
        return f"How to resolve: {title}?"

def calculate_quality_score(question: str, answer: str) -> float:
    """Calculate quality score for Q&A pair."""
    score = 0.0
    
    # Length scores
    if 30 <= len(question) <= 200:
        score += 1.0
    if 100 <= len(answer) <= 500:
        score += 2.0
    
    # Content quality indicators
    quality_indicators = [
        r'(?i)(install|pip|docker|command)',
        r'(?i)(configuration|config|setting)',
        r'(?i)(fix|solution|resolve)',
        r'(?i)(cuda|gpu|memory)',
        r'(?i)(model|checkpoint|weight)',
        r'```',  # Code blocks
        r'(?i)(vllm|llm)',
    ]
    
    for pattern in quality_indicators:
        if re.search(pattern, answer):
            score += 0.5
    
    return score

def main():
    print("🔧 Improving vLLM dataset quality...")
    
    # Load raw data
    input_file = Path("data/vllm_full_dataset.json")
    if not input_file.exists():
        print("❌ vllm_full_dataset.json not found!")
        return
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Processing {len(data.get('issues', []))} issues...")
    
    # Extract better Q&A pairs
    qa_pairs = extract_better_qa_pairs(data)
    
    print(f"✅ Extracted {len(qa_pairs)} quality Q&A pairs")
    
    # Save improved dataset
    output_dir = Path("data/training_datasets/period_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "improved_qa_examples.jsonl"
    with open(output_file, 'w') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + '\n')
    
    print(f"💾 Saved to: {output_file}")
    
    # Show some examples
    print("\n📝 Sample improved Q&A pairs:")
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n{i}. Q: {qa['instruction']}")
        print(f"   A: {qa['output'][:100]}...")
        print(f"   Quality: {qa['metadata']['quality_score']:.1f}")

if __name__ == "__main__":
    main()