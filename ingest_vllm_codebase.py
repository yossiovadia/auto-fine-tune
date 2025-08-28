#!/usr/bin/env python3
"""
Ingest vLLM codebase for code-aware adaptive learning.
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

class VLLMCodebaseProcessor:
    """Process vLLM codebase into training examples."""
    
    def __init__(self, output_dir: str = "data/codebase"):
        self.output_dir = Path(output_dir)
        self.vllm_dir = Path("data/vllm")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clone_vllm_repo(self):
        """Clone the vLLM repository."""
        print("📦 Cloning vLLM repository...")
        
        if self.vllm_dir.exists():
            print("♻️ vLLM repo already exists, pulling latest changes...")
            try:
                subprocess.run(["git", "pull"], cwd=self.vllm_dir, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("⚠️ Git pull failed, repo might be dirty")
        else:
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/vllm-project/vllm.git", 
                    str(self.vllm_dir)
                ], check=True, capture_output=True)
                print("✅ Successfully cloned vLLM repository")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to clone: {e}")
                return False
        
        return True
        
    def extract_python_files(self) -> List[Path]:
        """Find all Python files in the vLLM repository."""
        python_files = []
        
        # Focus on key directories
        key_dirs = [
            "vllm",  # Main package
            "examples",  # Usage examples
            "tests",  # Test patterns
        ]
        
        for dir_name in key_dirs:
            dir_path = self.vllm_dir / dir_name
            if dir_path.exists():
                python_files.extend(dir_path.rglob("*.py"))
        
        # Filter out __pycache__ and other irrelevant files
        python_files = [f for f in python_files if "__pycache__" not in str(f)]
        python_files = [f for f in python_files if ".git" not in str(f)]
        
        print(f"📁 Found {len(python_files)} Python files")
        return python_files
        
    def parse_python_file(self, file_path: Path) -> Optional[Dict]:
        """Parse a Python file and extract useful information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return None
                
            file_info = {
                "file_path": str(file_path.relative_to(self.vllm_dir)),
                "functions": [],
                "classes": [],
                "docstring": ast.get_docstring(tree),
                "imports": [],
                "error_handling": []
            }
            
            # Extract functions, classes, and other elements
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "line_number": node.lineno
                    }
                    file_info["functions"].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "line_number": node.lineno,
                        "methods": []
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append({
                                "name": item.name,
                                "docstring": ast.get_docstring(item)
                            })
                    
                    file_info["classes"].append(class_info)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info["imports"].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_info["imports"].append(node.module)
                        
                # Look for error handling patterns
                elif isinstance(node, ast.ExceptHandler):
                    if node.type and isinstance(node.type, ast.Name):
                        file_info["error_handling"].append(node.type.id)
                        
            return file_info
            
        except Exception as e:
            print(f"⚠️ Error parsing {file_path}: {e}")
            return None
            
    def generate_code_qa_examples(self, file_info: Dict) -> List[Dict]:
        """Generate Q&A examples from code information."""
        examples = []
        file_path = file_info["file_path"]
        
        # Module-level questions
        if file_info["docstring"]:
            examples.append({
                "instruction": f"What does the {file_path} module do in vLLM?",
                "input": f"File: {file_path}",
                "output": file_info["docstring"].strip(),
                "type": "code_documentation",
                "metadata": {
                    "source": "module_docstring",
                    "file": file_path
                }
            })
        
        # Function-level questions
        for func in file_info["functions"]:
            if func["docstring"] and len(func["docstring"]) > 20:
                examples.append({
                    "instruction": f"How does the {func['name']} function work in vLLM?",
                    "input": f"Function: {func['name']} in {file_path}",
                    "output": func["docstring"].strip(),
                    "type": "code_function",
                    "metadata": {
                        "source": "function_docstring", 
                        "file": file_path,
                        "function": func["name"],
                        "line": func["line_number"]
                    }
                })
                
            # Generate usage examples for key functions
            if any(keyword in func["name"].lower() for keyword in ["init", "create", "load", "run", "execute"]):
                examples.append({
                    "instruction": f"How to use vLLM's {func['name']} function?",
                    "input": f"Function: {func['name']}({', '.join(func['args'])})",
                    "output": f"The {func['name']} function in {file_path} takes parameters: {', '.join(func['args'])}. " + 
                             (func["docstring"] or "This function is part of vLLM's core functionality."),
                    "type": "code_usage",
                    "metadata": {
                        "source": "function_signature",
                        "file": file_path,
                        "function": func["name"]
                    }
                })
        
        # Class-level questions
        for cls in file_info["classes"]:
            if cls["docstring"] and len(cls["docstring"]) > 20:
                examples.append({
                    "instruction": f"What is the {cls['name']} class used for in vLLM?",
                    "input": f"Class: {cls['name']} in {file_path}",
                    "output": cls["docstring"].strip(),
                    "type": "code_class",
                    "metadata": {
                        "source": "class_docstring",
                        "file": file_path,
                        "class": cls["name"],
                        "line": cls["line_number"]
                    }
                })
        
        # Error handling questions
        if file_info["error_handling"]:
            error_types = list(set(file_info["error_handling"]))
            examples.append({
                "instruction": f"What errors are handled in {file_path}?",
                "input": f"File: {file_path}",
                "output": f"This vLLM module handles the following error types: {', '.join(error_types)}. " +
                         "These are common errors that can occur during vLLM operations.",
                "type": "code_error_handling",
                "metadata": {
                    "source": "error_analysis",
                    "file": file_path,
                    "error_types": error_types
                }
            })
            
        return examples
        
    def process_codebase(self):
        """Process the entire vLLM codebase."""
        if not self.clone_vllm_repo():
            return
            
        python_files = self.extract_python_files()
        all_examples = []
        processed_files = []
        
        print("🔍 Processing Python files...")
        for i, file_path in enumerate(python_files[:50]):  # Process first 50 files for now
            if i % 10 == 0:
                print(f"  Processed {i}/{len(python_files[:50])} files...")
                
            file_info = self.parse_python_file(file_path)
            if file_info:
                processed_files.append(file_info)
                examples = self.generate_code_qa_examples(file_info)
                all_examples.extend(examples)
        
        print(f"✅ Generated {len(all_examples)} code-based training examples")
        
        # Save processed data
        output_file = self.output_dir / "vllm_code_examples.jsonl"
        with open(output_file, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
                
        # Save file analysis
        analysis_file = self.output_dir / "vllm_file_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
            
        print(f"💾 Saved {len(all_examples)} examples to {output_file}")
        print(f"📊 Saved file analysis to {analysis_file}")
        
        return all_examples

def main():
    print("🚀 vLLM Codebase Ingestion for Adaptive Learning")
    print("=" * 60)
    
    processor = VLLMCodebaseProcessor()
    examples = processor.process_codebase()
    
    if examples:
        print(f"\n🎯 Code-aware training data ready!")
        print(f"📈 {len(examples)} new training examples from vLLM source code")
        print(f"🔄 This enables the model to understand vLLM implementation details")
        
        # Show sample examples
        print("\n📝 Sample code-based training examples:")
        for i, example in enumerate(examples[:3], 1):
            print(f"\n{i}. Type: {example['type']}")
            print(f"   Q: {example['instruction']}")
            print(f"   A: {example['output'][:100]}...")
    else:
        print("❌ Failed to process codebase")

if __name__ == "__main__":
    main()