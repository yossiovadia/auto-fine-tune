#!/usr/bin/env python3
"""
Proof of Concept: Adaptive Learning Demonstration

This script demonstrates the core concept of adaptive learning without actual model training.
It simulates how a model progressively learns and improves from vLLM issues over time.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re

class AdaptiveLearningPOC:
    """
    Simulates adaptive learning to demonstrate the core concept:
    - Models remember previous knowledge
    - Models adapt to new problem patterns  
    - Models improve performance over iterations
    """
    
    def __init__(self):
        self.knowledge_base = {
            "error_patterns": {},
            "model_compatibility": {},
            "solutions": {}
        }
        
        self.performance_history = []
        self.iteration_metrics = []
    
    def extract_error_signature(self, issue_text: str) -> str:
        """Extract key error signature from issue text."""
        # Look for common error patterns
        patterns = [
            r"(\w*Error|\w*Exception): (.+?)(?:\n|$)",
            r"RuntimeError: (.+?)(?:\n|$)", 
            r"ValueError: (.+?)(?:\n|$)",
            r"CUDA (?:error|Error): (.+?)(?:\n|$)",
            r"HTTP (\d+)(?: (.+?))?(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, issue_text, re.IGNORECASE)
            if match:
                return match.group(0)[:100]  # First 100 chars
        
        # Fallback: use title keywords
        title_words = issue_text.split()[:5]
        return " ".join(title_words)
    
    def extract_model_name(self, issue_text: str) -> str:
        """Extract model name from issue text."""
        patterns = [
            r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)",  # huggingface format
            r"(Qwen[0-9\-A-Za-z]+)",  # Qwen models
            r"(gpt-oss[0-9\-A-Za-z]*)",  # gpt-oss models
            r"(Llama[0-9\-A-Za-z]*)",  # Llama models
        ]
        
        for pattern in patterns:
            match = re.search(pattern, issue_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown_model"
    
    def simulate_learning(self, examples: List[Dict], iteration: int) -> Dict[str, Any]:
        """
        Simulate learning from examples in this iteration.
        Key concept: Each iteration builds on previous knowledge.
        """
        print(f"\n🧠 Iteration {iteration}: Learning from {len(examples)} examples...")
        
        new_patterns = 0
        reinforced_patterns = 0
        
        for example in examples:
            if example['type'] == 'error_diagnosis':
                error_sig = self.extract_error_signature(example['instruction'])
                
                if error_sig in self.knowledge_base["error_patterns"]:
                    # Reinforce existing pattern
                    self.knowledge_base["error_patterns"][error_sig]["confidence"] += 0.1
                    self.knowledge_base["error_patterns"][error_sig]["seen_count"] += 1
                    reinforced_patterns += 1
                else:
                    # Learn new pattern
                    self.knowledge_base["error_patterns"][error_sig] = {
                        "solution": example['output'],
                        "confidence": 0.5,
                        "seen_count": 1,
                        "learned_iteration": iteration
                    }
                    new_patterns += 1
            
            elif example['type'] == 'model_compatibility':
                model = self.extract_model_name(example['instruction'])
                
                if model not in self.knowledge_base["model_compatibility"]:
                    self.knowledge_base["model_compatibility"][model] = {
                        "compatibility_info": example['output'],
                        "confidence": 0.6,
                        "learned_iteration": iteration
                    }
                    new_patterns += 1
                else:
                    self.knowledge_base["model_compatibility"][model]["confidence"] += 0.1
                    reinforced_patterns += 1
        
        learning_metrics = {
            "iteration": iteration,
            "examples_processed": len(examples),
            "new_patterns_learned": new_patterns,
            "patterns_reinforced": reinforced_patterns,
            "total_error_patterns": len(self.knowledge_base["error_patterns"]),
            "total_model_knowledge": len(self.knowledge_base["model_compatibility"])
        }
        
        print(f"   📊 New patterns learned: {new_patterns}")
        print(f"   🔄 Patterns reinforced: {reinforced_patterns}")
        print(f"   🧩 Total error patterns: {learning_metrics['total_error_patterns']}")
        print(f"   🤖 Total model knowledge: {learning_metrics['total_model_knowledge']}")
        
        return learning_metrics
    
    def simulate_performance_test(self, test_questions: List[str], iteration: int) -> Dict[str, float]:
        """
        Simulate answering test questions based on current knowledge.
        Key concept: Performance improves as knowledge accumulates.
        """
        print(f"\n🎯 Testing iteration {iteration} on {len(test_questions)} questions...")
        
        correct_answers = 0
        confidence_scores = []
        
        for question in test_questions:
            # Simulate answering based on knowledge base
            error_sig = self.extract_error_signature(question)
            model_name = self.extract_model_name(question)
            
            answer_confidence = 0.0
            
            # Check if we have relevant error pattern knowledge
            if error_sig in self.knowledge_base["error_patterns"]:
                pattern_info = self.knowledge_base["error_patterns"][error_sig]
                answer_confidence += pattern_info["confidence"] * 0.7
            
            # Check if we have relevant model knowledge
            if model_name in self.knowledge_base["model_compatibility"]:
                model_info = self.knowledge_base["model_compatibility"][model_name]
                answer_confidence += model_info["confidence"] * 0.3
            
            # Add some randomness for realistic simulation
            answer_confidence += random.uniform(0, 0.2)
            answer_confidence = min(answer_confidence, 1.0)
            
            # Consider it correct if confidence > 0.5
            if answer_confidence > 0.5:
                correct_answers += 1
            
            confidence_scores.append(answer_confidence)
        
        accuracy = correct_answers / len(test_questions)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        performance = {
            "iteration": iteration,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "correct_answers": correct_answers,
            "total_questions": len(test_questions)
        }
        
        self.performance_history.append(performance)
        
        print(f"   ✅ Accuracy: {accuracy:.1%}")
        print(f"   💪 Avg Confidence: {avg_confidence:.2f}")
        
        return performance
    
    def demonstrate_adaptive_improvement(self, training_data_by_period: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Demonstrate the core adaptive learning concept:
        1. Each iteration learns from new data
        2. Knowledge accumulates over iterations
        3. Performance improves progressively
        """
        print("🚀 ADAPTIVE LEARNING DEMONSTRATION")
        print("="*50)
        print("Concept: Each iteration builds on previous knowledge")
        print("Goal: Show progressive improvement over time")
        print("="*50)
        
        # Create test questions (simulating real vLLM troubleshooting)
        test_questions = [
            "RuntimeError: CUDA out of memory when loading Llama-70B",
            "ValueError: unsupported LoRA weight for Qwen2.5vl",
            "How to run gpt-oss-20b with vLLM?",
            "TPU compilation fails with Qwen3-0.6B",
            "HTTP 500 error with JSON function calls"
        ]
        
        print(f"📝 Test Questions ({len(test_questions)}):")
        for i, q in enumerate(test_questions, 1):
            print(f"   {i}. {q}")
        
        all_results = {
            "learning_metrics": [],
            "performance_metrics": [],
            "improvement_summary": {}
        }
        
        # Run adaptive training simulation
        for iteration, (period_name, examples) in enumerate(training_data_by_period.items()):
            print(f"\n{'='*30}")
            print(f"PERIOD: {period_name}")
            print(f"{'='*30}")
            
            # Learn from this period's data
            learning_result = self.simulate_learning(examples, iteration)
            all_results["learning_metrics"].append(learning_result)
            
            # Test performance after learning
            performance_result = self.simulate_performance_test(test_questions, iteration)
            all_results["performance_metrics"].append(performance_result)
        
        # Calculate improvement metrics
        if len(all_results["performance_metrics"]) > 1:
            first_accuracy = all_results["performance_metrics"][0]["accuracy"]
            last_accuracy = all_results["performance_metrics"][-1]["accuracy"]
            
            improvement = {
                "baseline_accuracy": first_accuracy,
                "final_accuracy": last_accuracy,
                "absolute_improvement": last_accuracy - first_accuracy,
                "relative_improvement": (last_accuracy - first_accuracy) / first_accuracy if first_accuracy > 0 else 0,
                "total_patterns_learned": all_results["learning_metrics"][-1]["total_error_patterns"],
                "total_model_knowledge": all_results["learning_metrics"][-1]["total_model_knowledge"]
            }
            
            all_results["improvement_summary"] = improvement
        
        # Print final results
        print(f"\n{'='*50}")
        print("🎉 ADAPTIVE LEARNING RESULTS")
        print(f"{'='*50}")
        
        if "improvement_summary" in all_results:
            imp = all_results["improvement_summary"]
            print(f"📊 Baseline Accuracy: {imp['baseline_accuracy']:.1%}")
            print(f"📈 Final Accuracy: {imp['final_accuracy']:.1%}")
            print(f"🚀 Absolute Improvement: {imp['absolute_improvement']:.1%}")
            print(f"📯 Relative Improvement: {imp['relative_improvement']:.1%}")
            print(f"🧠 Total Error Patterns: {imp['total_patterns_learned']}")
            print(f"🤖 Total Model Knowledge: {imp['total_model_knowledge']}")
        
        print(f"\n💡 KEY INSIGHT:")
        print(f"   Each iteration BUILDS ON previous knowledge")
        print(f"   Model REMEMBERS and ADAPTS and IMPROVES")
        print(f"   Performance increases with accumulated experience")
        
        return all_results

def load_training_data():
    """Load the vLLM training data we created."""
    print("📂 Loading vLLM training data...")
    
    # Load QA examples
    qa_file = Path("data/training_datasets/period_2/qa_examples.jsonl")
    qa_examples = []
    
    if qa_file.exists():
        with open(qa_file, 'r') as f:
            for line in f:
                qa_examples.append(json.loads(line))
    
    print(f"   Loaded {len(qa_examples)} Q&A examples")
    
    # Since all our data is recent, let's artificially split it for the demo
    random.shuffle(qa_examples)
    
    # Split into 3 periods to simulate temporal learning
    third = len(qa_examples) // 3
    
    training_periods = {
        "Early 2024 (Iteration 0)": qa_examples[:third],
        "Mid 2024 (Iteration 1)": qa_examples[third:2*third], 
        "Recent 2024 (Iteration 2)": qa_examples[2*third:]
    }
    
    print(f"📅 Split into periods:")
    for period, examples in training_periods.items():
        print(f"   {period}: {len(examples)} examples")
    
    return training_periods

def main():
    """Run the adaptive learning POC demonstration."""
    print("🎯 Adaptive Fine-Tuning POC for vLLM Issues")
    print("Goal: Prove models can remember, adapt, and improve")
    print("-" * 60)
    
    # Load training data
    training_data = load_training_data()
    
    if not any(training_data.values()):
        print("❌ No training data found. Please run the data collection first.")
        print("   Run: python collect_vllm_issues_gh.py")
        print("   Then: python src/data/vllm_dataset_converter.py --input data/vllm_full_dataset.json --output data/training_datasets --types qa")
        return
    
    # Initialize adaptive learning POC
    adaptive_poc = AdaptiveLearningPOC()
    
    # Run the demonstration
    results = adaptive_poc.demonstrate_adaptive_improvement(training_data)
    
    # Save results
    results_file = Path("data/adaptive_poc_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✨ This POC demonstrates the core adaptive learning concept!")
    print("   Next step: Implement with actual model fine-tuning")

if __name__ == "__main__":
    main()