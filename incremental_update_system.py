#!/usr/bin/env python3
"""
Incremental update system for live codebase changes.
Addresses the core problem: "Most tools don't remember, don't adapt, and don't improve with feedback"
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalUpdateSystem:
    """System for continuously updating the model with new vLLM changes."""
    
    def __init__(self, repo_path: str = "data/vllm", check_interval_hours: int = 24):
        self.repo_path = Path(repo_path)
        self.check_interval = timedelta(hours=check_interval_hours)
        self.last_update_file = Path("data/last_update.json")
        self.changes_log = Path("data/incremental_changes.log")
        
    def get_last_update_info(self) -> Dict:
        """Get information about the last update."""
        if self.last_update_file.exists():
            with open(self.last_update_file, 'r') as f:
                return json.load(f)
        return {
            "last_commit": None,
            "last_update_time": None,
            "changes_since_last": 0
        }
    
    def save_update_info(self, commit_hash: str, changes_count: int):
        """Save update information."""
        update_info = {
            "last_commit": commit_hash,
            "last_update_time": datetime.now().isoformat(),
            "changes_since_last": changes_count
        }
        self.last_update_file.parent.mkdir(parents=True, exist_ok=True)  # Create data directory if it doesn't exist
        with open(self.last_update_file, 'w') as f:
            json.dump(update_info, f, indent=2)
    
    def get_latest_commit(self) -> Optional[str]:
        """Get the latest commit hash from the vLLM repository."""
        try:
            result = subprocess.run([
                "git", "rev-parse", "HEAD"
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get latest commit: {e}")
            return None
    
    def get_recent_changes(self, since_commit: Optional[str] = None) -> List[Dict]:
        """Get recent changes in the repository."""
        try:
            # Get commits since last update
            if since_commit:
                cmd = ["git", "log", f"{since_commit}..HEAD", "--oneline", "--max-count=50"]
            else:
                cmd = ["git", "log", "--since=1.week.ago", "--oneline", "--max-count=50"]
            
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            
            changes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash, *message_parts = line.split()
                    message = ' '.join(message_parts)
                    changes.append({
                        "commit": commit_hash,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return changes
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get recent changes: {e}")
            return []
    
    def get_changed_files(self, since_commit: Optional[str] = None) -> List[str]:
        """Get list of files that changed."""
        try:
            if since_commit:
                cmd = ["git", "diff", "--name-only", f"{since_commit}..HEAD"]
            else:
                cmd = ["git", "diff", "--name-only", "HEAD~10..HEAD"]
            
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
            # Filter for Python files
            python_files = [f for f in files if f.endswith('.py')]
            return python_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get changed files: {e}")
            return []
    
    def analyze_change_impact(self, changes: List[Dict], changed_files: List[str]) -> Dict:
        """Analyze the impact of changes for training data generation."""
        impact = {
            "critical_changes": [],
            "new_features": [],
            "bug_fixes": [],
            "documentation": [],
            "tests": [],
            "should_retrain": False
        }
        
        # Analyze commit messages
        for change in changes:
            message = change["message"].lower()
            
            if any(word in message for word in ["fix", "bug", "error", "crash"]):
                impact["bug_fixes"].append(change)
            elif any(word in message for word in ["add", "new", "feature", "implement"]):
                impact["new_features"].append(change)
            elif any(word in message for word in ["doc", "readme", "comment"]):
                impact["documentation"].append(change)
            elif any(word in message for word in ["test", "unittest"]):
                impact["tests"].append(change)
        
        # Analyze changed files
        critical_files = [
            "vllm/engine/", "vllm/model_executor/", "vllm/core/",
            "vllm/worker/", "vllm/attention/"
        ]
        
        for file_path in changed_files:
            if any(critical in file_path for critical in critical_files):
                impact["critical_changes"].append(file_path)
        
        # Determine if retraining is needed
        impact["should_retrain"] = (
            len(impact["critical_changes"]) > 0 or
            len(impact["bug_fixes"]) > 3 or
            len(impact["new_features"]) > 1
        )
        
        return impact
    
    def generate_incremental_training_data(self, changes: List[Dict], changed_files: List[str]) -> List[Dict]:
        """Generate new training examples from recent changes."""
        training_examples = []
        
        for change in changes:
            # Create Q&A about the change
            if "fix" in change["message"].lower():
                training_examples.append({
                    "instruction": f"What was fixed in vLLM commit {change['commit'][:8]}?",
                    "input": f"Commit: {change['commit']}",
                    "output": f"This commit fixed: {change['message']}",
                    "type": "version_history",
                    "metadata": {
                        "source": "incremental_update",
                        "commit": change["commit"],
                        "timestamp": change["timestamp"]
                    }
                })
            
            elif "add" in change["message"].lower() or "new" in change["message"].lower():
                training_examples.append({
                    "instruction": f"What new feature was added in vLLM?",
                    "input": f"Recent change: {change['message']}",
                    "output": f"vLLM recently added: {change['message']}. This enhances vLLM's capabilities.",
                    "type": "feature_update",
                    "metadata": {
                        "source": "incremental_update",
                        "commit": change["commit"],
                        "timestamp": change["timestamp"]
                    }
                })
        
        # Create questions about changed files
        for file_path in changed_files[:10]:  # Limit to first 10
            training_examples.append({
                "instruction": f"What recent changes were made to {file_path} in vLLM?",
                "input": f"File: {file_path}",
                "output": f"The file {file_path} was recently updated in vLLM. Check the latest commits for specific changes to this component.",
                "type": "file_update",
                "metadata": {
                    "source": "incremental_update",
                    "file": file_path,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return training_examples
    
    def check_for_updates(self) -> Dict:
        """Check for updates and determine if retraining is needed."""
        logger.info("🔍 Checking for vLLM repository updates...")
        
        # Pull latest changes
        try:
            subprocess.run(["git", "pull"], cwd=self.repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("Git pull failed, continuing with local state")
        
        # Get current state
        current_commit = self.get_latest_commit()
        last_update_info = self.get_last_update_info()
        
        if not current_commit:
            return {"error": "Could not get current commit"}
        
        # Check if there are new changes
        changes = self.get_recent_changes(last_update_info.get("last_commit"))
        changed_files = self.get_changed_files(last_update_info.get("last_commit"))
        
        if not changes:
            logger.info("✅ No new changes since last update")
            return {
                "status": "no_changes",
                "current_commit": current_commit,
                "changes_count": 0
            }
        
        # Analyze impact
        impact = self.analyze_change_impact(changes, changed_files)
        
        # Generate new training data
        new_training_data = self.generate_incremental_training_data(changes, changed_files)
        
        # Save incremental data
        if new_training_data:
            incremental_file = Path(f"data/incremental_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            incremental_file.parent.mkdir(parents=True, exist_ok=True)  # Create data directory if it doesn't exist
            with open(incremental_file, 'w') as f:
                for example in new_training_data:
                    f.write(json.dumps(example) + '\n')
            logger.info(f"💾 Saved {len(new_training_data)} incremental examples to {incremental_file}")
        
        # Log changes
        with open(self.changes_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {len(changes)} changes, {len(changed_files)} files\n")
        
        # Update state
        self.save_update_info(current_commit, len(changes))
        
        logger.info(f"📊 Found {len(changes)} changes, {len(changed_files)} files changed")
        logger.info(f"🤖 Should retrain: {impact['should_retrain']}")
        
        return {
            "status": "changes_found",
            "current_commit": current_commit,
            "changes_count": len(changes),
            "files_changed": len(changed_files),
            "impact": impact,
            "training_data_file": str(incremental_file) if new_training_data else None,
            "should_retrain": impact["should_retrain"]
        }
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring for changes."""
        logger.info(f"🔄 Starting continuous monitoring (checking every {self.check_interval.total_seconds()/3600:.1f} hours)")
        
        while True:
            try:
                result = self.check_for_updates()
                
                if result.get("should_retrain"):
                    logger.info("🚨 Significant changes detected - retraining recommended!")
                    # Here you could trigger automatic retraining
                    
                logger.info(f"😴 Sleeping for {self.check_interval.total_seconds()/3600:.1f} hours...")
                time.sleep(self.check_interval.total_seconds())
                
            except KeyboardInterrupt:
                logger.info("👋 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main function for testing the incremental update system."""
    print("🔄 vLLM Incremental Update System")
    print("=" * 50)
    
    updater = IncrementalUpdateSystem()
    
    # Run a single check
    result = updater.check_for_updates()
    
    print(f"\n📊 Update Check Results:")
    print(f"Status: {result.get('status')}")
    print(f"Changes: {result.get('changes_count', 0)}")
    print(f"Files changed: {result.get('files_changed', 0)}")
    print(f"Should retrain: {result.get('should_retrain', False)}")
    
    if result.get("training_data_file"):
        print(f"💾 New training data: {result['training_data_file']}")
    
    # Show impact breakdown
    if result.get("impact"):
        impact = result["impact"]
        print(f"\n🎯 Change Impact:")
        print(f"  Bug fixes: {len(impact.get('bug_fixes', []))}")
        print(f"  New features: {len(impact.get('new_features', []))}")
        print(f"  Critical file changes: {len(impact.get('critical_changes', []))}")

if __name__ == "__main__":
    main()