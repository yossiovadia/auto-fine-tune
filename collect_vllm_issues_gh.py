#!/usr/bin/env python3
"""
Collect vLLM issues using GitHub CLI for better rate limits and easier data access.
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

def run_gh_command(cmd):
    """Run a gh command and return JSON result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def collect_all_vllm_issues(output_file="data/vllm_all_issues.json", max_issues=1000):
    """Collect all vLLM issues using GitHub CLI."""
    print(f"🚀 Collecting up to {max_issues} vLLM issues using GitHub CLI...")
    
    # Get issues in batches
    cmd = f"""gh issue list \\
        --repo vllm-project/vllm \\
        --state all \\
        --limit {max_issues} \\
        --json number,title,state,createdAt,closedAt,body,labels,comments"""
    
    print(f"Running: {cmd}")
    issues = run_gh_command(cmd)
    
    if not issues:
        print("❌ Failed to fetch issues")
        return None
    
    print(f"✅ Collected {len(issues)} issues")
    
    # Convert to our format and add metadata
    dataset = {
        "metadata": {
            "repository": "vllm-project/vllm",
            "collection_date": datetime.now().isoformat(),
            "collection_method": "github_cli",
            "total_issues": len(issues)
        },
        "issues": []
    }
    
    # Convert format
    for issue in issues:
        issue_data = {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"] or "",
            "state": issue["state"].lower(),
            "labels": [label["name"] for label in issue["labels"]],
            "created_at": issue["createdAt"],
            "updated_at": None,  # Not provided by gh issue list
            "closed_at": issue["closedAt"],
            "author": None,  # Not provided by gh issue list
            "url": f"https://github.com/vllm-project/vllm/issues/{issue['number']}",
            "comments_count": issue["comments"],
            "comments": []  # We'll fetch these separately if needed
        }
        dataset["issues"].append(issue_data)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {len(issues)} issues to {output_path}")
    
    # Print statistics
    closed_issues = sum(1 for issue in dataset["issues"] if issue["state"] == "closed")
    open_issues = len(dataset["issues"]) - closed_issues
    bug_issues = sum(1 for issue in dataset["issues"] if any("bug" in label.lower() for label in issue["labels"]))
    
    print(f"\n📊 Statistics:")
    print(f"   Total issues: {len(dataset['issues'])}")
    print(f"   Open: {open_issues}")
    print(f"   Closed: {closed_issues}")
    print(f"   Bug reports: {bug_issues}")
    
    return dataset

def get_issue_comments(issue_number):
    """Get comments for a specific issue."""
    cmd = f"gh issue view {issue_number} --repo vllm-project/vllm --json comments"
    result = run_gh_command(cmd)
    return result.get("comments", []) if result else []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect vLLM issues using GitHub CLI")
    parser.add_argument("--output", "-o", default="data/vllm_all_issues.json", help="Output file")
    parser.add_argument("--max-issues", "-n", type=int, default=1000, help="Maximum issues to collect")
    parser.add_argument("--include-comments", action="store_true", help="Include issue comments (slower)")
    
    args = parser.parse_args()
    
    # Check if gh is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ GitHub CLI (gh) not found. Please install it first.")
        print("   Visit: https://cli.github.com/")
        sys.exit(1)
    
    # Collect issues
    dataset = collect_all_vllm_issues(args.output, args.max_issues)
    
    if dataset and args.include_comments:
        print("\n🔍 Fetching comments for closed issues...")
        closed_issues = [issue for issue in dataset["issues"] if issue["state"] == "closed"]
        
        for i, issue in enumerate(closed_issues[:50]):  # Limit to first 50 for speed
            print(f"   Fetching comments for issue #{issue['number']} ({i+1}/50)")
            comments = get_issue_comments(issue["number"])
            issue["comments"] = comments
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/50 issues")
        
        # Save updated dataset
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Updated {args.output} with comments")
    
    print("\n🎉 Collection complete!")