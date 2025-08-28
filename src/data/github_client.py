#!/usr/bin/env python3
"""
GitHub API client for collecting vLLM project issues.
"""

import requests
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Generator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GitHubClient:
    """Client for fetching GitHub issues from vLLM repository."""
    
    def __init__(self, token: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token (optional, increases rate limits)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.base_url = "https://api.github.com"
        self.repo = "vllm-project/vllm"
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        
        # Set headers
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "vLLM-Adaptive-Training-POC"
        })
        
        if token:
            self.session.headers["Authorization"] = f"token {token}"
            logger.info("GitHub token provided - higher rate limits available")
        else:
            logger.warning("No GitHub token - limited to 60 requests/hour")
    
    def get_repository_info(self) -> Dict:
        """Get basic repository information."""
        url = f"{self.base_url}/repos/{self.repo}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def fetch_issues(
        self, 
        state: str = "all",  # "open", "closed", or "all"
        labels: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        per_page: int = 100,
        max_pages: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Fetch issues from the repository.
        
        Args:
            state: Issue state filter
            labels: Comma-separated label names
            since: ISO 8601 timestamp - only issues updated after this
            until: ISO 8601 timestamp - only issues updated before this
            per_page: Number of issues per page (max 100)
            max_pages: Maximum number of pages to fetch
            
        Yields:
            Individual issue dictionaries
        """
        url = f"{self.base_url}/repos/{self.repo}/issues"
        page = 1
        
        params = {
            "state": state,
            "per_page": min(per_page, 100),
            "sort": "created",
            "direction": "desc"
        }
        
        if labels:
            params["labels"] = labels
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        
        while True:
            params["page"] = page
            
            logger.info(f"Fetching page {page} of issues...")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            issues = response.json()
            
            if not issues:
                logger.info("No more issues found")
                break
            
            # Filter out pull requests (they have a 'pull_request' key)
            actual_issues = [issue for issue in issues if 'pull_request' not in issue]
            
            logger.info(f"Page {page}: {len(actual_issues)} issues (filtered {len(issues) - len(actual_issues)} PRs)")
            
            for issue in actual_issues:
                yield issue
            
            # Check if we should continue
            if max_pages and page >= max_pages:
                logger.info(f"Reached maximum pages limit: {max_pages}")
                break
            
            if len(issues) < per_page:
                logger.info("Reached last page")
                break
            
            page += 1
            time.sleep(self.rate_limit_delay)
    
    def fetch_issue_comments(self, issue_number: int) -> List[Dict]:
        """Fetch all comments for a specific issue."""
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/comments"
        
        comments = []
        page = 1
        
        while True:
            params = {"page": page, "per_page": 100}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            page_comments = response.json()
            
            if not page_comments:
                break
            
            comments.extend(page_comments)
            
            if len(page_comments) < 100:
                break
            
            page += 1
            time.sleep(self.rate_limit_delay)
        
        return comments
    
    def collect_comprehensive_dataset(
        self,
        output_file: str,
        max_issues: Optional[int] = None,
        include_comments: bool = True,
        filter_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Collect comprehensive dataset of vLLM issues.
        
        Args:
            output_file: Path to save the collected data
            max_issues: Maximum number of issues to collect
            include_comments: Whether to fetch comments for each issue
            filter_labels: Only include issues with these labels
            
        Returns:
            Summary statistics of collected data
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total_issues": 0,
            "closed_issues": 0,
            "open_issues": 0,
            "bug_reports": 0,
            "issues_with_comments": 0,
            "total_comments": 0,
            "date_range": {"earliest": None, "latest": None}
        }
        
        collected_data = []
        
        # Determine label filter
        label_filter = ",".join(filter_labels) if filter_labels else None
        
        logger.info(f"Starting data collection for vLLM issues...")
        logger.info(f"Max issues: {max_issues or 'unlimited'}")
        logger.info(f"Include comments: {include_comments}")
        logger.info(f"Label filter: {label_filter or 'none'}")
        
        try:
            for i, issue in enumerate(self.fetch_issues(
                state="all",
                labels=label_filter,
                per_page=100
            )):
                if max_issues and i >= max_issues:
                    break
                
                # Extract basic issue info
                issue_data = {
                    "number": issue["number"],
                    "title": issue["title"],
                    "body": issue["body"] or "",
                    "state": issue["state"],
                    "labels": [label["name"] for label in issue["labels"]],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "closed_at": issue["closed_at"],
                    "author": issue["user"]["login"],
                    "url": issue["html_url"],
                    "comments_count": issue["comments"],
                    "comments": []
                }
                
                # Update statistics
                stats["total_issues"] += 1
                
                if issue["state"] == "closed":
                    stats["closed_issues"] += 1
                else:
                    stats["open_issues"] += 1
                
                # Check for bug label
                if any("bug" in label.lower() for label in issue_data["labels"]):
                    stats["bug_reports"] += 1
                
                # Track date range
                created_date = issue["created_at"]
                if not stats["date_range"]["earliest"] or created_date < stats["date_range"]["earliest"]:
                    stats["date_range"]["earliest"] = created_date
                if not stats["date_range"]["latest"] or created_date > stats["date_range"]["latest"]:
                    stats["date_range"]["latest"] = created_date
                
                # Fetch comments if requested
                if include_comments and issue["comments"] > 0:
                    try:
                        comments = self.fetch_issue_comments(issue["number"])
                        issue_data["comments"] = comments
                        stats["issues_with_comments"] += 1
                        stats["total_comments"] += len(comments)
                        logger.info(f"Issue #{issue['number']}: {len(comments)} comments")
                    except Exception as e:
                        logger.warning(f"Failed to fetch comments for issue #{issue['number']}: {e}")
                
                collected_data.append(issue_data)
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    logger.info(f"Collected {i + 1} issues...")
        
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        except Exception as e:
            logger.error(f"Error during collection: {e}")
            raise
        
        # Save collected data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "repository": self.repo,
                    "collection_date": datetime.now(timezone.utc).isoformat(),
                    "statistics": stats
                },
                "issues": collected_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Collection complete! Saved {len(collected_data)} issues to {output_path}")
        logger.info(f"Statistics: {stats}")
        
        return stats


def main():
    """Example usage of the GitHub client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect vLLM GitHub issues")
    parser.add_argument("--output", "-o", default="data/vllm_issues.json",
                        help="Output file path")
    parser.add_argument("--max-issues", "-n", type=int,
                        help="Maximum number of issues to collect")
    parser.add_argument("--token", "-t",
                        help="GitHub personal access token")
    parser.add_argument("--include-comments", action="store_true",
                        help="Include issue comments")
    parser.add_argument("--bug-only", action="store_true",
                        help="Only collect bug reports")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create client
    client = GitHubClient(token=args.token)
    
    # Get repository info
    try:
        repo_info = client.get_repository_info()
        logger.info(f"Repository: {repo_info['full_name']}")
        logger.info(f"Description: {repo_info['description']}")
        logger.info(f"Stars: {repo_info['stargazers_count']}")
        logger.info(f"Open issues: {repo_info['open_issues_count']}")
    except Exception as e:
        logger.error(f"Failed to get repository info: {e}")
        return
    
    # Collect data
    filter_labels = ["bug"] if args.bug_only else None
    
    stats = client.collect_comprehensive_dataset(
        output_file=args.output,
        max_issues=args.max_issues,
        include_comments=args.include_comments,
        filter_labels=filter_labels
    )
    
    print(f"\n📊 Collection Summary:")
    print(f"Total issues: {stats['total_issues']}")
    print(f"Closed/Open: {stats['closed_issues']}/{stats['open_issues']}")
    print(f"Bug reports: {stats['bug_reports']}")
    if args.include_comments:
        print(f"Issues with comments: {stats['issues_with_comments']}")
        print(f"Total comments: {stats['total_comments']}")
    print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")


if __name__ == "__main__":
    main()