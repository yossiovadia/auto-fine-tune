#!/bin/bash
# Demo Reset Script - Backup existing agents and remove them for clean demo

echo "🔄 Resetting demo environment..."

# Create backup directories
mkdir -p ~/.claude/agents/backup
mkdir -p .claude/agents/backup

# Backup and remove user-level agents
for agent in git-agent; do
  if [ -f ~/.claude/agents/$agent.md ]; then
    echo "📦 Backing up user-level agent: $agent"
    cp ~/.claude/agents/$agent.md ~/.claude/agents/backup/
    rm ~/.claude/agents/$agent.md
    echo "🗑️  Removed ~/.claude/agents/$agent.md"
  fi
done

# Backup and remove project-level agents  
for agent in desktop-remote-executor; do
  if [ -f .claude/agents/$agent.md ]; then
    echo "📦 Backing up project-level agent: $agent"
    cp .claude/agents/$agent.md .claude/agents/backup/
    rm .claude/agents/$agent.md
    echo "🗑️  Removed .claude/agents/$agent.md"
  fi
done

echo "✅ Demo environment reset complete - ready for live agent creation!"