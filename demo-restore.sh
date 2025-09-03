#!/bin/bash
# Demo Restore Script - Restore backed up agents after demo completion

echo "🔄 Restoring demo environment..."

# Restore user-level agents
if [ -d ~/.claude/agents/backup ]; then
  if [ "$(ls -A ~/.claude/agents/backup 2>/dev/null)" ]; then
    echo "📥 Restoring user-level agents..."
    cp ~/.claude/agents/backup/* ~/.claude/agents/ 2>/dev/null || true
    echo "🗑️  Cleaning up user-level backup directory"
    rm -rf ~/.claude/agents/backup
  else
    echo "📂 No user-level agents to restore"
    rmdir ~/.claude/agents/backup 2>/dev/null || true
  fi
fi

# Restore project-level agents
if [ -d .claude/agents/backup ]; then
  if [ "$(ls -A .claude/agents/backup 2>/dev/null)" ]; then
    echo "📥 Restoring project-level agents..."
    cp .claude/agents/backup/* .claude/agents/ 2>/dev/null || true
    echo "🗑️  Cleaning up project-level backup directory"
    rm -rf .claude/agents/backup
  else
    echo "📂 No project-level agents to restore"
    rmdir .claude/agents/backup 2>/dev/null || true
  fi
fi

echo "✅ Demo environment restored - all original agents are back!"