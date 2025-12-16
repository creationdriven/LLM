#!/usr/bin/env python3
"""
Example: Migrating checkpoints between versions.

This script demonstrates:
- Checking checkpoint versions
- Migrating checkpoints to new versions
- Batch migration
- Creating migration reports
"""

import os
from llm_common.checkpoint import (
    get_checkpoint_version,
    validate_checkpoint,
    migrate_checkpoint_file,
    CHECKPOINT_VERSION,
)
from llm_common.migration import (
    batch_migrate_checkpoints,
    analyze_checkpoint_directory,
    create_migration_report,
    find_checkpoints,
)

# Example checkpoint paths (create dummy ones for demo)
EXAMPLE_CHECKPOINTS = [
    "examples/gpt_model.pt",
    "examples/bert_classifier.pt",
    "examples/roberta_classifier.pt",
]


def check_checkpoint_version(path: str):
    """Check version of a checkpoint."""
    print(f"\nChecking: {path}")
    if not os.path.exists(path):
        print(f"  ⚠️  File not found (will be created when you run training examples)")
        return
    
    version = get_checkpoint_version(path)
    is_valid, error = validate_checkpoint(path)
    
    print(f"  Version: {version}")
    print(f"  Valid: {is_valid}")
    if error:
        print(f"  Error: {error}")


def migrate_single_checkpoint():
    """Demonstrate single checkpoint migration."""
    print("\n" + "="*60)
    print("Single Checkpoint Migration")
    print("="*60)
    
    checkpoint_path = "examples/gpt_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Run examples/train_gpt.py first to create a checkpoint")
        return
    
    # Check current version
    current_version = get_checkpoint_version(checkpoint_path)
    print(f"\nCurrent version: {current_version}")
    print(f"Target version: {CHECKPOINT_VERSION}")
    
    if current_version == CHECKPOINT_VERSION:
        print("✓ Checkpoint already at latest version")
        return
    
    # Migrate
    migrated_path = migrate_checkpoint_file(
        checkpoint_path,
        output_path="examples/gpt_model_migrated.pt",
        target_version=CHECKPOINT_VERSION
    )
    
    print(f"✓ Migrated checkpoint saved to: {migrated_path}")


def batch_migration_example():
    """Demonstrate batch migration."""
    print("\n" + "="*60)
    print("Batch Checkpoint Migration")
    print("="*60)
    
    # Find all checkpoints in examples directory
    checkpoints = find_checkpoints("examples")
    
    if not checkpoints:
        print("⚠️  No checkpoints found in examples/ directory")
        print("   Run training examples first to create checkpoints")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        version = get_checkpoint_version(cp)
        print(f"  {cp} (version {version})")
    
    # Analyze directory
    print("\nAnalyzing checkpoint directory...")
    analysis = analyze_checkpoint_directory("examples")
    
    print(f"\nAnalysis Results:")
    print(f"  Total checkpoints: {analysis['total_checkpoints']}")
    print(f"  Valid: {len(analysis['valid'])}")
    print(f"  Invalid: {len(analysis['invalid'])}")
    
    if analysis['versions']:
        print(f"\nVersion distribution:")
        for version, count in analysis['versions'].items():
            print(f"  Version {version}: {count} checkpoints")


def create_report_example():
    """Demonstrate migration report creation."""
    print("\n" + "="*60)
    print("Migration Report")
    print("="*60)
    
    report = create_migration_report("examples", output_file="examples/migration_report.txt")
    print(report)
    print("\n✓ Report saved to examples/migration_report.txt")


def main():
    print("="*60)
    print("Checkpoint Migration Examples")
    print("="*60)
    
    # Check versions of example checkpoints
    print("\n1. Checking Checkpoint Versions:")
    for checkpoint in EXAMPLE_CHECKPOINTS:
        check_checkpoint_version(checkpoint)
    
    # Single migration
    migrate_single_checkpoint()
    
    # Batch migration
    batch_migration_example()
    
    # Create report
    create_report_example()
    
    print("\n" + "="*60)
    print("✅ Migration examples complete!")
    print("="*60)
    print("\nNote: Run training examples first to create checkpoints for migration.")


if __name__ == "__main__":
    main()

