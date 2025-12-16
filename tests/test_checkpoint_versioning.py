"""
Tests for checkpoint versioning and migration.

Tests checkpoint format versioning, migration utilities, and compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
from llm_common.checkpoint import (
    save_model,
    load_model,
    CHECKPOINT_VERSION,
    SUPPORTED_VERSIONS,
    migrate_checkpoint,
    get_checkpoint_version,
    validate_checkpoint,
)
from llm_common.migration import (
    migrate_checkpoint_file,
    batch_migrate_checkpoints,
    analyze_checkpoint_directory,
    create_migration_report,
)


def test_checkpoint_versioning():
    """Test checkpoint versioning."""
    print("\nTesting checkpoint versioning...")
    
    from gpt_kit import GPTModel, create_model_config
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Save with version
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, version=CHECKPOINT_VERSION)
        
        # Check version
        version = get_checkpoint_version(temp_path)
        assert version == CHECKPOINT_VERSION
        
        # Validate
        is_valid, error = validate_checkpoint(temp_path)
        assert is_valid, f"Checkpoint validation failed: {error}"
        
        print("  ✓ Checkpoint versioning passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_checkpoint_migration():
    """Test checkpoint migration."""
    print("\nTesting checkpoint migration...")
    
    from gpt_kit import GPTModel, create_model_config
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Create old-format checkpoint (version 0.9)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        old_path = f.name
    
    # Create old format checkpoint manually
    old_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 1,
        'loss': 0.5,
        # No version field (old format)
    }
    torch.save(old_checkpoint, old_path)
    
    try:
        # Check version
        version = get_checkpoint_version(old_path)
        assert version == '0.9'  # Should default to old version
        
        # Migrate
        migrated = migrate_checkpoint(old_checkpoint, '0.9', CHECKPOINT_VERSION)
        assert migrated['version'] == CHECKPOINT_VERSION
        
        # Save migrated checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            migrated_path = f.name
        torch.save(migrated, migrated_path)
        
        # Load migrated checkpoint
        loaded_model, loaded_config, _ = load_model(migrated_path, GPTModel, device="cpu")
        assert loaded_config == config
        
        print("  ✓ Checkpoint migration passed")
        return True
    finally:
        for path in [old_path]:
            if os.path.exists(path):
                os.remove(path)
        if 'migrated_path' in locals() and os.path.exists(migrated_path):
            os.remove(migrated_path)


def test_auto_migration():
    """Test automatic migration during load."""
    print("\nTesting automatic migration...")
    
    from gpt_kit import GPTModel, create_model_config
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Create old-format checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        old_path = f.name
    
    old_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    torch.save(old_checkpoint, old_path)
    
    try:
        # Load with auto-migration
        loaded_model, loaded_config, _ = load_model(
            old_path, GPTModel, device="cpu", auto_migrate=True
        )
        assert loaded_config == config
        
        print("  ✓ Automatic migration passed")
        return True
    finally:
        if os.path.exists(old_path):
            os.remove(old_path)


def test_checkpoint_validation():
    """Test checkpoint validation."""
    print("\nTesting checkpoint validation...")
    
    from gpt_kit import GPTModel, create_model_config
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Valid checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        valid_path = f.name
    
    try:
        save_model(model, valid_path, config)
        is_valid, error = validate_checkpoint(valid_path)
        assert is_valid, f"Valid checkpoint failed validation: {error}"
        
        # Invalid checkpoint (missing fields)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            invalid_path = f.name
        
        invalid_checkpoint = {'version': CHECKPOINT_VERSION}  # Missing required fields
        torch.save(invalid_checkpoint, invalid_path)
        
        is_valid, error = validate_checkpoint(invalid_path)
        assert not is_valid
        assert error is not None
        
        print("  ✓ Checkpoint validation passed")
        return True
    finally:
        for path in [valid_path, invalid_path]:
            if os.path.exists(path):
                os.remove(path)


def test_migration_utilities():
    """Test migration utility functions."""
    print("\nTesting migration utilities...")
    
    from gpt_kit import GPTModel, create_model_config
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Create old checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        old_path = f.name
    
    old_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    torch.save(old_checkpoint, old_path)
    
    try:
        # Test migrate_checkpoint_file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            migrated_path = f.name
        
        migrated_file = migrate_checkpoint_file(
            old_path,
            output_path=migrated_path,
            target_version=CHECKPOINT_VERSION,
            model_class=GPTModel
        )
        
        assert os.path.exists(migrated_file)
        version = get_checkpoint_version(migrated_file)
        assert version == CHECKPOINT_VERSION
        
        print("  ✓ Migration utilities passed")
        return True
    finally:
        for path in [old_path, migrated_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    print("="*60)
    print("CHECKPOINT VERSIONING TESTS")
    print("="*60)
    
    results = []
    results.append(("Checkpoint Versioning", test_checkpoint_versioning()))
    results.append(("Checkpoint Migration", test_checkpoint_migration()))
    results.append(("Automatic Migration", test_auto_migration()))
    results.append(("Checkpoint Validation", test_checkpoint_validation()))
    results.append(("Migration Utilities", test_migration_utilities()))
    
    print("\n" + "="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} versioning tests passed")
    
    if passed == total:
        print("✅ All checkpoint versioning tests passed!")
        sys.exit(0)
    else:
        print("❌ Some checkpoint versioning tests failed")
        sys.exit(1)

