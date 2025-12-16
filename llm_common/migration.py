"""
Checkpoint migration utilities.

Provides utilities for migrating checkpoints between different format versions.
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .checkpoint import (
    CHECKPOINT_VERSION,
    SUPPORTED_VERSIONS,
    migrate_checkpoint,
    get_checkpoint_version,
    validate_checkpoint,
    save_model,
    load_model
)

logger = logging.getLogger(__name__)


def migrate_checkpoint_file(
    input_path: str,
    output_path: Optional[str] = None,
    target_version: str = CHECKPOINT_VERSION,
    model_class: Optional[Any] = None
) -> str:
    """
    Migrate a checkpoint file to a new version.
    
    Args:
        input_path: Path to input checkpoint file
        output_path: Path to save migrated checkpoint (None = overwrite input)
        target_version: Target version to migrate to
        model_class: Optional model class for validation
        
    Returns:
        Path to migrated checkpoint file
        
    Example:
        >>> migrated_path = migrate_checkpoint_file(
        ...     "old_checkpoint.pt",
        ...     "new_checkpoint.pt",
        ...     model_class=GPTModel
        ... )
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")
    
    # Get current version
    current_version = get_checkpoint_version(input_path)
    if current_version is None:
        raise ValueError(f"Could not determine checkpoint version: {input_path}")
    
    if current_version == target_version:
        logger.info(f"Checkpoint already at version {target_version}")
        return input_path
    
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Migrate
    migrated_checkpoint = migrate_checkpoint(checkpoint, current_version, target_version)
    
    # Validate if model class provided
    if model_class is not None:
        try:
            model = model_class(migrated_checkpoint['config'])
            model.load_state_dict(migrated_checkpoint['model_state_dict'])
            logger.info("Migrated checkpoint validated successfully")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
    
    # Save migrated checkpoint
    output = output_path if output_path else input_path
    torch.save(migrated_checkpoint, output)
    logger.info(f"Migrated checkpoint saved to {output}")
    
    return output


def batch_migrate_checkpoints(
    checkpoint_paths: List[str],
    target_version: str = CHECKPOINT_VERSION,
    output_dir: Optional[str] = None,
    backup: bool = True
) -> List[str]:
    """
    Migrate multiple checkpoint files.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        target_version: Target version to migrate to
        output_dir: Optional output directory (None = overwrite originals)
        backup: Whether to create backups before overwriting
        
    Returns:
        List of migrated checkpoint paths
        
    Example:
        >>> migrated = batch_migrate_checkpoints(
        ...     ["checkpoint1.pt", "checkpoint2.pt"],
        ...     output_dir="migrated/"
        ... )
    """
    migrated_paths = []
    
    for checkpoint_path in checkpoint_paths:
        try:
            # Create backup if requested
            if backup and output_dir is None:
                backup_path = f"{checkpoint_path}.backup"
                import shutil
                shutil.copy2(checkpoint_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Determine output path
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = Path(checkpoint_path).name
                output_path = os.path.join(output_dir, filename)
            else:
                output_path = None
            
            # Migrate
            migrated_path = migrate_checkpoint_file(
                checkpoint_path,
                output_path=output_path,
                target_version=target_version
            )
            migrated_paths.append(migrated_path)
            
        except Exception as e:
            logger.error(f"Failed to migrate {checkpoint_path}: {e}")
    
    return migrated_paths


def find_checkpoints(directory: str, pattern: str = "*.pt") -> List[str]:
    """
    Find all checkpoint files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "*.pt")
        
    Returns:
        List of checkpoint file paths
    """
    checkpoint_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                checkpoint_paths.append(os.path.join(root, file))
    
    return checkpoint_paths


def analyze_checkpoint_directory(
    directory: str
) -> Dict[str, Any]:
    """
    Analyze all checkpoints in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary with analysis results
    """
    checkpoints = find_checkpoints(directory)
    
    analysis = {
        'total_checkpoints': len(checkpoints),
        'versions': {},
        'invalid': [],
        'valid': []
    }
    
    for checkpoint_path in checkpoints:
        is_valid, error = validate_checkpoint(checkpoint_path)
        version = get_checkpoint_version(checkpoint_path)
        
        if is_valid:
            analysis['valid'].append(checkpoint_path)
            if version:
                analysis['versions'][version] = analysis['versions'].get(version, 0) + 1
        else:
            analysis['invalid'].append({
                'path': checkpoint_path,
                'error': error,
                'version': version
            })
    
    return analysis


def create_migration_report(
    directory: str,
    output_file: Optional[str] = None
) -> str:
    """
    Create a migration report for checkpoints in a directory.
    
    Args:
        directory: Directory to analyze
        output_file: Optional file to save report to
        
    Returns:
        Report string
    """
    analysis = analyze_checkpoint_directory(directory)
    
    report_lines = [
        "=" * 60,
        "Checkpoint Migration Report",
        "=" * 60,
        f"\nDirectory: {directory}",
        f"Total checkpoints: {analysis['total_checkpoints']}",
        f"\nVersion distribution:",
    ]
    
    for version, count in analysis['versions'].items():
        report_lines.append(f"  Version {version}: {count} checkpoints")
    
    if analysis['invalid']:
        report_lines.append(f"\nInvalid checkpoints: {len(analysis['invalid'])}")
        for invalid in analysis['invalid']:
            report_lines.append(f"  {invalid['path']}: {invalid['error']}")
    
    report_lines.append(f"\nValid checkpoints: {len(analysis['valid'])}")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_file}")
    
    return report

