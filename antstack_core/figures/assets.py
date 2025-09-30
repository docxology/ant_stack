"""Asset management utilities for scientific publications.

Comprehensive asset organization and file management:
- Figure file organization and directory structure
- Asset copying and symlinking for builds  
- Path normalization and resolution
- Asset manifest generation and validation

Following .cursorrules principles:
- Relative paths in Markdown with absolute URIs for provenance
- Consistent asset organization across papers
- Automated file management and validation
- Clean separation of generated and static assets

References:
- Asset management best practices: https://doi.org/10.1371/journal.pcbi.1005510
- Reproducible research workflows: https://doi.org/10.1038/s41559-017-0160
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import json


class AssetManager:
    """Manager for publication assets and file organization."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.assets_dir = self.base_dir / "assets"
        self.figures_dir = self.assets_dir / "figures"
        self.mermaid_dir = self.assets_dir / "mermaid"
        self.tmp_dir = self.assets_dir / "tmp_images"
        
        # Asset tracking
        self.managed_assets: Set[Path] = set()
        self.asset_manifest: Dict[str, Dict] = {}
    
    def setup_directory_structure(self) -> None:
        """Create standard asset directory structure."""
        directories = [
            self.assets_dir,
            self.figures_dir,
            self.mermaid_dir,
            self.tmp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create .gitkeep files for empty directories
        for directory in directories:
            gitkeep = directory / ".gitkeep"
            if not any(directory.iterdir()) and not gitkeep.exists():
                gitkeep.touch()
    
    def register_asset(
        self,
        asset_path: Path,
        asset_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Register an asset and return its managed path.
        
        Args:
            asset_path: Path to the asset file
            asset_type: Type of asset ('figure', 'mermaid', 'data', etc.)
            metadata: Optional metadata dictionary
            
        Returns:
            Relative path suitable for use in Markdown
        """
        asset_path = Path(asset_path)
        
        # Determine target directory based on type
        if asset_type == 'mermaid':
            target_dir = self.mermaid_dir
        elif asset_type in ['figure', 'plot', 'diagram']:
            target_dir = self.figures_dir
        elif asset_type == 'tmp':
            target_dir = self.tmp_dir
        else:
            target_dir = self.assets_dir
        
        # Generate managed path
        managed_path = target_dir / asset_path.name
        
        # Copy or symlink the asset if needed
        if not managed_path.exists() or managed_path.stat().st_mtime < asset_path.stat().st_mtime:
            if managed_path.exists():
                managed_path.unlink()
            shutil.copy2(asset_path, managed_path)
        
        # Register in tracking
        self.managed_assets.add(managed_path)
        
        # Update manifest
        relative_path = os.path.relpath(managed_path, self.base_dir)
        self.asset_manifest[relative_path] = {
            "type": asset_type,
            "source": str(asset_path.absolute()),
            "size_bytes": managed_path.stat().st_size,
            "metadata": metadata or {}
        }
        
        return relative_path
    
    def organize_figure_assets(self, source_dir: Optional[Path] = None) -> List[str]:
        """Organize figure assets from source directory.
        
        Args:
            source_dir: Source directory to scan (defaults to base_dir)
            
        Returns:
            List of organized asset paths
        """
        if source_dir is None:
            source_dir = self.base_dir
        
        source_dir = Path(source_dir)
        organized_assets = []
        
        # Find image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.eps'}
        
        for ext in image_extensions:
            for image_file in source_dir.rglob(f'*{ext}'):
                # Skip already managed assets
                if image_file.is_relative_to(self.assets_dir):
                    continue
                
                # Determine asset type
                if 'mermaid' in str(image_file).lower():
                    asset_type = 'mermaid'
                elif any(keyword in str(image_file).lower() for keyword in ['plot', 'chart', 'graph']):
                    asset_type = 'plot'
                else:
                    asset_type = 'figure'
                
                relative_path = self.register_asset(image_file, asset_type)
                organized_assets.append(relative_path)
        
        return organized_assets
    
    def copy_figure_files(self, file_patterns: List[str]) -> Dict[str, str]:
        """Copy specific figure files based on patterns.
        
        Args:
            file_patterns: List of file patterns to match
            
        Returns:
            Dictionary mapping source paths to destination paths
        """
        copied_files = {}
        
        for pattern in file_patterns:
            # Handle glob patterns
            if '*' in pattern or '?' in pattern:
                matches = list(self.base_dir.rglob(pattern))
            else:
                # Handle specific files
                file_path = self.base_dir / pattern
                matches = [file_path] if file_path.exists() else []
            
            for match in matches:
                if match.is_file():
                    relative_path = self.register_asset(match, 'figure')
                    copied_files[str(match)] = relative_path
        
        return copied_files
    
    def generate_asset_manifest(self, output_file: Optional[Path] = None) -> Dict:
        """Generate comprehensive asset manifest.
        
        Args:
            output_file: Optional file to save manifest JSON
            
        Returns:
            Asset manifest dictionary
        """
        manifest = {
            "base_directory": str(self.base_dir.absolute()),
            "generated_at": str(os.path.getmtime(__file__)),  # Rough timestamp
            "asset_count": len(self.asset_manifest),
            "total_size_bytes": sum(
                asset["size_bytes"] for asset in self.asset_manifest.values()
            ),
            "assets": self.asset_manifest
        }
        
        if output_file:
            output_file = Path(output_file)
            with open(output_file, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        return manifest
    
    def validate_assets(self) -> Dict[str, List[str]]:
        """Validate all managed assets.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        for relative_path, asset_info in self.asset_manifest.items():
            full_path = self.base_dir / relative_path
            
            # Check if file exists
            if not full_path.exists():
                issues.append(f"Missing asset: {relative_path}")
                continue
            
            # Check file size consistency
            current_size = full_path.stat().st_size
            recorded_size = asset_info.get("size_bytes", 0)
            
            if current_size != recorded_size:
                warnings.append(
                    f"Size mismatch for {relative_path}: "
                    f"current={current_size}, recorded={recorded_size}"
                )
            
            # Check for zero-size files
            if current_size == 0:
                warnings.append(f"Zero-size asset: {relative_path}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "validated_assets": len(self.asset_manifest)
        }
    
    def cleanup_temporary_assets(self) -> int:
        """Clean up temporary assets and return count of removed files."""
        removed_count = 0
        
        # Clean tmp directory
        if self.tmp_dir.exists():
            for tmp_file in self.tmp_dir.iterdir():
                if tmp_file.is_file():
                    tmp_file.unlink()
                    removed_count += 1
        
        # Remove orphaned assets (not in manifest)
        for assets_subdir in [self.figures_dir, self.mermaid_dir]:
            if assets_subdir.exists():
                for asset_file in assets_subdir.iterdir():
                    if asset_file.is_file():
                        relative_path = os.path.relpath(asset_file, self.base_dir)
                        if relative_path not in self.asset_manifest:
                            asset_file.unlink()
                            removed_count += 1
        
        return removed_count


def organize_figure_assets(base_dir: Union[str, Path], patterns: Optional[List[str]] = None) -> Dict[str, str]:
    """Organize figure assets in a directory.
    
    Args:
        base_dir: Base directory for asset organization
        patterns: Optional file patterns to match
        
    Returns:
        Dictionary mapping source to destination paths
    """
    manager = AssetManager(base_dir)
    manager.setup_directory_structure()
    
    if patterns:
        return manager.copy_figure_files(patterns)
    else:
        organized = manager.organize_figure_assets()
        return {asset: asset for asset in organized}


def copy_figure_files(source_files: List[str], dest_dir: Union[str, Path]) -> Dict[str, str]:
    """Copy figure files to managed asset directory.
    
    Args:
        source_files: List of source file paths
        dest_dir: Destination directory
        
    Returns:
        Dictionary mapping source to destination paths
    """
    manager = AssetManager(dest_dir)
    manager.setup_directory_structure()
    
    copied = {}
    for source_file in source_files:
        source_path = Path(source_file)
        if source_path.exists():
            dest_relative = manager.register_asset(source_path, 'figure')
            copied[source_file] = dest_relative
    
    return copied
