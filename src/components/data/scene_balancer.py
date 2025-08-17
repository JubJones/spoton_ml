"""
Scene Balancer for RF-DETR Training
Dynamic scene balancing and cross-scene validation strategies
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import random
from pathlib import Path

from .scene_analyzer import SceneCharacteristics, MTMCCSceneAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BalancingStrategy:
    """Configuration for scene balancing strategy."""
    
    # Balancing method
    method: str = "difficulty_weighted"  # "uniform", "difficulty_weighted", "adaptive", "curriculum"
    
    # Weights for different difficulty levels
    easy_scene_weight: float = 0.8
    medium_scene_weight: float = 1.0
    hard_scene_weight: float = 1.5
    
    # Curriculum learning parameters
    curriculum_enabled: bool = False
    curriculum_epochs: int = 10  # Epochs to transition from easy to hard
    curriculum_start_ratio: float = 0.8  # Start with 80% easy scenes
    curriculum_end_ratio: float = 0.2    # End with 20% easy scenes
    
    # Cross-scene validation
    cross_scene_validation: bool = True
    validation_scene_ratio: float = 0.2  # 20% of scenes for validation
    validation_strategy: str = "stratified"  # "random", "stratified", "hardest"
    
    # Adaptive balancing
    adaptive_enabled: bool = False
    adaptation_frequency: int = 500  # Steps between adaptations
    performance_threshold: float = 0.05  # Performance improvement threshold
    
    # Scene diversity
    ensure_diversity: bool = True
    min_scenes_per_batch: int = 2  # Minimum different scenes per batch
    camera_diversity: bool = True   # Ensure camera diversity
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ["uniform", "difficulty_weighted", "adaptive", "curriculum"]:
            raise ValueError(f"Unknown balancing method: {self.method}")
        
        if not 0 < self.validation_scene_ratio < 1:
            raise ValueError(f"validation_scene_ratio must be in (0, 1), got {self.validation_scene_ratio}")
        
        if self.curriculum_enabled:
            if not 0 < self.curriculum_start_ratio <= 1:
                raise ValueError(f"curriculum_start_ratio must be in (0, 1], got {self.curriculum_start_ratio}")
            if not 0 < self.curriculum_end_ratio <= 1:
                raise ValueError(f"curriculum_end_ratio must be in (0, 1], got {self.curriculum_end_ratio}")


@dataclass
class SceneBatch:
    """Information about a balanced scene batch."""
    
    # Batch composition
    scene_ids: List[str] = field(default_factory=list)
    camera_ids: List[str] = field(default_factory=list)
    sample_indices: List[int] = field(default_factory=list)
    
    # Batch statistics
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)  # easy, medium, hard counts
    avg_difficulty: float = 0.0
    scene_diversity: float = 0.0  # 0-1, higher = more diverse
    
    # Sampling weights
    scene_weights: Dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get batch information summary."""
        return {
            'batch_size': len(self.sample_indices),
            'unique_scenes': len(set(self.scene_ids)),
            'unique_cameras': len(set(self.camera_ids)),
            'avg_difficulty': self.avg_difficulty,
            'difficulty_distribution': self.difficulty_distribution.copy(),
            'scene_diversity': self.scene_diversity
        }


class SceneBalancer:
    """
    Advanced scene balancer for RF-DETR training on MTMMC dataset.
    Implements multiple balancing strategies and cross-scene validation.
    """
    
    def __init__(
        self,
        scene_analyzer: MTMCCSceneAnalyzer,
        strategy: BalancingStrategy
    ):
        """
        Initialize scene balancer.
        
        Args:
            scene_analyzer: Configured scene analyzer with analyzed scenes
            strategy: Balancing strategy configuration
        """
        self.scene_analyzer = scene_analyzer
        self.strategy = strategy
        
        # Extract scene information
        self.scenes = scene_analyzer.scene_characteristics
        self.scene_ids = list(self.scenes.keys())
        
        # Initialize balancing state
        self.current_epoch = 0
        self.training_scenes = []
        self.validation_scenes = []
        self.scene_weights = {}
        self.performance_history = []
        
        # Split scenes for training/validation
        self._initialize_scene_splits()
        
        # Initialize weights
        self._initialize_scene_weights()
        
        logger.info(f"Initialized Scene Balancer:")
        logger.info(f"  Total scenes: {len(self.scene_ids)}")
        logger.info(f"  Training scenes: {len(self.training_scenes)}")
        logger.info(f"  Validation scenes: {len(self.validation_scenes)}")
        logger.info(f"  Balancing method: {strategy.method}")
    
    def _initialize_scene_splits(self):
        """Initialize training/validation scene splits."""
        
        if not self.strategy.cross_scene_validation:
            # Use all scenes for training
            self.training_scenes = self.scene_ids.copy()
            self.validation_scenes = []
            return
        
        # Determine number of validation scenes
        total_scenes = len(self.scene_ids)
        val_scene_count = max(1, int(total_scenes * self.strategy.validation_scene_ratio))
        
        if self.strategy.validation_strategy == "random":
            # Random split
            shuffled_scenes = self.scene_ids.copy()
            random.shuffle(shuffled_scenes)
            self.validation_scenes = shuffled_scenes[:val_scene_count]
            self.training_scenes = shuffled_scenes[val_scene_count:]
            
        elif self.strategy.validation_strategy == "stratified":
            # Stratified split by difficulty
            scenes_by_difficulty = self._group_scenes_by_difficulty()
            
            self.validation_scenes = []
            self.training_scenes = []
            
            for difficulty, scene_list in scenes_by_difficulty.items():
                n_val = max(1, int(len(scene_list) * self.strategy.validation_scene_ratio))
                shuffled = scene_list.copy()
                random.shuffle(shuffled)
                
                self.validation_scenes.extend(shuffled[:n_val])
                self.training_scenes.extend(shuffled[n_val:])
            
        elif self.strategy.validation_strategy == "hardest":
            # Use hardest scenes for validation
            scene_difficulties = [
                (scene_id, self.scenes[scene_id].difficulty_score)
                for scene_id in self.scene_ids
            ]
            scene_difficulties.sort(key=lambda x: x[1], reverse=True)
            
            self.validation_scenes = [sid for sid, _ in scene_difficulties[:val_scene_count]]
            self.training_scenes = [sid for sid, _ in scene_difficulties[val_scene_count:]]
        
        logger.info(f"Scene split strategy: {self.strategy.validation_strategy}")
        logger.info(f"Training scenes: {self.training_scenes}")
        logger.info(f"Validation scenes: {self.validation_scenes}")
    
    def _group_scenes_by_difficulty(self) -> Dict[str, List[str]]:
        """Group scenes by difficulty level."""
        
        groups = {'easy': [], 'medium': [], 'hard': []}
        
        for scene_id in self.scene_ids:
            difficulty = self.scenes[scene_id].difficulty_score
            
            if difficulty < 0.3:
                groups['easy'].append(scene_id)
            elif difficulty < 0.7:
                groups['medium'].append(scene_id)
            else:
                groups['hard'].append(scene_id)
        
        return groups
    
    def _initialize_scene_weights(self):
        """Initialize scene sampling weights based on strategy."""
        
        if self.strategy.method == "uniform":
            # Uniform weighting
            for scene_id in self.training_scenes:
                self.scene_weights[scene_id] = 1.0
                
        elif self.strategy.method == "difficulty_weighted":
            # Weight by difficulty level
            for scene_id in self.training_scenes:
                difficulty = self.scenes[scene_id].difficulty_score
                
                if difficulty < 0.3:
                    self.scene_weights[scene_id] = self.strategy.easy_scene_weight
                elif difficulty < 0.7:
                    self.scene_weights[scene_id] = self.strategy.medium_scene_weight
                else:
                    self.scene_weights[scene_id] = self.strategy.hard_scene_weight
        
        elif self.strategy.method in ["adaptive", "curriculum"]:
            # Start with uniform weights, will be adapted
            for scene_id in self.training_scenes:
                self.scene_weights[scene_id] = 1.0
        
        logger.info(f"Initialized scene weights: {len(self.scene_weights)} scenes")
    
    def update_epoch(self, epoch: int):
        """Update balancer for new epoch."""
        
        self.current_epoch = epoch
        
        # Update weights based on strategy
        if self.strategy.method == "curriculum":
            self._update_curriculum_weights()
        
        logger.debug(f"Updated balancer for epoch {epoch}")
    
    def _update_curriculum_weights(self):
        """Update weights for curriculum learning."""
        
        if not self.strategy.curriculum_enabled:
            return
        
        # Calculate curriculum progress (0 to 1)
        progress = min(1.0, self.current_epoch / self.strategy.curriculum_epochs)
        
        # Calculate current easy scene ratio
        start_ratio = self.strategy.curriculum_start_ratio
        end_ratio = self.strategy.curriculum_end_ratio
        current_easy_ratio = start_ratio + (end_ratio - start_ratio) * progress
        
        # Group scenes by difficulty
        difficulty_groups = self._group_scenes_by_difficulty()
        
        # Calculate weights based on curriculum progress
        easy_scenes = [s for s in difficulty_groups['easy'] if s in self.training_scenes]
        medium_scenes = [s for s in difficulty_groups['medium'] if s in self.training_scenes]
        hard_scenes = [s for s in difficulty_groups['hard'] if s in self.training_scenes]
        
        total_scenes = len(easy_scenes) + len(medium_scenes) + len(hard_scenes)
        
        if total_scenes == 0:
            return
        
        # Target number of easy scenes
        target_easy_count = int(total_scenes * current_easy_ratio)
        target_hard_count = int(total_scenes * (1 - current_easy_ratio))
        
        # Assign weights
        easy_weight = target_easy_count / max(len(easy_scenes), 1)
        medium_weight = (total_scenes - target_easy_count - target_hard_count) / max(len(medium_scenes), 1)
        hard_weight = target_hard_count / max(len(hard_scenes), 1)
        
        for scene_id in self.training_scenes:
            difficulty = self.scenes[scene_id].difficulty_score
            
            if difficulty < 0.3:
                self.scene_weights[scene_id] = easy_weight
            elif difficulty < 0.7:
                self.scene_weights[scene_id] = medium_weight
            else:
                self.scene_weights[scene_id] = hard_weight
        
        logger.debug(f"Curriculum weights updated: easy_ratio={current_easy_ratio:.3f}")
    
    def sample_balanced_batch(
        self,
        dataset_indices: List[int],
        scene_mapping: Dict[int, str],  # Maps dataset index to scene_id
        camera_mapping: Dict[int, str],  # Maps dataset index to camera_id
        batch_size: int,
        current_step: Optional[int] = None
    ) -> SceneBatch:
        """
        Sample a balanced batch based on current strategy.
        
        Args:
            dataset_indices: Available dataset indices
            scene_mapping: Mapping from dataset index to scene ID
            camera_mapping: Mapping from dataset index to camera ID
            batch_size: Target batch size
            current_step: Current training step (for adaptive strategies)
            
        Returns:
            Balanced scene batch
        """
        
        # Filter to training scenes only
        training_indices = [
            idx for idx in dataset_indices
            if scene_mapping.get(idx) in self.training_scenes
        ]
        
        if not training_indices:
            logger.warning("No training scene indices available")
            return SceneBatch()
        
        # Sample batch based on strategy
        if self.strategy.method in ["uniform", "difficulty_weighted", "curriculum"]:
            sampled_indices = self._sample_weighted_batch(
                training_indices, scene_mapping, camera_mapping, batch_size
            )
        elif self.strategy.method == "adaptive":
            sampled_indices = self._sample_adaptive_batch(
                training_indices, scene_mapping, camera_mapping, batch_size, current_step
            )
        else:
            # Fallback to uniform sampling
            sampled_indices = random.sample(
                training_indices, min(batch_size, len(training_indices))
            )
        
        # Create batch information
        batch = SceneBatch()
        batch.sample_indices = sampled_indices
        batch.scene_ids = [scene_mapping.get(idx, 'unknown') for idx in sampled_indices]
        batch.camera_ids = [camera_mapping.get(idx, 'unknown') for idx in sampled_indices]
        
        # Calculate batch statistics
        batch = self._calculate_batch_statistics(batch)
        
        return batch
    
    def _sample_weighted_batch(
        self,
        training_indices: List[int],
        scene_mapping: Dict[int, str],
        camera_mapping: Dict[int, str],
        batch_size: int
    ) -> List[int]:
        """Sample batch using scene weights."""
        
        # Group indices by scene
        scene_indices = defaultdict(list)
        for idx in training_indices:
            scene_id = scene_mapping.get(idx)
            if scene_id:
                scene_indices[scene_id].append(idx)
        
        # Calculate sampling probabilities
        scene_probs = []
        available_scenes = []
        
        for scene_id, indices in scene_indices.items():
            if scene_id in self.scene_weights:
                weight = self.scene_weights[scene_id]
                # Weight by number of available samples in scene
                prob = weight * len(indices)
                scene_probs.append(prob)
                available_scenes.append(scene_id)
        
        if not scene_probs:
            return random.sample(training_indices, min(batch_size, len(training_indices)))
        
        # Normalize probabilities
        total_prob = sum(scene_probs)
        scene_probs = [p / total_prob for p in scene_probs]
        
        # Sample batch ensuring diversity if required
        sampled_indices = []
        
        if self.strategy.ensure_diversity and len(available_scenes) > 1:
            # Ensure minimum number of different scenes
            min_scenes = min(self.strategy.min_scenes_per_batch, len(available_scenes))
            
            # First, sample from different scenes
            sampled_scenes = set()
            while len(sampled_scenes) < min_scenes and len(sampled_indices) < batch_size:
                # Sample scene based on probabilities
                scene_idx = np.random.choice(len(available_scenes), p=scene_probs)
                scene_id = available_scenes[scene_idx]
                
                if scene_id not in sampled_scenes:
                    # Sample one index from this scene
                    available_from_scene = scene_indices[scene_id]
                    idx = random.choice(available_from_scene)
                    sampled_indices.append(idx)
                    sampled_scenes.add(scene_id)
        
        # Fill remaining batch slots
        remaining_size = batch_size - len(sampled_indices)
        remaining_indices = [
            idx for idx in training_indices 
            if idx not in sampled_indices
        ]
        
        if remaining_size > 0 and remaining_indices:
            # Continue weighted sampling for remaining slots
            while len(sampled_indices) < batch_size and remaining_indices:
                # Recalculate probabilities for remaining indices
                remaining_scene_counts = defaultdict(int)
                for idx in remaining_indices:
                    scene_id = scene_mapping.get(idx)
                    if scene_id:
                        remaining_scene_counts[scene_id] += 1
                
                remaining_probs = []
                remaining_scenes = []
                
                for scene_id, count in remaining_scene_counts.items():
                    if scene_id in self.scene_weights:
                        prob = self.scene_weights[scene_id] * count
                        remaining_probs.append(prob)
                        remaining_scenes.append(scene_id)
                
                if remaining_probs:
                    total_prob = sum(remaining_probs)
                    remaining_probs = [p / total_prob for p in remaining_probs]
                    
                    # Sample scene
                    scene_idx = np.random.choice(len(remaining_scenes), p=remaining_probs)
                    target_scene = remaining_scenes[scene_idx]
                    
                    # Sample index from target scene
                    scene_candidates = [
                        idx for idx in remaining_indices
                        if scene_mapping.get(idx) == target_scene
                    ]
                    
                    if scene_candidates:
                        selected_idx = random.choice(scene_candidates)
                        sampled_indices.append(selected_idx)
                        remaining_indices.remove(selected_idx)
                else:
                    break
        
        return sampled_indices
    
    def _sample_adaptive_batch(
        self,
        training_indices: List[int],
        scene_mapping: Dict[int, str],
        camera_mapping: Dict[int, str],
        batch_size: int,
        current_step: Optional[int]
    ) -> List[int]:
        """Sample batch using adaptive strategy based on performance."""
        
        # Check if we should adapt weights
        if (current_step and 
            self.strategy.adaptive_enabled and 
            current_step % self.strategy.adaptation_frequency == 0):
            self._adapt_scene_weights()
        
        # Use weighted sampling with current weights
        return self._sample_weighted_batch(
            training_indices, scene_mapping, camera_mapping, batch_size
        )
    
    def _adapt_scene_weights(self):
        """Adapt scene weights based on performance history."""
        
        if len(self.performance_history) < 2:
            return
        
        # Calculate recent performance improvement
        recent_performance = self.performance_history[-5:]  # Last 5 measurements
        if len(recent_performance) < 2:
            return
        
        improvement = recent_performance[-1] - recent_performance[0]
        
        # If performance is not improving, increase hard scene weights
        if improvement < self.strategy.performance_threshold:
            logger.info("Adapting weights: increasing hard scene emphasis")
            
            for scene_id in self.training_scenes:
                difficulty = self.scenes[scene_id].difficulty_score
                
                if difficulty >= 0.7:  # Hard scenes
                    self.scene_weights[scene_id] *= 1.2
                elif difficulty < 0.3:  # Easy scenes
                    self.scene_weights[scene_id] *= 0.9
        
        # Normalize weights
        total_weight = sum(self.scene_weights.values())
        if total_weight > 0:
            for scene_id in self.scene_weights:
                self.scene_weights[scene_id] /= total_weight
    
    def _calculate_batch_statistics(self, batch: SceneBatch) -> SceneBatch:
        """Calculate statistics for the batch."""
        
        if not batch.sample_indices:
            return batch
        
        # Difficulty distribution
        difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        difficulties = []
        
        for scene_id in batch.scene_ids:
            if scene_id in self.scenes:
                difficulty = self.scenes[scene_id].difficulty_score
                difficulties.append(difficulty)
                
                if difficulty < 0.3:
                    difficulty_counts['easy'] += 1
                elif difficulty < 0.7:
                    difficulty_counts['medium'] += 1
                else:
                    difficulty_counts['hard'] += 1
        
        batch.difficulty_distribution = difficulty_counts
        batch.avg_difficulty = np.mean(difficulties) if difficulties else 0.0
        
        # Scene diversity (Shannon entropy of scene distribution)
        scene_counts = {}
        for scene_id in batch.scene_ids:
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        
        if scene_counts:
            total_samples = sum(scene_counts.values())
            probs = [count / total_samples for count in scene_counts.values()]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)
            max_entropy = np.log(len(scene_counts))
            batch.scene_diversity = entropy / max(max_entropy, 1e-8)
        
        # Scene weights
        for scene_id in set(batch.scene_ids):
            batch.scene_weights[scene_id] = self.scene_weights.get(scene_id, 1.0)
        
        batch.total_weight = sum(
            self.scene_weights.get(scene_id, 1.0) for scene_id in batch.scene_ids
        )
        
        return batch
    
    def get_validation_scenes(self) -> List[str]:
        """Get list of validation scene IDs."""
        return self.validation_scenes.copy()
    
    def get_training_scenes(self) -> List[str]:
        """Get list of training scene IDs."""
        return self.training_scenes.copy()
    
    def update_performance(self, performance_metric: float):
        """Update performance history for adaptive strategies."""
        
        self.performance_history.append(performance_metric)
        
        # Keep history size manageable
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-25:]
    
    def get_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive balancing statistics."""
        
        training_difficulties = [
            self.scenes[scene_id].difficulty_score
            for scene_id in self.training_scenes
            if scene_id in self.scenes
        ]
        
        validation_difficulties = [
            self.scenes[scene_id].difficulty_score
            for scene_id in self.validation_scenes
            if scene_id in self.scenes
        ]
        
        stats = {
            'strategy': self.strategy.method,
            'total_scenes': len(self.scenes),
            'training_scenes': len(self.training_scenes),
            'validation_scenes': len(self.validation_scenes),
            'current_epoch': self.current_epoch,
            'scene_weights': self.scene_weights.copy(),
            'training_difficulty_stats': {
                'mean': np.mean(training_difficulties) if training_difficulties else 0,
                'std': np.std(training_difficulties) if training_difficulties else 0,
                'min': min(training_difficulties) if training_difficulties else 0,
                'max': max(training_difficulties) if training_difficulties else 0
            },
            'validation_difficulty_stats': {
                'mean': np.mean(validation_difficulties) if validation_difficulties else 0,
                'std': np.std(validation_difficulties) if validation_difficulties else 0,
                'min': min(validation_difficulties) if validation_difficulties else 0,
                'max': max(validation_difficulties) if validation_difficulties else 0
            }
        }
        
        return stats


def create_scene_balancer(
    scene_analyzer: MTMCCSceneAnalyzer,
    method: str = "difficulty_weighted",
    cross_scene_validation: bool = True,
    curriculum_learning: bool = False,
    **kwargs
) -> SceneBalancer:
    """
    Convenience function to create scene balancer.
    
    Args:
        scene_analyzer: Configured scene analyzer
        method: Balancing method
        cross_scene_validation: Enable cross-scene validation
        curriculum_learning: Enable curriculum learning
        **kwargs: Additional strategy parameters
        
    Returns:
        Configured scene balancer
    """
    
    strategy = BalancingStrategy(
        method=method,
        cross_scene_validation=cross_scene_validation,
        curriculum_enabled=curriculum_learning,
        **kwargs
    )
    
    return SceneBalancer(scene_analyzer, strategy)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Would normally use real scene analyzer with analyzed scenes
    from .scene_analyzer import create_scene_analyzer
    
    # Mock scene analysis
    print("🧪 Testing Scene Balancer with mock data")
    
    analyzer = create_scene_analyzer()
    
    # Add mock scene characteristics
    from .scene_analyzer import SceneCharacteristics
    
    mock_scenes = {
        'scene_01': SceneCharacteristics(
            scene_id='scene_01',
            difficulty_score=0.2,
            avg_person_count=3.0,
            crowd_density_level='low'
        ),
        'scene_02': SceneCharacteristics(
            scene_id='scene_02', 
            difficulty_score=0.5,
            avg_person_count=8.0,
            crowd_density_level='medium'
        ),
        'scene_03': SceneCharacteristics(
            scene_id='scene_03',
            difficulty_score=0.8,
            avg_person_count=15.0,
            crowd_density_level='high'
        )
    }
    
    analyzer.scene_characteristics = mock_scenes
    
    # Create balancer with curriculum learning
    balancer = create_scene_balancer(
        analyzer,
        method="curriculum",
        curriculum_learning=True,
        curriculum_epochs=5
    )
    
    # Test batch sampling
    mock_dataset_indices = list(range(100))
    mock_scene_mapping = {i: f'scene_{(i % 3) + 1:02d}' for i in range(100)}
    mock_camera_mapping = {i: f'cam_{(i % 6) + 1:02d}' for i in range(100)}
    
    # Sample batches across epochs
    for epoch in range(3):
        balancer.update_epoch(epoch)
        
        batch = balancer.sample_balanced_batch(
            mock_dataset_indices,
            mock_scene_mapping,
            mock_camera_mapping,
            batch_size=16
        )
        
        print(f"\nEpoch {epoch} batch:")
        print(f"  Batch size: {len(batch.sample_indices)}")
        print(f"  Unique scenes: {len(set(batch.scene_ids))}")
        print(f"  Avg difficulty: {batch.avg_difficulty:.3f}")
        print(f"  Difficulty distribution: {batch.difficulty_distribution}")
        print(f"  Scene diversity: {batch.scene_diversity:.3f}")
    
    # Get balancing statistics
    stats = balancer.get_balancing_statistics()
    print(f"\nBalancing Statistics:")
    print(f"  Method: {stats['strategy']}")
    print(f"  Training scenes: {stats['training_scenes']}")
    print(f"  Validation scenes: {stats['validation_scenes']}")
    
    print("✅ Scene Balancer validation completed")