#!/usr/bin/env python3
import os
import sys
import logging
import shutil
import glob
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from video_augmentor import augment_video
from video_processor import generate_video_prompt

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_orchestrator.log')
        ]
    )

# Logger will be initialized after setup_logging() is called
logger = None

DATASETS_BASE = "/root/synphony/datasets"

def get_next_episode_number(video_folder):
    """Find the next available episode number in a video folder."""
    existing_files = glob.glob(os.path.join(video_folder, "episode_*.mp4"))
    if not existing_files:
        return 0
    
    episode_numbers = []
    for file in existing_files:
        filename = os.path.basename(file)
        episode_num = filename.replace("episode_", "").replace(".mp4", "")
        try:
            episode_numbers.append(int(episode_num))
        except ValueError:
            continue
    
    return max(episode_numbers) + 1 if episode_numbers else 0

def copy_parquet_file(src_episode_num, dest_episode_num, data_folder):
    """Copy parquet file from source episode to destination episode."""
    src_parquet = os.path.join(data_folder, f"episode_{src_episode_num:06d}.parquet")
    dest_parquet = os.path.join(data_folder, f"episode_{dest_episode_num:06d}.parquet")
    
    if os.path.exists(src_parquet):
        try:
            shutil.copy2(src_parquet, dest_parquet)
            logger.info(f"Copied parquet file: {src_parquet} -> {dest_parquet}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy parquet file {src_parquet}: {e}")
            return False
    else:
        logger.warning(f"Source parquet file not found: {src_parquet}")
        return False

def process_dataset(dataset_name, n_episodes):
    """Process a dataset to generate n_episodes new synthetic videos."""
    dataset_path = os.path.join(DATASETS_BASE, dataset_name)
    videos_path = os.path.join(dataset_path, "videos", "chunk-000")
    data_path = os.path.join(dataset_path, "data", "chunk-000")
    
    if not os.path.exists(videos_path):
        logger.error(f"Videos path not found: {videos_path}")
        return False
    
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        return False
    
    # Get all observation folders (camera types)
    observation_folders = [d for d in os.listdir(videos_path) 
                          if os.path.isdir(os.path.join(videos_path, d)) and d.startswith("observation.images.")]
    
    if not observation_folders:
        logger.error(f"No observation folders found in {videos_path}")
        return False
    
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Number of episodes to generate: {n_episodes}")
    logger.info(f"Found camera types: {observation_folders}")
    
    # First, validate source episodes exist
    logger.info("Step 0: Validating source episodes...")
    for i in range(n_episodes):
        src_parquet = os.path.join(data_path, f"episode_{i:06d}.parquet")
        if not os.path.exists(src_parquet):
            logger.error(f"Source episode {i:06d} not found: {src_parquet}")
            return False
    
    # Copy parquet files for all new episodes
    logger.info("Step 1: Copying parquet files...")
    first_obs_folder = observation_folders[0]
    first_obs_path = os.path.join(videos_path, first_obs_folder)
    next_episode_num = get_next_episode_number(first_obs_path)
    
    for i in range(n_episodes):
        src_episode = i  # Use episodes 000000, 000001, etc.
        dest_episode = next_episode_num + i
        copy_parquet_file(src_episode, dest_episode, data_path)
    
    # Then, generate videos for each camera type
    logger.info("Step 2: Generating synthetic videos...")
    total_success = 0
    total_attempts = 0
    
    # Process one video at a time (episode by episode, then camera by camera)
    for i in range(n_episodes):
        src_episode = i
        dest_episode = next_episode_num + i
        
        logger.info(f"Processing episode {src_episode:06d} -> {dest_episode:06d}")
        
        for obs_folder in observation_folders:
            obs_path = os.path.join(videos_path, obs_folder)
            logger.info(f"  Camera type: {obs_folder}")
            
            src_video_path = os.path.join(obs_path, f"episode_{src_episode:06d}.mp4")
            dest_video_path = os.path.join(obs_path, f"episode_{dest_episode:06d}.mp4")
            
            if not os.path.exists(src_video_path):
                logger.warning(f"Source video not found: {src_video_path}")
                continue
            
            total_attempts += 1
            logger.info(f"Generating: {src_video_path} -> {dest_video_path}")
            
            # Generate prompt using OpenAI
            prompt = generate_video_prompt(src_video_path)
            logger.info(f"Generated prompt: {prompt[:100]}...")
            
            # Generate synthetic video
            try:
                success = augment_video(
                    prompt=prompt,
                    input_video_path=src_video_path,
                    output_video_path=dest_video_path
                )
                
                if success:
                    total_success += 1
                    logger.info(f"✓ Successfully generated: {dest_video_path}")
                else:
                    logger.error(f"✗ Failed to generate: {dest_video_path}")
                    
            except Exception as e:
                logger.error(f"✗ Exception generating {dest_video_path}: {str(e)}")
    
    logger.info(f"Dataset {dataset_name} completed: {total_success}/{total_attempts} videos generated successfully")
    return total_success == total_attempts

def orchestrate_video_generation(dataset_episodes_list):
    """Orchestrate video generation for multiple datasets.
    
    Args:
        dataset_episodes_list: List of tuples [(dataset_name, n_episodes), ...]
    """
    logger.info("Starting video generation orchestration...")
    logger.info(f"Datasets to process: {dataset_episodes_list}")
    
    success_count = 0
    total_datasets = len(dataset_episodes_list)
    
    for dataset_name, n_episodes in dataset_episodes_list:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset_name} ({n_episodes} episodes)")
            logger.info(f"{'='*60}")
            
            success = process_dataset(dataset_name, n_episodes)
            if success:
                success_count += 1
                logger.info(f"✓ Successfully completed dataset: {dataset_name}")
            else:
                logger.error(f"✗ Failed to complete dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"✗ Exception processing dataset {dataset_name}: {str(e)}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ORCHESTRATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total datasets: {total_datasets}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_datasets - success_count}")
    logger.info(f"Success rate: {(success_count/total_datasets*100):.1f}%" if total_datasets > 0 else "No datasets processed")
    
    return success_count == total_datasets

def main():
    """Main orchestrator function"""
    global logger
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("Video Generation Orchestrator")
        print("Usage: python test_video_augmentor.py <dataset_name1:episodes1> [dataset_name2:episodes2] ...")
        print("")
        print("Example:")
        print("  python test_video_augmentor.py stack_rings_blue:3 so101_test:2")
        print("  python test_video_augmentor.py dice-art-run1:5")
        print("")
        print("This will generate new synthetic episodes for each dataset using the first N episodes as source.")
        print("Available datasets:")
        if os.path.exists(DATASETS_BASE):
            for item in os.listdir(DATASETS_BASE):
                if os.path.isdir(os.path.join(DATASETS_BASE, item)):
                    print(f"  - {item}")
        sys.exit(0)
    
    # Parse dataset:episodes arguments
    dataset_episodes_list = []
    for arg in sys.argv[1:]:
        if ':' not in arg:
            logger.error(f"Invalid argument format: {arg}. Use dataset_name:episodes")
            sys.exit(1)
        
        dataset_name, episodes_str = arg.split(':', 1)
        try:
            n_episodes = int(episodes_str)
            if n_episodes <= 0:
                raise ValueError("Episodes must be positive")
            dataset_episodes_list.append((dataset_name, n_episodes))
        except ValueError as e:
            logger.error(f"Invalid episodes count for {dataset_name}: {episodes_str}")
            sys.exit(1)
    
    logger.info(f"Starting video generation orchestration...")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Run orchestration
    success = orchestrate_video_generation(dataset_episodes_list)
    
    if success:
        logger.info("🎉 All datasets processed successfully!")
    else:
        logger.error("❌ Some datasets failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)