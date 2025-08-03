#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append('/root/synphony')
from app import download_huggingface_dataset, collect_lerobot_files

async def test_download():
    """Test downloading multiple robot datasets"""
    datasets_to_test = [
        "https://huggingface.co/datasets/siyavash/so101_test"
    ]
    
    all_valid_videos = []
    
    for dataset_url in datasets_to_test:
        print(f"\nDownloading dataset: {dataset_url}")
        
        try:
            # Download the dataset
            dataset_path = await download_huggingface_dataset(dataset_url)
            print(f"✓ Dataset downloaded to: {dataset_path}")
            
            # Collect video and parquet files
            print(f"Collecting files...")
            videos_dict, parquet_paths = await collect_lerobot_files(dataset_path)
            
            print(f"Found videos by camera:")
            for camera, videos in videos_dict.items():
                print(f"  {camera}: {len(videos)} videos")
                for i, video in enumerate(videos[:2]):  # Show first 2
                    video_size = os.path.getsize(video)
                    print(f"    {i}: {os.path.basename(video)} ({video_size} bytes)")
                if len(videos) > 2:
                    print(f"    ... and {len(videos) - 2} more")
            
            print(f"Found {len(parquet_paths)} parquet files")
            
            # Check for corrupted videos (very small files)
            all_videos = []
            for camera, video_list in videos_dict.items():
                all_videos.extend(video_list)
            
            corrupted_videos = []
            valid_videos = []
            
            for video in all_videos:
                size = os.path.getsize(video)
                if size < 1000:  # Less than 1KB is likely corrupted
                    corrupted_videos.append((video, size))
                else:
                    valid_videos.append((video, size))
            
            print(f"Video integrity check:")
            print(f"  Valid videos: {len(valid_videos)}")
            print(f"  Corrupted videos: {len(corrupted_videos)}")
            
            all_valid_videos.extend(valid_videos)
            
        except Exception as e:
            print(f"❌ Error downloading dataset {dataset_url}: {e}")
    
    return all_valid_videos

if __name__ == "__main__":
    valid_videos = asyncio.run(test_download())
    if valid_videos:
        print(f"\n✓ Success! Found {len(valid_videos)} total valid videos across all datasets")
        print(f"Sample videos for testing:")
        for video, size in valid_videos[:5]:
            print(f"  {video}: {size} bytes")
    else:
        print(f"\n❌ No valid videos found for testing")