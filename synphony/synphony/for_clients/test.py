#!/usr/bin/env python3

import os
from video_augmentor import augment_video

def main():
    # Base paths
    input_base_path = "/root/synphony/datasets/so100_hackathon_shirt/videos/chunk-000/observation.images.overhead"
    output_base_path = "/root/synphony/datasets/so100_hackathon_shirt/videos/chunk-000/observation.images.augmented"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    # Process videos 0 through 4
    for i in range(4):
        input_video_path = os.path.join(input_base_path, f"episode_{i:06d}.mp4")
        output_video_path = os.path.join(output_base_path, f"augmented_episode_{i:06d}.mp4")
        
        prompt = f"robot arms fold clothes"
        
        print(f"Processing video {i}: {input_video_path} -> {output_video_path}")
        
        success = augment_video(prompt, input_video_path, output_video_path)
        
        if success:
            print(f"Successfully processed episode {i}")
        else:
            print(f"Failed to process episode {i}")

if __name__ == "__main__":
    main()