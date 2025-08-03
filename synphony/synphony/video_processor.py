#!/usr/bin/env python3
import os
import cv2
import base64
import tempfile
import shutil
import logging
import asyncio
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def extract_frames_from_video(video_path, num_frames=6):
    """Extract equally spaced frames from a video file using FFmpeg (async)."""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        # Get video duration
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *duration_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"Could not get video duration: {video_path}")
            return []
        try:
            duration = float(stdout.decode().strip())
        except ValueError:
            logger.error(f"Invalid duration format: {stdout.decode().strip()}")
            return []
        if duration <= 0:
            logger.error(f"Invalid video duration: {duration}")
            return []
        time_intervals = [i * duration / num_frames for i in range(num_frames)]
        frames = []
        for i, time_pos in enumerate(time_intervals):
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', video_path, '-ss', str(time_pos),
                '-vframes', '1', '-q:v', '2', frame_path
            ]
            proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0 and os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None and frame.size > 0:
                    frames.append(frame)
                else:
                    logger.warning(f"Could not load extracted frame: {frame_path}")
            else:
                logger.warning(f"FFmpeg failed to extract frame at {time_pos}s: {stderr.decode()}")
        if not frames:
            logger.error(f"Could not extract any frames from {video_path}")
            return []
        logger.info(f"Extracted {len(frames)} frames from {video_path} using FFmpeg")
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return []
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp directory {temp_dir}: {cleanup_error}")

def encode_frame_to_base64(frame):
    """Encode a CV2 frame to base64 string."""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception as e:
        logger.error(f"Error encoding frame to base64: {e}")
        return None

async def analyze_video_with_openai(frames):
    """Analyze video frames using OpenAI Vision API to generate a realistic description (async)."""
    try:
        if not frames:
            logger.error("No frames provided for analysis")
            return "a video sequence"
        logger.info(f"Analyzing {len(frames)} frames with OpenAI")
        encoded_frames = []
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                logger.warning(f"Skipping invalid frame {i}")
                continue
            encoded_frame = encode_frame_to_base64(frame)
            if encoded_frame:
                encoded_frames.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_frame}",
                        "detail": "low"
                    }
                })
            else:
                logger.warning(f"Failed to encode frame {i}")
        if not encoded_frames:
            logger.error("No frames could be encoded for OpenAI analysis")
            return "a video sequence"
        logger.info(f"Successfully encoded {len(encoded_frames)} frames for OpenAI analysis")
        prompt = """Analyze these 6 frames from a video sequence and provide a realistic, detailed description suitable for a video diffusion model. Focus on:

1. The main subject/object and their actions
2. The environment and setting
3. Camera angle and perspective
4. Movement and motion patterns
5. Visual style and lighting and colors

Provide a concise but descriptive prompt that would help a video diffusion model recreate a similar video. Keep it under 200 words and focus on visual elements that are important for video generation."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + encoded_frames
            }
        ]
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        description = response.choices[0].message.content.strip()
        logger.info(f"Generated video description: {description[:100]}...")
        return description
    except Exception as e:
        logger.error(f"Error analyzing video with OpenAI: {e}")
        return "a robot is performing manipulation tasks"

async def generate_video_prompt(video_path):
    """Generate OpenAI prompt for video diffusion model based on video content (async)."""
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is not set")
        return "a robot is performing manipulation tasks"
    frames = await extract_frames_from_video(video_path, num_frames=6)
    if frames:
        return await analyze_video_with_openai(frames)
    else:
        logger.warning(f"Could not extract frames from {video_path}, using fallback prompt")
        return "a robot is performing manipulation tasks"