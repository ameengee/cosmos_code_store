import subprocess
import os
import logging
import socket

def find_free_port():
    """Find a free port for torchrun."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def augment_video(prompt, input_video_path, output_video_path):
    """
    Process a video using the cosmos-transfer1 script.
    
    Args:
        prompt (str): The text prompt for video generation
        input_video_path (str): Path to the input video file
        output_video_path (str): Path where the output video will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
        env['CHECKPOINT_DIR'] = env.get('CHECKPOINT_DIR', './checkpoints')
        env['NUM_GPU'] = env.get('NUM_GPU', '1')
        
        # Add CUDA memory management settings
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Change to cosmos-transfer1 directory
        os.chdir('/root/cosmos-transfer1')
        
        # Set PYTHONPATH
        env['PYTHONPATH'] = os.getcwd()
        
        # Find free port for torchrun
        free_port = find_free_port()
        
        # Build the command
        # Remove .mp4 extension from output name since cosmos-transfer1 adds it automatically
        output_name = os.path.basename(output_video_path)
        if output_name.endswith('.mp4'):
            output_name = output_name[:-4]
            
        cmd = [
            'torchrun',
            f'--nproc_per_node={env["NUM_GPU"]}',
            '--nnodes=1',
            '--node_rank=0',
            f'--master_port={free_port}',
            'cosmos_transfer1/diffusion/inference/transfer.py',
            '--prompt', prompt,
            '--checkpoint_dir', env['CHECKPOINT_DIR'],
            '--video_save_folder', os.path.dirname(output_video_path),
            '--video_save_name', output_name,
            '--input_video_path', input_video_path,
            '--controlnet_specs', 'assets/augment.json',
            '--offload_text_encoder_model',
            '--num_steps', '20',
            '--offload_guardrail_models',
            '--num_gpus', env['NUM_GPU']
        ]
        
        logging.info(f"Starting video processing: {input_video_path} -> {output_video_path}")
        
        # Run the command
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Verify the output file exists
            if os.path.exists(output_video_path):
                logging.info(f"Video processing completed successfully: {output_video_path}")
                file_size = os.path.getsize(output_video_path)
                logging.info(f"Generated video file size: {file_size} bytes")
            else:
                logging.error(f"Video processing claimed success but output file not found: {output_video_path}")
                # Check if a .mp4.mp4 file was created instead
                double_ext_path = output_video_path + '.mp4'
                if os.path.exists(double_ext_path):
                    logging.info(f"Found double extension file, renaming: {double_ext_path} -> {output_video_path}")
                    os.rename(double_ext_path, output_video_path)
                else:
                    return False
            
            # Clean up any .txt files generated alongside the video
            output_dir = os.path.dirname(output_video_path)
            video_basename = os.path.splitext(os.path.basename(output_video_path))[0]
            txt_pattern = os.path.join(output_dir, f"{video_basename}*.txt")
            
            import glob
            for txt_file in glob.glob(txt_pattern):
                try:
                    os.remove(txt_file)
                    logging.info(f"Cleaned up generated text file: {txt_file}")
                except Exception as cleanup_error:
                    logging.warning(f"Could not remove text file {txt_file}: {cleanup_error}")
            
            return True
        else:
            logging.error(f"Video processing failed. Error: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        return False