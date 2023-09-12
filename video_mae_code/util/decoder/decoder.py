# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import math
import random

import numpy as np
import av
import torch
import torchvision.io as io
import io as io_module
import cv2

import av
import torch
import numpy as np
import ffmpeg

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    return new_frames

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

# Keeping the rest of decode the same, just changing where it access the container
def pyav_decode(container, target_fps):
    frames = []
    container.streams.video[0].thread_type = "AUTO"
    for frame in container.decode(video=0):
        img = frame.to_image()
        img_array = np.asarray(img)
        frames.append(torch.from_numpy(img_array).permute(2, 0, 1))
    frames = torch.stack(frames)
    return frames

def decode_ffmpeg(video_path, start=0, num_sec=2, num_frames=16):
    try:
        if '.webm' in video_path:
            video = cv2.VideoCapture(video_path)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = math.floor(float(frame_count / fps))
        
        # Get video metadata using ffprobe
        decode_all_video = False
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=width,height,duration,r_frame_rate')
        video_info = next((s for s in probe['streams'] if 'width' in s and 'height' in s), None)
        
        if video_info is None:
            raise ValueError("No video stream information found in the input video.")
        
        width = int(video_info['width'])
        height = int(video_info['height'])
        r_frame_rate = video_info['r_frame_rate'].split('/')
        
        if '.webm' in video_path:
            end = duration
        else:
            end = math.floor(float(video_info['duration']))
            
        fps = int(r_frame_rate[0]) / int(r_frame_rate[1])
        
        start_seek = random.randint(start, int(max(start, end - num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=num_sec + 0.1)
            .filter('fps', fps=fps)
        )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video_copy = video.copy()
        video = torch.from_numpy(video_copy)
        if video.shape[1] < num_frames:
            zeros = torch.zeros((3, num_frames - video.shape[1], height, width), dtype=torch.uint8)
            video = torch.cat((video, zeros), axis=1)
        frames = video
        
    except Exception as e:
        print("Failed to decode by ffmpeg with exception: {}".format(e))
        return None
    
    return frames, fps, decode_all_video

# Original decode function
def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    try:
        assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
        # Convert the bytes to a tensor.
        video_tensor = torch.from_numpy(np.frombuffer(np.copy(container), dtype=np.uint8))

        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        # The video_meta is empty, fetch the meta data from the raw video.
        if len(video_meta) == 0:
            # Tracking the meta info for selective decoding in the future.

            meta = io._probe_video_from_memory(video_tensor)
            # Using the information from video_meta to perform selective decoding.
            video_meta["video_timebase"] = meta.video_timebase
            video_meta["video_numerator"] = meta.video_timebase.numerator
            video_meta["video_denominator"] = meta.video_timebase.denominator
            video_meta["has_video"] = meta.has_video
            video_meta["video_duration"] = meta.video_duration
            video_meta["video_fps"] = meta.video_fps
            video_meta["audio_timebas"] = meta.audio_timebase
            video_meta["audio_numerator"] = meta.audio_timebase.numerator
            video_meta["audio_denominator"] = meta.audio_timebase.denominator
            video_meta["has_audio"] = meta.has_audio
            video_meta["audio_duration"] = meta.audio_duration
            video_meta["audio_sample_rate"] = meta.audio_sample_rate

        fps = video_meta["video_fps"]
        if not rigid_decode_all_video: # In pretraining we skip this
            if (
                video_meta["has_video"]
                and video_meta["video_denominator"] > 0
                and video_meta["video_duration"] > 0
            ):
                # try selective decoding.
                decode_all_video = False
                clip_size = sampling_rate * num_frames / target_fps * fps
                start_idx, end_idx = get_start_end_idx(
                    fps * video_meta["video_duration"],
                    clip_size,
                    clip_idx,
                    num_clips,
                    use_offset=use_offset,
                )
                # Convert frame index to pts.
                pts_per_frame = video_meta["video_denominator"] / fps
                video_start_pts = int(start_idx * pts_per_frame)
                video_end_pts = int(end_idx * pts_per_frame)

        # Decode the raw video with the tv decoder.
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

        if v_frames.shape == torch.Size([0]):
            # failed selective decoding
            decode_all_video = True
            video_start_pts, video_end_pts = 0, -1
            v_frames, _ = io._read_video_from_memory(
                video_tensor,
                seek_frame_margin=1.0,
                read_video_stream="visual" in modalities,
                video_width=0,
                video_height=0,
                video_min_dimension=max_spatial_scale,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase_numerator=video_meta["video_numerator"],
                video_timebase_denominator=video_meta["video_denominator"],
            )
    except Exception as e:
        print("Failed to decode by torchvision with exception: {}".format(e))
        return None

    # Return None if the frames was not decoded successfully.
    if v_frames is None or v_frames.size(0) == 0:
        return None, fps, decode_all_video
    return v_frames, fps, decode_all_video

# New decode function with pyav because torchvision.io is not working
def decode(
    container_bytes,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    try:
        assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)

        # Convert bytes to a readable buffer and open it with PyAV
        container_buffer = io_module.BytesIO(container_bytes)
        container = av.open(container_buffer)
        
        # Print size of the video in MB
        size_MB = len(container_bytes) / (1024 * 1024)

        # Print duration of the video in seconds
        video_stream = container.streams.video[0]
        video_duration_sec = video_stream.duration * float(video_stream.time_base)

        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        # The video_meta is empty, fetch the meta data from the raw video.
        if len(video_meta) == 0:
            # Tracking the meta info for selective decoding in the future.
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate)
            time_base = video_stream.time_base
            video_meta["video_timebase"] = time_base
            video_meta["video_numerator"] = time_base.numerator
            video_meta["video_denominator"] = time_base.denominator
            video_meta["has_video"] = True  # Assuming the container has a video stream
            video_meta["video_duration"] = video_stream.duration * time_base
            video_meta["video_fps"] = fps

        fps = video_meta["video_fps"]

        # Determine total frames in video
        total_frames = video_stream.frames
        
        # If less than 120 frames, raise an exception (or handle it as you see fit)
        window_length = int(fps * 1.85)
        max_frames = 450
        if video_duration_sec > 40:
            print("total_frames, fps: ", total_frames, fps)
            raise ValueError("Video is too long")
        if total_frames < window_length + 1:
            raise ValueError("Video of fps {} has less than {} frames".format(fps, window_length + 1))
        if total_frames > max_frames:
            print("total_frames: ", total_frames)
            raise ValueError("Video has greater than {} frames".format(max_frames))
        if total_frames < 16:
            raise ValueError("Video must contain at least 16 frames")

        # Select starting point
        start_frame = np.random.randint(0, int(total_frames - window_length - 1))

        # PyAV decoding
        frames_list = []
        frame_count = 0

        count = 0
        for frame in container.decode(video=0):
            if frame_count >= start_frame and frame_count < start_frame + window_length:
                img = frame.to_image()
                img_array = np.array(img)
                frames_list.append(img_array)
            frame_count += 1
            if frame_count >= start_frame + window_length:
                break
            count += 1

        v_frames = torch.from_numpy(np.stack(frames_list))

    except Exception as e:
        print("Failed to decode with PyAV with exception: {}".format(e))
        raise e

    # Return None if the frames were not decoded successfully.
    if v_frames is None or v_frames.size(0) == 0:
        return None, fps, decode_all_video
    return v_frames, fps, decode_all_video


# decode_with_pyav first attempt
def decode_with_pyav(
    video_path,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    target_fps=30,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    def get_container_info(container):
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0] if len(container.streams.audio) > 0 else None

        video_meta = {
            "video_timebase": video_stream.time_base,
            "video_numerator": video_stream.time_base.numerator,
            "video_denominator": video_stream.time_base.denominator,
            "has_video": True,
            "video_duration": video_stream.duration * video_stream.time_base,
            "video_fps": video_stream.average_rate,
            "has_audio": audio_stream is not None,
        }
        if audio_stream is not None:
            video_meta.update({
                "audio_timebase": audio_stream.time_base,
                "audio_numerator": audio_stream.time_base.numerator,
                "audio_denominator": audio_stream.time_base.denominator,
                "audio_duration": audio_stream.duration * audio_stream.time_base,
                "audio_sample_rate": audio_stream.sample_rate,
            })
        return video_meta

    try:
        container = av.open(video_path)
        video_meta = get_container_info(container)

        fps = video_meta["video_fps"]
        clip_size = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"],
            clip_size,
            clip_idx,
            num_clips,
            use_offset=use_offset,
        )

        video_stream = container.streams.video[0]

        if max_spatial_scale > 0:
            height, width = video_stream.height, video_stream.width
            if height < width:
                new_height = max_spatial_scale
                new_width = int(width * (max_spatial_scale / height))
            else:
                new_width = max_spatial_scale
                new_height = int(height * (max_spatial_scale / width))

            video_stream.codec_context.height = new_height
            video_stream.codec_context.width = new_width

        container.seek(start_idx / fps, any_stream=video_stream)
        frames = []
        frame_count = 0
        for frame in container.decode(video_stream):
            if frame_count % sampling_rate == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            frame_count += 1
            if frame_count >= end_idx:
                break

        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames)

    except Exception as e:
        print("Failed to decode by pyav with exception: {}".format(e))
        return None

    return frames, fps, rigid_decode_all_video

# decode_with_pyav second attempt
def decode_with_pyav(
    video_path,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    def get_container_info(container):
        video_meta = {}
        for stream in container.streams:
            if stream.type == "video":
                video_meta["video_fps"] = stream.average_rate
                video_meta["video_duration"] = stream.duration * stream.time_base
                video_meta["video_timebase"] = stream.time_base
            elif stream.type == "audio":
                video_meta["audio_sample_rate"] = stream.sample_rate
                video_meta["audio_duration"] = stream.duration * stream.time_base
                video_meta["audio_timebase"] = stream.time_base
        return video_meta

    try:
        assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_meta = get_container_info(container)

        fps = video_meta["video_fps"]
        total_frames = int(video_meta["video_duration"] * fps)
        clip_size = sampling_rate * num_frames
        start_idx, end_idx = get_start_end_idx(
            total_frames, clip_size, clip_idx, num_clips, use_offset=use_offset
        )

        if max_spatial_scale > 0:
            orig_width, orig_height = video_stream.width, video_stream.height
            aspect_ratio = orig_width / orig_height
            if aspect_ratio >= 1:
                new_width = max_spatial_scale
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_spatial_scale
                new_width = int(new_height * aspect_ratio)

            video_stream.codec_context.width = new_width
            video_stream.codec_context.height = new_height

        selected_frames = []
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx >= start_idx and frame_idx < end_idx:
                if (frame_idx - start_idx) % sampling_rate == 0:
                    selected_frames.append(frame.to_ndarray(format="rgb24"))

        frames = np.stack(selected_frames, axis=0)
        frames = torch.from_numpy(frames)

        container.close()

    except Exception as e:
        print("Failed to decode with PyAV with exception: {}".format(e))
        return None, fps, None

    return frames, fps, None

