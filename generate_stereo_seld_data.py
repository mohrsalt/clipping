import glob
import tqdm
import pandas as pd
import os
import random
import librosa
import math
import cv2
import numpy as np
import soundfile as sf
import py360convert

from utils import CheckOnOffFast, set_range_minus180to180, fold_back_azimuth, E2PFast


def select_random_recording_start_deg(metadata_paths, metadata_path_weights, len_frames):
    # params
    first_frame = 0
    min_deg, max_deg = 0, 359

    # select metadata_path (i.e., recording) from metadata_paths with weights based on frame length
    metadata_path = random.choices(metadata_paths, weights=metadata_path_weights)[0]
    metadata_path_idx = metadata_paths.index(metadata_path)
    max_frame = metadata_path_weights[metadata_path_idx]
    # select start_frame
    start_frame = random.randint(first_frame, max_frame - len_frames)
    # select deg
    deg = random.randint(min_deg, max_deg)

    return metadata_path, start_frame, deg


def make_audio_video(start_frame, deg, metadata_path, len_frames, tag_dataset, source_dir, target_dir, bin_make_video):
    # data params
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    sr = 24000
    # input video params
    in_height = 960
    in_width = 1920
    # viewing angle
    out_u_deg = deg
    out_v_deg = 0
    # perspective video params
    out_height = 360
    out_width = 640
    out_u_fov_deg = 100
    out_v_fov_deg = 2 * np.arctan(np.tan((np.pi / 180) * (out_u_fov_deg / 2)) * out_height / out_width) / np.pi * 180

    # get input paths
    wav_path = metadata_path.replace("metadata_", "foa_").replace(".csv", ".wav")
    video_path = metadata_path.replace("metadata_", "video_").replace(".csv", ".mp4")

    # set output paths
    new_wav_path = wav_path.replace(
        "{}/".format(source_dir),
        "{}/{}/".format(target_dir, tag_dataset)
        ).replace(
            "foa_",
            "stereo_"
            ).replace(
                ".wav",
                "_deg{:03}_start{:04}.wav".format(deg, start_frame)
                )
    os.makedirs(os.path.dirname(new_wav_path), exist_ok=True)
    if bin_make_video:
        new_video_path = video_path.replace(
            "{}/".format(source_dir),
            "{}/{}/".format(target_dir, tag_dataset)
            ).replace(
                ".mp4",
                "_deg{:03}_start{:04}.mp4".format(deg, start_frame)
                )
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)

    # make audio file
    audio_data, _ = librosa.load(wav_path, sr=sr, mono=False)
    cut_audio_data = audio_data[:, int(start_frame * 0.1 * sr): int((start_frame + len_frames) * 0.1 * sr)]
    cut_audio_data_W = cut_audio_data[0, :]
    cut_audio_data_Y = cut_audio_data[1, :]
    cut_audio_data_Z = cut_audio_data[2, :]
    cut_audio_data_X = cut_audio_data[3, :]

    cut_audio_len = len(cut_audio_data_W)
    deg_list = np.ones(cut_audio_len) * out_u_deg
    rad_list = deg_list / 180 * np.pi

    cut_audio_data_newX = np.multiply(np.cos(rad_list), cut_audio_data_X) - np.multiply(np.sin(rad_list), cut_audio_data_Y)
    cut_audio_data_newY = np.multiply(np.sin(rad_list), cut_audio_data_X) + np.multiply(np.cos(rad_list), cut_audio_data_Y)

    two_ch_cut_audio_data = np.zeros_like(cut_audio_data[0:2, :])
    two_ch_cut_audio_data[0, :] = cut_audio_data_W + cut_audio_data_newY  # left
    two_ch_cut_audio_data[1, :] = cut_audio_data_W - cut_audio_data_newY  # right
    if np.max(np.abs(two_ch_cut_audio_data)) > 0.9999:
        with open("log_make_audio_video.txt", "a") as f:
            f.write("Not made because of clipping {}: {}\n".format(np.max(np.abs(two_ch_cut_audio_data)), new_wav_path))
        return 1
    else:
        sf.write(new_wav_path, two_ch_cut_audio_data.T, sr)

    if bin_make_video:
        # make video file
        video_only = cv2.VideoCapture(video_path)
        frame_count = int(video_only.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = video_only.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(new_video_path, fmt, frame_rate, (out_width, out_height))

        e2p_fast = E2PFast(in_hw=(in_height, in_width),
                        fov_deg=(out_u_fov_deg, out_v_fov_deg),
                        u_deg=out_u_deg, v_deg=out_v_deg,
                        out_hw=(out_height, out_width))

        # make video from start to start + len
        start_video_frame = int(np.round(start_frame * 0.1 * frame_rate))
        len_video_frames = int(np.round(len_frames * 0.1 * frame_rate))
        video_only.set(cv2.CAP_PROP_POS_FRAMES, start_video_frame)
        for index in range(len_video_frames):
            ret, equi = video_only.read()
            perspective = e2p_fast.e2p(equi)
            # perspective = py360convert.e2p(equi, fov_deg=(out_u_fov_deg, out_v_fov_deg), u_deg=out_u_deg, v_deg=out_v_deg, out_hw=(out_height, out_width))  # same as above but slow, keep as reference
            writer.write(perspective)
        writer.release()
        video_only.release()

    return 0


def main():
    source_dir = "../data_dcase2023_task3"
    target_dir = "../dataset"

    metadata_path_part0 = glob.glob("{}/metadata_dev/dev-train-sony/fold3_room21_mix01[3-9].csv".format(source_dir))
    metadata_path_part1 = glob.glob("{}/metadata_dev/dev-train-sony/fold3_room21_mix02[0-9].csv".format(source_dir))
    metadata_path_part2 = glob.glob("{}/metadata_dev/dev-train-sony/fold3_room22_mix*.csv".format(source_dir))
    metadata_path_part3 = glob.glob("{}/metadata_dev/dev-train-tau/fold3_room*_mix*.csv".format(source_dir))
    metadata_path_part4 = glob.glob("{}/metadata_dev/dev-test-*/*.csv".format(source_dir))
    metadata_paths = sorted(metadata_path_part0 + metadata_path_part1 + metadata_path_part2 + metadata_path_part3 + metadata_path_part4)

    # params
    len_frames = 50  # 50 frames in metadata = 5 seconds
    tag_dataset = "DCASE2025_Task3_Stereo_SELD_Dataset_Repro"
    total_stereo_files = 30000
    random_seed = 20250305
    bin_make_video = True  # if making only audio, set False

    random.seed(random_seed)

    check_on_off_fast = CheckOnOffFast()

    metadata_path_weights = []
    for metadata_path in metadata_paths:
        wav_path = metadata_path.replace("metadata_", "foa_").replace(".csv", ".wav")
        duration_a = librosa.get_duration(path=wav_path, sr=None)
        max_frame_a = math.floor(duration_a * 10)

        if bin_make_video:
            video_path = metadata_path.replace("metadata_", "video_").replace(".csv", ".mp4")
            video = cv2.VideoCapture(video_path)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            duration_v = frame_count / frame_rate
            max_frame_v = math.floor(duration_v * 10)
            video.release()
        else:
            max_frame_v = max_frame_a  # dummy

        max_frame = min(max_frame_a, max_frame_v)
        metadata_path_weights.append(max_frame)

    count_stereo_files = 0
    with tqdm.tqdm(total=total_stereo_files) as pbar:
        while True:
            metadata_path, start_frame, deg = select_random_recording_start_deg(metadata_paths, metadata_path_weights, len_frames)

            metadata_df = pd.read_csv(metadata_path, sep=",", header=None)
            cut_metadata_df = metadata_df[(metadata_df[0] >= start_frame) & (metadata_df[0] < start_frame + len_frames)]
            cut_metadata_df.loc[:, 0] = cut_metadata_df[0] - start_frame

            cut_df_perspective = pd.DataFrame(columns=["frame", "class", "source", "azimuth", "elevation", "distance", "onscreen", "x_on", "y_on"])
            for row in cut_metadata_df.values:
                frame, category, source, azi, ele, dist = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
                is_onscreen, x_on, y_on = check_on_off_fast.check(deg, azi, ele)  # x_on, y_on: (x, y) in the screen of w640 and h360
                cut_df_perspective.loc[len(cut_df_perspective.index)] = [frame, category, source,
                                                                         fold_back_azimuth(set_range_minus180to180(azi + deg)), ele, dist,
                                                                         is_onscreen, x_on, y_on]

            new_metadata_path = metadata_path.replace(
                "{}/".format(source_dir),
                "{}/{}/".format(target_dir, tag_dataset)
                ).replace(
                    ".csv",
                    "_deg{:03}_start{:04}.csv".format(deg, start_frame)
                    )

            if os.path.exists(new_metadata_path):
                pass  # no count
            else:
                ret_make_audio_video = make_audio_video(start_frame, deg, metadata_path, len_frames, tag_dataset, source_dir, target_dir, bin_make_video)
                if ret_make_audio_video == 1:
                    pass  # no count
                elif ret_make_audio_video == 0:
                    os.makedirs(os.path.dirname(new_metadata_path), exist_ok=True)
                    cut_df_perspective_simple = cut_df_perspective[["frame", "class", "source", "azimuth", "distance", "onscreen"]]
                    cut_df_perspective_simple_int = cut_df_perspective_simple.astype(int)
                    cut_df_perspective_simple_int.to_csv(new_metadata_path, sep=',', index=False, header=True)

                    count_stereo_files += 1
                    pbar.update(1)

            if count_stereo_files >= total_stereo_files:
                break


if __name__ == "__main__":
    main()
