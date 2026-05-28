import orbslam3
import argparse
import os
import cv2
import logging

logger = logging.getLogger("ORB-SLAM3")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Single directory, rgb frames are named frame0000.jpg, depth frames are named depth0000.png (replica)
def get_frame_paths_single_dir(dataset_path):
    file_names = sorted(os.listdir(dataset_path))
    file_paths = [os.path.join(dataset_path, f) for f in file_names]
    mid = len(file_paths) // 2
    rgb_paths = file_paths[mid:]
    depth_paths = file_paths[:mid]
    return rgb_paths, depth_paths


# Separate directories, rgb frames are in rgb/0000.jpg, depth frames are in depth/0000.png (realsense)
def get_frame_paths_separate_dirs(dataset_path):
    rgb_directory = os.path.join(dataset_path, 'rgb')
    depth_directory = os.path.join(dataset_path, 'depth')
    rgb_file_names = sorted(os.listdir(rgb_directory))
    depth_file_names = sorted(os.listdir(depth_directory))
    rgb_paths = [os.path.join(rgb_directory, f) for f in rgb_file_names]
    depth_paths = [os.path.join(depth_directory, f) for f in depth_file_names]
    return rgb_paths, depth_paths


def write_full_trajectory(traj, out_path):
    with open(out_path, 'w') as f:
        for arr in traj:
            flattened = ' '.join(map(str, arr.flatten()))
            f.write(flattened + '\n')


def write_keyframe_trajectory(times, poses, out_path):
    with open(out_path, 'w') as f:
        for t, p in zip(times, poses):
            pose_str = ' '.join(map(str, p.flatten()))
            f.write(f"{t} {pose_str}\n")


def main():
    p = argparse.ArgumentParser(description='ORB-SLAM3 RGBD')
    p.add_argument('--vocab', default='./third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt', help='Path to ORB vocabulary file')
    p.add_argument('--settings', default='./settings/replica.yaml' , help='Path to ORB-SLAM3 settings file')
    p.add_argument('--dataset', default='/mnt/data/Replica/room0/results', help='Path to dataset or folder')
    p.add_argument('--data-layout', choices=['single', 'separate'], default='single')
    p.add_argument('--use-viewer', action='store_true', help='Enable ORB-SLAM3 viewer')
    p.add_argument('--keyframes-only', action='store_true', help='Save keyframe trajectory only')
    p.add_argument('--output', help='Output file path (overrides defaults)')
    args = p.parse_args()

    # Get frame paths
    dataset = args.dataset
    if args.data_layout == 'single':
        rgb_frames, depth_frames = get_frame_paths_single_dir(dataset)
    else:
        rgb_frames, depth_frames = get_frame_paths_separate_dirs(dataset)

    if not rgb_frames or not depth_frames:
        logger.error('No frames found (rgb: %d, depth: %d)', len(rgb_frames), len(depth_frames))
        return

    if len(rgb_frames) != len(depth_frames):
        logger.warning('Counts differ (rgb: %d, depth: %d)', len(rgb_frames), len(depth_frames))

    # Initialize SLAM
    slam = orbslam3.system(args.vocab, args.settings, orbslam3.Sensor.RGBD)
    slam.set_use_viewer(bool(args.use_viewer))
    slam.initialize()
    logger.info('SLAM initialized')

    # Process frames
    for i, (r, d) in enumerate(zip(rgb_frames[1000:], depth_frames[1000:])):
        rgb = cv2.imread(r)
        depth = cv2.imread(d, -1)
        timestamp = float(i)
        slam.process_image_rgbd(rgb, depth, timestamp)

        # print(f'Keyframes: {slam.get_all_keyframe_times()} ')
        # print(f'Keyframe Map IDs: {slam.get_all_keyframe_map_ids()} ')

        # # Poll and print any map events produced by the C++ core
        # try:
        #     events = slam.get_map_events()
        #     if events:
        #         logger.info('Map events at frame %d: %s', i, events)
        # except Exception:
        #     logger.exception('Failed to poll map events')

        # # Trigger a dataset change halfway through to exercise multi-session APIs
        # if i == len(rgb_frames) // 2:
        #     print(f'Keyframes: {slam.get_all_keyframe_times()} ')
        #     print(f'Len KFs: {len(slam.get_all_keyframe_times())} ')
        #     print(f'Keyframe Map IDs: {slam.get_all_keyframe_map_ids()} ')
        #     logger.info('Calling change_dataset() at frame %d', i)
        #     try:

        #         slam.change_dataset()
        #         # Optionally reset the active map after dataset change
        #         slam.reset_active_map()
        #     except Exception:
        #         logger.exception('Change/reset dataset failed')

        #     print(f'Keyframes: {slam.get_all_keyframe_times()} ')
        #     print(f'Keyframe Map IDs: {slam.get_all_keyframe_map_ids()} ')

    # Wait for viewer to close if enabled
    if args.use_viewer:
        try:
            while not slam.viewer_should_quit():
                pass
        except KeyboardInterrupt:
            logger.info('Viewer interrupted, shutting down')

    # Shutdown SLAM
    slam.shutdown()
    while not slam.is_shutdown():
        pass

    # After shutdown, poll any remaining map events (and show that atlas was saved if configured)
    try:
        final_events = slam.get_map_events()
        if final_events:
            logger.info('Final map events after shutdown: %s', final_events)
    except Exception:
        logger.exception('Failed to retrieve final map events')

    # print(f'Keyframes: {slam.get_all_keyframe_times()} ')
    # print(f'Keyframe Map IDs: {slam.get_all_keyframe_map_ids()} ')

    # Save trajectory
    # If --output is provided, use it. Otherwise, default to dataset-based paths depending on keyframes-only flag
    out_path = args.output
    if not out_path:
        if args.keyframes_only:
            out_path = os.path.join(dataset, 'traj_keyframes.txt')
        else:
            out_path = os.path.join(dataset, 'traj_full.txt')

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.keyframes_only:
        try:
            times = slam.get_all_keyframe_times()
            poses = slam.get_all_keyframe_poses()
            if len(times) > 0 and len(poses) > 0:
                write_keyframe_trajectory(times, poses, out_path)
                logger.info(f'Wrote keyframe trajectory with {len(times)} keyframes to {out_path}')
            else:
                logger.info('No keyframe trajectory available')
        except Exception:
            logger.exception('Failed to retrieve/write keyframe trajectory')
    else:
        try:
            logger.info('Retrieving full trajectory')
            traj = slam.get_active_frame_poses()
            logger.info(f'Full trajectory retrieved with {len(traj)} frames')
            if len(traj) > 0:
                write_full_trajectory(traj, out_path)
                logger.info(f'Wrote full trajectory with {len(traj)} frames to {out_path}')
            else:
                logger.info('No full trajectory available')
        except Exception:
            logger.exception('Failed to retrieve/write full trajectory')


if __name__ == '__main__':
    main()
