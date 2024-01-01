import cv2
import argparse

def simple_video_test(input_path, output_path, target_fps):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_skip = int(original_fps / target_fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            out.write(frame)
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample FPS of a video file without changing its duration.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the processed video.')
    parser.add_argument('--fps', type=int, required=True, help='Target frames per second for the output video.')
    args = parser.parse_args()

    simple_video_test(args.input, args.output, args.fps)
