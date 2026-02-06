import cv2
import argparse
import os
import sys
from pathlib import Path
from anonymizer import FaceAnonymizer
from tqdm import tqdm

def process_image(input_path, output_path, anonymizer, mode, intensity):
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    processed_frame, faces_count = anonymizer.process_frame(frame, mode, intensity)
    
    cv2.imwrite(str(output_path), processed_frame)
    print(f"Processed {input_path}: {faces_count} faces found. Saved to {output_path}")

def process_video(input_path, output_path, anonymizer, mode, intensity):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0  # Fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try different codecs (mp4v is standard, avc1 is often better for browsers)
    codecs = ['mp4v', 'XVID', 'MJPG']
    out = None
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if out.isOpened():
            break
        out.release()
    
    if not out or not out.isOpened():
        print(f"Error: Could not initialize VideoWriter with any of {codecs} for {output_path}")
        cap.release()
        return

    print(f"Processing video: {input_path}")
    # Use tqdm if total_frames is valid, else just a plain loop
    if total_frames > 0:
        pbar = tqdm(total_frames=total_frames, desc="Frames")
    else:
        pbar = None
        print("Note: Video length unknown, processing all frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, _ = anonymizer.process_frame(frame, mode, intensity)
        out.write(processed_frame)
        if pbar: pbar.update(1)

    if pbar: pbar.close()
    cap.release()
    out.release()
    print(f"Finished processing video. Saved to {output_path}")

def process_webcam(anonymizer, mode, intensity):
    # Try multiple camera indices and backends
    # CAP_DSHOW is often faster/required on Windows for some webcams
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        # Fallback to default
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam (tried index 0)")
        return

    current_mode = mode
    print("\nWebcam Mode Active")
    print("Controls:")
    print("  'q' - Quit")
    print("  'b' - Switch to Blur")
    print("  'o' - Switch to Blur Oval")
    print("  'p' - Switch to Pixelate")
    print("  'k' - Switch to Blackout")
    print(f"Current Mode: {current_mode}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            break
        
        processed_frame, count = anonymizer.process_frame(frame, current_mode, intensity)
        
        # UI Overlay
        # Rectangle background for text for better visibility
        cv2.rectangle(processed_frame, (0, 0), (250, 80), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"Faces: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Mode: {current_mode}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Advanced Face Anonymizer', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            current_mode = 'blur'
            print("Switched to: blur")
        elif key == ord('o'):
            current_mode = 'blur_oval'
            print("Switched to: blur_oval")
        elif key == ord('p'):
            current_mode = 'pixelate'
            print("Switched to: pixelate")
        elif key == ord('k'):
            current_mode = 'blackout'
            print("Switched to: blackout")

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Advanced Face Anonymizer - Blur, Pixelate or Blackout faces.")
    parser.add_argument("-i", "--input", help="Path to input image, video, or folder. Use 'webcam' for live mode.")
    parser.add_argument("-o", "--output", help="Path to output file or folder.")
    parser.add_argument("-m", "--mode", choices=['blur', 'blur_oval', 'pixelate', 'blackout'], default='blur', help="Anonymization mode.")
    parser.add_argument("-s", "--intensity", type=int, default=99, help="Intensity of the effect (default 99).")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum detection confidence (0-1).")

    args = parser.parse_args()

    # If no input, show help
    if not args.input and not (len(sys.argv) > 1):
        parser.print_help()
        # Fallback for default image if it exists
        default_img = Path('data/TestFaceImage.jpg')
        if default_img.exists():
            print(f"\nNo input specified. Running on default test image: {default_img}")
            args.input = str(default_img)
        else:
            return

    anonymizer = FaceAnonymizer(min_detection_confidence=args.confidence)

    if args.input.lower() == 'webcam':
        process_webcam(anonymizer, args.mode, args.intensity)
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist.")
        return

    # Handle output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate default output name
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_anonymized{input_path.suffix}"
        else:
            output_path = input_path.parent / f"{input_path.name}_anonymized"

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)
        
        files = list(input_path.glob('*'))
        extensions = ('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')
        valid_files = [f for f in files if f.suffix.lower() in extensions]
        
        print(f"Processing directory: {input_path}")
        for file in tqdm(valid_files, desc="Processing Files"):
            file_output = output_path / f"{file.stem}_anonymized{file.suffix}"
            if file.suffix.lower() in ('.mp4', '.avi', '.mov'):
                process_video(file, file_output, anonymizer, args.mode, args.intensity)
            else:
                process_image(file, file_output, anonymizer, args.mode, args.intensity)
    
    elif input_path.is_file():
        if input_path.suffix.lower() in ('.mp4', '.avi', '.mov'):
            process_video(input_path, output_path, anonymizer, args.mode, args.intensity)
        else:
            process_image(input_path, output_path, anonymizer, args.mode, args.intensity)

if __name__ == "__main__":
    main()