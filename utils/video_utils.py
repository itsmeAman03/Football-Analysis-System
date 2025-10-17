import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames,width,height,fps


def save_video(output_video_frames,width,height,fps,output_video_path): 

    # Define codec and create VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames: 
        out.write(frame)
    out.release()

if __name__ == '__main__':
    print("MAIN")