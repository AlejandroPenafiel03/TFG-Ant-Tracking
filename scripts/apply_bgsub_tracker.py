from bgsub_ant_tracker import analize_video
import os
import argparse
import yaml

def list_videos(folder):
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
    files = []
    for fname in os.listdir(folder):
        _, ext = os.path.splitext(fname)
        if ext in VIDEO_EXTS:
            files.append(os.path.join(folder, fname))
    return sorted(files)

def run_script(videos_dir, output_dir, skip_corrupted_frames):
    videos = list_videos(videos_dir)
    if not videos:
        print("No se encontraron vídeos en:", videos_dir)
        return
    with open(os.path.join(output_dir, "ant_flow.csv"), "a", encoding="utf-8") as f:
        f.write("video_label;up_count;down_count\n")
    for video_path in videos:
        try:
            up_count, down_count, detections = analize_video(video_path, verbose=True, skip_corrupted_frames=skip_corrupted_frames)
            with open(os.path.join(output_dir, "ant_flow.csv"), "a", encoding="utf-8") as f:
                f.write(os.path.basename(video_path) + f";{up_count};{down_count}\n")
            with open(os.path.join(output_dir, "detections_"+ os.path.basename(video_path).split(".")[0] +".yaml"), "w") as f:
                yaml.dump(detections, f)
        except Exception as e:
            print(f"Error al procesar {video_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Carpeta con los videos a analizar")
    parser.add_argument("--output_dir", type=str, default="./", help="Carpeta donde se guardarán los resultados")
    parser.add_argument("--skip_corrupted_frames", type=int, default=30, help="Número máximo de frames corruptos consecutivos a saltar antes de abortar el análisis del video")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_script(args.input_dir, args.output_dir, args.skip_corrupted_frames)