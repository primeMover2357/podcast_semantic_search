import whisper
import os
import argparse

def transcribe_with_timestamps(mp3_path, output_dir):
    model = whisper.load_model("base")  # Use 'small', 'medium', or 'large' for better accuracy (slower)
    result = model.transcribe(mp3_path, verbose=True)  # verbose=True for timestamps

    # Format with timestamps (group into ~200-word chunks for search project)
    chunks = []
    current_chunk = ""
    current_start = None
    for segment in result['segments']:
        start = f"[{segment['start']:.2f}]"  # Format as seconds; convert to HH:MM:SS if needed
        text = segment['text'].strip()
        if not current_start:
            current_start = start
        current_chunk += f" {text}"
        if len(current_chunk.split()) > 200:  # Chunk size
            chunks.append(f"{current_start} {current_chunk.strip()}")
            current_chunk = ""
            current_start = None
    if current_chunk:
        chunks.append(f"{current_start} {current_chunk.strip()}")

    # Save to txt
    base_name = os.path.basename(mp3_path).replace('.mp3', '.txt')
    output_path = os.path.join(output_dir, base_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(chunks))  # Double newline separator
    print(f"Transcript saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir', default='podcasts', help='Directory with MP3 files')
    parser.add_argument('--output_dir', default='transcripts', help='Output directory for TXT')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for mp3 in glob.glob(os.path.join(args.mp3_dir, '*.mp3')):
        transcribe_with_timestamps(mp3, args.output_dir)