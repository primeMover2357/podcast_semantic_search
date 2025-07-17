import whisper
import os
import glob
import argparse

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_with_timestamps(mp3_path, output_dir):
    model = whisper.load_model("small")  # Use 'small', 'medium', or 'large' for better accuracy (slower)
    result = model.transcribe(mp3_path, verbose=True)  # verbose=True for timestamps

    # Group segments into ~200-word chunks
    chunks = []
    current_chunk = ""
    current_start = None
    word_count = 0
    for segment in result['segments']:
        start = segment['start']  # Start time in seconds
        text = segment['text'].strip()
        if not current_start:
            current_start = start
        current_chunk += f" {text}"
        word_count += len(text.split())
        if word_count > 200:  # Chunk size
            chunks.append({
                'timestamp': current_start,
                'text': current_chunk.strip()
            })
            current_chunk = ""
            current_start = None
            word_count = 0
    if current_chunk:
        chunks.append({
            'timestamp': current_start,
            'text': current_chunk.strip()
        })

    # Save to txt with [HH:MM:SS] format
    base_name = os.path.basename(mp3_path).replace('.mp3', '.txt')
    output_path = os.path.join(output_dir, base_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            timestamp_hms = seconds_to_hms(chunk['timestamp'])
            f.write(f"[{timestamp_hms}] {chunk['text']}\n\n")
    
    print(f"Transcript saved: {output_path}")
    print(f"Generated {len(chunks)} chunks with timestamps:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{seconds_to_hms(chunk['timestamp'])}] {chunk['text'][:50]}...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir', default='podcasts', help='Directory with MP3 files')
    parser.add_argument('--output_dir', default='transcripts', help='Output directory for TXT')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for mp3 in glob.glob(os.path.join(args.mp3_dir, '*.mp3')):
        transcribe_with_timestamps(mp3, args.output_dir)