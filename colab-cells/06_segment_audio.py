"""
Cell 06: Speech Segmentation using VAD
Mục đích:
  - Detect speech segments using Silero VAD
  - Loại bỏ im lặng
  - Chia audio thành clips 3-10 giây
"""

# ============================================================================
# CELL 06: SPEECH SEGMENTATION (VAD)
# ============================================================================

print("🔊 Speech Segmentation using Silero VAD...")

import os
import sys
import json
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Use venv
venv_python = "/content/venv/bin/python"
sys.path.insert(0, '/content')

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Get audio files (separated or original)
if config.get('separated', False):
    audio_files = {Path(p): Path(v) for p, v in config['separated_files'].items()}
    print("✅ Using separated vocals")
else:
    audio_files = {Path(p): Path(p) for p in config['audio_files']}
    print("✅ Using original audio files")

speakers = config['speakers']

# ------------------------------------------------------------------------------
# 0. Check for existing backup in Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for existing segments backup in Drive...")
print("="*70)

segments_backup_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/segments_backup"
config_backup_path = "/content/drive/MyDrive/F5TTS_Vietnamese/segments_config_backup.json"
use_backup = False

if os.path.exists(segments_backup_dir) and os.path.exists(config_backup_path):
    print(f"✅ Found backup in Drive: {segments_backup_dir}")
    
    # Check backup contents
    import shutil
    backup_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(segments_backup_dir)
        for filename in filenames
    ) / (1024 * 1024)  # MB
    
    num_backup_files = sum(
        len(filenames)
        for dirpath, dirnames, filenames in os.walk(segments_backup_dir)
    )
    
    print(f"📊 Backup info:")
    print(f"   Files: {num_backup_files}")
    print(f"   Size: {backup_size:.1f} MB")
    print()
    print("="*70)
    print("💬 Bạn muốn sử dụng dữ liệu backup hay chạy lại?")
    print("="*70)
    print("   1. Sử dụng backup từ Drive (nhanh, tiết kiệm thời gian)")
    print("   2. Chạy lại từ đầu (sẽ ghi đè backup cũ)")
    print("="*70)
    
    user_choice = input("Lựa chọn của bạn (1/2, mặc định=1): ").strip()
    
    if user_choice == "2":
        print("\n✅ Sẽ chạy lại từ đầu và tạo backup mới...")
        use_backup = False
    else:
        print("\n✅ Sử dụng backup từ Drive...")
        use_backup = True
else:
    print("📝 Không tìm thấy backup. Sẽ xử lý từ đầu...")
    use_backup = False

# ------------------------------------------------------------------------------
# Restore from backup if user chose to
# ------------------------------------------------------------------------------
if use_backup:
    print("\n" + "="*70)
    print("📦 Restoring segments from Drive backup...")
    print("="*70)
    
    import shutil
    
    # Create local segments directory
    segments_dir = "/content/processed/segments"
    os.makedirs(segments_dir, exist_ok=True)
    
    # Copy all files from backup to local
    print("⏳ Copying files...")
    if os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
    shutil.copytree(segments_backup_dir, segments_dir)
    
    # Load config from backup
    with open(config_backup_path, 'r') as f:
        backup_config = json.load(f)
    
    # Update current config with backup data
    config['segments_dir'] = segments_dir
    config['all_segments'] = backup_config.get('all_segments', {})
    config['extracted_segments'] = backup_config.get('extracted_segments', [])
    config['total_segments'] = backup_config.get('total_segments', 0)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Restored {config['total_segments']} segments from backup")
    print(f"📁 Location: {segments_dir}")
    
    # Display statistics
    print("\n" + "="*70)
    print("📊 Restored Data Summary:")
    print("="*70)
    
    speaker_stats = {}
    for seg in config['extracted_segments']:
        speaker = seg['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0, 'duration': 0}
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['duration'] += seg['duration']
    
    for speaker, stats in speaker_stats.items():
        print(f"\n   {speaker}:")
        print(f"      Segments: {stats['count']}")
        print(f"      Duration: {stats['duration'] / 60:.1f} minutes")
    
    print("\n" + "="*70)
    print("✅ RESTORE COMPLETE! Skipping to next cell...")
    print("="*70)
    print(f"📝 Next Steps:")
    print(f"   Run Cell 07 to transcribe these segments")
    print(f"   Or skip to Cell 08 if you already have transcriptions")
    print("="*70)
    
    # Skip the rest of this cell
    import sys
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. Load Silero VAD Model
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📥 Loading Silero VAD Model...")
print("="*70)

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)

get_speech_timestamps, _, read_audio, _, _ = utils

print("✅ Silero VAD loaded!")

# ------------------------------------------------------------------------------
# 2. Chunked Processing Function (for large files)
# ------------------------------------------------------------------------------

def process_audio_chunked(audio_path, model, get_speech_timestamps, chunk_duration=300):
    """
    Process large audio file in chunks for VAD to avoid memory issues
    
    Args:
        audio_path: Path to audio file
        model: Silero VAD model
        get_speech_timestamps: VAD function
        chunk_duration: Duration per chunk in seconds (default: 5 minutes)
    
    Returns:
        List of speech segments with timestamps in seconds
    """
    # Get file info WITHOUT loading full file
    info = torchaudio.info(str(audio_path))
    sr = info.sample_rate
    total_frames = info.num_frames
    total_duration = total_frames / sr
    
    chunk_frames = int(chunk_duration * sr)
    all_segments = []
    
    # Calculate number of chunks
    num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
    
    print(f"  Duration: {total_duration/60:.1f} minutes")
    print(f"  Processing in {num_chunks} chunks ({chunk_duration/60:.1f} min each)...")
    
    # Process each chunk with progress bar
    for chunk_idx in tqdm(range(num_chunks), desc=f"  Processing chunks", leave=False):
        start_frame = chunk_idx * chunk_frames
        end_frame = min(start_frame + chunk_frames, total_frames)
        
        # Load only this chunk
        wav, chunk_sr = torchaudio.load(
            str(audio_path),
            frame_offset=start_frame,
            num_frames=end_frame - start_frame
        )
        
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz for VAD
        if chunk_sr != 16000:
            resampler = torchaudio.transforms.Resample(chunk_sr, 16000)
            wav = resampler(wav)
            chunk_sr = 16000
        
        # VAD on chunk
        speech_timestamps = get_speech_timestamps(
            wav.squeeze(),
            model,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=2000,
            min_silence_duration_ms=500,
            window_size_samples=512
        )
        
        # Adjust timestamps to global position
        time_offset = start_frame / sr
        
        for ts in speech_timestamps:
            all_segments.append({
                'start': ts['start'] / 16000 + time_offset,
                'end': ts['end'] / 16000 + time_offset
            })
    
    return all_segments

# ------------------------------------------------------------------------------
# 3. Detect Speech Segments
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Detecting speech segments...")
print("="*70)

all_segments = {}
total_segments = 0

for original_path, audio_path in tqdm(audio_files.items(), desc="Processing files"):
    speaker_name = speakers[str(original_path)]['name']
    
    print(f"\n📁 Processing: {audio_path.name} (Speaker: {speaker_name})")
    
    try:
        # Check file size to decide processing method
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        file_duration = torchaudio.info(str(audio_path)).num_frames / torchaudio.info(str(audio_path)).sample_rate
        
        # Use chunked processing for files > 50MB or > 10 minutes
        if file_size_mb > 50 or file_duration > 600:
            print(f"  Large file detected ({file_size_mb:.1f} MB, {file_duration/60:.1f} min)")
            print(f"  Using chunked processing to save memory...")
            
            # Process in chunks
            speech_timestamps = process_audio_chunked(
                audio_path,
                model,
                get_speech_timestamps,
                chunk_duration=300  # 5 minutes per chunk
            )
        else:
            # Small file: use direct processing
            print(f"  Small file ({file_size_mb:.1f} MB), processing directly...")
            wav = read_audio(str(audio_path), sampling_rate=16000)
            
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                sampling_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=2000,
                min_silence_duration_ms=500,
                window_size_samples=512
            )
            
            # Convert to same format as chunked processing
            speech_timestamps = [
                {
                    'start': ts['start'] / 16000,
                    'end': ts['end'] / 16000
                }
                for ts in speech_timestamps
            ]
        
        print(f"  Found {len(speech_timestamps)} speech segments")
        
        # Smart filtering and splitting
        # - Accept segments: 1-30 seconds
        # - Segments > 10s: split into smaller chunks with overlap to reduce noise
        MIN_DURATION = 1.0
        MAX_DURATION = 30.0
        SPLIT_THRESHOLD = 10.0  # Split segments longer than this
        TARGET_CHUNK_SIZE = 8.0  # Target size for split chunks
        OVERLAP = 0.5  # 0.5s overlap to avoid cutting words
        
        segments = []
        split_count = 0
        
        for ts in speech_timestamps:
            duration = ts['end'] - ts['start']
            
            # Skip too short or too long
            if duration < MIN_DURATION:
                continue
            if duration > MAX_DURATION:
                continue
            
            # If segment is short enough, keep as-is
            if duration <= SPLIT_THRESHOLD:
                segments.append({
                    'start': ts['start'],
                    'end': ts['end'],
                    'duration': duration
                })
            else:
                # Split long segment into smaller chunks with overlap
                # This reduces noise and prevents quality degradation
                num_chunks = int(duration / TARGET_CHUNK_SIZE) + 1
                chunk_size = duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = ts['start'] + (i * chunk_size)
                    chunk_end = min(ts['start'] + ((i + 1) * chunk_size) + OVERLAP, ts['end'])
                    chunk_duration = chunk_end - chunk_start
                    
                    if chunk_duration >= MIN_DURATION:
                        segments.append({
                            'start': chunk_start,
                            'end': chunk_end,
                            'duration': chunk_duration
                        })
                        split_count += 1
        
        original_count = len(speech_timestamps)
        filtered_count = len([ts for ts in speech_timestamps if MIN_DURATION <= (ts['end'] - ts['start']) <= MAX_DURATION])
        final_count = len(segments)
        
        print(f"  After filtering ({MIN_DURATION}-{MAX_DURATION}s): {filtered_count} segments")
        if split_count > 0:
            print(f"  Split {split_count} long segments (>{SPLIT_THRESHOLD}s) into smaller chunks")
        print(f"  Final segments: {final_count}")
        
        # Calculate retention rate
        original_duration = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        kept_duration = sum(s['duration'] for s in segments)
        retention_rate = kept_duration / original_duration if original_duration > 0 else 0
        
        if retention_rate < 0.5:
            print(f"  ⚠️  WARNING: Low retention rate {retention_rate*100:.1f}%")
            print(f"     Original: {original_duration/60:.1f} min → Kept: {kept_duration/60:.1f} min")
            print(f"     💡 Consider adjusting MIN_DURATION or MAX_DURATION")
        
        all_segments[str(audio_path)] = {
            'speaker': speaker_name,
            'original_file': str(original_path),
            'audio_file': str(audio_path),
            'segments': segments,
            'total_duration': sum(s['duration'] for s in segments)
        }
        
        total_segments += len(segments)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ CRITICAL ERROR in VAD processing!")
        print(f"{'='*70}")
        print(f"File: {audio_path}")
        print(f"Speaker: {speaker_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 This error prevents proper data preparation.")
        print(f"   Please fix the issue and re-run this cell.")
        print(f"{'='*70}")
        sys.exit(1)  # Stop execution on critical error

# ------------------------------------------------------------------------------
# 4. Extract and Save Segments
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✂️  Extracting audio segments...")
print("="*70)

segments_dir = "/content/processed/segments"
os.makedirs(segments_dir, exist_ok=True)

extracted_segments = []

for audio_path, info in tqdm(all_segments.items(), desc="Extracting segments"):
    speaker_name = info['speaker']
    segments = info['segments']
    audio_file_path = Path(info['audio_file'])
    
    # Create speaker directory
    speaker_dir = os.path.join(segments_dir, speaker_name, "wavs")
    os.makedirs(speaker_dir, exist_ok=True)
    
    if not segments:
        print(f"  ⚠️  No segments to extract for {audio_file_path.name}")
        continue
    
    print(f"\n  📁 Extracting {len(segments)} segments from {audio_file_path.name}...")
    
    # Get original sample rate WITHOUT loading full file
    info = torchaudio.info(str(audio_file_path))
    original_sr = info.sample_rate
    target_sr = 24000
    
    print(f"  Sample rate: {original_sr} Hz → {target_sr} Hz")
    
    # Extract each segment individually (load only what we need)
    for i, seg in enumerate(tqdm(segments, desc=f"    Extracting", leave=False)):
        try:
            # Calculate frame positions at original sample rate
            start_frame = int(seg['start'] * original_sr)
            end_frame = int(seg['end'] * original_sr)
            num_frames = end_frame - start_frame
            
            # Load ONLY this segment from file
            segment_wav, segment_sr = torchaudio.load(
                str(audio_file_path),
                frame_offset=start_frame,
                num_frames=num_frames
            )
            
            # Ensure mono
            if segment_wav.shape[0] > 1:
                segment_wav = segment_wav.mean(dim=0, keepdim=True)
            
            # Resample to 24kHz if needed
            if segment_sr != target_sr:
                resampler = torchaudio.transforms.Resample(segment_sr, target_sr)
                segment_wav = resampler(segment_wav)
            
            # Convert to numpy for soundfile
            segment_audio = segment_wav.squeeze().numpy()
            
            # Save segment
            segment_filename = f"{audio_file_path.stem}_seg{i:04d}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            
            sf.write(segment_path, segment_audio, target_sr)
            
            extracted_segments.append({
                'path': segment_path,
                'speaker': speaker_name,
                'duration': seg['duration'],
                'original_file': str(audio_path)
            })
            
        except Exception as e:
            print(f"    ⚠️  Warning: Failed to extract segment {i}: {e}")
            # Non-critical: continue with other segments
            continue

print(f"\n✅ Extracted {len(extracted_segments)} segments")

# Critical validation: Check if we have any segments
if len(extracted_segments) == 0:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: No segments extracted!")
    print(f"{'='*70}")
    print(f"⚠️  Possible causes:")
    print(f"   1. VAD filter too strict (try adjusting MIN/MAX_DURATION)")
    print(f"   2. Audio has no speech detected")
    print(f"   3. Audio format not supported")
    print(f"   4. All segments were too short or too long")
    print(f"\n💡 Cannot proceed without segments. Please investigate.")
    print(f"{'='*70}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 5. Update Configuration
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Saving configuration...")
print("="*70)

config['segments_dir'] = segments_dir
config['all_segments'] = all_segments
config['extracted_segments'] = extracted_segments
config['total_segments'] = len(extracted_segments)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Backup main config to Drive (separate from segments backup)
drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
with open(drive_config, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Configuration saved")

# ------------------------------------------------------------------------------
# 6. Display Statistics
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ SEGMENTATION COMPLETE!")
print("="*70)

# Calculate statistics by speaker
speaker_stats = {}
for seg in extracted_segments:
    speaker = seg['speaker']
    if speaker not in speaker_stats:
        speaker_stats[speaker] = {
            'count': 0,
            'total_duration': 0,
            'segments': []
        }
    speaker_stats[speaker]['count'] += 1
    speaker_stats[speaker]['total_duration'] += seg['duration']
    speaker_stats[speaker]['segments'].append(seg)

print(f"""
📊 Overall Statistics:
   Total Segments: {len(extracted_segments)}
   Total Duration: {sum(s['duration'] for s in extracted_segments) / 60:.1f} minutes
   
👥 Per Speaker:
""")

# Calculate overall retention rate
total_input_duration = 0
total_output_duration = sum(s['duration'] for s in extracted_segments)

for audio_path, info in all_segments.items():
    file_duration = torchaudio.info(info['audio_file']).num_frames / torchaudio.info(info['audio_file']).sample_rate
    total_input_duration += file_duration

overall_retention = total_output_duration / total_input_duration if total_input_duration > 0 else 0

for speaker, stats in speaker_stats.items():
    print(f"\n   {speaker}:")
    print(f"      Segments: {stats['count']}")
    print(f"      Duration: {stats['total_duration'] / 60:.1f} minutes")
    print(f"      Avg length: {stats['total_duration'] / stats['count']:.1f} seconds")

print(f"\n{'='*70}")
print(f"📊 DATA QUALITY CHECK")
print(f"{'='*70}")
print(f"Input audio: {total_input_duration/60:.1f} minutes")
print(f"Output segments: {total_output_duration/60:.1f} minutes")
print(f"Retention rate: {overall_retention*100:.1f}%")

if overall_retention < 0.3:
    print(f"\n{'='*70}")
    print(f"⚠️  CRITICAL WARNING: Very low retention rate!")
    print(f"{'='*70}")
    print(f"   Expected: 60-80% retention for good quality")
    print(f"   Got: {overall_retention*100:.1f}%")
    print(f"   You're losing {(1-overall_retention)*100:.1f}% of your data!")
    print(f"\n   💡 Recommendations:")
    print(f"      1. Check VAD parameters (MIN_DURATION, MAX_DURATION)")
    print(f"      2. Verify audio quality (should have clear speech)")
    print(f"      3. Consider using different audio files")
    print(f"{'='*70}")
    
    proceed = input("\nContinue with this low retention rate? (y/n, default=n): ").strip().lower()
    if proceed != 'y':
        print("\nStopping. Please adjust parameters and re-run.")
        sys.exit(1)
elif overall_retention < 0.5:
    print(f"\n⚠️  WARNING: Low retention rate ({overall_retention*100:.1f}%)")
    print(f"   Expected: 60-80% for optimal results")
    print(f"   💡 Consider adjusting VAD filter parameters")
else:
    print(f"\n✅ Retention rate looks good!")

print(f"""
📁 Output Directory:
   {segments_dir}
   
📝 File Structure:
   segments/
   ├── speaker_name/
   │   └── wavs/
   │       ├── audio_seg0001.wav
   │       ├── audio_seg0002.wav
   │       └── ...

🎧 Sample Segment:
""")

# Play first segment
if extracted_segments:
    sample_seg = extracted_segments[0]
    print(f"   Speaker: {sample_seg['speaker']}")
    print(f"   Duration: {sample_seg['duration']:.1f}s")
    from IPython.display import Audio, display
    display(Audio(sample_seg['path']))

# ------------------------------------------------------------------------------
# 7. Backup to Drive
# ------------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"💾 Backing up segments to Drive...")
print(f"{'='*70}")

import shutil

segments_backup_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/segments_backup"
config_backup_path = "/content/drive/MyDrive/F5TTS_Vietnamese/segments_config_backup.json"

# Remove old backup if exists
if os.path.exists(segments_backup_dir):
    print("⏳ Removing old backup...")
    shutil.rmtree(segments_backup_dir)

# Copy segments to Drive
print("⏳ Copying segments to Drive...")
shutil.copytree(segments_dir, segments_backup_dir)

# Save config backup
backup_config = {
    'segments_dir': segments_dir,
    'all_segments': all_segments,
    'extracted_segments': extracted_segments,
    'total_segments': len(extracted_segments)
}

with open(config_backup_path, 'w') as f:
    json.dump(backup_config, f, indent=2)

# Calculate backup size
backup_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk(segments_backup_dir)
    for filename in filenames
) / (1024 * 1024)  # MB

print(f"✅ Backup complete!")
print(f"   Location: {segments_backup_dir}")
print(f"   Files: {len(extracted_segments)}")
print(f"   Size: {backup_size:.1f} MB")

print(f"\n{'='*70}")
print(f"📝 Next Steps:")
print(f"   Run Cell 07 to transcribe these segments")
print(f"   Or skip to Cell 08 if you already have transcriptions")
print(f"{'='*70}")
print(f"🎉 Ready to proceed to Cell 07!")
print(f"{'='*70}")



