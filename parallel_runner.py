#!/usr/bin/env python3
"""Parallel runner to compare serial vs multiprocessing subtitle translation.

Usage:
    python parallel_runner.py reference_1.srt -l Kannada --workers 10 --chunk-size 100

This script does NOT modify `subtitle_generator.py`. It imports `SubtitleGenerator` and
processes chunks in serial and in parallel (ProcessPoolExecutor) to compare timings.
"""
import argparse
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

# Import the existing generator (must be in PYTHONPATH / same folder)
from subtitle_generator import SubtitleGenerator


def _process_chunk_worker(chunk_text: str, target_language: str, project_id: str = None) -> List[Dict[str, str]]:
    """Worker function to run inside each process. Creates its own SubtitleGenerator.

    Returns list of translations (ordered) for the chunk.
    """
    gen = SubtitleGenerator(project_id=project_id, target_language=target_language)
    # Use the internal retry processing for chunk text
    translations = gen._process_chunk_with_retry(chunk_text, chunk_num=0, max_depth=2, depth=0)
    return translations


def run_serial(subtitle_chunks: List[List[Dict[str, str]]], target_language: str, project_id: str = None) -> List[Dict[str, str]]:
    start = time.time()
    all_translations = []
    for i, chunk in enumerate(subtitle_chunks, 1):
        print(f"[serial] Processing chunk {i}/{len(subtitle_chunks)} ({len(chunk)} lines) ...")
        chunk_text = "\n".join([s['text'] for s in chunk])
        translations = _process_chunk_worker(chunk_text, target_language, project_id)
        all_translations.extend(translations)
    elapsed = time.time() - start
    return all_translations, elapsed


def run_parallel(subtitle_chunks: List[List[Dict[str, str]]], target_language: str, workers: int = 10, project_id: str = None) -> List[Dict[str, str]]:
    start = time.time()
    all_translations_by_index = [None] * len(subtitle_chunks)  # preserve chunk order

    with ProcessPoolExecutor(max_workers=workers) as exec:
        futures = {}
        for i, chunk in enumerate(subtitle_chunks):
            chunk_text = "\n".join([s['text'] for s in chunk])
            fut = exec.submit(_process_chunk_worker, chunk_text, target_language, project_id)
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                translations = fut.result()
            except Exception as e:
                print(f"[parallel] Chunk {idx+1} failed: {e}")
                raise
            all_translations_by_index[idx] = translations
            print(f"[parallel] Chunk {idx+1} done (received {len(translations)} translations)")

    # Flatten preserving original chunk order
    all_translations = []
    for chunk_list in all_translations_by_index:
        all_translations.extend(chunk_list)

    elapsed = time.time() - start
    return all_translations, elapsed


def write_output_srt(output_path: str, original_subtitles: List[Dict[str, str]], translations: List[Dict[str, str]], target_language: str):
    # Use SubtitleGenerator helper to format SRT
    gen = SubtitleGenerator(project_id=None, target_language=target_language)
    srt_content = gen.create_srt_output(original_subtitles, translations)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)


def main():
    parser = argparse.ArgumentParser(description='Compare serial vs parallel translation performance')
    parser.add_argument('input_file')
    parser.add_argument('-l', '--language', default='Hindi')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--project-id', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return

    # Parse original SRT
    gen = SubtitleGenerator(project_id=args.project_id, target_language=args.language)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    original_subtitles = gen._extract_srt_with_timecodes(srt_content)

    subtitle_chunks = gen.split_subtitles_into_chunks(original_subtitles, chunk_size=args.chunk_size)
    print(f"Found {len(original_subtitles)} subtitles -> {len(subtitle_chunks)} chunk(s) (chunk size {args.chunk_size})")

    # Serial
    serial_translations, serial_time = run_serial(subtitle_chunks, args.language, args.project_id)
    serial_out = os.path.splitext(args.input_file)[0] + f"_serial_{args.chunk_size}.srt"
    write_output_srt(serial_out, original_subtitles, serial_translations, args.language)
    print(f"Serial run time: {serial_time:.1f}s -> output: {serial_out}")

    # Parallel
    parallel_translations, parallel_time = run_parallel(subtitle_chunks, args.language, workers=args.workers, project_id=args.project_id)
    parallel_out = os.path.splitext(args.input_file)[0] + f"_parallel_w{args.workers}_{args.chunk_size}.srt"
    write_output_srt(parallel_out, original_subtitles, parallel_translations, args.language)
    print(f"Parallel run time (workers={args.workers}): {parallel_time:.1f}s -> output: {parallel_out}")

    print(f"Speedup: {serial_time / parallel_time:.2f}x")


if __name__ == '__main__':
    main()
