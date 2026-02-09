#!/usr/bin/env python3
"""
Subtitle Generator with Translation
Processes transcripts using Vertex AI LLM to generate well-formatted subtitles with translations.
"""

import os
import re
import argparse
import time
from typing import List, Tuple, Dict
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel


class SubtitleGenerator:
    """Generate and format subtitles from transcript with translations."""

    # Word pairs that should not be split
    WORD_PAIRS = [
        r'\b(I|he|she|it|we|they|you)\s+(am|is|are|was|were|be|been|have|has|had|do|does|did|will|would|should|could|can|may|might)\b',
        r'\b(is|was|will|should|could|would|can|may)\s+(being|going|coming|doing|making|taking|giving)\b',
        r'\b(has|have|had)\s+(been|got|made|taken|given|done|come|gone)\b',
        r'\b(the|a|an)\s+\w+\b',
        r'\b(in|on|at|by|for|with|about|as|of|to|from)\s+\w+\b',
        r'\b(and|but|or|nor|yet|so|because|though|although)\s+\w+\b',
        r'\b\w+\s+(is|was|are|were|be|being|been)\s+(being|coming|doing|making|taking|giving|going)\b',
    ]

    def __init__(self, project_id: str = None, location: str = "us-central1", target_language: str = "Hindi"):
        """
        Initialize the subtitle generator.
        
        Args:
            project_id: GCP project ID
            location: Vertex AI location
            target_language: Target language for translation
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.target_language = target_language
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel("gemini-2.0-flash")
        
    def read_transcript(self, file_path: str) -> str:
        """Read transcript from file (txt or srt)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If it's an SRT file, extract only the text content
        if file_path.lower().endswith('.srt'):
            content = self._extract_text_from_srt(content)
        
        return content

    def _extract_srt_with_timecodes(self, srt_content: str) -> List[Dict[str, str]]:
        """Extract SRT subtitles with original timecodes preserved.
        
        Returns list of dicts with keys: 'start', 'end', 'text'
        """
        subtitles = []
        lines = srt_content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
            
            # Try to find sequence number (should be digits only)
            if re.match(r'^\d+$', lines[i].strip()):
                i += 1
                
                # Next line should be timecode
                if i < len(lines) and '-->' in lines[i]:
                    timecode_line = lines[i].strip()
                    parts = timecode_line.split('-->')
                    start = parts[0].strip() if len(parts) > 0 else '00:00:00,000'
                    end = parts[1].strip() if len(parts) > 1 else '00:00:05,000'
                    i += 1
                    
                    # Collect text lines until we hit an empty line
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if text_lines:
                        subtitles.append({
                            'start': start,
                            'end': end,
                            'text': ' '.join(text_lines)
                        })
                else:
                    i += 1
            else:
                i += 1
        
        return subtitles

    def _extract_text_from_srt(self, srt_content: str) -> str:
        """Extract ALL text content from SRT format for chunking."""
        subtitles = self._extract_srt_with_timecodes(srt_content)
        return ' '.join([s['text'] for s in subtitles])
    
    def split_subtitles_into_chunks(self, subtitles: List[Dict[str, str]], chunk_size: int = 100) -> List[List[Dict[str, str]]]:
        """Split list of subtitles into chunks by count (not by character length).
        
        Args:
            subtitles: List of subtitle dicts {start, end, text}
            chunk_size: Number of subtitles per chunk
            
        Returns:
            List of subtitle chunks
        """
        chunks = []
        for i in range(0, len(subtitles), chunk_size):
            chunks.append(subtitles[i:i+chunk_size])
        return chunks

    def split_into_chunks(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks of approximately chunk_size characters, breaking at sentence boundaries."""
        chunks = []
        current_chunk = ""
        sentences = text.split('. ')
        
        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ('. ' if i < len(sentences) - 1 else '')
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ('. ' if i < len(sentences) - 1 else '')
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _escape_unescaped_newlines_in_json_strings(self, s: str) -> str:
        """Escape literal newlines inside JSON string literals so json.loads can parse them.

        This walks the text and replaces raw newline characters occurring while inside
        a double-quoted string with the two-character sequence \n, while preserving
        existing escaped sequences and other JSON structure.
        """
        out = []
        in_string = False
        escape = False

        for ch in s:
            if in_string:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == '\\':
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                    out.append(ch)
                    continue
                if ch == '\n' or ch == '\r':
                    out.append('\\n')
                    continue
                out.append(ch)
            else:
                out.append(ch)
                if ch == '"':
                    in_string = True

        return ''.join(out)

    def can_split_here(self, text: str, position: int) -> bool:
        """Check if we can safely split at this position without breaking word pairs."""
        if position <= 0 or position >= len(text):
            return False
        
        # Get context around the split position
        start = max(0, position - 30)
        end = min(len(text), position + 30)
        context = text[start:end]
        
        # Check if any word pair pattern would be broken
        for pattern in self.WORD_PAIRS:
            matches = list(re.finditer(pattern, context, re.IGNORECASE))
            for match in matches:
                match_start = start + match.start()
                match_end = start + match.end()
                # If our split position is within a word pair, don't split
                if match_start < position < match_end:
                    return False
        
        return True

    def split_subtitle(self, text: str, max_chars: int = 42) -> List[str]:
        """
        Split text into subtitle lines respecting formatting rules.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per line
            
        Returns:
            List of subtitle lines (max 2)
        """
        text = text.strip()
        
        # If text fits in one line
        if len(text) <= max_chars:
            return [text]
        
        # Try to split at natural points
        lines = []
        
        # First try: split at space near max_chars
        space_positions = [i for i, c in enumerate(text) if c == ' ']
        
        best_split = None
        for pos in space_positions:
            if pos <= max_chars and self.can_split_here(text, pos):
                if best_split is None or abs(pos - max_chars) < abs(best_split - max_chars):
                    best_split = pos
        
        if best_split and len(lines) < 2:
            first_line = text[:best_split].strip()
            remaining = text[best_split:].strip()
            
            if len(first_line) <= max_chars:
                lines.append(first_line)
                
                # Second line
                if len(remaining) <= max_chars:
                    lines.append(remaining)
                else:
                    # If remaining is still too long, try to split it
                    second_best = None
                    for pos in space_positions:
                        if pos > best_split and pos - best_split <= max_chars:
                            if self.can_split_here(text, pos):
                                second_best = pos
                    
                    if second_best:
                        lines.append(text[best_split:second_best].strip())
                    else:
                        # Fallback: just truncate with ellipsis
                        lines.append(remaining[:max_chars-3] + "...")
        else:
            # Fallback: simple split
            words = text.split()
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    if len(lines) < 2:
                        current_line = word
                    else:
                        # Max 2 lines reached
                        break
            
            if current_line and len(lines) < 2:
                lines.append(current_line)
        
        # Ensure max 2 lines
        return lines[:2]

    def generate_subtitles_and_translations(self, transcript: str) -> List[Dict[str, str]]:
        """
        Generate translations for transcript using Vertex AI.
        Transcript should be newline-separated lines to translate.
        
        Args:
            transcript: Newline-separated text lines to translate
            
        Returns:
            List of subtitle dictionaries with original and translated text
        """
        # Split transcript into lines and add explicit indices to avoid LLM confusion
        lines = transcript.strip().split('\n')
        
        # Create indexed text for clarity
        indexed_text = '\n'.join([f"[{i+1}] {line}" for i, line in enumerate(lines)])
        
        # Prepare the prompt for LLM
        prompt = f"""You are a professional translator. Translate the following English text into {self.target_language}.
For each numbered line, provide a JSON entry with matching index, "original" (English) and "translated" ({self.target_language}) fields.

CRITICAL RULES:
- Return ONLY valid JSON array
- Match line indices EXACTLY: [1] maps to output index 1, [2] to index 2, etc.
- Do NOT reorder, merge, or combine lines
- Each input line [N] MUST have exactly one output entry with "index": N
- Do NOT skip any lines
- Maintain exact ordering and count

Format your response as:
[
  {{"index": 1, "original": "line 1 text", "translated": "line 1 translation"}},
  {{"index": 2, "original": "line 2 text", "translated": "line 2 translation"}}
]

Text to translate (note the [N] indices - PRESERVE THEM):
{indexed_text}

Generate translations now. Return ONLY the JSON array, nothing else."""

        print("📝 Generating translations with Vertex AI (fresh model)...")

        # Instantiate a fresh model for each call to avoid retained conversation/context
        model = GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        # Try to extract the actual generated text from the response
        response_text = None
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    part = candidate.content.parts[0]
                    response_text = getattr(part, 'text', None) or str(candidate)
        except Exception:
            response_text = None

        if not response_text:
            # Fallbacks
            response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Remove markdown code block markers if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        if response_text:
            import json
            
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)

                    # Attempt to repair common issues (literal newlines in strings)
                    try:
                        subtitles = json.loads(json_str)
                    except json.JSONDecodeError:
                        repaired = self._escape_unescaped_newlines_in_json_strings(json_str)
                        subtitles = json.loads(repaired)
                    
                    # Validate response and sort by index
                    if isinstance(subtitles, list) and len(subtitles) > 0:
                        # Sort by index to ensure correct order
                        if all('index' in s for s in subtitles):
                            subtitles = sorted(subtitles, key=lambda x: x.get('index', 0))
                        
                        print(f"✅ Generated {len(subtitles)} translations")
                        return subtitles
                    else:
                        print("⚠️ Response parsed but no translations generated")
                        raise ValueError("Empty translation list from LLM")
                else:
                    # If regex fails, try parsing the entire response
                    print("⚠️ Could not find JSON array, trying full response...")
                    subtitles = json.loads(response_text.strip())
                    if isinstance(subtitles, list) and len(subtitles) > 0:
                        # Sort by index if available
                        if all('index' in s for s in subtitles):
                            subtitles = sorted(subtitles, key=lambda x: x.get('index', 0))
                        print(f"✅ Generated {len(subtitles)} translations")
                        return subtitles
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error: {e}")
                print(f"Response preview: {response_text[:500]}")
                raise ValueError(f"Failed to parse JSON from LLM response: {e}")
        
        raise ValueError("No response text received from LLM")

    def create_srt_output(self, original_subtitles: List[Dict[str, str]], translations: List[Dict[str, str]]) -> str:
        """
        Create SRT format output by mapping translations to original subtitles.
        
        Args:
            original_subtitles: Original subtitles with timecodes {start, end, text}
            translations: Translations {original, translated}
            
        Returns:
            SRT formatted string with original timecodes and translations
        """
        srt_content = ""
        sequence_number = 1
        
        # Map translations to original subtitles by index
        for i, orig in enumerate(original_subtitles):
            start_time = orig['start']
            end_time = orig['end']
            original_text = orig['text']
            
            # Get corresponding translation if available
            translated_text = ""
            if i < len(translations):
                translated_text = translations[i].get('translated', '')
            
            srt_content += f"{sequence_number}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{original_text}\n"
            srt_content += f"[{self.target_language}]: {translated_text}\n"
            srt_content += "\n"
            
            sequence_number += 1
        
        return srt_content

    def _format_timecode(self, seconds: float) -> str:
        """Format seconds to SRT timecode format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _process_chunk_with_retry(self, chunk: str, chunk_num: int = 0, max_depth: int = 2, depth: int = 0) -> List[Dict[str, str]]:
        """
        Process a chunk, and if it fails, recursively split it in half and retry.
        
        Args:
            chunk: The chunk text to process
            chunk_num: Chunk number (for logging)
            max_depth: Maximum recursion depth (to avoid infinite recursion)
            depth: Current recursion depth
            
        Returns:
            List of subtitle dictionaries for this chunk
        """
        try:
            return self.generate_subtitles_and_translations(chunk)
        except Exception as e:
            # If we've hit max depth, re-raise the error
            if depth >= max_depth:
                raise ValueError(f"Chunk {chunk_num} failed after {max_depth} recursive splits: {e}")
            
            # Split the chunk in half and retry each half
            mid_point = len(chunk) // 2
            
            # Find a good split point (at a sentence boundary)
            for i in range(mid_point - 100, mid_point + 100):
                if i > 0 and i < len(chunk) and chunk[i:i+2] == '. ':
                    mid_point = i + 2
                    break
            
            print(f"  ⚠️ Chunk {chunk_num} failed, splitting into 2 sub-chunks and retrying...")
            
            chunk1 = chunk[:mid_point].strip()
            chunk2 = chunk[mid_point:].strip()
            
            results = []
            
            # Process first half recursively
            if chunk1:
                results.extend(self._process_chunk_with_retry(chunk1, chunk_num, max_depth, depth + 1))
            
            # Process second half recursively
            if chunk2:
                results.extend(self._process_chunk_with_retry(chunk2, chunk_num, max_depth, depth + 1))
            
            return results

    def process(self, input_file: str, output_file: str = None, target_language: str = None) -> str:
        """
        Process SRT file and generate subtitle file with translations.
        Chunks by subtitle count (e.g., 100 subtitles per chunk) to preserve structure.
        
        Args:
            input_file: Path to input SRT file
            output_file: Path to output SRT file (optional)
            target_language: Target language for translation (optional)
            
        Returns:
            Path to generated SRT file
        """
        start_time = time.time()
        
        # Update target language if provided
        if target_language:
            self.target_language = target_language
        
        # Set default output file
        if not output_file:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_subtitles.srt"
        
        print(f"📖 Reading SRT file: {input_file}")
        
        # Parse original SRT with timecodes
        with open(input_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        original_subtitles = self._extract_srt_with_timecodes(srt_content)
        
        print(f"📊 Found {len(original_subtitles)} subtitle(s)")
        print(f"🌐 Target language: {self.target_language}")
        
        # Split subtitles into chunks by count (100 per chunk)
        subtitle_chunks = self.split_subtitles_into_chunks(original_subtitles, chunk_size=100)
        print(f"📦 Processing {len(subtitle_chunks)} chunk(s)...")
        
        # Collect all translations in order
        all_translations = []
        
        for i, chunk in enumerate(subtitle_chunks, 1):
            print(f"  Processing chunk {i}/{len(subtitle_chunks)} ({len(chunk)} subtitle(s))...")
            
            # Create text input: each subtitle on a new line
            chunk_text = '\n'.join([s['text'] for s in chunk])
            
            try:
                # Get translations for this chunk
                chunk_translations = self._process_chunk_with_retry(chunk_text, chunk_num=i, max_depth=2, depth=0)
                all_translations.extend(chunk_translations)
            except Exception as e:
                print(f"  ❌ Chunk {i} failed: {e}")
                raise
        
        # Create output with original timecodes + translations
        srt_output = self.create_srt_output(original_subtitles, all_translations)
        
        print(f"💾 Writing subtitles to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(srt_output)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"✨ Subtitles translated successfully!")
        print(f"⏱️  Total processing time: {minutes}m {seconds}s")
        return str(output_file)


def main():
    """CLI interface for subtitle generator."""
    parser = argparse.ArgumentParser(
        description='Generate formatted subtitles with translations using Vertex AI'
    )
    parser.add_argument(
        'input_file',
        help='Input transcript file (.txt or .srt)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output SRT file path (optional)',
        default=None
    )
    parser.add_argument(
        '--language',
        '-l',
        help='Target language for translation (default: Hindi)',
        default='Hindi'
    )
    parser.add_argument(
        '--project-id',
        help='GCP Project ID (uses GOOGLE_CLOUD_PROJECT env var if not provided)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found")
        return
    
    # Create generator and process
    generator = SubtitleGenerator(
        project_id=args.project_id,
        target_language=args.language
    )
    
    output_file = generator.process(
        input_file=args.input_file,
        output_file=args.output,
        target_language=args.language
    )
    
    print(f"\n✅ Output file: {output_file}")


if __name__ == "__main__":
    main()
