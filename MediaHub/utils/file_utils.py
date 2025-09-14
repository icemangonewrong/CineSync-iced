import re
import os
import json
import builtins
import unicodedata
import time
from typing import Tuple, Optional, Dict, List, Set, Union, Any
from functools import lru_cache
from MediaHub.utils.logging_utils import log_message
from MediaHub.config.config import *
from MediaHub.utils.parser.extractor import extract_all_metadata
from MediaHub.utils.parser.parse_anime import is_anime_filename

# Cache for parsed metadata to avoid redundant parsing
_metadata_cache = {}

# Cache for keywords dataz
_keywords_cache = None

# ============================================================================
# ONEDRIVE UPLOAD DETECTION FUNCTIONS
# ============================================================================

def is_file_upload_complete(file_path: str, stability_time: int = 30, max_wait: int = 300) -> bool:
    """
    Check if a file has finished uploading to OneDrive by monitoring size stability.
    
    Args:
        file_path: Path to the file to check
        stability_time: Seconds to wait for size stability (default: 30)
        max_wait: Maximum time to wait for upload completion (default: 300)
        
    Returns:
        True if file appears stable (upload complete), False if still uploading
    """
    if not os.path.exists(file_path):
        return False
        
    try:
        start_time = time.time()
        initial_size = os.path.getsize(file_path)
        
        log_message(f"Checking upload completion for {file_path} (initial size: {initial_size / (1024*1024):.2f}MB)", level="DEBUG")
        
        # Wait for the stability period
        time.sleep(stability_time)
        
        # Check if we've exceeded max wait time
        if time.time() - start_time > max_wait:
            log_message(f"Max wait time ({max_wait}s) exceeded for {file_path}, proceeding with current size", level="WARNING")
            return True
            
        final_size = os.path.getsize(file_path)
        
        # If size hasn't changed, upload is likely complete
        if initial_size == final_size:
            log_message(f"File size stable for {stability_time}s: {file_path} ({final_size / (1024*1024):.2f}MB)", level="DEBUG")
            return True
        else:
            log_message(f"File still uploading: {file_path} (size changed from {initial_size / (1024*1024):.2f}MB to {final_size / (1024*1024):.2f}MB)", level="INFO")
            return False
            
    except (OSError, IOError) as e:
        log_message(f"Error checking upload completion for {file_path}: {e}", level="ERROR")
        return True  # Assume complete on error to avoid infinite waiting

def wait_for_upload_completion(file_path: str, max_attempts: int = 10, stability_time: int = 30) -> bool:
    """
    Wait for a file to complete uploading with multiple retry attempts.
    
    Args:
        file_path: Path to the file to wait for
        max_attempts: Maximum number of attempts to check
        stability_time: Seconds to wait for size stability per attempt
        
    Returns:
        True if upload completed, False if still uploading after max attempts
    """
    for attempt in range(max_attempts):
        if is_file_upload_complete(file_path, stability_time):
            return True
        log_message(f"Upload not complete, attempt {attempt + 1}/{max_attempts}: {file_path}", level="INFO")
        
    log_message(f"File may still be uploading after {max_attempts} attempts: {file_path}", level="WARNING")
    return False

# ============================================================================
# MAIN STRUCTURED PARSING FUNCTIONS
# ============================================================================

def parse_media_file(filename: str) -> Dict[str, Any]:
    """
    Parse a media filename and return comprehensive structured information.
    This is the main function to use for new code.

    Args:
        filename: The filename to parse

    Returns:
        Dictionary with parsed information including:
        - title: Cleaned title
        - year: Release year
        - resolution: Video resolution (1080p, 720p, etc.)
        - quality_source: Source quality (BluRay, WEB-DL, etc.)
        - video_codec: Video codec (x264, x265, etc.)
        - audio_codecs: List of audio codecs
        - audio_channels: List of audio channels
        - release_group: Release group
        - is_dubbed: Whether it's dubbed
        - season: Season number (for TV shows)
        - episode: Episode number (for TV shows)
        - episode_title: Episode title (for TV shows)
        - languages: List of languages
        - is_repack: Whether it's a repack
        - is_anime: Whether it's anime content
        - container: File container format
        - hdr: HDR information
        - is_proper: Whether it's a proper release

    Examples:
        >>> parse_media_file("1923.S02E05.Only.Gunshots.to.GuideUs.1080p.Webrip.10bit.DDP5.1.x265-HODL.mkv")
        {
            "title": "1923",
            "year": None,
            "season": 2,
            "episode": 5,
            "episode_title": "Only Gunshots to GuideUs",
            "resolution": "1080p",
            "quality_source": "Webrip",
            "video_codec": "X265",
            "audio_codecs": ["DDP"],
            "audio_channels": ["5.1"],
            "release_group": "HODL",
            "container": "mkv",
            "is_anime": false
        }
    """
    try:
        # Use the unified parser
        metadata = extract_all_metadata(filename)
        result = metadata.to_dict()

        # Normalize Unicode characters in the title for better API compatibility
        if 'title' in result and result['title']:
            result['title'] = normalize_unicode_characters(result['title'])

        return result
    except Exception as e:
        log_message(f"Error parsing media file '{filename}': {e}", "ERROR")
        return {"title": filename, "error": str(e)}

def parse_media_file_json(filename: str, indent: int = 2) -> str:
    """
    Parse a media filename and return JSON string with structured information.

    Args:
        filename: The filename to parse
        indent: JSON indentation (None for compact)

    Returns:
        JSON string with parsed information
    """
    try:
        result = parse_media_file(filename)
        return json.dumps(result, indent=indent, default=str)
    except Exception as e:
        log_message(f"Error parsing media file to JSON '{filename}': {e}", "ERROR")
        return json.dumps({"title": filename, "error": str(e)}, indent=indent)

# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS (Updated to use structured parser)
# ============================================================================

def extract_year(query: str) -> Optional[int]:
    """Extract year from query string using the unified parser."""
    if not isinstance(query, str):
        return None
    try:
        metadata = extract_all_metadata(query)
        return metadata.year
    except Exception:
        return None

def extract_movie_name_and_year(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract movie name and year from filename using the unified parser."""
    if not isinstance(filename, str) or not filename.strip():
        return None, None
    try:
        metadata = extract_all_metadata(filename)
        title = metadata.title
        year = metadata.year
        return title, str(year) if year else None
    except Exception:
        return None, None

def extract_resolution_from_filename(filename: str) -> Optional[str]:
    """Extract resolution from filename using the unified parser."""
    if not isinstance(filename, str) or not filename.strip():
        return None
    try:
        metadata = extract_all_metadata(filename)
        return metadata.resolution
    except Exception:
        return None

def extract_title(filename: str) -> str:
    """Extract title from filename using the unified parser with Unicode normalization."""
    if not isinstance(filename, str) or not filename.strip():
        return ""
    try:
        metadata = extract_all_metadata(filename)
        title = metadata.title
        if title:
            # Normalize Unicode characters for better API compatibility
            title = normalize_unicode_characters(title)
            return title
        return "Unknown Title"
    except Exception:
        return "Unknown Title"

def _is_clean_title(query: str) -> bool:
    """
    Check if a query string appears to be already a clean title.

    A clean title typically:
    - Contains only letters, numbers, spaces, and basic punctuation
    - Has no file extensions
    - Has no technical terms like resolution, codecs, etc.
    - Has no years in parentheses or brackets
    - Has no dots used as separators (except in abbreviations)
    """
    if not query or not query.strip():
        return False

    query = query.strip()

    # Check for file extensions
    if re.search(r'\.(mkv|mp4|avi|mov|wmv|flv|webm|m4v|mpg|mpeg|ts|m2ts)$', query, re.IGNORECASE):
        return False

    # Check for technical terms that indicate it's a filename
    technical_terms = [
        r'\b\d{3,4}p\b',  # Resolution like 1080p, 720p
        r'\b(BluRay|WEB-DL|WEBRip|HDTV|x264|x265|H264|H265|HEVC|AAC|AC3|DTS)\b',  # Technical terms
        r'\b(MULTI|DUAL|REPACK|PROPER|EXTENDED|UNCUT|DIRECTORS|THEATRICAL)\b',  # Release terms
        r'\b[A-Z]{2,}-[A-Z0-9]+\b',  # Release group patterns like "RARBG", "LOST"
        r'\bS\d{1,2}\.E\d{1,2}\b',  # Season/episode patterns like S01.E01
        r'\bS\d{1,2}E\d{1,2}\b',    # Season/episode patterns like S01E01
        r'\bS\d{1,2}-S\d{1,2}\b',   # Season ranges like S01-S08
        r'\bS\d{1,2}-\d{1,2}\b',    # Season ranges like S1-25, S01-25
        r'\b\d{1,2}-\d{1,2}\b',     # Plain number ranges like 1-25, 01-25 (season ranges)
        r'\b\d{1,2}x\d{1,2}\b',     # Season x Episode patterns like 1x02
        r'\bS\d{1,2}\b',            # Season patterns like S01, S02, S03
        r'\bE\d{1,3}\b',            # Episode patterns like E01, E02
        r'\bSeason\s+\d+\b',        # Season patterns like "Season 1"
        r'\b[Ss]eason\d+\b',         # Season patterns like "season09", "Season09"
        r'\bS\s+\d+\b',             # Season patterns like "S 09"
    ]

    for pattern in technical_terms:
        if re.search(pattern, query, re.IGNORECASE):
            return False

    # Check for years in brackets/parentheses (common in filenames)
    if re.search(r'[\[\(]\d{4}[\]\)]', query):
        return False

    # Check for excessive dots (indicating dot-separated filename format)
    if query.count('.') > 2:  # Allow some dots for abbreviations like "U.S.A."
        return False

    # Check for patterns that look like filename separators
    if re.search(r'[._-]{2,}', query):  # Multiple consecutive separators
        return False

    # If none of the filename indicators are found, it's likely a clean title
    return True

def clean_query(query: str) -> Dict[str, Any]:
    """
    Parse media filename and return comprehensive structured information.

    Returns:
        Dictionary with complete parsing results including:
        - title: Cleaned title
        - year: Release year
        - resolution: Video resolution
        - quality_source: Source quality
        - video_codec: Video codec
        - audio_codecs: Audio codecs
        - audio_channels: Audio channels
        - languages: Languages
        - season: Season number (for TV shows)
        - episode: Episode number (for TV shows)
        - episode_title: Episode title (for TV shows)
        - release_group: Release group
        - And all other attributes from the unified parser
    """
    if not isinstance(query, str):
        log_message(f"Invalid query type: {type(query)}. Expected string.", "ERROR", "stderr")
        return {"title": "", "error": "Invalid input type"}

    # Check cache first to avoid redundant parsing
    if query in _metadata_cache:
        return _metadata_cache[query]

    # Check if the query is already a clean title (no file extensions, technical terms, or complex patterns)
    # If it looks like a clean title, return it as-is to avoid unnecessary parsing
    if _is_clean_title(query):
        log_message(f"Query appears to be already clean, returning as-is: '{query}'", level="DEBUG")
        result = {"title": query, "episodes": [], "seasons": [], "episode_identifier": None}
        _metadata_cache[query] = result
        return result

    try:
        # Use the unified parser
        metadata = extract_all_metadata(query)
        result = metadata.to_dict()

        # Normalize Unicode characters in the title for better API compatibility
        if 'title' in result and result['title']:
            result['title'] = normalize_unicode_characters(result['title'])

        # Add alternative title if available
        if metadata.alternative_title:
            result['alternative_title'] = normalize_unicode_characters(metadata.alternative_title)

        # Add legacy compatibility fields
        result['episodes'] = [metadata.episode] if metadata.episode else []
        result['seasons'] = [metadata.season] if metadata.season else []

        # For episode_identifier, create if we have episode info
        if metadata.episode and metadata.season:
            result['episode_identifier'] = f"S{metadata.season:02d}E{metadata.episode:02d}"
        elif metadata.episode:
            # If we have episode but no season, just use episode number
            result['episode_identifier'] = f"E{metadata.episode:02d}"
        else:
            result['episode_identifier'] = None

        result['show_name'] = result['title'] if metadata.season or metadata.episode else None
        result['create_season_folder'] = bool(metadata.season or metadata.episode)
        result['is_extra'] = False
        result['dubbed'] = metadata.is_dubbed
        result['subbed'] = metadata.is_subbed
        result['repack'] = metadata.is_repack
        result['proper'] = metadata.is_proper
        result['quality'] = metadata.quality_source
        result['codec'] = metadata.video_codec
        result['audio'] = metadata.audio_codecs
        result['channels'] = metadata.audio_channels
        result['group'] = metadata.release_group

        # Add season/episode numbers as strings for compatibility
        if metadata.season:
            result['season_number'] = f"{metadata.season:02d}"

        if metadata.episode:
            result['episode_number'] = f"{metadata.episode:02d}"

        # Reduce logging overhead for performance
        log_message(f"Final parsed result: title='{result.get('title')}', episode='{result.get('episode_identifier')}'", level="DEBUG")

        # Cache the result to avoid redundant parsing
        _metadata_cache[query] = result

        # Limit cache size to prevent memory issues
        if len(_metadata_cache) > 1000:
            # Remove oldest entries (simple FIFO approach)
            oldest_keys = list(_metadata_cache.keys())[:100]
            for key in oldest_keys:
                del _metadata_cache[key]

        return result

    except Exception as e:
        log_message(f"Error using parser for query cleaning: {e}", "ERROR")
        error_result = {"title": query, "error": str(e), "episodes": [], "seasons": []}
        _metadata_cache[query] = error_result
        return error_result

# ============================================================================
# UNICODE AND CHARACTER NORMALIZATION FUNCTIONS
# ============================================================================

def remove_accents(input_str: str) -> str:
    """
    Removes accented characters from a string by normalizing it to NFD form
    and then filtering out combining characters.

    Args:
        input_str: The string from which to remove accents.

    Returns:
        The string with accents removed.
    """
    if not isinstance(input_str, str):
        return input_str

    # Normalize the string to NFD (Normalization Form Canonical Decomposition).
    # This separates base characters from their diacritical marks (accents).
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

def normalize_unicode_characters(text: str) -> str:
    """
    Normalize Unicode characters to their ASCII equivalents for better TMDB matching and logging.

    This function handles special Unicode characters that might appear in filenames
    but cause issues with API searches or console output, such as:
    - Modifier Letter Colon (꞉) -> Regular Colon (:)
    - Various Unicode punctuation -> ASCII equivalents
    - Accented characters -> Base characters (é -> e, ñ -> n)

    Args:
        text: Text containing potentially problematic Unicode characters

    Returns:
        Text with normalized characters, safe for ASCII output
    """
    if not isinstance(text, str):
        return ""

    # Handle None or empty strings gracefully
    if not text:
        return ""

    # First, handle specific problematic characters
    character_replacements = {
        # Colon variants
        '\ua789': ':',  # MODIFIER LETTER COLON -> COLON
        '\u02d0': ':',  # MODIFIER LETTER TRIANGULAR COLON -> COLON
        '\uff1a': ':',  # FULLWIDTH COLON -> COLON
        '\u2236': ':',  # RATIO -> COLON
        '\u2237': ':',  # PROPORTION -> COLON
        '\u205a': ':',  # TWO DOT PUNCTUATION -> COLON
        '\u02f8': ':',  # MODIFIER LETTER RAISED COLON -> COLON

        # Space variants
        '\u2009': ' ',  # THIN SPACE -> REGULAR SPACE
        '\u00a0': ' ',  # NON-BREAKING SPACE -> REGULAR SPACE
        '\u2000': ' ',  # EN QUAD -> REGULAR SPACE
        '\u2001': ' ',  # EM QUAD -> REGULAR SPACE
        '\u2002': ' ',  # EN SPACE -> REGULAR SPACE
        '\u2003': ' ',  # EM SPACE -> REGULAR SPACE
        '\u2004': ' ',  # THREE-PER-EM SPACE -> REGULAR SPACE
        '\u2005': ' ',  # FOUR-PER-EM SPACE -> REGULAR SPACE
        '\u2006': ' ',  # SIX-PER-EM SPACE -> REGULAR SPACE
        '\u2007': ' ',  # FIGURE SPACE -> REGULAR SPACE
        '\u2008': ' ',  # PUNCTUATION SPACE -> REGULAR SPACE
        '\u200a': ' ',  # HAIR SPACE -> REGULAR SPACE
        '\u202f': ' ',  # NARROW NO-BREAK SPACE -> REGULAR SPACE
        '\u205f': ' ',  # MEDIUM MATHEMATICAL SPACE -> REGULAR SPACE

        # Dash/hyphen variants
        '\u2013': '-',  # EN DASH -> HYPHEN
        '\u2014': '-',  # EM DASH -> HYPHEN
        '\u2015': '-',  # HORIZONTAL BAR -> HYPHEN
        '\u2212': '-',  # MINUS SIGN -> HYPHEN
        '\u2010': '-',  # HYPHEN -> HYPHEN-MINUS
        '\u2011': '-',  # NON-BREAKING HYPHEN -> HYPHEN

        # Quote variants
        '\u2018': "'",  # LEFT SINGLE QUOTATION MARK -> APOSTROPHE
        '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK -> APOSTROPHE
        '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK -> QUOTATION MARK
        '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK -> QUOTATION MARK
        '\u2032': "'",  # PRIME -> APOSTROPHE
        '\u2033': '"',  # DOUBLE PRIME -> QUOTATION MARK

        # Fraction characters
        '\u00bc': '1/4',  # FRACTION ONE QUARTER -> 1/4
        '\u00bd': '1/2',  # FRACTION ONE HALF -> 1/2
        '\u00be': '3/4',  # FRACTION THREE QUARTERS -> 3/4
        '\u2153': '1/3',  # FRACTION ONE THIRD -> 1/3
        '\u2154': '2/3',  # FRACTION TWO THIRDS -> 2/3
        '\u2155': '1/5',  # FRACTION ONE FIFTH -> 1/5
        '\u2156': '2/5',  # FRACTION TWO FIFTHS -> 2/5
        '\u2157': '3/5',  # FRACTION THREE FIFTHS -> 3/5
        '\u2158': '4/5',  # FRACTION FOUR FIFTHS -> 4/5
        '\u2159': '1/6',  # FRACTION ONE SIXTH -> 1/6
        '\u215a': '5/6',  # FRACTION FIVE SIXTHS -> 5/6
        '\u215b': '1/8',  # FRACTION ONE EIGHTH -> 1/8
        '\u215c': '3/8',  # FRACTION THREE EIGHTHS -> 3/8
        '\u215d': '5/8',  # FRACTION FIVE EIGHTHS -> 5/8
        '\u215e': '7/8',  # FRACTION SEVEN EIGHTHS -> 7/8
        '\u2150': '1/7',  # FRACTION ONE SEVENTH -> 1/7
        '\u2151': '1/9',  # FRACTION ONE NINTH -> 1/9
        '\u2152': '1/10', # FRACTION ONE TENTH -> 1/10
        '\u2189': '0/3',  # FRACTION ZERO THIRDS -> 0/3

        # Other common problematic characters
        '\u2026': '...',  # HORIZONTAL ELLIPSIS -> THREE DOTS
        '\u00b7': '.',    # MIDDLE DOT -> PERIOD
        '\u2022': '*',    # BULLET -> ASTERISK
        '\u00d7': 'x',    # MULTIPLICATION SIGN -> x
    }

    # Apply specific character replacements with proper spacing for fractions
    for unicode_char, replacement in character_replacements.items():
        if unicode_char in ['\u00bc', '\u00bd', '\u00be', '\u2153', '\u2154', '\u2155',
                           '\u2156', '\u2157', '\u2158', '\u2159', '\u215a', '\u215b',
                           '\u215c', '\u215d', '\u215e', '\u2150', '\u2151', '\u2152', '\u2189']:
            # For fraction characters, add space before if preceded by a digit
            import re
            pattern = r'(\d)(' + re.escape(unicode_char) + r')'
            text = re.sub(pattern, r'\1 ' + replacement, text)
            # Handle any remaining fraction characters without preceding digits
            text = text.replace(unicode_char, replacement)
        else:
            text = text.replace(unicode_char, replacement)

    # Use the improved remove_accents function for accent removal
    text = remove_accents(text)

    # Final cleanup: ensure we have clean ASCII
    try:
        # Try to encode as ASCII to catch any remaining problematic characters
        text.encode('ascii')
        return text
    except UnicodeEncodeError:
        # If there are still non-ASCII characters, use transliteration
        # This is a more aggressive approach for stubborn characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    # Remove empty parentheses that result from removing non-ASCII content
    import re
    text = re.sub(r'\(\s*\)', '', text)  # Remove empty parentheses
    text = re.sub(r'\[\s*\]', '', text)  # Remove empty brackets
    text = re.sub(r'\{\s*\}', '', text)  # Remove empty braces
    text = re.sub(r'\s+', ' ', text)     # Clean up multiple spaces
    text = text.strip()                  # Remove leading/trailing spaces

    return text

# ============================================================================
# SYMLINK RESOLUTION FUNCTIONS
# ============================================================================

def resolve_symlink_to_source(file_path: str) -> str:
    """
    Resolve a symlinked file to its actual source path.

    Args:
        file_path: Path to the file (may be a symlink)

    Returns:
        The actual source path if it's a symlink, otherwise the original path
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return file_path

    try:
        # Check if the path exists and is a symlink
        if os.path.islink(file_path):
            # Resolve the symlink to get the actual source path
            resolved_path = os.path.realpath(file_path)
            log_message(f"Resolved symlink: {file_path} -> {resolved_path}", level="DEBUG")
            return resolved_path
        else:
            # Not a symlink, return original path
            return file_path
    except (OSError, IOError) as e:
        log_message(f"Error resolving symlink {file_path}: {e}", level="WARNING")
        return file_path

def get_source_directory_from_symlink(file_path: str) -> str:
    """
    Get the source directory from a symlinked file.

    Args:
        file_path: Path to the file (may be a symlink)

    Returns:
        The source directory path if it's a symlink, otherwise the directory of the original path
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return ""

    try:
        # Resolve the symlink to get the actual source path
        resolved_path = resolve_symlink_to_source(file_path)

        # Get the directory of the resolved path
        source_dir = os.path.dirname(resolved_path)

        if os.path.islink(file_path):
            log_message(f"Source directory for symlink {file_path}: {source_dir}", level="DEBUG")

        return source_dir
    except Exception as e:
        log_message(f"Error getting source directory from {file_path}: {e}", level="WARNING")
        return os.path.dirname(file_path) if file_path else ""

# ============================================================================
# UTILITY FUNCTIONS (Keep these as they are utility functions)
# ============================================================================

def sanitize_windows_filename(filename: str) -> str:
    """Sanitize a filename to be compatible with Windows filesystem."""
    if not isinstance(filename, str):
        return ""

    if not filename.strip():
        return "sanitized_filename"

    # Windows filename restrictions: \ / : * ? " < > |
    replacements = {
        ':': ' -', '/': '-', '\\': '-', '*': 'x', '?': '',
        '"': "'", '<': '(', '>': ')', '|': '-'
    }

    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    filename = re.sub(r'[\\/:*?"<>|]', '', filename)
    filename = filename.strip(' .')

    if not filename:
        filename = "sanitized_filename"

    return filename

def normalize_query(query: str) -> str:
    """Normalize query string for comparison purposes."""
    if not isinstance(query, str):
        log_message(f"Invalid query type: {type(query)}. Expected string.", "ERROR", "stderr")
        return ""

    normalized_query = re.sub(r'[._-]', ' ', query)
    normalized_query = re.sub(r'[^\w\s\(\)-]', '', normalized_query)
    normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()

    return normalized_query

def should_skip_processing(filename: str) -> bool:
    """
    Determine if a file should be skipped from MediaHub processing.
    Returns True if the file should be skipped (metadata files)
    """
    if not isinstance(filename, str):
        return False

    # Skip only metadata files - allow .srt and .strm to be processed
    return filename.lower().endswith(('.sub', '.idx', '.vtt'))

def _load_keywords():
    """Load keywords from keywords.json file."""
    global _keywords_cache
    if _keywords_cache is not None:
        return _keywords_cache

    try:
        keywords_path = os.path.join(os.path.dirname(__file__), 'keywords.json')
        with open(keywords_path, 'r', encoding='utf-8') as f:
            _keywords_cache = json.load(f)
            return _keywords_cache
    except Exception as e:
        log_message(f"Error loading keywords from keywords.json: {e}", level="ERROR")
        _keywords_cache = {'extras_patterns': []}
        return _keywords_cache

def _is_extras_by_name(filename: str, file_path: str = None) -> bool:
    """
    Check if a file should be considered an extra based on its name patterns.

    Args:
        filename: The filename to check
        file_path: Optional full file path to also check

    Returns:
        bool: True if the file matches extras patterns
    """
    try:
        keywords_data = _load_keywords()
        extras_patterns = keywords_data.get('extras_patterns', [])

        # Check filename
        filename_lower = filename.lower()
        for pattern in extras_patterns:
            if pattern.lower() in filename_lower:
                return True

        if file_path:
            file_path_lower = file_path.lower()
            for pattern in extras_patterns:
                if pattern.lower() in file_path_lower:
                    return True

        return False

    except Exception as e:
        log_message(f"Error checking extras patterns: {e}", level="ERROR")
        return False

def is_extras_file(file: str, file_path: str, is_movie: bool = False) -> bool:
    """
    Determine if the file is an extra based on size limits and name patterns.

    Args:
        file: Filename to check
        file_path: Full path to the file
        is_movie: True if processing movie files, False for show files

    Returns:
        bool: True if file should be skipped based on size limits or name patterns
    """
    if not isinstance(file, str):
