import os
import argparse
import threading
import time
import sys
import traceback
import signal
from MediaHub.utils.logging_utils import log_message
from MediaHub.config.config import *
from MediaHub.monitor.polling_monitor import start_polling_monitor
from MediaHub.processors.symlink_creator import create_symlinks
from MediaHub.utils.rclone_utils import wait_for_mount, check_rclone_mount, is_rclone_mount_enabled
from MediaHub.processors.process_db import *
from MediaHub.utils.global_events import terminate_flag, set_shutdown, terminate_subprocesses
from MediaHub.utils.webdav_api import start_webdav_server
from MediaHub.utils.plex_utils import check_dashboard_availability

LOCK_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'app.lock')

def parse_season_episode(season_episode):
    """Parse season and episode from string in format SxxExx or similar."""
    if not season_episode:
        return None, None
    season_episode = season_episode.replace(' ', '').lower()
    match = re.search(r's(\d{1,2})e(\d{1,2})', season_episode, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def remove_lock_file():
    """Remove the lock file if it exists."""
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
            log_message("Lock file removed successfully", level="INFO")
        except Exception as e:
            log_message(f"Error removing lock file: {e}", level="ERROR")

def initialize_db_with_mount_check():
    """Initialize database with mount check."""
    try:
        if is_rclone_mount_enabled() and not check_rclone_mount():
            wait_for_mount()
        initialize_file_database()
        return True
    except Exception as e:
        log_message(f"Error initializing database: {e}", level="ERROR")
        return False

def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM gracefully."""
    log_message("Received interrupt signal, initiating shutdown...", level="WARNING")
    set_shutdown()
    terminate_subprocesses()
    remove_lock_file()
    sys.exit(0)

def main(dest_dir):
    parser = argparse.ArgumentParser(description="Create symlinks for files from src_dirs in dest_dir.")
    parser.add_argument("--auto-select", action="store_true", help="Automatically chooses the first option without prompting the user")
    parser.add_argument("--use-source-db", action="store_true", help="Use source files database to find unprocessed files (faster for auto mode)")
    parser.add_argument("single_path", nargs="?", help="Single path to process instead of using SOURCE_DIRS from environment variables")
    parser.add_argument("--force", action="store_true", help="Force recreate symlinks even if they already exist")
    parser.add_argument("--force-show", action="store_true", help="Force process file as a TV show regardless of naming pattern")
    parser.add_argument("--force-movie", action="store_true", help="Force process file as a movie regardless of naming pattern")
    parser.add_argument("--force-extra", action="store_true", help="Force an extra file to be considered as a Movie/Show")
    parser.add_argument("--disable-monitor", action="store_true", help="Disable polling monitor and symlink cleanup processes")
    parser.add_argument("--monitor-only", action="store_true", help="Start only the polling monitor without processing existing files")
    parser.add_argument("--imdb", type=str, help="Direct IMDb ID for the show")
    parser.add_argument("--tmdb", type=int, help="Direct TMDb ID for the show")
    parser.add_argument("--tvdb", type=int, help="Direct TVDb ID for the show")
    parser.add_argument("--season-episode", type=str, help="Specify season and episode numbers in format SxxExx (e.g., S03E15)")
    parser.add_argument("--skip", action="store_true", help="Skip processing the file and mark it as 'Skipped by user' in the database")
    parser.add_argument("--batch-apply", action="store_true", help="Apply the first manual selection to all subsequent files in the batch")
    parser.add_argument("--manual-search", action="store_true", help="Allow manual TMDB search when automatic search fails")

    db_group = parser.add_argument_group('Database Management')
    db_group.add_argument("--reset", action="store_true",
                         help="Reset the database to its initial state")
    db_group.add_argument("--status", action="store_true",
                         help="Display database statistics")
    db_group.add_argument("--vacuum", action="store_true",
                         help="Perform database vacuum to optimize storage and performance")
    db_group.add_argument("--verify", action="store_true",
                         help="Verify database integrity and check for corruption")
    db_group.add_argument("--export", metavar="FILE",
                         help="Export database contents to a CSV file")
    db_group.add_argument("--import", metavar="FILE", dest="import_file",
                         help="Import database contents from a CSV file")
    db_group.add_argument("--search", metavar="PATTERN",
                         help="Search for files in database matching the given pattern")
    db_group.add_argument("--optimize", action="store_true",
                         help="Optimize database indexes and analyze tables")
    db_group.add_argument("--update-database", action="store_true",
                         help="Update database entries using TMDB API calls")

    args = parser.parse_args()

    # Parse season and episode numbers if provided
    season_number, episode_number = parse_season_episode(args.season_episode)

    # Resolve symlink if single_path is provided
    if args.single_path:
        original_path = args.single_path
        resolved_path = resolve_symlink_to_source(args.single_path)
        if resolved_path != original_path:
            log_message(f"Resolved symlink path: {original_path} -> {resolved_path}", level="INFO")
            args.single_path = resolved_path

    # Ensure --force-show and --force-movie aren't used together
    if args.force_show and args.force_movie:
        log_message("Error: Cannot use --force-show and --force-movie together", level="ERROR")
        exit(1)

    if args.vacuum:
        vacuum_database()
        return

    if args.verify:
        verify_database_integrity()
        return

    if args.export:
        export_database(args.export)
        return

    if args.import_file:
        import_database(args.import_file)
        return

    if args.search:
        search_database(args.search)
        return

    if args.optimize:
        optimize_database()
        return

    if args.update_database:
        update_database_to_new_format()
        return

    if args.reset:
        if input("Are you sure you want to reset the database? This will delete all entries. (Y/N): ").lower() == 'y':
            reset_database()
            return

    if args.status:
        stats = get_database_stats()
        if stats:
            log_message("Database Statistics:", level="INFO")
            log_message(f"Total Records: {stats['total_records']}", level="INFO")
            log_message(f"Archived Records: {stats['archived_records']}", level="INFO")
            log_message(f"Main DB Size: {stats['main_db_size']:.2f} MB", level="INFO")
            log_message(f"Archive DB Size: {stats['archive_db_size']:.2f} MB", level="INFO")
        return

    # Handle monitor-only mode
    if args.monitor_only:
        log_message("Starting in monitor-only mode", level="INFO")
        # Check dashboard availability even in monitor-only mode
        check_dashboard_availability()
        # Initialize database
        if not initialize_db_with_mount_check():
            log_message("Failed to initialize database. Exiting.", level="ERROR")
            return
        # Start only the polling monitor
        start_polling_monitor()
        return

    if not os.path.exists(LOCK_FILE):
        # Wait for mount if needed and initialize database
        if not initialize_db_with_mount_check():
            log_message("Failed to initialize database. Exiting.", level="ERROR")
            return

        # Skip heavy background processes for single file operations
        is_single_file_operation = args.single_path and os.path.isfile(args.single_path) if args.single_path else False

        if not args.disable_monitor and not is_single_file_operation:
            log_message("Starting background processes...", level="INFO")
            log_message("RealTime-Monitoring is enabled", level="INFO")

            # Define the callback function to be called once the background task finishes
            def on_missing_files_check_done():
                log_message("Database import completed.", level="INFO")

            # Function to run the missing files check and call the callback when done
            def display_missing_files_with_callback(dest_dir, callback):
                try:
                    if not dest_dir or not os.path.exists(dest_dir):
                        log_message(f"Invalid or non-existent destination directory: {dest_dir}", level="ERROR")
                        return
                    missing_files_list = display_missing_files_with_mount_check(dest_dir)

                    if missing_files_list:
                        log_message(f"Found {len(missing_files_list)} missing files. Attempting to recreate symlinks.", level="INFO")
                        # Get source directories for create_symlinks
                        src_dirs_str = get_setting_with_client_lock('SOURCE_DIR', '', 'string')
                        if not src_dirs_str:
                            log_message("Source directories not configured. Cannot recreate symlinks.", level="ERROR")
                            return
                        src_dirs = src_dirs_str.split(',')
                        if not src_dirs:
                            log_message("Source directories not configured. Cannot recreate symlinks.", level="ERROR")
                            return

                        for source_file_path, expected_dest_path in missing_files_list:
                            log_message(f"Attempting to recreate symlink for missing file: {source_file_path}", level="INFO")
                            create_symlinks(src_dirs=src_dirs, dest_dir=dest_dir, single_path=source_file_path, force=True, mode='create', auto_select=True, use_source_db=args.use_source_db)
                    else:
                        log_message("No missing files found.", level="INFO")

                    callback()
                except Exception as e:
                    log_message(f"Error in display_missing_files_with_callback: {str(e)}", level="ERROR")
                    log_message(traceback.format_exc(), level="DEBUG")

            # Run missing files check in a separate thread
            missing_files_thread = threading.Thread(name="missing_files_check", target=display_missing_files_with_callback, args=(dest_dir, on_missing_files_check_done))
            missing_files_thread.daemon = True
            missing_files_thread.start()

            # Symlink cleanup
            cleanup_thread = threading.Thread(target=run_symlink_cleanup, args=(dest_dir,))
            cleanup_thread.daemon = True
            cleanup_thread.start()
        elif is_single_file_operation:
            log_message("Single file operation detected - skipping background processes and dashboard checks for faster startup", level="INFO")
        else:
            log_message("RealTime-Monitoring is disabled", level="INFO")
            # Check dashboard availability even when monitoring is disabled
            check_dashboard_availability()

    src_dirs_str = get_setting_with_client_lock('SOURCE_DIR', '', 'string')
    dest_dir = get_setting_with_client_lock('DESTINATION_DIR', '', 'string')
    if not src_dirs_str or not dest_dir:
        log_message("Source or destination directory not set in environment variables.", level="ERROR")
        exit(1)
    src_dirs = src_dirs_str.split(',')

    # Wait for mount before creating symlinks if needed
    if is_rclone_mount_enabled() and not check_rclone_mount():
        wait_for_mount()

    try:
        # Check if this is a single file operation for optimization
        is_single_file_operation = args.single_path and os.path.isfile(args.single_path) if args.single_path else False

        # Check upload completion for single file operations (skip if --force is used)
        if is_single_file_operation and not args.force:
            from MediaHub.utils.file_utils import wait_for_upload_completion
            from MediaHub.config.config import get_upload_max_attempts, get_upload_stability_wait
            if not wait_for_upload_completion(args.single_path, max_attempts=get_upload_max_attempts(), stability_time=get_upload_stability_wait()):
                log_message(f"Skipping {args.single_path}: Upload not confirmed complete", level="WARNING")
                return  # Exit early for single file if upload incomplete

        # Filter source directories for stable files in auto mode
        filtered_src_dirs = []
        if not args.single_path:  # Only filter for directory scanning
            from MediaHub.utils.file_utils import wait_for_upload_completion
            from MediaHub.config.config import get_upload_max_attempts, get_upload_stability_wait
            for src_dir in src_dirs:
                for root, _, files in os.walk(src_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if not wait_for_upload_completion(file_path, max_attempts=get_upload_max_attempts(), stability_time=get_upload_stability_wait()):
                            log_message(f"Skipping {file_path}: Upload not confirmed complete", level="INFO")
                            continue
                        filtered_src_dirs.append(src_dir)  # Add directory if at least one file is stable
                        break  # Move to next directory after finding one stable file
                if is_shutdown_requested():
                    break
            filtered_src_dirs = list(set(filtered_src_dirs))  # Remove duplicates
            if not filtered_src_dirs:
                log_message("No source directories contain stable files, skipping symlink creation", level="WARNING")
                return

        # Start RealTime-Monitoring in main thread if not disabled and not single file operation
        if not args.disable_monitor and not is_single_file_operation:
            start_webdav_server()
            log_message("Starting RealTime-Monitoring...", level="INFO")
            monitor_thread = threading.Thread(target=start_polling_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            time.sleep(2)
            create_symlinks(
                src_dirs=filtered_src_dirs if not args.single_path else src_dirs,
                dest_dir=dest_dir,
                auto_select=args.auto_select,
                single_path=args.single_path,
                force=args.force,
                mode='create',
                tmdb_id=args.tmdb,
                imdb_id=args.imdb,
                tvdb_id=args.tvdb,
                force_show=args.force_show,
                force_movie=args.force_movie,
                season_number=season_number,
                episode_number=episode_number,
                force_extra=args.force_extra,
                skip=args.skip,
                batch_apply=args.batch_apply,
                manual_search=args.manual_search,
                use_source_db=args.use_source_db
            )

            while monitor_thread.is_alive() and not terminate_flag.is_set():
                time.sleep(0.1)

            if terminate_flag.is_set():
                log_message("Termination requested, stopping monitor thread", level="INFO")
        else:
            if is_single_file_operation:
                log_message("Single file operation - skipping monitoring services for faster processing", level="INFO")
            create_symlinks(
                src_dirs=src_dirs,
                dest_dir=dest_dir,
                auto_select=args.auto_select,
                single_path=args.single_path,
                force=args.force,
                mode='create',
                tmdb_id=args.tmdb,
                imdb_id=args.imdb,
                tvdb_id=args.tvdb,
                force_show=args.force_show,
                force_movie=args.force_movie,
                season_number=season_number,
                episode_number=episode_number,
                force_extra=args.force_extra,
                skip=args.skip,
                batch_apply=args.batch_apply,
                manual_search=args.manual_search,
                use_source_db=args.use_source_db
            )
    except KeyboardInterrupt:
        log_message("Keyboard interrupt received, cleaning up and exiting...", level="INFO")
        set_shutdown()
        terminate_subprocesses()
        remove_lock_file()
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    dest_dir = get_setting_with_client_lock('DESTINATION_DIR', '', 'string')
    if not dest_dir:
        log_message("DESTINATION_DIR environment variable not set.", level="ERROR")
        sys.exit(1)

    main(dest_dir)
