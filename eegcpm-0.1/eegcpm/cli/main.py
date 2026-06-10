"""Main CLI entry point for EEGCPM."""

import sys
import argparse
from pathlib import Path

from .status import status_command
from .preprocess import preprocess_command
from .resume import resume_command
from .import_qc import import_qc_command
from .epochs import epochs_command
from .source_reconstruct import source_reconstruct_command
from .connectivity import connectivity_command


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EEGCPM - EEG Connectome Predictive Modeling Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check processing status
  eegcpm status --project /path/to/project

  # Run preprocessing on all subjects
  eegcpm preprocess --config config.yaml --project /path/to/project

  # Resume failed/incomplete workflows
  eegcpm resume --project /path/to/project --subject sub-001

For more information: https://github.com/clivehywong/eegcpm-dev
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show processing status',
        description='Display workflow status for subjects and tasks'
    )
    status_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    status_parser.add_argument(
        '--subject',
        type=str,
        help='Filter by subject ID'
    )
    status_parser.add_argument(
        '--task',
        type=str,
        help='Filter by task name'
    )
    status_parser.add_argument(
        '--pipeline',
        type=str,
        help='Filter by pipeline name'
    )
    status_parser.add_argument(
        '--status-filter',
        type=str,
        choices=['pending', 'in_progress', 'completed', 'failed', 'skipped'],
        help='Filter by status'
    )
    status_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed step information'
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Run preprocessing pipeline',
        description='Execute preprocessing on subjects'
    )
    preprocess_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Pipeline configuration file (YAML)'
    )
    preprocess_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    preprocess_parser.add_argument(
        '--subjects',
        type=Path,
        help='Text file with subject IDs (one per line)'
    )
    preprocess_parser.add_argument(
        '--subject',
        type=str,
        help='Process single subject'
    )
    preprocess_parser.add_argument(
        '--task',
        type=str,
        help='Task name (if not in config)'
    )
    preprocess_parser.add_argument(
        '--task-config',
        type=str,
        help='Task config name for ERP QC (default: auto-detect from task name)'
    )
    preprocess_parser.add_argument(
        '--pipeline',
        type=str,
        default='standard',
        help='Pipeline name (default: standard)'
    )
    preprocess_parser.add_argument(
        '--eegcpm-root',
        type=Path,
        help='EEGCPM workspace root (configs & state; outputs go to {project}/derivatives/preprocessing/{pipeline}/)'
    )
    preprocess_parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess even if already completed'
    )
    preprocess_parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel jobs (default: 1)'
    )

    # Resume command
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume incomplete/failed workflows',
        description='Resume processing from last successful step'
    )
    resume_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    resume_parser.add_argument(
        '--subject',
        type=str,
        help='Resume specific subject (otherwise resume all incomplete)'
    )
    resume_parser.add_argument(
        '--task',
        type=str,
        help='Resume specific task'
    )
    resume_parser.add_argument(
        '--pipeline',
        type=str,
        help='Resume specific pipeline'
    )
    resume_parser.add_argument(
        '--from-step',
        type=str,
        help='Resume from specific step (default: last completed)'
    )

    # Import QC command
    import_parser = subparsers.add_parser(
        'import-qc',
        help='Import QC metrics from JSON to state database',
        description='Scan derivatives for QC JSON files and update workflow state'
    )
    import_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    import_parser.add_argument(
        '--derivatives',
        type=Path,
        help='Derivatives path (default: project/derivatives)'
    )
    import_parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        help='Pipeline name to assign to imported workflows'
    )
    import_parser.add_argument(
        '--task',
        type=str,
        help='Filter by task name'
    )
    import_parser.add_argument(
        '--subject',
        type=str,
        help='Filter by subject ID'
    )
    import_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-import existing entries (overwrite)'
    )
    import_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be imported without making changes'
    )

    # Epochs extraction command
    epochs_parser = subparsers.add_parser(
        'epochs',
        help='Extract epochs from preprocessed data',
        description='Create epochs by combining runs for each subject/session/task'
    )
    epochs_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    epochs_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Task/epochs config file (YAML)'
    )
    epochs_parser.add_argument(
        '--preprocessing',
        type=str,
        required=True,
        help='Preprocessing pipeline variant (e.g., optimal, standard)'
    )
    epochs_parser.add_argument(
        '--task',
        type=str,
        help='Task name (optional if task_name is in config)'
    )
    epochs_parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to process (e.g., NDARAA306NT2)'
    )
    epochs_parser.add_argument(
        '--sessions',
        type=str,
        nargs='+',
        help='Specific sessions to process (e.g., 01)'
    )
    epochs_parser.add_argument(
        '--runs',
        type=str,
        nargs='+',
        help='Specific runs to combine (e.g., 1 2 3). If not specified, combines all runs.'
    )

    # Source reconstruction command
    source_parser = subparsers.add_parser(
        'source-reconstruct',
        help='Run source reconstruction',
        description='Perform source reconstruction on epoched data'
    )
    source_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    source_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Source reconstruction config file (YAML)'
    )
    source_parser.add_argument(
        '--preprocessing',
        type=str,
        required=True,
        help='Preprocessing pipeline variant (e.g., optimal, standard)'
    )
    source_parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task name (e.g., contrastdetection)'
    )
    source_parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to process (e.g., NDARAA306NT2)'
    )
    source_parser.add_argument(
        '--sessions',
        type=str,
        nargs='+',
        help='Specific sessions to process (e.g., 01)'
    )
    source_parser.add_argument(
        '--task-config',
        type=str,
        help='Task config name for condition grouping (default: auto-detect from task name, use "none" to skip)'
    )

    # Connectivity analysis command
    connectivity_parser = subparsers.add_parser(
        'connectivity',
        help='Run connectivity analysis',
        description='Compute functional connectivity from source ROI timecourses'
    )
    connectivity_parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    connectivity_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Connectivity config file (YAML)'
    )
    connectivity_parser.add_argument(
        '--preprocessing',
        type=str,
        required=True,
        help='Preprocessing pipeline variant (e.g., optimal, standard)'
    )
    connectivity_parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task name (e.g., contrastdetection)'
    )
    connectivity_parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source variant (e.g., eLORETA-CONN32)'
    )
    connectivity_parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to process (e.g., NDARAA306NT2)'
    )
    connectivity_parser.add_argument(
        '--sessions',
        type=str,
        nargs='+',
        help='Specific sessions to process (e.g., 01)'
    )
    connectivity_parser.add_argument(
        '--task-config',
        type=str,
        help='Task config name for condition grouping (default: auto-detect from task name, use "none" to skip)'
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'status':
            status_command(args)
        elif args.command == 'preprocess':
            preprocess_command(args)
        elif args.command == 'resume':
            resume_command(args)
        elif args.command == 'import-qc':
            import_qc_command(args)
        elif args.command == 'epochs':
            epochs_command(args)
        elif args.command == 'source-reconstruct':
            source_reconstruct_command(args)
        elif args.command == 'connectivity':
            connectivity_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
