"""Status command for EEGCPM CLI."""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import box

from ..workflow.state import WorkflowStateManager, ProcessingStatus


def status_command(args):
    """
    Display workflow status.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    console = Console()

    # Get state database path
    db_path = args.project / "derivatives" / ".eegcpm" / "state.db"

    if not db_path.exists():
        console.print(f"[yellow]No workflow state found at {db_path}[/yellow]")
        console.print("Run preprocessing to initialize workflow tracking.")
        return

    # Load state manager
    manager = WorkflowStateManager(db_path)

    # Get summary
    summary = manager.get_summary()

    console.print(f"\n[bold]EEGCPM Workflow Status[/bold]")
    console.print(f"Project: {args.project}")
    console.print(f"State DB: {db_path}\n")

    # Print summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Total workflows: {summary['total_workflows']}")
    console.print(f"  Subjects: {summary['n_subjects']}")
    console.print(f"  Tasks: {summary['n_tasks']}")
    console.print(f"  Pipelines: {summary['n_pipelines']}\n")

    console.print("[bold]Status Breakdown:[/bold]")
    for status, count in summary['status_counts'].items():
        color = {
            'completed': 'green',
            'in_progress': 'yellow',
            'failed': 'red',
            'pending': 'cyan',
            'skipped': 'dim'
        }.get(status, 'white')
        console.print(f"  [{color}]{status:12s}[/{color}]: {count}")

    # Get filtered workflows
    status_filter = ProcessingStatus(args.status_filter) if args.status_filter else None
    workflows = manager.get_all_states(
        subject_id=args.subject,
        task=args.task,
        pipeline=args.pipeline,
        status=status_filter,
        stage=getattr(args, "stage", None),
    )

    if not workflows:
        console.print("\n[yellow]No workflows match the specified filters.[/yellow]")
        return

    # Create table
    table = Table(
        title=f"\nWorkflow Details ({len(workflows)} workflows)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("Subject", style="cyan", no_wrap=True)
    table.add_column("Task", style="magenta")
    table.add_column("Pipeline", style="blue")
    table.add_column("Status", style="white")
    table.add_column("Steps", justify="right")
    table.add_column("Updated", style="dim")

    for workflow in workflows:
        # Status color
        status_colors = {
            ProcessingStatus.COMPLETED: 'green',
            ProcessingStatus.IN_PROGRESS: 'yellow',
            ProcessingStatus.FAILED: 'red',
            ProcessingStatus.PENDING: 'cyan',
            ProcessingStatus.SKIPPED: 'dim'
        }
        status_color = status_colors.get(workflow.status, 'white')
        status_text = f"[{status_color}]{workflow.status.value}[/{status_color}]"

        # Step summary
        completed = len(workflow.get_completed_steps())
        total = len(workflow.steps)
        failed = len(workflow.get_failed_steps())

        if failed > 0:
            steps_text = f"{completed}/{total} ({failed} failed)"
        else:
            steps_text = f"{completed}/{total}"

        # Updated time
        updated = workflow.updated_at.strftime("%Y-%m-%d %H:%M") if workflow.updated_at else "N/A"

        table.add_row(
            workflow.subject_id,
            workflow.task,
            workflow.pipeline,
            status_text,
            steps_text,
            updated
        )

    console.print(table)

    # Verbose mode: show step details
    if args.verbose and workflows:
        console.print("\n[bold]Step Details:[/bold]\n")

        for workflow in workflows[:5]:  # Limit to first 5 for readability
            console.print(f"[cyan]{workflow.subject_id}[/cyan] / [magenta]{workflow.task}[/magenta] / [blue]{workflow.pipeline}[/blue]")

            step_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            step_table.add_column("Step", style="white")
            step_table.add_column("Status", style="white")
            step_table.add_column("Duration", justify="right", style="dim")
            step_table.add_column("Output", style="dim")

            for step in workflow.steps:
                status_colors = {
                    ProcessingStatus.COMPLETED: 'green',
                    ProcessingStatus.IN_PROGRESS: 'yellow',
                    ProcessingStatus.FAILED: 'red',
                    ProcessingStatus.PENDING: 'cyan',
                    ProcessingStatus.SKIPPED: 'dim'
                }
                step_status_color = status_colors.get(step.status, 'white')
                step_status = f"[{step_status_color}]{step.status.value}[/{step_status_color}]"

                # Duration
                if step.started_at and step.completed_at:
                    duration = (step.completed_at - step.started_at).total_seconds()
                    duration_text = f"{duration:.1f}s"
                else:
                    duration_text = "-"

                # Output path
                output = step.output_path if step.output_path else "-"
                if len(output) > 40:
                    output = "..." + output[-37:]

                step_table.add_row(
                    step.step_name,
                    step_status,
                    duration_text,
                    output
                )

                # Show error if failed
                if step.status == ProcessingStatus.FAILED and step.error_message:
                    console.print(f"    [red]Error: {step.error_message}[/red]")

            console.print(step_table)
            console.print()

        if len(workflows) > 5:
            console.print(f"[dim]... and {len(workflows) - 5} more workflows (use filters to narrow down)[/dim]\n")
