"""Utilities for executing EEGCPM commands from the UI."""

import subprocess
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import streamlit as st


class ProcessExecutor:
    """Execute subprocess commands with real-time output streaming."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self.is_running: bool = False
        self.return_code: Optional[int] = None

    def _enqueue_output(self, pipe, queue):
        """Read output from pipe and put into queue."""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    queue.put(line)
        finally:
            pipe.close()

    def start(
        self,
        command: list,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Start executing a command.

        Parameters
        ----------
        command : list
            Command and arguments as list
        cwd : Path, optional
            Working directory
        env : dict, optional
            Environment variables

        Returns
        -------
        bool
            True if process started successfully
        """
        if self.is_running:
            return False

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=env
            )

            # Start thread to read output
            thread = threading.Thread(
                target=self._enqueue_output,
                args=(self.process.stdout, self.output_queue),
                daemon=True
            )
            thread.start()

            self.is_running = True
            self.return_code = None
            return True

        except Exception as e:
            st.error(f"Failed to start process: {e}")
            return False

    def get_output_lines(self, max_lines: int = 100) -> list:
        """Get available output lines from queue."""
        lines = []
        try:
            while not self.output_queue.empty() and len(lines) < max_lines:
                lines.append(self.output_queue.get_nowait())
        except queue.Empty:
            pass
        return lines

    def poll(self) -> Optional[int]:
        """Check if process is still running."""
        if self.process is None:
            return None

        ret = self.process.poll()
        if ret is not None:
            self.is_running = False
            self.return_code = ret
        return ret

    def terminate(self):
        """Terminate the running process."""
        if self.process and self.is_running:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.is_running = False


def run_eegcpm_command(
    command: str,
    args: Dict[str, Any],
    log_container: Any,
    status_container: Any
) -> bool:
    """
    Run an EEGCPM CLI command with live output display.

    Parameters
    ----------
    command : str
        EEGCPM command (e.g., 'source-reconstruct', 'epochs')
    args : dict
        Command arguments
    log_container : streamlit container
        Container for displaying logs
    status_container : streamlit container
        Container for status messages

    Returns
    -------
    bool
        True if command completed successfully
    """
    # Build command
    cmd = [sys.executable, '-m', 'eegcpm.cli.main', command]

    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            cmd.append(f'--{key}')
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))

    # Display command
    status_container.code(' '.join(cmd), language='bash')

    # Initialize executor
    executor = ProcessExecutor()

    if not executor.start(cmd):
        status_container.error("❌ Failed to start command")
        return False

    status_container.info("🚀 Process started...")

    # Create log display
    log_text = log_container.empty()
    all_logs = []

    # Monitor process
    while executor.is_running or not executor.output_queue.empty():
        # Get new output lines
        new_lines = executor.get_output_lines()
        all_logs.extend(new_lines)

        # Update display (show last 50 lines)
        display_logs = all_logs[-50:]
        log_text.code(''.join(display_logs), language='log')

        # Check if process finished
        executor.poll()

        # Small delay to prevent UI overload
        time.sleep(0.1)

    # Final status
    if executor.return_code == 0:
        status_container.success("✅ Command completed successfully")
        return True
    else:
        status_container.error(f"❌ Command failed with exit code {executor.return_code}")
        return False


def create_source_config(
    project_root: Path,
    variant_name: str,
    preprocessing: str,
    task: str,
    epochs_variant: Optional[str],
    method: str,
    parcellation: str,
    template: str,
    spacing: str,
    snr: float,
    loose: float,
    depth: float,
    roi_radius: float,
    save_stc: bool,
    save_roi_tc: bool,
    generate_qc: bool
) -> Path:
    """
    Create source reconstruction configuration file.

    Returns
    -------
    Path
        Path to created config file
    """
    import yaml

    config_dict = {
        "stage": "source",
        "variant": variant_name,
        "depends_on": {
            "preprocessing": preprocessing,
            "task": task,
            "epochs": epochs_variant if epochs_variant else None
        },
        "method": method,
        "forward": {
            "template": template,
            "spacing": spacing
        },
        "inverse": {
            "snr": snr,
            "loose": loose,
            "depth": depth
        },
        "parcellation": parcellation,
        "roi_radius": roi_radius,
        "subjects": "all",
        "output": {
            "save_stc": save_stc,
            "save_roi_tc": save_roi_tc,
            "generate_qc": generate_qc
        }
    }

    # Save config
    config_dir = project_root / "eegcpm" / "configs" / "source"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_filename = f"{variant_name.lower().replace(' ', '_').replace('-', '_')}.yaml"
    config_path = config_dir / config_filename

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_path
