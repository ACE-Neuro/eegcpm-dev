"""
HTML report builders for QC results.

Provides HTML report generation with embedded plots, metrics tables, and
index pages with subject navigation.
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional

from .base import QCMetric, QCResult


class HTMLReportBuilder:
    """
    Build HTML reports with embedded plots and metrics.

    Uses inline CSS and base64-encoded images for self-contained HTML files.
    """

    # Inline CSS styles
    CSS_STYLES = """
    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0;
        padding: 20px;
        background: #f9fafb;
        color: #1f2937;
        line-height: 1.6;
    }

    h1 {
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 28px;
        color: #111827;
    }

    h2 {
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 20px;
        color: #111827;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 10px;
    }

    h3 {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 16px;
        color: #374151;
    }

    p {
        margin: 10px 0;
        font-size: 14px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background: white;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    th {
        background: #f3f4f6;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
    }

    td {
        padding: 12px;
        border-bottom: 1px solid #e5e7eb;
        font-size: 14px;
    }

    tr:last-child td {
        border-bottom: none;
    }

    tr:hover {
        background: #f9fafb;
    }

    .metric-card {
        background: white;
        border-radius: 6px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .metric-name {
        font-weight: 500;
        color: #374151;
        font-size: 14px;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 600;
        margin-right: 10px;
    }

    .metric-unit {
        font-size: 12px;
        color: #6b7280;
    }

    .status-ok {
        color: #22c55e;
        font-weight: 600;
    }

    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }

    .status-bad {
        color: #ef4444;
        font-weight: 600;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 600;
    }

    .status-badge.ok {
        background: #dcfce7;
        color: #166534;
    }

    .status-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-badge.bad {
        background: #fee2e2;
        color: #991b1b;
    }

    .figure-container {
        margin: 25px 0;
        background: white;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .figure-container img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10px 0;
        border-radius: 4px;
    }

    .figure-caption {
        font-size: 13px;
        color: #6b7280;
        font-style: italic;
        margin-top: 10px;
    }

    .notes {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin: 15px 0;
        border-radius: 4px;
    }

    .notes-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 8px;
        font-size: 13px;
    }

    .notes-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .notes-list li {
        padding: 6px 0;
        font-size: 13px;
        color: #1e40af;
    }

    .notes-list li:before {
        content: "• ";
        margin-right: 8px;
    }

    .metadata {
        background: #f3f4f6;
        padding: 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #6b7280;
        margin: 15px 0;
    }
    """

    def __init__(self, title: str = "QC Report"):
        """
        Initialize HTML report builder.

        Args:
            title: Report title (shown in heading)
        """
        self.title = title
        self.sections: List[str] = []

    def add_header(self, text: str, level: int = 2) -> "HTMLReportBuilder":
        """
        Add header/heading section.

        Args:
            text: Header text
            level: Header level (1-6, default 2)

        Returns:
            Self for method chaining
        """
        self.sections.append(f"<h{level}>{self._escape_html(text)}</h{level}>")
        return self

    def add_text(self, text: str) -> "HTMLReportBuilder":
        """
        Add paragraph text.

        Args:
            text: Text content

        Returns:
            Self for method chaining
        """
        self.sections.append(f"<p>{self._escape_html(text)}</p>")
        return self

    def add_metrics_table(self, metrics: List[QCMetric]) -> "HTMLReportBuilder":
        """
        Add metrics as HTML table.

        Args:
            metrics: List of QCMetric objects

        Returns:
            Self for method chaining
        """
        if not metrics:
            return self

        html = "<table>\n  <tr>\n"
        html += "    <th>Metric</th>\n"
        html += "    <th>Value</th>\n"
        html += "    <th>Status</th>\n"
        html += "  </tr>\n"

        for metric in metrics:
            status_class = f"status-{metric.status}"
            value = metric.value
            if isinstance(value, (int, float)):
                if value != 0 and abs(value) < 0.01:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            if metric.unit:
                value_str += f" {metric.unit}"

            html += "  <tr>\n"
            html += f"    <td>{self._escape_html(metric.name)}</td>\n"
            html += f"    <td>{value_str}</td>\n"
            html += f'    <td><span class="status-badge {metric.status}">{metric.status.upper()}</span></td>\n'
            html += "  </tr>\n"

        html += "</table>"
        self.sections.append(html)
        return self

    def add_figure(
        self, name: str, fig_bytes: bytes, caption: str = ""
    ) -> "HTMLReportBuilder":
        """
        Add matplotlib figure (PNG bytes or base64 string) as embedded image.

        Args:
            name: Figure identifier/name
            fig_bytes: PNG bytes or base64-encoded string
            caption: Optional figure caption

        Returns:
            Self for method chaining
        """
        # Handle both bytes and base64 strings
        if isinstance(fig_bytes, str):
            # Already base64 encoded
            encoded = fig_bytes
        else:
            # Convert bytes to base64
            encoded = base64.b64encode(fig_bytes).decode("ascii")

        data_uri = f"data:image/png;base64,{encoded}"

        html = f'<div class="figure-container">\n'
        html += f'  <img src="{data_uri}" alt="{self._escape_html(name)}">\n'
        if caption:
            html += f'  <div class="figure-caption">{self._escape_html(caption)}</div>\n'
        html += "</div>"

        self.sections.append(html)
        return self

    def add_notes(self, notes: List[str]) -> "HTMLReportBuilder":
        """
        Add notes/comments section.

        Args:
            notes: List of note strings

        Returns:
            Self for method chaining
        """
        if not notes:
            return self

        html = '<div class="notes">\n'
        html += '  <div class="notes-title">Notes</div>\n'
        html += '  <ul class="notes-list">\n'
        for note in notes:
            html += f"    <li>{self._escape_html(note)}</li>\n"
        html += "  </ul>\n"
        html += "</div>"

        self.sections.append(html)
        return self

    def add_raw_html(self, html: str) -> "HTMLReportBuilder":
        """
        Add raw HTML content without escaping.

        Use this method to add custom HTML elements that need to preserve
        markup (e.g., links, styled divs, iframes).

        Args:
            html: Raw HTML string to insert

        Returns:
            Self for method chaining

        Warning:
            Only use this with trusted content to avoid XSS vulnerabilities
        """
        self.sections.append(html)
        return self

    def build(self) -> str:
        """
        Build complete HTML document.

        Returns:
            Complete HTML string with DOCTYPE, inline CSS, and content
        """
        html = "<!DOCTYPE html>\n"
        html += '<html lang="en">\n'
        html += "<head>\n"
        html += '  <meta charset="UTF-8">\n'
        html += '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += f"  <title>{self._escape_html(self.title)}</title>\n"
        html += "  <style>\n"
        html += self.CSS_STYLES
        html += "  </style>\n"
        html += "</head>\n"
        html += "<body>\n"
        html += f"<h1>{self._escape_html(self.title)}</h1>\n"

        for section in self.sections:
            html += section + "\n"

        html += "</body>\n"
        html += "</html>"

        return html

    def save(self, path: Path) -> Path:
        """
        Save HTML report to file.

        Args:
            path: Path to save HTML file

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())
        return path

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


class QCIndexBuilder:
    """
    Build index.html with subject sidebar and iframe content.

    Creates a responsive layout with left sidebar for subject navigation
    and right content area for viewing individual QC reports.
    """

    # Inline CSS for index
    CSS_INDEX = """
    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        display: flex;
        height: 100vh;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #f9fafb;
    }

    .sidebar {
        width: 320px;
        background: white;
        overflow-y: auto;
        border-right: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .sidebar-header {
        padding: 15px;
        border-bottom: 1px solid #e5e7eb;
        background: #f9fafb;
    }

    .sidebar h2 {
        font-size: 16px;
        margin: 0;
        color: #111827;
        font-weight: 600;
    }

    .subject-table-wrapper {
        flex: 1;
        overflow-y: auto;
    }

    .subject-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .subject-table thead th {
        position: sticky;
        top: 0;
        background: #f9fafb;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        color: #6b7280;
        border-bottom: 1px solid #e5e7eb;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .subject-table tbody tr {
        cursor: pointer;
        transition: background-color 0.15s;
        border-bottom: 1px solid #f3f4f6;
    }

    .subject-table tbody tr:hover {
        background: #f9fafb;
    }

    .subject-table tbody tr:active {
        background: #f3f4f6;
    }

    .subject-table tbody tr.active {
        background: #eff6ff;
    }

    .subject-table tbody td {
        padding: 10px 12px;
        color: #374151;
    }

    .subject-table tbody td.status-cell {
        width: 50px;
        text-align: center;
    }

    .subject-status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    .status-ok {
        background: #22c55e;
    }

    .status-warning {
        background: #f59e0b;
    }

    .status-bad {
        background: #ef4444;
    }

    .content {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #f9fafb;
    }

    .content-header {
        background: white;
        padding: 15px 20px;
        border-bottom: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .content-header h1 {
        margin: 0;
        font-size: 20px;
        color: #111827;
    }

    .content-iframe-wrapper {
        flex: 1;
        overflow: hidden;
        position: relative;
    }

    .content-iframe-wrapper iframe {
        width: 100%;
        height: 100%;
        border: none;
    }

    .content-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #9ca3af;
        font-size: 16px;
    }

    /* Scrollbar styling */
    .subject-table-wrapper::-webkit-scrollbar {
        width: 8px;
    }

    .subject-table-wrapper::-webkit-scrollbar-track {
        background: transparent;
    }

    .subject-table-wrapper::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }

    .subject-table-wrapper::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        body {
            flex-direction: column;
        }

        .sidebar {
            width: 100%;
            height: auto;
            max-height: 200px;
            border-right: none;
            border-bottom: 1px solid #e5e7eb;
        }

        .content {
            min-height: 400px;
        }
    }
    """

    def __init__(self, title: str = "QC Index"):
        """
        Initialize index builder.

        Args:
            title: Page title
        """
        self.title = title
        self.subjects: List[Dict[str, str]] = []  # [{id, status, filename}, ...]

    def add_subject(self, subject_id: str, status: str, filename: str) -> "QCIndexBuilder":
        """
        Add subject to navigation list.

        Args:
            subject_id: Subject identifier
            status: Status ("ok", "warning", "bad")
            filename: HTML filename for this subject's report

        Returns:
            Self for method chaining
        """
        self.subjects.append(
            {
                "id": subject_id,
                "status": status,
                "filename": filename,
            }
        )
        return self

    def build(self) -> str:
        """
        Build index HTML with sidebar and iframe.

        Returns:
            Complete HTML string
        """
        html = "<!DOCTYPE html>\n"
        html += '<html lang="en">\n'
        html += "<head>\n"
        html += '  <meta charset="UTF-8">\n'
        html += '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += f"  <title>{self._escape_html(self.title)}</title>\n"
        html += "  <style>\n"
        html += self.CSS_INDEX
        html += "  </style>\n"
        html += "</head>\n"
        html += "<body>\n"

        # Sidebar with table
        html += '  <div class="sidebar">\n'
        html += '    <div class="sidebar-header">\n'
        html += f"      <h2>{self._escape_html(self.title)}</h2>\n"
        html += '    </div>\n'
        html += '    <div class="subject-table-wrapper">\n'
        html += '      <table class="subject-table">\n'
        html += '        <thead>\n'
        html += '          <tr>\n'
        html += '            <th>Status</th>\n'
        html += '            <th>Subject ID</th>\n'
        html += '          </tr>\n'
        html += '        </thead>\n'
        html += '        <tbody>\n'

        for subject in self.subjects:
            status = subject["status"]
            subject_id = subject["id"]
            filename = subject["filename"]

            html += f'          <tr onclick="loadReport(\'{self._escape_html(filename)}\', this)">\n'
            html += f'            <td class="status-cell">\n'
            html += f'              <span class="subject-status-indicator status-{status}"></span>\n'
            html += f'            </td>\n'
            html += f'            <td>{self._escape_html(subject_id)}</td>\n'
            html += f'          </tr>\n'

        html += '        </tbody>\n'
        html += '      </table>\n'
        html += '    </div>\n'
        html += '  </div>\n'

        # Content area
        html += '  <div class="content">\n'
        html += '    <div class="content-header">\n'
        html += '      <h1 id="report-title">QC Report</h1>\n'
        html += "    </div>\n"
        html += '    <div class="content-iframe-wrapper">\n'
        html += '      <iframe name="content-frame"></iframe>\n'
        html += "    </div>\n"
        html += "  </div>\n"

        # JavaScript for row click handling
        html += "  <script>\n"
        html += "    function loadReport(filename, row) {\n"
        html += "      // Load report in iframe\n"
        html += "      document.querySelector('iframe[name=\"content-frame\"]').src = filename;\n"
        html += "      \n"
        html += "      // Update active row\n"
        html += "      document.querySelectorAll('.subject-table tbody tr').forEach(tr => {\n"
        html += "        tr.classList.remove('active');\n"
        html += "      });\n"
        html += "      row.classList.add('active');\n"
        html += "    }\n"
        html += "  </script>\n"

        html += "</body>\n"
        html += "</html>"

        return html

    def save(self, path: Path) -> Path:
        """
        Save index HTML to file.

        Args:
            path: Path to save HTML file

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())
        return path

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
