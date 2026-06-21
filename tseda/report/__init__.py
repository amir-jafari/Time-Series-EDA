"""
tseda.report — HTML and console EDA report generation.

Single-call API:

    from tseda.report import HTMLReport, ConsoleReport
    HTMLReport().generate(ts, "report.html", period=12)
    ConsoleReport().generate(ts, period=12)

Classes
-------
HTMLReport
    Self-contained HTML report with embedded figures.
ConsoleReport
    Plain-text EDA summary printed to stdout.
"""
from tseda.report.console_report import ConsoleReport
from tseda.report.html_report import HTMLReport

__all__ = ["HTMLReport", "ConsoleReport"]