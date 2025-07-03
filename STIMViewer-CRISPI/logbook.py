# logbook.py
from PyQt5.QtWidgets import QGridLayout, QPushButton, QWidget, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import pyqtSignal


class Logbook(QWidget):
    # Class-level variable to hold the global instance.
    instance = None
    log_buffer = []
    export_count = 0

    # Need to define a pyqtSignal since these need to be done in background threads

    newLogpyqtSignal = pyqtSignal(str)

    closed = pyqtSignal()

    def __init__(self, logger=None, log_widget=None):
        super().__init__()
        self.setWindowTitle("Logbook")
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize the log_widget. If none provided, create a QTextEdit.
        self.log_widget = log_widget if log_widget else QTextEdit()
        self.log_widget.setReadOnly(True)
        self.layout.addWidget(self.log_widget)

        self.logger = logger

        self.paused = False

        self.newLogpyqtSignal.connect(self.write_log_pyqtSlot)

        self.log_init()

        # Set the global instance to this object.
        Logbook.instance = self

        # Flush the log buffer to the logbook
        for message in Logbook.log_buffer:
            self.newLogpyqtSignal.emit(message)
        Logbook.log_buffer.clear()

    def closeEvent(self, event):
        # Override close event: hide instead of closing.
        event.ignore()
        self.hide()
        self.closed.emit()

    def log_init(self):
        self.pause_resume_button = QPushButton("Pause Logging")
        self.pause_resume_button.setCheckable(True)
        self.pause_resume_button.clicked.connect(self.pause_resume_logging)


        self.export = QPushButton("Export Logbook")
        self.export.clicked.connect(self.export_logbook_to_file)

        grid = QGridLayout()
        grid.addWidget(self.pause_resume_button, 2, 0, 2, 1)
        grid.addWidget(self.export, 2, 2, 2, 1)

        self.layout.addStretch()
        self.layout.addLayout(grid)

    def pause_resume_logging(self):
        if self.pause_resume_button.isChecked():
            self.paused = True
            self.pause_resume_button.setText("Resume Logging")
        else:
            self.paused = False
            self.pause_resume_button.setText("Pause Logging")


    def write_log_pyqtSlot(self, log):
        """
        Write the log to the log widget.
        Uses HTML formatting so that colored messages are rendered properly.
        """
        if not self.paused:
            self.log_widget.insertHtml(log)
            self.log_widget.moveCursor(QTextCursor.End)

    def write_log(self, log):
        self.newLogpyqtSignal.emit(log)


    @classmethod
    def _log_generic(cls, level, message):
        """
        Generic logging method using HTML formatting for colored log levels.
        Levels: EMER, ALRT, CRIT, ERRO, WARN, NOTI, INFO, DBUG.
        """
        level_config = {
            "EMER": ("EMERGENCY:", "red"),
            "ALRT": ("ALERT:", "orange"),
            "CRIT": ("CRITICAL:", "darkred"),
            "ERRO": ("ERROR:", "red"),
            "WARN": ("WARNING:", "goldenrod"),
            "NOTI": ("NOTIFICATION:", "blue"),
            "INFO": ("INFORMATIONAL:", "black"),
            "DBUG": ("DEBUG:", "gray"),
        }
        prefix, color = level_config.get(level, ("", "black"))
        # Build an HTML-formatted log message.
        html = f"<span style='color: {color};'><b>{prefix}</b></span> {message}<br>"
        if cls.instance:
            cls.instance.write_log(html)
    

    # Convenience methods for each log level:
    @classmethod
    def log_EMER(cls, message):
        cls._log_generic("EMER", message)

    @classmethod
    def log_ALRT(cls, message):
        cls._log_generic("ALRT", message)

    @classmethod
    def log_CRIT(cls, message):
        cls._log_generic("CRIT", message)

    @classmethod
    def log_ERRO(cls, message):
        cls._log_generic("ERRO", message)

    @classmethod
    def log_WARN(cls, message):
        cls._log_generic("WARN", message)

    @classmethod
    def log_NOTI(cls, message):
        cls._log_generic("NOTI", message)

    @classmethod
    def log_INFO(cls, message):
        cls._log_generic("INFO", message)

    @classmethod
    def log_DBUG(cls, message):
        cls._log_generic("DBUG", message)

    def export_logbook_to_file(self):
        """
        Export the log to a file.
        Each export is appended to the file with an export header and separator.
        """
        Logbook.export_count += 1
        # Get all logs from the widget as plain text.
        log_text = self.log_widget.toPlainText()
        lines = log_text.splitlines()
        file_path = "export_log.txt"
        try:
            with open(file_path, "a") as f:
                f.write(f"Export: {Logbook.export_count}\n")
                f.write("\n".join(lines))
                f.write("\n" + ("-" * 40) + "\n")
            self.write_log(f"<br><i>Log exported successfully to {file_path}</i><br>")
        except Exception as e:
            self.write_log(f"<br><i>Error exporting log: {str(e)}</i><br>")
        print("Logbook exported to file")