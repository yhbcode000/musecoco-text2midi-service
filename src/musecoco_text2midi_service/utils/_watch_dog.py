import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    """Handler for file change events."""
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback

    def on_modified(self, event):
        if event.src_path == self.file_path:
            print(f"File {self.file_path} has changed.")
            self.callback()

def watch_file(file_path, callback):
    """Watches a file for changes and triggers the callback."""
    event_handler = FileChangeHandler(file_path, callback)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    observer.start()
    