"""
Debug output redirection utilities.
"""
import sys
import contextlib
from typing import Optional
from config import Config

class TeeOutput:
    """A class that writes to both stdout and a file simultaneously."""
    
    def __init__(self, file_handle):
        self.file_handle = file_handle
        self.original_stdout = sys.stdout
        
    def write(self, text):
        # Write to both stdout and file
        self.original_stdout.write(text)
        self.original_stdout.flush()  # Ensure immediate display
        self.file_handle.write(text)
        self.file_handle.flush()  # Ensure immediate write to file
        
    def flush(self):
        self.original_stdout.flush()
        self.file_handle.flush()

class DebugOutputRedirector:
    """Context manager to redirect debug output to a file with streaming."""
    
    def __init__(self, config: Config):
        self.config = config
        self.original_stdout = None
        self.file_handle = None
        self.tee_output = None
        
    def __enter__(self):
        if self.config.REDIRECT_DEBUG_TO_FILE:
            self.original_stdout = sys.stdout
            self.file_handle = open(self.config.DEBUG_OUTPUT_FILE, 'w', encoding='utf-8')
            self.tee_output = TeeOutput(self.file_handle)
            sys.stdout = self.tee_output
            print(f"Debug output streaming to: {self.config.DEBUG_OUTPUT_FILE}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.REDIRECT_DEBUG_TO_FILE and self.original_stdout:
            sys.stdout = self.original_stdout
            if self.file_handle:
                self.file_handle.close()
                print(f"Debug output completed. File saved: {self.config.DEBUG_OUTPUT_FILE}")

@contextlib.contextmanager
def debug_output_redirect(config: Config):
    """Context manager to redirect debug output to file if configured."""
    redirector = DebugOutputRedirector(config)
    with redirector:
        yield