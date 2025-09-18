import os
import json
import re
import csv
import pandas as pd
import shutil

class SnippetProcessor:
    """
    A utility class for processing datasets, specifically for extracting English user messages
    from JSON files where the assistant's response contains valid code snippets.
    """
    def __init__(self):
        self.snippets = [] 

    def is_code_snippet(self, text):
        """
            Check if the text contains a valid multi-line or inline code snippet.
            Check if the snippet is empty or not.
        """
        # Detect triple backticks (multi-line code snippets)
        snippets = re.findall(r"```.*?```", text, re.DOTALL)  # Match code blocks wrapped in triple backticks

        # # Exclude snippets that are empty (e.g., "``````") or start with invalid patterns
        # Only User show ```Absolutely! xxx```
        valid_snippets = [
            snippet for snippet in snippets
            if snippet.strip("`").strip()  # Ensure content inside backticks is not empty
            and not re.match(r"```[\s\S]*?(Absolutely|^[A-Za-z]+$)", snippet.splitlines()[0], re.IGNORECASE)
        ] 
        
        # Store valid snippets in self.snippets, only if valid_snippets is non-empty
        if valid_snippets:
            if not hasattr(self, "snippets"):
                self.snippets = []  # Initialize self.snippets if it doesn't exist
            self.snippets = valid_snippets  # Overwrite with the current valid snippets 
        return bool(valid_snippets)
            
    def cal_snippets(self, content):
        """
        Extract all valid snippets from the provided content.
        """
        if self.is_code_snippet(content):
            return self.snippets  # Return valid snippets extracted by is_code_snippet
        else:
            return []  # Return an empty list if no valid snippets are found

    def count_snippets_num(self, text):
        """Count the number of valid code snippets (wrapped in triple backticks) in the text."""
        if not text:
            return 0
        # Find all code snippets in the content 
        valid_snippets = self.cal_snippets(text)
        return len(valid_snippets)
    
    def count_snippet_lines(self, snippet):
        """
        Calculate the number of non-empty lines in a code snippet.
        The snippet is expected to be enclosed in triple backticks (```).

        :param snippet: Code snippet as a string.
        :return: Number of non-empty lines in the snippet.
        """
        # Ensure the snippet is not empty
        if not snippet:
            return 0

        # Remove enclosing triple backticks if they exist
        if snippet.startswith("```") and snippet.endswith("```"):
            snippet_content = snippet[3:-3].strip()  # Remove the first and last triple backticks and strip whitespace
        else:
            snippet_content = snippet.strip()  # Strip whitespace if no enclosing backticks

        # Split the snippet content into lines and count non-empty ones
        non_empty_lines = sum(1 for line in snippet_content.splitlines() if line.strip())
        return non_empty_lines
    
    def get_language_tag(self, snippet):
        """
        Extract the language tag from a code snippet.
        Returns the language tag if present, otherwise None.
        """
        # Match triple backticks followed by a language tag, ensuring a newline (\n) follows the tag
        match = re.match(r"```([\w#+-]+)\n", snippet)
        if match:
            language_tag = match.group(1).lower()  # Extract and return the language tag in lowercase
            return language_tag
        return None  # No language tag
    
    def save_snippets(self, snippets, folder, snippet_type):
        """
        Writes snippets into a JSON file in the specified folder.
        Each file contains up to 5000 snippets, and the naming is sequential.
        The snippet_type parameter determines whether the snippets are 'tagged' or 'untagged'.
        :param snippets: List of snippets to be written to a file.
        :param folder: Directory where the file will be saved.
        :param snippet_type: Type of snippets, either 'tag' or 'untag' (default: 'untag').
        """
        if snippet_type not in ['tag', 'untag']:
            raise ValueError("Invalid snippet_type. Expected 'tag' or 'untag'.")

        # Determine file prefix based on the snippet type
        file_prefix = "tagged_snippets_" if snippet_type == "tag" else "untagged_snippets_"

        # Count existing JSON files to determine the next file number
        existing_files = [f for f in os.listdir(folder) if f.startswith(file_prefix) and f.endswith(".json")]
        next_file_number = len(existing_files) + 1

        # Define the file name and path
        file_name = f"{file_prefix}{next_file_number}.json"
        file_path = os.path.join(folder, file_name)

        # Write snippets to the JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(snippets, json_file, indent=4)
    
    def clear_directory(self, path):
        """
        Removes all files and subdirectories in the specified directory.
        Args:
            path (str): The path to the directory to clear.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove file or symbolic link
                    print(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")