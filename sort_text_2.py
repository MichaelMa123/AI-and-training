import re
import json

# Function to remove sequential page numbers
def remove_page_numbers(text):
    # Split text into lines
    lines = text.splitlines()
    cleaned_lines = []
    
    # Initialize the expected page number
    expected_page_number = 11
    
    # Iterate through each line
    for line in lines:
        stripped_line = line.strip()

        # Check if the line is a number and matches the expected page number
        if stripped_line.isdigit() and int(stripped_line) == expected_page_number:
            # It's a page number, so we skip adding it to cleaned_lines
            expected_page_number += 1  # Increment expected page number for the next one
        else:
            # It's not a page number, add it to the cleaned lines
            cleaned_lines.append(line)

    # Join the cleaned lines back into a single text
    return '\n'.join(cleaned_lines)

# Function to classify the text into speech, system dialog, and narration
def classify_text(text):
    # Lists to store the different types of text
    speech = []
    system_dialog = []
    narration = []

    # Regex patterns to detect fully enclosed speech and system dialog
    speech_pattern = re.compile(r'“(.*?)”', re.DOTALL)  # Speech is within “ ”
    system_dialog_pattern = re.compile(r'<(.*?)>', re.DOTALL)  # System dialog is within < >
    
    # Regex pattern to detect the end of a sentence (., !, ?)
    sentence_end_pattern = re.compile(r'[.!?]')

    # Initialize an accumulator for narration
    current_narration = ""

    # Split the text into lines
    lines = text.splitlines()

    # Process each line
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Check if the line contains fully enclosed speech
        speech_match = speech_pattern.findall(line)
        if speech_match:
            # Add fully enclosed speech (even if it contains multiple sentences)
            speech.extend(speech_match)
            continue

        # Check if the line contains fully enclosed system dialog
        system_dialog_match = system_dialog_pattern.findall(line)
        if system_dialog_match:
            # Add fully enclosed system dialog (even if it contains multiple sentences)
            system_dialog.extend(system_dialog_match)
            continue

        # Accumulate lines for narration
        current_narration += " " + line

        # Check if the narration has reached the end of a sentence (with punctuation)
        if sentence_end_pattern.search(current_narration):
            # Add the accumulated narration as a complete entry
            narration.append(current_narration.strip())
            current_narration = ""  # Reset narration accumulator

    # If there is any leftover narration (without sentence-ending punctuation), store it
    if current_narration:
        narration.append(current_narration.strip())

    # Return the categorized text
    return {
        "speech": speech,
        "system_dialog": system_dialog,
        "narration": narration
    }

# Function to process text from a file and save the result in JSON format
def process_text_file(input_file, output_json_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove page numbers from the text
    cleaned_text = remove_page_numbers(text)

    # Classify the cleaned text into speech, system dialog, and narration
    classified_text = classify_text(cleaned_text)

    # Save the classified text to a JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(classified_text, f, indent=4)

    print(f"Processed text saved to {output_json_file}")

# Example usage
if __name__ == "__main__":
    # Path to your input text file
    input_file = '86--EIGHTY-SIX Volume-1.txt'  # Change this to your file path
    output_json_file = 'output9.json'  # Output JSON file

    # Process the text and save to JSON
    process_text_file(input_file, output_json_file)
