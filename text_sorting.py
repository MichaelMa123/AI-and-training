import re
import json

# Function to classify the text into speech, system dialog, and narration
def classify_text(text):
    # Lists to store the different types of text
    speech = []
    system_dialog = []
    narration = []

    # Regex patterns to detect speech and system dialog
    speech_pattern = re.compile(r'“(.*?)”', re.DOTALL)  # Speech is within “ ”
    system_dialog_pattern = re.compile(r'<(.*?)>', re.DOTALL)  # System dialog is within < >

    # Split the text into lines
    lines = text.splitlines()

    # Classify each line
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Check if the line matches the speech pattern
        speech_match = speech_pattern.findall(line)
        if speech_match:
            speech.extend(speech_match)
            continue  # Skip to next line

        # Check if the line matches the system dialog pattern
        system_match = system_dialog_pattern.findall(line)
        if system_match:
            system_dialog.extend(system_match)
            continue  # Skip to next line

        # If it's not speech or system dialog, treat it as narration
        narration.append(line)

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

    # Classify the text into speech, system dialog, and narration
    classified_text = classify_text(text)

    # Save the classified text to a JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(classified_text, f, indent=4)

    print(f"Processed text saved to {output_json_file}")

# Example usage
if __name__ == "__main__":
    # Path to your input text file
    input_file = '86--EIGHTY-SIX Volume-1.txt'  # Change this to your file path
    output_json_file = 'output3.json'  # Output JSON file

    # Process the text and save to JSON
    process_text_file(input_file, output_json_file)