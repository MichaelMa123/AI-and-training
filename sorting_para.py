import re
import json

# Function to remove sequential page numbers
def remove_page_numbers(text):
    # Split text into lines
    lines = text.splitlines()
    cleaned_lines = []
    
    # Initialize the expected page number
    expected_page_number = 1
    
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

    # State tracking
    in_speech = False
    in_system_dialog = False

    # Initialize accumulators
    current_narration = ""
    current_speech = ""
    current_system_dialog = ""

    # Split the text into lines
    lines = text.splitlines()

    # Process each line
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        while line:
            # Handle speech mode (Unicode-aware for straight and curly quotes)
            if in_speech:
                # Look for closing speech quote (curly or straight quotes)
                if '”' in line or '"' in line:
                    quote_type = '”' if '”' in line else '"'
                    current_speech += " " + line.split(quote_type, 1)[0] + quote_type
                    speech.append(current_speech.strip())
                    current_speech = ""
                    in_speech = False  # Exit speech mode
                    # Process remaining part of line as narration
                    line = line.split(quote_type, 1)[1]
                else:
                    current_speech += " " + line
                    line = ""  # Finish processing this line in speech mode
                    continue

            # Handle system dialog mode (Unicode-aware for angle brackets)
            if in_system_dialog:
                # Look for closing system dialog
                if '>' in line:
                    current_system_dialog += " " + line.split(">", 1)[0] + ">"
                    system_dialog.append(current_system_dialog.strip())
                    current_system_dialog = ""
                    in_system_dialog = False  # Exit system dialog mode
                    # Process remaining part of line as narration
                    line = line.split(">", 1)[1]
                else:
                    current_system_dialog += " " + line
                    line = ""  # Finish processing this line in system dialog mode
                    continue

            # Check if entering speech (curly or straight quotes)
            if '“' in line or '"' in line:
                quote_type = '“' if '“' in line else '"'
                # Save any accumulated narration
                if current_narration:
                    narration.append(current_narration.strip())
                    current_narration = ""
                # Enter speech mode
                current_speech = line.split(quote_type, 1)[1]
                in_speech = True
                line = line.split(quote_type, 1)[1]  # Continue processing the line after opening speech
                continue

            # Check if entering system dialog
            if '<' in line:
                # Save any accumulated narration
                if current_narration:
                    narration.append(current_narration.strip())
                    current_narration = ""
                # Enter system dialog mode
                current_system_dialog = line.split("<", 1)[1]
                in_system_dialog = True
                line = line.split("<", 1)[1]  # Continue processing the line after opening system dialog
                continue

            # Accumulate narration if not in speech or system dialog mode
            current_narration += " " + line
            line = ""  # Finish processing this line

    # If there is any leftover narration, store it
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
        json.dump(classified_text, f, ensure_ascii=False, indent=4)

    print(f"Processed text saved to {output_json_file}")

# Example usage
if __name__ == "__main__":
    # Path to your input text file
    input_file = '86--EIGHTY-SIX Volume-1.txt'  # Change this to your file path
    output_json_file = 'output15.json'  # Output JSON file

    # Process the text and save to JSON
    process_text_file(input_file, output_json_file)
