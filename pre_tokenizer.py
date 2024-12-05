import json

# Step 1: Load the original JSON structure from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Step 2: Convert multiple lists (speech, system_dialog, and narration) into id-text-tag format
def convert_to_id_text_tag(data):
    result = []
    id_counter = 1  # Initialize an ID counter
    
    # Helper function to convert a list to id-text-tag format and append to result
    def add_to_result(category_list, tag):
        nonlocal id_counter  # Use the outer `id_counter` variable
        for text in category_list:
            result.append({
                "id": id_counter,
                "text": text,
                "tag": tag  # Add the tag (either 'speech', 'system_dialog', or 'narration')
            })
            id_counter += 1

    # Extract and process the speech, system_dialog, and narration lists
    speech_list = data.get('speech', [])
    system_dialog_list = data.get('system_dialog', [])
    narration_list = data.get('narration', [])

    # Convert each list and append to result with its respective tag
    add_to_result(speech_list, "speech")
    add_to_result(system_dialog_list, "system_dialog")
    add_to_result(narration_list, "narration")
    
    return result

# Step 3: Save the converted result into a new JSON file
def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # Pretty-print the JSON

# Main function to handle the conversion
def process_json_to_id_text_tag(input_file, output_file):
    # Load the original JSON
    data = load_json(input_file)
    
    # Convert the lists to the new format
    converted_data = convert_to_id_text_tag(data)
    
    # Save the converted data to a new file
    save_json(converted_data, output_file)
    print(f"Converted data saved to {output_file}")

# Example usage
input_file = 'output15.json'  # Path to the original JSON file
output_file = 'converted_data_with_tags.json'  # Path to save the new format
process_json_to_id_text_tag(input_file, output_file)
