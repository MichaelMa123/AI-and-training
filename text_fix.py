import json
import re

def fix_json_format(input_file: str, output_file: str):
    # Step 1: Read the raw file content as text
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Step 2: Print the first few lines of content for inspection (optional for debugging)
    print("First 200 characters of the file for inspection:")
    print(content[:200])

    # Step 3: Replace single quotes with double quotes (JSON requires double quotes)
    content = re.sub(r"'", r'"', content)

    # Step 4: Remove any invalid characters at the beginning or end (e.g., BOM markers)
    content = content.strip()

    # Step 5: Attempt to load the fixed content as JSON
    try:
        data = json.loads(content)  # Parse the fixed content
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON after fixing format: {e}")
        return

    # Step 6: Write the fixed JSON back to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Fixed JSON saved to {output_file}")

# Example usage
input_file = 'output10.json'  # Replace with your actual JSON file path
output_file = 'output_fixed10.json'  # Replace with your desired output file path
fix_json_format(input_file, output_file)
