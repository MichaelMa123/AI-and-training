import re
import json

# Function to classify the text into speech, system dialog, and narration
def classify_text(text):
    # Lists to store the different types of text
    speech = []
    system_dialog = []
    narration = []

    # Regex patterns to detect speech (within curly quotes or straight quotes) and system dialog
    speech_pattern = re.compile(r'[“"](.*?)[”"]', re.DOTALL)  # Match both curly and straight quotes for speech
    system_dialog_pattern = re.compile(r'<(.*?)>', re.DOTALL)  # System dialog is within < >

    # Split the text into lines
    lines = text.splitlines()

    # Classify each line
    for line in lines:
        line = line.strip()
        if not line or line.isdigit():  # Skip empty lines or page numbers
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

# Function to process text from a file, fix Unicode sequences, classify the text, and save it in JSON format
def process_text_file(input_text, output_json_file):
    # Fix all Unicode sequences (e.g., \u2026 -> …)
    input_text = input_text.encode('utf-8').decode('unicode_escape')

    # Classify the text into speech, system dialog, and narration
    classified_text = classify_text(input_text)

    # Save the classified text to a JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(classified_text, f, indent=4)

    print(f"Processed text saved to {output_json_file}")

# Example usage
if __name__ == "__main__":
    # Your input text (example)
    input_text = """
<RMI M1A4 Juggernaut OS Version 8.15>
A rumbling cacophony mixed into the noise of the radio transmission.
“Handler One to Undertaker. Enemy interception force is visible on radar. We’ve confirmed a battalion-size unit of Anti-Tank Artillery types as well as a force of Dragoon types of similar size.”
“Acknowledged, Undertaker. I can sense them from here.”
“Command is transferred to the commanding officer on the field, effective immediately. Show gratitude to your homeland with your flesh and blood and defend the Republic with your very life.”
“Roger.”
“…I’m sorry, you guys. I’m so sorry.”
<End transmission>
<Cockpit sealed>
<Power pack activated. Actuator engaged. Joint-lock mechanism released.>
<Stabilizer: operating normally. FCS: compatible. Vetronics: off-line. Enemy scouting mode: passive.>
“Undertaker to all units. Handler One has relinquished command. Henceforth, Undertaker will take command of the operation.”
“Acknowledged, Alpha Leader. Same as always, right, Reaper? What did our cowardly wuss of an owner say in the end there?”
“That they’re sorry.”

12

The voice at the other end of the Para-RAID burst into laughter.
“Ha, those white pigs never change. They drive us out, lock us up, and then plug their ears and say they’re sorry? The hell… All units, you heard him. If we gotta march to our deaths anyway, at the very least, it might not be so bad with our trusty Reaper there to guide us.”
“Sixty seconds till contact with the enemy… The bombardment’s coming. Break through the enemy’s bombardment zone at maximum combat speed.”
“Let’s do this, boys!”
"""

    # Path to save the output JSON file
    output_json_file = 'sorted_text.json'

    # Process the text and save to JSON
    process_text_file(input_text, output_json_file)
