#python script to refine my raw wahtsapp chat txt file so that there is only conversation:

import re
import os
import glob

def clean_whatsapp_chat(input_folder, output_filename):
    """
    Cleans all raw WhatsApp chat exports in the given folder by removing 
    timestamps and system messages, leaving ONLY the "Name: Message" format.
    Merges them into a single massive dataset.
    """
    # Find all .txt files in the input folder
    search_pattern = os.path.join(input_folder, "*.txt")
    chat_files = glob.glob(search_pattern)
    
    if not chat_files:
        print(f"Error: Could not find any .txt files in '{input_folder}'.")
        return

    print(f"Found {len(chat_files)} chat files to process...")
    clean_lines = []
    
    # This Regex catches BOTH Android and iOS WhatsApp timestamp formats
    timestamp_pattern = re.compile(r'^\[?\d{1,2}/\d{1,2}/\d{2,4}[,\.]\s\d{1,2}:\d{2}(?:[:\.]\d{2})?\s*[a-zA-Z]*\]?\s*(?:-\s*)?')

    for file_path in chat_files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  -> Skipping {file_path} due to error: {e}")
            continue

        for line in lines:
            # 1. Ignore annoying system messages
            line_lower = line.lower()
            if "end-to-end encrypted" in line_lower or "omitted" in line_lower or "messages and calls are" in line_lower:
                continue
                
            # 2. Remove the timestamp using the regex
            cleaned_line = re.sub(timestamp_pattern, '', line).strip()
            
            # 3. Only save the line if it actually contains text
            if cleaned_line:
                clean_lines.append(cleaned_line)
                
        # Add a visual separator between different chats in the dataset (optional but helpful)
        clean_lines.append("\n--- END OF CHAT ---\n")

    # Write the massive merged dataset
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_lines))
        
    print(f"\nSuccess! Merged and cleaned {len(clean_lines)} lines of conversation.")
    print(f"Your huge training data is saved as: {output_filename}")

if __name__ == "__main__":
    # INSTRUCTIONS:
    # 1. Create a folder named 'raw_chats' inside your data folder.
    # 2. Export your WhatsApp chats and put ALL those .txt files inside 'raw_chats'
    # 3. Run this script to merge them into one giant clean dataset!
    
    # Create the folder if it doesn't exist
    input_folder = 'raw_chats'
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created folder '{input_folder}'. Please drop your chat .txt files in there and run again!")
    else:
        output_file = 'clean_chat.txt'
        print("Starting bulk data cleaner...")
        clean_whatsapp_chat(input_folder, output_file)