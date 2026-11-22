"""
Instagram DM JSON Parser for You-GPT
=====================================
Parses the official Instagram JSON data export (from Meta Accounts Center)
and converts all direct messages into the clean dialogue format:
  <Contact>: <Message>
  Parshwa Nahar: <Response>

Handles:
- Meta's latin1/utf-8 encoding quirk for emojis and special characters
- Skips shared reel links, audio/video call notifications, and attachment placeholders
- Merges into your dataset automatically
"""

import os, json, glob, re

def fix_encoding(text):
    """Fixes Meta's infamous latin1 encoding bug on Unicode characters and emojis."""
    if not text:
        return ""
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def is_valid_message(content):
    """Filters out automated messages, media links, and call notifications."""
    if not content or len(content.strip()) == 0:
        return False
        
    c_lower = content.lower().strip()
    
    # System notifications & reactions
    if re.search(r'reacted\s+.*to\s+(your|a)\s+message', c_lower) or re.search(r'changed\s+the\s+group', c_lower):
        return False
        
    system_phrases = [
        "liked a message", "started an audio call",
        "started a video chat", "ended the call", "sent an attachment",
        "shared a story", "shared a reel", "shared a post", "voted in the poll",
        "pinned a message", "left the group"
    ]
    if any(p in c_lower for p in system_phrases):
        return False
        
    # Pure URL / Reel links
    if re.match(r'^https?://(www\.)?instagram\.com/(reel|p|stories)/', c_lower):
        return False
        
    return True

def parse_instagram_folder(inbox_path, output_txt_path, your_ig_name="parshwa"):
    """
    Scans the 'inbox' directory extracted from the Instagram ZIP export.
    Every conversation has its own subfolder containing one or more 'message_*.json' files.
    """
    if not os.path.exists(inbox_path):
        print(f"Error: Folder '{inbox_path}' not found!")
        print("Tip: Extract your Instagram zip file and point to the 'messages/inbox' folder.")
        return

    # Find all conversation directories
    chat_dirs = [d for d in glob.glob(os.path.join(inbox_path, "*")) if os.path.isdir(d)]
    
    if not chat_dirs:
        print(f"No conversation folders found in '{inbox_path}'.")
        return

    print(f"Found {len(chat_dirs)} Instagram conversation folders to process...")
    
    all_turns = []
    skipped_count = 0
    total_chats_with_data = 0
    
    for c_dir in chat_dirs:
        # Find all message_*.json in this conversation
        msg_files = glob.glob(os.path.join(c_dir, "message_*.json"))
        if not msg_files:
            continue
            
        # Instagram numbers message files: message_1 is newest, message_2 is older, etc.
        # So we sort by number descending to get oldest file first
        def get_msg_num(f_path):
            m = re.search(r'message_(\d+)\.json$', f_path)
            return int(m.group(1)) if m else 1
            
        msg_files.sort(key=get_msg_num, reverse=True)
        
        chat_lines = []
        for jf in msg_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                continue
                
            messages = data.get('messages', [])
            # Inside each file, messages are newest-first; reverse to make them oldest-first
            messages.reverse()
            
            for msg in messages:
                content = fix_encoding(msg.get('content', ''))
                sender = fix_encoding(msg.get('sender_name', ''))
                
                if not is_valid_message(content):
                    skipped_count += 1
                    continue
                    
                if your_ig_name.lower() in sender.lower():
                    speaker = "Parshwa Nahar"
                else:
                    speaker = sender.split()[0] if sender else "Friend"
                    # Remove emojis or weird characters from speaker name
                    speaker = re.sub(r'[^\w\s]', '', speaker).strip() or "Friend"
                    
                clean_content = content.replace('\r\n', ' ').replace('\n', ' ').strip()
                if clean_content:
                    chat_lines.append(f"{speaker}: {clean_content}")
                    
        if len(chat_lines) >= 2:
            all_turns.extend(chat_lines)
            all_turns.append("\n--- END OF CHAT ---\n")
            total_chats_with_data += 1

    print(f"Successfully processed {len(all_turns)} message lines (filtered {skipped_count:,} attachments/reels).")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_turns))
        
    print(f"Saved parsed Instagram chats to: {output_txt_path}")

if __name__ == "__main__":
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-detect common extraction paths
    candidate_paths = [
        os.path.join(script_dir, "raw_chats", "your_instagram_activity", "messages", "inbox"),
        os.path.join(script_dir, "raw_chats", "instagram", "messages", "inbox"),
        os.path.join(script_dir, "raw_chats", "messages", "inbox"),
        os.path.join(script_dir, "your_instagram_activity", "messages", "inbox"),
    ]
    
    inbox_dir = None
    if len(sys.argv) > 1:
        inbox_dir = sys.argv[1]
    else:
        for c in candidate_paths:
            if os.path.exists(c):
                inbox_dir = c
                break
                
    output_file = os.path.join(script_dir, "raw_chats", "instagram_clean.txt")
    
    if inbox_dir and os.path.exists(inbox_dir):
        print(f"Found Instagram inbox in: {inbox_dir}")
        parse_instagram_folder(inbox_dir, output_file)
    else:
        print("\n" + "=" * 60)
        print("INSTRUCTIONS FOR WHEN YOUR INSTAGRAM ZIP ARRIVES:")
        print("=" * 60)
        print("1. Download the Instagram ZIP export from Meta.")
        print("2. Extract the ZIP file.")
        print("3. Inside the extracted folder, find: 'messages/inbox'")
        print(f"4. Copy or move that 'inbox' folder to:")
        print(f"   {inbox_dir}")
        print("5. Run: python e:\\you-gpt\\data\\parse_instagram.py")
        print("=" * 60)
