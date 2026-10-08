import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import yaml

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("generate_codeswitch_scripts")

# 50 Control (neutral, casual, workplace, daily life) and 50 Fraud-register (urgency, family, bank, credentials)
# Strictly scripted scenarios — NEVER real fraud call recordings.
SENTENCE_TEMPLATES = [
    # --- CONTROL / NEUTRAL SENTENCES (1 to 50) ---
    {"text": "Mera laptop start nahi ho raha, can you check the battery?", "register": "control", "pair": "hi-en"},
    {"text": "Main market ja raha hoon to buy some fresh vegetables.", "register": "control", "pair": "hi-en"},
    {"text": "Kal subah meeting schedule ki hai, please send the presentation.", "register": "control", "pair": "hi-en"},
    {"text": "Chalo lunch ke liye chalte hain, I am very hungry today.", "register": "control", "pair": "hi-en"},
    {"text": "Weather forecast check kiya tha, it might rain this evening.", "register": "control", "pair": "hi-en"},
    {"text": "Flight tickets book kar li hain, departure is at six in the morning.", "register": "control", "pair": "hi-en"},
    {"text": "Yeh movie trailer bohot interesting lag raha hai, let us book tickets.", "register": "control", "pair": "hi-en"},
    {"text": "Office ke baad gym jaana hai, need to do some workout.", "register": "control", "pair": "hi-en"},
    {"text": "Coffee peene chalte hain, let us take a short break.", "register": "control", "pair": "hi-en"},
    {"text": "Project report submit kar diya tha, now waiting for feedback.", "register": "control", "pair": "hi-en"},
    {"text": "Traffic bohot zyada hai, I will reach in thirty minutes.", "register": "control", "pair": "hi-en"},
    {"text": "Naya phone order kiya hai, delivery is expected tomorrow.", "register": "control", "pair": "hi-en"},
    {"text": "Doctor ne bola hai walk karne, so I go every evening.", "register": "control", "pair": "hi-en"},
    {"text": "Books table par rakhi hain, you can read them whenever you want.", "register": "control", "pair": "hi-en"},
    {"text": "Wi-Fi connection slow chal raha hai, please restart the router.", "register": "control", "pair": "hi-en"},
    {"text": "Weekend par trip plan karte hain, maybe somewhere near the hills.", "register": "control", "pair": "hi-en"},
    {"text": "Mera luggage pack ho gaya hai, just need to call a cab.", "register": "control", "pair": "hi-en"},
    {"text": "Yeh naya cafe try karte hain, they serve excellent pasta.", "register": "control", "pair": "hi-en"},
    {"text": "Electricity bill pay kar diya tha, receipt email par aayi hai.", "register": "control", "pair": "hi-en"},
    {"text": "Doorbell ring hui thi, maybe courier boy has arrived.", "register": "control", "pair": "hi-en"},
    {"text": "Stationery items khatam ho gaye hain, make a shopping list.", "register": "control", "pair": "hi-en"},
    {"text": "Purani photos dekh raha tha, those were really golden days.", "register": "control", "pair": "hi-en"},
    {"text": "Dinner ready ho gaya hai, please come to the dining table.", "register": "control", "pair": "hi-en"},
    {"text": "Keys drawer mein rakhi hain, do not forget to lock the door.", "register": "control", "pair": "hi-en"},
    {"text": "Sunday ko cricket match dekhenge, all friends are coming over.", "register": "control", "pair": "hi-en"},
    {"text": "Subah jaldi uthna hai, set the alarm for five AM.", "register": "control", "pair": "hi-en"},
    {"text": "Recipe bohot simple lagti hai, let us cook paneer tonight.", "register": "control", "pair": "hi-en"},
    {"text": "Window close kar do, cold air is coming inside.", "register": "control", "pair": "hi-en"},
    {"text": "New car test drive ke liye booked hai, on Saturday morning.", "register": "control", "pair": "hi-en"},
    {"text": "Garden mein water de diya tha, the flowers look lovely.", "register": "control", "pair": "hi-en"},
    {"text": "Music volume thoda kam karo, I am trying to read.", "register": "control", "pair": "hi-en"},
    {"text": "Passport renewal form fill kar diya hai, appointment is next week.", "register": "control", "pair": "hi-en"},
    {"text": "Bakery se fresh bread le aana, and a carton of milk.", "register": "control", "pair": "hi-en"},
    {"text": "Room clean karna padega, guests are arriving tomorrow.", "register": "control", "pair": "hi-en"},
    {"text": "Conference call connect nahi ho rahi, share the link again.", "register": "control", "pair": "hi-en"},
    {"text": "Chai thandi ho rahi hai, please drink it quickly.", "register": "control", "pair": "hi-en"},
    {"text": "Grocery store se fruits lene hain, like apples and bananas.", "register": "control", "pair": "hi-en"},
    {"text": "Document print nikaal liya hai, sign it on the last page.", "register": "control", "pair": "hi-en"},
    {"text": "Train time par chal rahi hai, no delay reported so far.", "register": "control", "pair": "hi-en"},
    {"text": "Shirt iron karni hai, wear formal clothes for the seminar.", "register": "control", "pair": "hi-en"},
    {"text": "Swimming pool clean ho gaya hai, we can go for a swim.", "register": "control", "pair": "hi-en"},
    {"text": "Mobile recharge khatam ho gaya, please do an online recharge.", "register": "control", "pair": "hi-en"},
    {"text": "Bache ground par khel rahe hain, they seem very happy.", "register": "control", "pair": "hi-en"},
    {"text": "Notebook bag mein rakh li hai, pens are in the pouch.", "register": "control", "pair": "hi-en"},
    {"text": "Library se novel borrow ki hai, it has received great reviews.", "register": "control", "pair": "hi-en"},
    {"text": "Shoes rack par rakh do, clean the floor properly.", "register": "control", "pair": "hi-en"},
    {"text": "Newspaper delivery late aayi aaj, let me check the headlines.", "register": "control", "pair": "hi-en"},
    {"text": "Battery charging par laga di hai, full charge in one hour.", "register": "control", "pair": "hi-en"},
    {"text": "Water bottle freeze mein rakh do, cold water feels refreshing.", "register": "control", "pair": "hi-en"},
    {"text": "Watch repair shop par deni hai, the strap got damaged.", "register": "control", "pair": "hi-en"},

    # --- FRAUD-REGISTER SENTENCES (urgency, family/credentials/money framing) (51 to 100) ---
    {"text": "Beta main police station mein hoon, send twenty thousand rupees immediately.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapka bank account block ho gaya hai, share the OTP right now.", "register": "fraud", "pair": "hi-en"},
    {"text": "Mera phone break ho gaya hai, please transfer money to this hospital account.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapki electricity power cut ho jayegi, pay the pending bill instantly.", "register": "fraud", "pair": "hi-en"},
    {"text": "Main bohot badi emergency mein phans gaya hoon, do not tell anyone at home.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapka credit card KYC expire ho gaya hai, click the verification link.", "register": "fraud", "pair": "hi-en"},
    {"text": "Hospital mein admission fees jama karni hai, hurry up please.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapke parcel mein illegal items mile hain, customs officer is on line.", "register": "fraud", "pair": "hi-en"},
    {"text": "Mera wallet kidnap ho gaya hai, urgently UPI ten thousand rupees.", "register": "fraud", "pair": "hi-en"},
    {"text": "Sim card deactivate hone wala hai, confirm your Aadhaar number now.", "register": "fraud", "pair": "hi-en"},
    {"text": "Bhai car accident ho gaya hai, need five lakh rupees for surgery.", "register": "fraud", "pair": "hi-en"},
    {"text": "Bank manager bol raha hoon, your account has been compromised.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapki lottery prize release karni hai, transfer processing tax immediately.", "register": "fraud", "pair": "hi-en"},
    {"text": "Maine naya number liya hai emergency ke liye, send funds on this Google Pay.", "register": "fraud", "pair": "hi-en"},
    {"text": "Tax department se notice aaya hai, penalty pay karo within ten minutes.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapka son custody mein hai, if you disconnect call we will file FIR.", "register": "fraud", "pair": "hi-en"},
    {"text": "Salary account freeze ho gaya hai, verify card details quickly.", "register": "fraud", "pair": "hi-en"},
    {"text": "Lawyer ki fee deni hai abhi, otherwise bail will be rejected.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapke debit card se suspicious transaction hua hai, tell me the code.", "register": "fraud", "pair": "hi-en"},
    {"text": "Station par luggage seize ho gaya hai, fine clearance requires cash.", "register": "fraud", "pair": "hi-en"},
    {"text": "Court warrant issue ho chuka hai, avoid arrest by settling right now.", "register": "fraud", "pair": "hi-en"},
    {"text": "Ambulance driver payment maang raha hai, transfer fifty dollars urgently.", "register": "fraud", "pair": "hi-en"},
    {"text": "Loan approval cancel ho jayega, submit processing charges right away.", "register": "fraud", "pair": "hi-en"},
    {"text": "Papa ki tabiyat critical hai ICU mein, wire the hospital deposit.", "register": "fraud", "pair": "hi-en"},
    {"text": "CBI verification team call kar rahi hai, share screen via remote app.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapka online delivery courier hold par hai, pay clearance charge now.", "register": "fraud", "pair": "hi-en"},
    {"text": "Friend ki bail ke liye collateral chahiye, do not ask questions just transfer.", "register": "fraud", "pair": "hi-en"},
    {"text": "Pension scheme deactivate ho jayegi, update biometric details today.", "register": "fraud", "pair": "hi-en"},
    {"text": "Gas pipeline connection disconnect hoga, pay warning notice fine immediately.", "register": "fraud", "pair": "hi-en"},
    {"text": "Mera car breakdown highway par hua hai, mechanic needs instant cash.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapke PAN card par cyber crime case registered hai, cooperate with investigation.", "register": "fraud", "pair": "hi-en"},
    {"text": "Chacha ji jail mein hain, arrange fifteen thousand rupees bail amount.", "register": "fraud", "pair": "hi-en"},
    {"text": "Net banking password expire ho gaya hai, authenticate via security code.", "register": "fraud", "pair": "hi-en"},
    {"text": "Police raid chal rahi hai, send money to settle the inspector.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapka insurance claim expire ho jayega, pay the premium surcharge.", "register": "fraud", "pair": "hi-en"},
    {"text": "Hostel room se nikaal diya hai, need rent money by this evening.", "register": "fraud", "pair": "hi-en"},
    {"text": "Telecom department calling, your mobile number will be terminated within one hour.", "register": "fraud", "pair": "hi-en"},
    {"text": "Accident victim critical hai, if you do not pay he will register complaint.", "register": "fraud", "pair": "hi-en"},
    {"text": "Mera handbag train mein chori ho gaya, buy me a return ticket right now.", "register": "fraud", "pair": "hi-en"},
    {"text": "Flight ticket cancel hone wali hai, pay rebooking penalty immediately.", "register": "fraud", "pair": "hi-en"},
    {"text": "Aapke account par money laundering probe shuru hui hai, transfer balance to safety account.", "register": "fraud", "pair": "hi-en"},
    {"text": "Sister hospital mein admit hai, medicine bill is pending since morning.", "register": "fraud", "pair": "hi-en"},
    {"text": "Job offer confirm karne ke liye, deposit registration fee immediately.", "register": "fraud", "pair": "hi-en"},
    {"text": "Income tax refund release karne ke liye, verify banking credentials.", "register": "fraud", "pair": "hi-en"},
    {"text": "Vehicle impound kar liya traffic police ne, send fine amount via UPI.", "register": "fraud", "pair": "hi-en"},
    {"text": "Colleague ka treatment chal raha hai, help with whatever amount you have.", "register": "fraud", "pair": "hi-en"},
    {"text": "Passport confiscate ho gaya immigration par, lawyer needs clearance deposit.", "register": "fraud", "pair": "hi-en"},
    {"text": "Mobile banking token revoke ho gaya hai, enter OTP sent to mobile.", "register": "fraud", "pair": "hi-en"},
    {"text": "Beti kidnap ho gayi hai, do not call police if you want her safe.", "register": "fraud", "pair": "hi-en"},
    {"text": "Secret investigation chal rahi hai, keep this conversation strictly confidential.", "register": "fraud", "pair": "hi-en"},
]

# Common Romanized Hindi words for rule-based token-level tagging fallback
ROMANIZED_HINDI_WORDS = {
    "mera", "meri", "mere", "main", "ja", "raha", "rahi", "hoon", "kal", "subah",
    "ki", "hai", "chalo", "ke", "liye", "chalte", "hain", "bohot", "tha", "thi",
    "the", "kar", "li", "yeh", "lag", "aaya", "aayi", "aaye", "baad", "peene",
    "diya", "zyada", "naya", "nayi", "ne", "bola", "par", "rakhi", "rakha", "slow",
    "chal", "khatam", "ho", "gaye", "purani", "dekh", "hua", "bache", "thandi",
    "thoda", "kam", "karo", "nikaal", "utna", "gaya", "beta", "station", "mein",
    "aapka", "aapki", "aapke", "hoga", "jayega", "jayegi", "badi", "phans", "karni",
    "mile", "kidnap", "hone", "wala", "wali", "bhai", "bol", "pension", "chacha",
    "papa", "tabiyat", "maang", "chahiye", "nahi", "ab", "abhi", "shuru", "beti",
    "khel", "lene", "paas", "sirf", "jaldi", "door", "paani", "khana", "ghar"
}


def tag_tokens_heuristic(tokens: List[str]) -> List[str]:
    """Tag tokens as 'hi' or 'en' based on vocabulary and linguistic heuristics.
    Punctuation is stripped for lookup; words not in Romanized Hindi dictionary default to English.
    """
    tags = []
    prev_tag = "hi"
    for tok in tokens:
        clean = tok.lower().strip(".,!?;:\"'()-")
        if not clean:
            tags.append(prev_tag)
            continue
        if clean in ROMANIZED_HINDI_WORDS:
            tag = "hi"
        else:
            tag = "en"
        tags.append(tag)
        prev_tag = tag
    return tags


def find_token_switches(tokens: List[str], tags: List[str]) -> List[Dict]:
    """Identify switch points at token transitions."""
    switches = []
    for i in range(len(tags) - 1):
        if tags[i] != tags[i + 1]:
            switches.append({
                "token_index": i,
                "from_token": tokens[i],
                "to_token": tokens[i + 1],
                "from_lang": tags[i],
                "to_lang": tags[i + 1],
            })
    return switches


def generate_scripts(output_dir: str = "data/processed") -> List[Dict]:
    """Generate script bank, tag switch points, and persist as CSV and JSONL."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, item in enumerate(SENTENCE_TEMPLATES):
        sentence_id = f"sent_{idx + 1:04d}"
        tokens = item["text"].split()
        tags = tag_tokens_heuristic(tokens)
        switches = find_token_switches(tokens, tags)

        rec = {
            "sentence_id": sentence_id,
            "text": item["text"],
            "tokens": tokens,
            "token_tags": tags,
            "codeswitch_points": switches,
            "num_switch_points": len(switches),
            "register": item["register"],
            "language_pair": item["pair"],
            "human_reviewed": False,  # Manual review flag per build spec
            "notes": "Scripted scenarios for research; no real fraud call audio used.",
        }
        records.append(rec)

    # Save JSONL
    jsonl_file = out_path / "codeswitch_scripts.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSV
    csv_file = out_path / "codeswitch_scripts.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sentence_id",
                "register",
                "language_pair",
                "num_switch_points",
                "human_reviewed",
                "text",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sentence_id": r["sentence_id"],
                "register": r["register"],
                "language_pair": r["language_pair"],
                "num_switch_points": r["num_switch_points"],
                "human_reviewed": r["human_reviewed"],
                "text": r["text"],
            })

    logger.info(
        f"Generated {len(records)} scripted sentences ({jsonl_file}, {csv_file}). "
        "HUMAN REVIEW REQUIRED: Native speakers should review sentences for naturalness."
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scripted code-switched sentences")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()
    generate_scripts(args.output_dir)

