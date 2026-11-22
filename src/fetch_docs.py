import requests
import urllib3
from pathlib import Path
from pypdf import PdfReader

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

docs = [
    {
        "name": "peca_2016.pdf",
        "url": "https://www.na.gov.pk/uploads/documents/1470910659_707.pdf",
        "version": "previous",
    },
    {
        "name": "peca_2025.pdf",
        "url": "https://www.na.gov.pk/uploads/documents/679b243193585_457.pdf",
        "version": "recent",
    },
    {
        "name": "hec_attestation_manual.pdf",
        "url": "https://www.hec.gov.pk/english/services/students/DAS/Documents/User-Manual.pdf",
        "version": "recent",
    },
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for doc in docs:
    output_path = Path("data/raw") / doc["name"]
    if not output_path.exists():
        print(f"Downloading {doc['name']} ({doc['version']})...")
        response = requests.get(doc["url"], headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)
    reader = PdfReader(output_path)
    print(f"Ready: {doc['name']} | Version: {doc['version']} | Pages: {len(reader.pages)}")

# Also save official NADRA FRC & CNIC guidelines as a structured reference document
nadra_path = Path("data/raw/nadra_frc_policy.txt")
nadra_text = """NADRA Family Registration Certificate (FRC) & Identity Rules (Official Summary)
Issued under Rule 19 of the NADRA (NIC) Rules, 2002.
1. Categories of FRC:
   - By Birth: Includes details of applicant's parents and siblings.
   - By Marriage: Includes details of applicant's spouse and children.
   - By Adoption: Includes details of legal guardian and adopted family members.
   - By All: Combines both alpha (birth) and beta (marriage) family members.
2. Eligibility & Documents Required:
   - Applicant must be 18 years or older with a valid 13-digit CNIC, NICOP, or POC.
   - Every included family member must be registered with NADRA and hold a 13-digit ID (CNIC, NICOP, POC, or Child Registration Certificate / B-Form).
   - Required documents: Original valid CNIC/NICOP of applicant, ID numbers/copies of family members, Marriage Certificate (for By Marriage), and birth certificates where applicable.
3. Application Channels & Fees:
   - Apply online via NADRA Pak-Identity portal (digital PDF sent via email within 24 hours) or in-person at any NADRA Registration Center (NRC).
   - Standard FRC fee (By Birth, By Marriage, By Adoption): Rs. 1,000.
   - Combined 'By All' FRC fee: Rs. 2,000.
"""
nadra_path.write_text(nadra_text.strip(), encoding="utf-8")
print("Ready: nadra_frc_policy.txt | Version: recent")