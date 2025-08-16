#!/usr/bin/env python3
"""
Fix character positions in the sample benchmark data.
This script calculates the correct character positions for ground truth snippets.
"""

import json
from pathlib import Path

# Define the sample documents exactly as they appear in setup_legalbench_data.py
SAMPLE_DOCUMENTS = {
    "contract_001.txt": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on January 1, 2024, between TechCorp Inc., a Delaware corporation ("Company"), and John Smith ("Employee").

SECTION 1 - EMPLOYMENT TERMS

1.1 Position: Employee shall serve as Senior Software Engineer.

1.2 Duties: Employee's duties include:
- Software development and maintenance
- Code review and technical documentation
- Collaboration with cross-functional teams
- Mentoring junior developers

1.3 Reporting: Employee shall report to the Chief Technology Officer.

SECTION 2 - COMPENSATION

2.1 Base Salary: Company shall pay Employee an annual base salary of $120,000.

2.2 Benefits: Employee is eligible for:
- Health insurance with 90% company contribution
- 401(k) plan with 4% company match
- 15 days paid vacation annually
- Stock option plan participation

2.3 Performance Bonus: Employee may receive annual performance bonus up to 20% of base salary.

SECTION 3 - CONFIDENTIALITY

3.1 Non-Disclosure: Employee agrees to maintain confidentiality of all proprietary information.

3.2 Return of Property: Upon termination, Employee must return all company property.

SECTION 4 - TERMINATION

4.1 At-Will Employment: Employment may be terminated by either party with two weeks notice.

4.2 Severance: If terminated without cause, Employee receives two weeks severance pay.

This Agreement constitutes the entire agreement between the parties.

Signed:
Company: TechCorp Inc.
Employee: John Smith
Date: January 1, 2024""",

    "nda_002.txt": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is made between Innovate Solutions LLC ("Disclosing Party") and Beta Testing Corp ("Receiving Party") on February 15, 2024.

ARTICLE I - DEFINITION OF CONFIDENTIAL INFORMATION

Confidential Information includes:
- Technical data, algorithms, and source code
- Business plans and financial information
- Customer lists and market research
- Any information marked as confidential

ARTICLE II - OBLIGATIONS

2.1 Non-Disclosure: Receiving Party shall not disclose Confidential Information to third parties.

2.2 Limited Use: Information may only be used for evaluation purposes.

2.3 Standard of Care: Receiving Party shall use same degree of care as with own confidential information.

ARTICLE III - EXCLUSIONS

This Agreement does not apply to information that:
- Is publicly available
- Was known prior to disclosure
- Is independently developed
- Is required to be disclosed by law

ARTICLE IV - TERM

This Agreement remains in effect for 3 years from the date of execution.

ARTICLE V - REMEDIES

Breach of this Agreement may result in irreparable harm, entitling Disclosing Party to seek injunctive relief.

ARTICLE VI - GOVERNING LAW

This Agreement shall be governed by the laws of California.

Signed:
Innovate Solutions LLC
Beta Testing Corp
Date: February 15, 2024""",

    "license_003.txt": """SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("License") is between DataSoft Inc. ("Licensor") and Enterprise Client Corp ("Licensee") effective March 1, 2024.

SECTION 1 - GRANT OF LICENSE

1.1 License Grant: Licensor grants Licensee a non-exclusive, non-transferable license to use the DataAnalytics Pro software.

1.2 Permitted Uses:
- Installation on up to 50 workstations
- Use for internal business operations only
- Creating backup copies for archival purposes

1.3 Restrictions:
- No modification or reverse engineering
- No redistribution or sublicensing
- No use for competitive products

SECTION 2 - PAYMENT TERMS

2.1 License Fee: Licensee shall pay $50,000 annually.

2.2 Payment Schedule: Payment due within 30 days of invoice.

2.3 Late Fees: Overdue payments subject to 1.5% monthly interest.

SECTION 3 - SUPPORT AND MAINTENANCE

3.1 Technical Support: Licensor provides email support during business hours.

3.2 Updates: Licensee receives minor updates at no additional cost.

3.3 Major Versions: Upgrades to major versions require separate license.

SECTION 4 - WARRANTIES

4.1 Performance Warranty: Software will perform substantially as documented for 90 days.

4.2 Limitation: LICENSOR DISCLAIMS ALL OTHER WARRANTIES, EXPRESS OR IMPLIED.

SECTION 5 - LIABILITY

5.1 Limitation: Licensor's liability limited to license fees paid.

5.2 Exclusion: No liability for consequential or incidental damages.

SECTION 6 - TERMINATION

6.1 Termination for Breach: Either party may terminate for material breach with 30 days notice to cure.

6.2 Effect of Termination: Upon termination, Licensee must destroy all copies of software.

SECTION 7 - GENERAL

7.1 Entire Agreement: This License constitutes the entire agreement.

7.2 Assignment: License may not be assigned without written consent.

7.3 Governing Law: This License shall be governed by laws of New York.

Signed:
DataSoft Inc.
Enterprise Client Corp
Date: March 1, 2024"""
}


def find_text_position(document: str, search_text: str) -> tuple:
    """Find the exact character position of text in a document."""
    start = document.find(search_text)
    if start == -1:
        # Try with newlines normalized
        search_normalized = search_text.replace('\\n', '\n')
        start = document.find(search_normalized)
        if start == -1:
            return None, None
        end = start + len(search_normalized)
    else:
        end = start + len(search_text)
    return start, end


def validate_position(document: str, start: int, end: int, expected_text: str) -> bool:
    """Validate that the character positions extract the expected text."""
    extracted = document[start:end]
    # Normalize for comparison
    extracted_normalized = extracted.strip()
    expected_normalized = expected_text.strip()
    return extracted_normalized == expected_normalized


def calculate_correct_positions():
    """Calculate correct character positions for all test cases."""

    test_cases = [
        {
            "query": "What is the base salary mentioned in the employment agreement?",
            "file": "contract_001.txt",
            "text": "Company shall pay Employee an annual base salary of $120,000."
        },
        {
            "query": "How long does the non-disclosure agreement remain in effect?",
            "file": "nda_002.txt",
            "text": "This Agreement remains in effect for 3 years from the date of execution."
        },
        {
            "query": "What are the restrictions on the software license?",
            "file": "license_003.txt",
            "text": "- No modification or reverse engineering\n- No redistribution or sublicensing\n- No use for competitive products"
        },
        {
            "query": "What benefits are available to the employee?",
            "file": "contract_001.txt",
            "text": "- Health insurance with 90% company contribution\n- 401(k) plan with 4% company match\n- 15 days paid vacation annually\n- Stock option plan participation"
        },
        {
            "query": "What is the annual license fee for the software?",
            "file": "license_003.txt",
            "text": "Licensee shall pay $50,000 annually."
        }
    ]

    corrected_positions = []

    print("Calculating correct character positions...")
    print("=" * 60)

    for test_case in test_cases:
        file_name = test_case["file"]
        search_text = test_case["text"]

        if file_name in SAMPLE_DOCUMENTS:
            document = SAMPLE_DOCUMENTS[file_name]
            start, end = find_text_position(document, search_text)

            if start is not None:
                # Validate the extraction
                is_valid = validate_position(document, start, end, search_text)

                result = {
                    "query": test_case["query"],
                    "file_path": file_name,
                    "start_char": start,
                    "end_char": end,
                    "text": search_text,
                    "valid": is_valid
                }

                corrected_positions.append(result)

                print(f"✓ Query: {test_case['query'][:50]}...")
                print(f"  File: {file_name}")
                print(f"  Positions: {start}-{end}")
                print(f"  Valid: {is_valid}")
                print(f"  Preview: '{document[start:min(start+50, end)]}...'")
                print("-" * 40)
            else:
                print(f"✗ Could not find text for query: {test_case['query']}")
                print(f"  Looking for: '{search_text[:50]}...'")
                print("-" * 40)

    return corrected_positions


def generate_fixed_benchmark(corrected_positions):
    """Generate the fixed benchmark data structure."""

    # Group by query for the benchmark format
    benchmark = {
        "name": "sample_contracts_benchmark",
        "description": "Sample legal document Q&A benchmark for testing",
        "test_cases": []
    }

    # Map queries to expected answers
    expected_answers = {
        "What is the base salary mentioned in the employment agreement?": "The base salary is $120,000 annually.",
        "How long does the non-disclosure agreement remain in effect?": "The non-disclosure agreement remains in effect for 3 years.",
        "What are the restrictions on the software license?": "The license restricts modification, reverse engineering, redistribution, sublicensing, and use for competitive products.",
        "What benefits are available to the employee?": "Benefits include health insurance with 90% company contribution, 401(k) with 4% match, 15 days vacation, and stock options.",
        "What is the annual license fee for the software?": "The annual license fee is $50,000."
    }

    for pos in corrected_positions:
        if pos.get("valid", False):
            test_case = {
                "query": pos["query"],
                "ground_truth": [
                    {
                        "file_path": pos["file_path"],
                        "start_char": pos["start_char"],
                        "end_char": pos["end_char"],
                        "text": pos["text"]
                    }
                ],
                "expected_answer": expected_answers.get(pos["query"], "")
            }
            benchmark["test_cases"].append(test_case)

    return benchmark


if __name__ == "__main__":
    # Calculate correct positions
    corrected = calculate_correct_positions()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_count = sum(1 for p in corrected if p.get("valid", False))
    print(f"Total test cases: {len(corrected)}")
    print(f"Valid extractions: {valid_count}")

    if valid_count == len(corrected):
        print("\n✅ All character positions have been calculated correctly!")

        # Generate the fixed benchmark
        fixed_benchmark = generate_fixed_benchmark(corrected)

        # Save to file
        output_file = Path("fixed_sample_benchmark.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_benchmark, f, indent=2, ensure_ascii=False)

        print(f"\n📝 Fixed benchmark saved to: {output_file}")

        # Print the corrected positions for manual update
        print("\n" + "=" * 60)
        print("CORRECTED POSITIONS FOR setup_legalbench_data.py:")
        print("=" * 60)

        for pos in corrected:
            print(f"\n# {pos['query']}")
            print(f"start_char: {pos['start_char']}")
            print(f"end_char: {pos['end_char']}")
    else:
        print("\n⚠️ Some positions could not be calculated. Please check the text snippets.")
