#!/usr/bin/env python3
"""
LegalBench-RAG Data Setup Script

This script helps download and set up the LegalBench-RAG dataset for evaluation.
It can either download the pre-built dataset or help generate it from source datasets.

Usage:
    # Download pre-built dataset (recommended)
    python setup_legalbench_data.py --download

    # Get information about the dataset
    python setup_legalbench_data.py --info

    # Check if dataset exists locally
    python setup_legalbench_data.py --check
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import zipfile

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalBenchDatasetManager:
    """Manager for LegalBench-RAG dataset setup and validation."""

    def __init__(self, data_directory: str = "data"):
        # Handle relative paths from eval/scripts directory
        if not Path(data_directory).is_absolute():
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_directory = project_root / data_directory
        else:
            self.data_directory = Path(data_directory)
        self.corpus_path = self.data_directory / "legalbench_corpus"
        self.benchmarks_path = self.data_directory / "legalbench_benchmarks"

        # URLs for dataset download (these would need to be updated with actual URLs)
        self.dataset_info = {
            "download_url": "https://example.com/legalbench-rag-dataset.zip",  # Placeholder
            "paper_url": "https://arxiv.org/abs/2408.10343",
            "github_url": "https://github.com/zeroentropy-ai/legalbenchrag",
            "description": "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain"
        }

    def check_dataset_exists(self) -> Dict[str, Any]:
        """Check if the LegalBench-RAG dataset exists locally."""
        corpus_exists = self.corpus_path.exists()
        benchmarks_exists = self.benchmarks_path.exists()

        corpus_files = 0
        benchmark_files = 0

        if corpus_exists:
            corpus_files = len(list(self.corpus_path.rglob("*.txt")))

        if benchmarks_exists:
            benchmark_files = len(list(self.benchmarks_path.glob("*.json")))

        return {
            "corpus_exists": corpus_exists,
            "benchmarks_exists": benchmarks_exists,
            "corpus_files": corpus_files,
            "benchmark_files": benchmark_files,
            "corpus_path": str(self.corpus_path),
            "benchmarks_path": str(self.benchmarks_path),
            "ready_for_evaluation": corpus_exists and benchmarks_exists and corpus_files > 0 and benchmark_files > 0
        }

    def create_sample_dataset(self) -> bool:
        """Create a sample dataset for testing purposes."""
        logger.info("Creating sample LegalBench-RAG dataset for testing...")

        # Create directories
        self.corpus_path.mkdir(parents=True, exist_ok=True)
        self.benchmarks_path.mkdir(parents=True, exist_ok=True)

        # Create sample corpus documents
        sample_corpus = [
            {
                "filename": "contract_001.txt",
                "content": """EMPLOYMENT AGREEMENT

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
Date: January 1, 2024"""
            },
            {
                "filename": "nda_002.txt",
                "content": """NON-DISCLOSURE AGREEMENT

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
Date: February 15, 2024"""
            },
            {
                "filename": "license_003.txt",
                "content": """SOFTWARE LICENSE AGREEMENT

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

6.1 Term: License effective for one year, automatically renewable.

6.2 Termination for Breach: Either party may terminate for material breach with 30 days cure period.

6.3 Effect of Termination: Licensee must cease use and destroy all copies.

Executed by:
DataSoft Inc.
Enterprise Client Corp
Date: March 1, 2024"""
            }
        ]

        # Write corpus files
        for doc in sample_corpus:
            file_path = self.corpus_path / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            logger.info(f"Created corpus file: {file_path.name}")

        # Create sample benchmark
        sample_benchmark = {
            "benchmark_name": "sample_legal_contracts",
            "description": "Sample benchmark for testing legal contract understanding",
            "version": "1.0",
            "test_cases": [
                {
                    "query": "What is the base salary mentioned in the employment agreement?",
                    "ground_truth": [
                        {
                            "file_path": "contract_001.txt",
                            "start_char": 591,
                            "end_char": 652,
                            "text": "Company shall pay Employee an annual base salary of $120,000."
                        }
                    ],
                    "expected_answer": "The base salary is $120,000 annually."
                },
                {
                    "query": "How long does the non-disclosure agreement remain in effect?",
                    "ground_truth": [
                        {
                            "file_path": "nda_002.txt",
                            "start_char": 973,
                            "end_char": 1045,
                            "text": "This Agreement remains in effect for 3 years from the date of execution."
                        }
                    ],
                    "expected_answer": "The non-disclosure agreement remains in effect for 3 years."
                },
                {
                    "query": "What are the restrictions on the software license?",
                    "ground_truth": [
                        {
                            "file_path": "license_003.txt",
                            "start_char": 502,
                            "end_char": 612,
                            "text": "- No modification or reverse engineering\n- No redistribution or sublicensing\n- No use for competitive products"
                        }
                    ],
                    "expected_answer": "The license restricts modification, reverse engineering, redistribution, sublicensing, and use for competitive products."
                },
                {
                    "query": "What benefits are available to the employee?",
                    "ground_truth": [
                        {
                            "file_path": "contract_001.txt",
                            "start_char": 694,
                            "end_char": 845,
                            "text": "- Health insurance with 90% company contribution\n- 401(k) plan with 4% company match\n- 15 days paid vacation annually\n- Stock option plan participation"
                        }
                    ],
                    "expected_answer": "Benefits include health insurance with 90% company contribution, 401(k) with 4% match, 15 days vacation, and stock options."
                },
                {
                    "query": "What is the annual license fee for the software?",
                    "ground_truth": [
                        {
                            "file_path": "license_003.txt",
                            "start_char": 658,
                            "end_char": 694,
                            "text": "Licensee shall pay $50,000 annually."
                        }
                    ],
                    "expected_answer": "The annual license fee is $50,000."
                }
            ]
        }

        # Write benchmark file
        benchmark_file = self.benchmarks_path / "sample_contracts_benchmark.json"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(sample_benchmark, f, indent=2, ensure_ascii=False)

        logger.info(f"Created benchmark file: {benchmark_file.name}")
        logger.info("Sample dataset created successfully!")

        return True

    def download_dataset(self) -> bool:
        """Download legal datasets from available sources."""
        logger.info("🚀 Downloading real legal datasets...")

        try:
            # Download the real LegalBench-RAG dataset from Dropbox
            if not REQUESTS_AVAILABLE:
                logger.error("requests library not available. Install with: pip install requests")
                return False

            import subprocess

            # Convert Dropbox share link to direct download link
            dropbox_url = "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1"

            logger.info("📥 Downloading real LegalBench-RAG dataset from official source...")
            logger.info(f"🔗 URL: {dropbox_url}")

            # Create directories
            self.corpus_path.mkdir(parents=True, exist_ok=True)
            self.benchmarks_path.mkdir(parents=True, exist_ok=True)

            # Download the zip file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                logger.info("⬇️  Downloading dataset archive...")
                response = requests.get(dropbox_url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"📊 Downloaded: {percent:.1f}%")

                tmp_file_path = tmp_file.name

            logger.info("📦 Extracting dataset...")

            # Extract the zip file
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                # List contents first
                file_list = zip_ref.namelist()
                logger.info(f"📋 Archive contains {len(file_list)} files")

                # Extract to data directory
                zip_ref.extractall(self.data_directory)

                # Find corpus and benchmarks directories
                corpus_found = False
                benchmarks_found = False

                for file_path in file_list:
                    if 'corpus' in file_path.lower():
                        corpus_found = True
                    if 'benchmark' in file_path.lower():
                        benchmarks_found = True

                logger.info(f"✅ Extracted dataset")
                logger.info(f"📁 Corpus files found: {corpus_found}")
                logger.info(f"📁 Benchmark files found: {benchmarks_found}")

            # Clean up temporary file
            os.unlink(tmp_file_path)

            # Verify extraction worked
            status = self.check_dataset_exists()
            if status["ready_for_evaluation"]:
                logger.info("🎉 Real LegalBench-RAG dataset successfully downloaded and extracted!")
                return True
            else:
                logger.warning("⚠️  Dataset extraction may not have worked as expected")
                return False


        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            logger.info("💡 Falling back to sample dataset creation...")
            return self.create_sample_dataset()

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the dataset structure and content."""
        status = self.check_dataset_exists()

        if not status["ready_for_evaluation"]:
            return {
                "valid": False,
                "errors": ["Dataset not found or incomplete"],
                "status": status
            }

        errors = []
        warnings = []

        # Check corpus structure
        corpus_files = list(self.corpus_path.rglob("*.txt"))
        if len(corpus_files) == 0:
            errors.append("No text files found in corpus")

        # Check benchmark structure
        benchmark_files = list(self.benchmarks_path.glob("*.json"))
        if len(benchmark_files) == 0:
            errors.append("No JSON benchmark files found")

        # Validate benchmark files
        total_test_cases = 0
        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)

                if 'test_cases' not in benchmark_data:
                    errors.append(f"No test_cases in {benchmark_file.name}")
                else:
                    test_cases = benchmark_data['test_cases']
                    total_test_cases += len(test_cases)

                    # Validate test case structure
                    for i, test_case in enumerate(test_cases):
                        if 'query' not in test_case:
                            warnings.append(f"Test case {i} in {benchmark_file.name} missing query")
                        if 'ground_truth' not in test_case:
                            warnings.append(f"Test case {i} in {benchmark_file.name} missing ground_truth")

            except Exception as e:
                errors.append(f"Error reading {benchmark_file.name}: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "statistics": {
                "corpus_files": len(corpus_files),
                "benchmark_files": len(benchmark_files),
                "total_test_cases": total_test_cases
            },
            "status": status
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the LegalBench-RAG dataset."""
        return {
            "dataset_info": self.dataset_info,
            "local_status": self.check_dataset_exists(),
            "validation": self.validate_dataset() if self.check_dataset_exists()["ready_for_evaluation"] else None
        }


def print_dataset_status(info: Dict[str, Any]) -> None:
    """Print formatted dataset status information."""
    print("\n" + "="*80)
    print("📚 LEGALBENCH-RAG DATASET STATUS")
    print("="*80)

    # Dataset information
    dataset_info = info["dataset_info"]
    print(f"📄 Description: {dataset_info['description']}")
    print(f"📖 Paper: {dataset_info['paper_url']}")
    print(f"💻 GitHub: {dataset_info['github_url']}")

    # Local status
    local_status = info["local_status"]
    print(f"\n📁 Local Dataset Status:")
    print(f"   Corpus Path: {local_status['corpus_path']}")
    print(f"   Benchmarks Path: {local_status['benchmarks_path']}")
    print(f"   Corpus Files: {local_status['corpus_files']}")
    print(f"   Benchmark Files: {local_status['benchmark_files']}")

    if local_status["ready_for_evaluation"]:
        print(f"   ✅ Ready for evaluation")
    else:
        print(f"   ❌ Not ready for evaluation")

    # Validation results
    validation = info.get("validation")
    if validation:
        print(f"\n🔍 Dataset Validation:")
        if validation["valid"]:
            print(f"   ✅ Dataset is valid")
        else:
            print(f"   ❌ Dataset has issues:")
            for error in validation["errors"]:
                print(f"      - {error}")

        if validation["warnings"]:
            print(f"   ⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"      - {warning}")

        stats = validation["statistics"]
        print(f"   📊 Statistics:")
        print(f"      - Corpus Files: {stats['corpus_files']}")
        print(f"      - Benchmark Files: {stats['benchmark_files']}")
        print(f"      - Total Test Cases: {stats['total_test_cases']}")

    print("\n" + "="*80)


def main():
    """Main entry point for the dataset setup script."""
    parser = argparse.ArgumentParser(
        description="Setup and manage LegalBench-RAG dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset status
  python setup_legalbench_data.py --check

  # Get dataset information
  python setup_legalbench_data.py --info

  # Download dataset (when available)
  python setup_legalbench_data.py --download

  # Create sample dataset for testing
  python setup_legalbench_data.py --create-sample
        """
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if dataset exists locally"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Get information about the dataset"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official LegalBench-RAG dataset"
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset for testing"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset structure"
    )

    parser.add_argument(
        "--data-directory",
        default="data",
        help="Directory for dataset files"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If no action specified, show info
    if not any([args.check, args.info, args.download, args.create_sample, args.validate]):
        args.info = True

    # Create dataset manager
    manager = LegalBenchDatasetManager(args.data_directory)

    try:
        if args.check:
            status = manager.check_dataset_exists()
            print_dataset_status({"dataset_info": manager.dataset_info, "local_status": status})

        elif args.info:
            info = manager.get_dataset_info()
            print_dataset_status(info)

        elif args.download:
            success = manager.download_dataset()
            if success:
                print("✅ Dataset setup completed successfully!")
                info = manager.get_dataset_info()
                print_dataset_status(info)
            else:
                print("❌ Dataset download failed!")
                return 1

        elif args.create_sample:
            success = manager.create_sample_dataset()
            if success:
                print("✅ Sample dataset created successfully!")
                info = manager.get_dataset_info()
                print_dataset_status(info)
            else:
                print("❌ Sample dataset creation failed!")
                return 1

        elif args.validate:
            validation = manager.validate_dataset()
            print("\n📋 Dataset Validation Results:")
            if validation["valid"]:
                print("✅ Dataset is valid and ready for evaluation!")
            else:
                print("❌ Dataset validation failed:")
                for error in validation["errors"]:
                    print(f"   - {error}")
                return 1

        return 0

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
