#!/usr/bin/env python3
"""
RAG System Evaluation Runner

This script provides a command-line interface for evaluating RAG systems using:
1. LegalBench-RAG benchmark (full evaluation with downloaded dataset)
2. Quick evaluation (using sample documents and test cases)

Usage Examples:
    # Quick evaluation using sample documents
    python evaluate_rag.py --mode quick

    # Full LegalBench-RAG evaluation (requires downloaded dataset)
    python evaluate_rag.py --mode full --corpus-path data/legalbench_corpus --benchmarks-path data/legalbench_benchmarks

    # Evaluate specific benchmark file
    python evaluate_rag.py --mode full --benchmark-file cuad_benchmark.json

    # Save results to specific file
    python evaluate_rag.py --mode quick --output results/my_evaluation.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add backend and eval to path
eval_path = Path(__file__).parent.parent
backend_path = eval_path.parent / "backend"
sys.path.extend([str(eval_path), str(backend_path)])

# Set minimal environment variables to avoid settings issues
os.environ.setdefault("ALLOWED_ORIGINS", "[]")
os.environ.setdefault("DATABASE_URL", "sqlite:///temp.db")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "dummy")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")

from services.evaluation_service import LegalBenchRAGEvaluator, QuickEvaluator
from app.services.enhanced_document_service import EnhancedDocumentService
from app.services.enhanced_vectorstore import EnhancedVectorStore
from app.services.document_processor import DocumentProcessor

# For local evaluation without database
try:
    import faiss
    import numpy as np
    from tests.test_local_rag_run import LocalVectorStore, SimpleEmbeddingService
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# LangChain for OpenAI
try:
    from langchain_core.caches import BaseCache
    from langchain_openai import ChatOpenAI
    # Fix for Pydantic model rebuild issue
    ChatOpenAI.model_rebuild()
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluationRunner:
    """Main runner for RAG system evaluation."""

    def __init__(
        self,
        openai_api_key: str,
        use_local_vectorstore: bool = True,
        data_directory: str = "data"
    ):
        self.openai_api_key = openai_api_key
        self.use_local_vectorstore = use_local_vectorstore

        # Handle relative paths from eval/scripts directory
        if not Path(data_directory).is_absolute():
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_directory = project_root / data_directory
        else:
            self.data_directory = Path(data_directory)

        # Initialize services
        self.document_processor = DocumentProcessor(
            chunk_size=600,
            chunk_overlap=100,
            use_langchain=True
        )

        self.document_service = EnhancedDocumentService(self.document_processor)

        # Initialize embedding service
        self.embedding_service = SimpleEmbeddingService(openai_api_key)

        # Initialize vector store
        if use_local_vectorstore:
            self.vector_store = LocalVectorStore(self.embedding_service)
        else:
            # Use real database-backed vector store
            from app.services.enhanced_vectorstore import EnhancedVectorStore
            self.vector_store = EnhancedVectorStore(self.embedding_service)

        # Initialize LLM
        if LANGCHAIN_OPENAI_AVAILABLE:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4o-mini",
                temperature=0.1
            )
        else:
            self.llm = None
            logger.warning("LangChain OpenAI not available. Generation evaluation will be skipped.")

    async def setup_sample_documents(self) -> bool:
        """Set up sample documents for evaluation."""
        logger.info("Setting up sample documents...")

        # Ensure data directory exists
        self.data_directory.mkdir(exist_ok=True)

        # Check if we already have documents
        existing_files = list(self.data_directory.glob("*.txt")) + list(self.data_directory.glob("*.md"))
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing documents")

            # Process existing documents
            from tests.test_local_rag_run import MockUploadFile

            for file_path in existing_files:
                try:
                    mock_file = MockUploadFile(file_path)
                    enhanced_doc = await self.document_service.process_document_enhanced(
                        mock_file,
                        use_enhanced_models=True,
                        preserve_structure=True
                    )
                    await self.vector_store.store_enhanced_document(enhanced_doc)
                    logger.info(f"Processed {file_path.name}: {len(enhanced_doc.chunks)} chunks")

                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    return False

            return True

        # Create sample documents if none exist
        logger.info("Creating sample legal documents...")

        sample_docs = [
            {
                "filename": "software_development_agreement.txt",
                "content": """SOFTWARE DEVELOPMENT AGREEMENT

AGREEMENT made this 1st day of January, 2024, between ABC Corporation, a Delaware corporation ("Client"), and XYZ Development Services LLC, a California limited liability company ("Developer").

ARTICLE I - DEFINITIONS AND INTERPRETATION

1.1 "Agreement" means this Software Development Agreement, including all schedules, exhibits, and amendments.

1.2 "Client" means ABC Corporation and its affiliates, successors, and assigns.

1.3 "Developer" means XYZ Development Services LLC and its employees, contractors, and agents.

1.4 "Software" means the computer program, source code, object code, documentation, and related materials to be developed under this Agreement.

1.5 "Confidential Information" means any proprietary or confidential information disclosed by either party.

ARTICLE II - SCOPE OF WORK AND DELIVERABLES

2.1 Development Services
Developer agrees to provide the following services:
a) Requirements analysis and system design
b) Software architecture and technical specifications
c) Application development and programming
d) Quality assurance testing and debugging
e) Documentation and user training materials
f) Deployment and implementation support

2.2 Deliverables
The following deliverables shall be provided:
a) Technical requirements document
b) System architecture design
c) Source code and compiled application
d) User documentation and training materials
e) Testing reports and quality assurance documentation
f) Deployment guide and support documentation

2.3 Timeline
The project shall be completed in three phases:
Phase 1 (Months 1-2): Requirements and Design
Phase 2 (Months 3-5): Development and Testing
Phase 3 (Month 6): Deployment and Training

ARTICLE III - PAYMENT TERMS AND CONDITIONS

3.1 Total Contract Value
The total fee for all services under this Agreement is Seventy-Five Thousand Dollars ($75,000).

3.2 Payment Schedule
Payment shall be made according to the following schedule:
a) Thirty percent (30%) or $22,500 upon execution of this Agreement
b) Forty percent (40%) or $30,000 upon completion of Phase 2 milestones
c) Thirty percent (30%) or $22,500 upon final delivery and client acceptance

3.3 Late Payment Penalties
Any payment not made within thirty (30) days of the due date shall accrue interest at a rate of one and one-half percent (1.5%) per month or the maximum rate permitted by law, whichever is less.

3.4 Expenses
Client shall reimburse Developer for reasonable pre-approved expenses incurred in connection with the services.

ARTICLE IV - INTELLECTUAL PROPERTY RIGHTS

4.1 Ownership of Work Product
All intellectual property rights in the Software and related work product shall vest exclusively in Client upon full payment of all fees.

4.2 Developer Retained Rights
Developer retains ownership of its general methodologies, know-how, and pre-existing intellectual property.

4.3 License to Developer Tools
Client grants Developer a non-exclusive license to use any Client-provided tools or systems solely for performing services under this Agreement.

ARTICLE V - WARRANTIES AND REPRESENTATIONS

5.1 Developer Warranties
Developer warrants that:
a) The Software will perform substantially in accordance with specifications
b) The Software will be free from material defects for 90 days after delivery
c) All work will be performed in a professional and workmanlike manner
d) Developer has the authority to enter into this Agreement

5.2 Client Warranties
Client warrants that it has the authority to enter into this Agreement and will provide necessary cooperation and information.

5.3 Limitation of Liability
DEVELOPER'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE TOTAL AMOUNT PAID BY CLIENT UNDER THIS AGREEMENT.

ARTICLE VI - CONFIDENTIALITY

6.1 Confidentiality Obligations
Each party agrees to maintain the confidentiality of the other party's Confidential Information and use it solely for purposes of this Agreement.

6.2 Exceptions
Confidentiality obligations do not apply to information that is publicly available or independently developed.

ARTICLE VII - TERMINATION

7.1 Termination for Convenience
Either party may terminate this Agreement with thirty (30) days written notice.

7.2 Termination for Cause
Either party may terminate immediately upon material breach that remains uncured after fifteen (15) days written notice.

7.3 Effect of Termination
Upon termination, Client shall pay for all work completed and accepted through the termination date.

ARTICLE VIII - GENERAL PROVISIONS

8.1 Governing Law
This Agreement shall be governed by the laws of the State of California.

8.2 Entire Agreement
This Agreement constitutes the entire agreement between the parties and supersedes all prior negotiations and agreements.

8.3 Amendment
This Agreement may be amended only by written agreement signed by both parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

ABC CORPORATION                    XYZ DEVELOPMENT SERVICES LLC

By: _________________________      By: _________________________
Name: John Smith                   Name: Jane Doe
Title: Chief Technology Officer    Title: Managing Member
Date: January 1, 2024             Date: January 1, 2024"""
            },
            {
                "filename": "data_processing_agreement.txt",
                "content": """DATA PROCESSING AGREEMENT

This Data Processing Agreement ("DPA") is entered into between DataCorp Inc. ("Data Controller") and SecureCloud Services Ltd. ("Data Processor") effective as of March 15, 2024.

ARTICLE I - DEFINITIONS

1.1 "Personal Data" means any information relating to an identified or identifiable natural person.

1.2 "Processing" means any operation performed on Personal Data, including collection, recording, organization, structuring, storage, adaptation, retrieval, consultation, use, disclosure, dissemination, or erasure.

1.3 "Data Subject" means the natural person to whom Personal Data relates.

1.4 "GDPR" means the General Data Protection Regulation (EU) 2016/679.

ARTICLE II - SCOPE AND PURPOSE

2.1 Processing Activities
Data Processor will process Personal Data on behalf of Data Controller for the following purposes:
a) Cloud storage and backup services
b) Data analytics and reporting
c) Customer relationship management
d) Marketing automation and communication

2.2 Categories of Personal Data
The following categories of Personal Data may be processed:
a) Contact information (names, email addresses, phone numbers)
b) Demographic information (age, location, preferences)
c) Transaction data (purchase history, payment information)
d) Behavioral data (website interactions, app usage)

2.3 Categories of Data Subjects
Data Subjects include:
a) Customers and prospective customers
b) Website visitors and app users
c) Newsletter subscribers
d) Customer service contacts

ARTICLE III - DATA PROCESSOR OBLIGATIONS

3.1 Processing Instructions
Data Processor shall process Personal Data only on documented instructions from Data Controller, including transfers to third countries.

3.2 Confidentiality
Data Processor ensures that persons authorized to process Personal Data have committed themselves to confidentiality.

3.3 Security Measures
Data Processor shall implement appropriate technical and organizational measures, including:
a) Encryption of Personal Data in transit and at rest
b) Multi-factor authentication for system access
c) Regular security audits and penetration testing
d) Employee security training and background checks
e) Incident response and breach notification procedures

3.4 Sub-processing
Data Processor may engage sub-processors only with prior written authorization from Data Controller.

ARTICLE IV - DATA SUBJECT RIGHTS

4.1 Assistance with Rights Requests
Data Processor shall assist Data Controller in responding to Data Subject requests, including:
a) Access to Personal Data
b) Rectification of inaccurate data
c) Erasure of Personal Data
d) Restriction of processing
e) Data portability

4.2 Response Time
Data Processor shall respond to Data Controller's requests for assistance within 72 hours.

ARTICLE V - DATA TRANSFERS

5.1 International Transfers
Personal Data may be transferred to the following countries with adequate protection:
a) United States (under Privacy Shield or Standard Contractual Clauses)
b) United Kingdom (adequacy decision)
c) Canada (adequacy decision)

5.2 Safeguards
All international transfers shall be protected by appropriate safeguards as required by GDPR.

ARTICLE VI - DATA BREACH NOTIFICATION

6.1 Notification Requirement
Data Processor shall notify Data Controller without undue delay and in any case within 24 hours of becoming aware of a Personal Data breach.

6.2 Breach Information
Notifications shall include:
a) Description of the nature of the breach
b) Categories and approximate number of Data Subjects affected
c) Likely consequences of the breach
d) Measures taken or proposed to address the breach

ARTICLE VII - DATA PROTECTION IMPACT ASSESSMENTS

7.1 Assistance Requirement
Data Processor shall assist Data Controller in conducting Data Protection Impact Assessments when required.

7.2 Prior Consultation
Data Processor shall assist with prior consultation with supervisory authorities when necessary.

ARTICLE VIII - AUDITS AND INSPECTIONS

8.1 Audit Rights
Data Controller may conduct audits of Data Processor's compliance with this DPA, with reasonable notice.

8.2 Third-Party Audits
Data Processor maintains SOC 2 Type II and ISO 27001 certifications, which may satisfy audit requirements.

ARTICLE IX - DATA RETENTION AND DELETION

9.1 Retention Period
Personal Data shall be retained only for as long as necessary for the purposes outlined in this DPA.

9.2 Deletion Requirements
Upon termination of services, Data Processor shall delete or return all Personal Data within 30 days, unless retention is required by law.

ARTICLE X - LIABILITY AND INDEMNIFICATION

10.1 Liability Cap
Data Processor's liability for data protection violations shall not exceed $1,000,000 per incident.

10.2 Indemnification
Data Processor shall indemnify Data Controller against claims arising from Data Processor's violation of this DPA.

ARTICLE XI - TERM AND TERMINATION

11.1 Term
This DPA shall remain in effect for the duration of the underlying service agreement.

11.2 Survival
Obligations regarding confidentiality, data deletion, and liability shall survive termination.

ARTICLE XII - GOVERNING LAW

12.1 Applicable Law
This DPA shall be governed by the laws of the European Union and the jurisdiction where Data Controller is established.

IN WITNESS WHEREOF, the parties have executed this Data Processing Agreement.

DATACORP INC.                      SECURECLOUD SERVICES LTD.

By: _________________________      By: _________________________
Name: Sarah Johnson              Name: Michael Chen
Title: Chief Privacy Officer     Title: Chief Executive Officer
Date: March 15, 2024            Date: March 15, 2024"""
            },
            {
                "filename": "privacy_policy.md",
                "content": """# Privacy Policy

**Effective Date:** January 1, 2024
**Last Updated:** January 1, 2024

## 1. Introduction

TechSolutions Inc. ("we," "our," or "us") respects your privacy and is committed to protecting your personal information. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our software application and related services.

## 2. Information We Collect

### 2.1 Personal Information
We may collect the following types of personal information:

- **Account Information:** Name, email address, username, password
- **Contact Information:** Phone number, mailing address
- **Payment Information:** Credit card details, billing address (processed by third-party payment processors)
- **Profile Information:** Profile picture, preferences, settings

### 2.2 Usage Information
We automatically collect information about how you use our services:

- **Device Information:** IP address, browser type, operating system, device identifiers
- **Usage Data:** Pages visited, features used, time spent, click patterns
- **Log Data:** Server logs, error reports, performance metrics
- **Location Data:** General geographic location based on IP address

### 2.3 Cookies and Tracking Technologies
We use cookies and similar technologies to:

- Remember your preferences and settings
- Analyze usage patterns and improve our services
- Provide personalized content and recommendations
- Ensure security and prevent fraud

## 3. How We Use Your Information

We use your information for the following purposes:

### 3.1 Service Provision
- Provide, maintain, and improve our software application
- Process transactions and manage your account
- Provide customer support and respond to inquiries
- Send important notices about your account or services

### 3.2 Communication
- Send promotional emails and marketing communications (with your consent)
- Notify you about new features, updates, and security alerts
- Conduct surveys and collect feedback

### 3.3 Analytics and Improvement
- Analyze usage patterns to improve user experience
- Conduct research and development for new features
- Generate aggregated, anonymized statistics

### 3.4 Legal and Security
- Comply with legal obligations and respond to legal requests
- Protect against fraud, unauthorized access, and security threats
- Enforce our terms of service and other agreements

## 4. Information Sharing and Disclosure

We do not sell your personal information. We may share your information in the following circumstances:

### 4.1 Service Providers
We may share information with trusted third-party service providers who assist us in:

- Payment processing (Stripe, PayPal)
- Email communications (SendGrid, Mailchimp)
- Analytics and monitoring (Google Analytics, Mixpanel)
- Cloud hosting and storage (AWS, Google Cloud)

### 4.2 Business Transfers
In the event of a merger, acquisition, or sale of assets, your information may be transferred to the acquiring entity.

### 4.3 Legal Requirements
We may disclose information when required by law or to:

- Comply with court orders, subpoenas, or legal processes
- Protect our rights, property, or safety
- Prevent fraud or security threats
- Assist law enforcement investigations

### 4.4 Consent
We may share information with your explicit consent for specific purposes not covered in this policy.

## 5. Data Security

We implement comprehensive security measures to protect your information:

### 5.1 Technical Safeguards
- Encryption of data in transit and at rest using industry-standard protocols
- Secure data centers with physical access controls
- Regular security audits and penetration testing
- Multi-factor authentication for administrative access

### 5.2 Organizational Measures
- Employee training on data protection and privacy
- Strict access controls and need-to-know principles
- Regular review and update of security policies
- Incident response procedures for data breaches

### 5.3 Breach Notification
In the event of a data breach affecting your personal information, we will:

- Notify affected users within 72 hours of discovery
- Provide details about the breach and steps being taken
- Offer guidance on protective measures you can take
- Report to relevant authorities as required by law

## 6. Your Rights and Choices

You have the following rights regarding your personal information:

### 6.1 Access and Portability
- Request access to your personal information
- Receive a copy of your data in a structured, machine-readable format

### 6.2 Correction and Updates
- Correct inaccurate or incomplete information
- Update your account information and preferences

### 6.3 Deletion and Restriction
- Request deletion of your personal information (subject to legal requirements)
- Restrict certain types of processing

### 6.4 Communication Preferences
- Opt out of marketing communications
- Manage email notification preferences
- Control cookie settings in your browser

### 6.5 Exercising Your Rights
To exercise these rights, contact us at privacy@techsolutions.com or use the account settings in our application.

## 7. International Data Transfers

We may transfer your information to countries outside your residence, including the United States. We ensure adequate protection through:

- Standard Contractual Clauses approved by regulatory authorities
- Adequacy decisions by relevant data protection authorities
- Other appropriate safeguards as required by applicable law

## 8. Data Retention

We retain your information for as long as necessary to:

- Provide our services and maintain your account
- Comply with legal obligations and resolve disputes
- Pursue legitimate business interests

Specific retention periods:
- Account information: Until account deletion plus 30 days
- Usage data: 2 years from collection
- Payment information: As required by financial regulations
- Marketing data: Until you opt out plus 30 days

## 9. Children's Privacy

Our services are not intended for children under 13 years of age. We do not knowingly collect personal information from children under 13. If we learn that we have collected such information, we will delete it immediately.

## 10. Third-Party Links and Services

Our application may contain links to third-party websites or integrate with third-party services. This Privacy Policy does not apply to those external sites or services. We encourage you to review their privacy policies.

## 11. Changes to This Privacy Policy

We may update this Privacy Policy from time to time. We will:

- Notify you of material changes via email or in-app notification
- Post the updated policy on our website with a new effective date
- Maintain previous versions for your reference

Your continued use of our services after changes take effect constitutes acceptance of the updated policy.

## 12. Contact Information

If you have questions, concerns, or requests regarding this Privacy Policy or our privacy practices, please contact us:

**TechSolutions Inc.**
**Data Protection Officer**
Email: privacy@techsolutions.com
Phone: 1-800-TECH-HELP
Address: 123 Innovation Drive, Tech City, CA 94000

**EU Representative** (for GDPR matters):
Email: eu-privacy@techsolutions.com
Address: 456 Data Street, Digital City, Dublin, Ireland

---

© 2024 TechSolutions Inc. All rights reserved."""
            }
        ]

        # Create sample documents
        for doc in sample_docs:
            file_path = self.data_directory / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            logger.info(f"Created sample document: {file_path.name}")

        # Process the created documents
        from tests.test_local_rag_run import MockUploadFile

        for doc in sample_docs:
            file_path = self.data_directory / doc["filename"]
            try:
                mock_file = MockUploadFile(file_path)
                enhanced_doc = await self.document_service.process_document_enhanced(
                    mock_file,
                    use_enhanced_models=True,
                    preserve_structure=True
                )
                await self.vector_store.store_enhanced_document(enhanced_doc)
                logger.info(f"Processed {file_path.name}: {len(enhanced_doc.chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                return False

        return True

    async def run_quick_evaluation(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run quick evaluation using sample documents."""
        logger.info("Starting quick evaluation...")

        # Setup sample documents
        setup_success = await self.setup_sample_documents()
        if not setup_success:
            logger.error("Failed to setup sample documents")
            return {"error": "Failed to setup sample documents"}

        # Create quick evaluator
        evaluator = QuickEvaluator(
            self.document_service,
            self.vector_store,
            self.llm,
            self.data_directory
        )

        # Run evaluation
        results = await evaluator.run_quick_evaluation()

        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")

        return results

    async def run_full_evaluation(
        self,
        corpus_path: str,
        benchmarks_path: str,
        benchmark_file: Optional[str] = None,
        max_test_cases: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run full LegalBench-RAG evaluation."""
        logger.info("Starting full LegalBench-RAG evaluation...")

        # Create full evaluator
        evaluator = LegalBenchRAGEvaluator(
            self.document_service,
            self.vector_store,
            self.llm,
            corpus_path=corpus_path,
            benchmarks_path=benchmarks_path
        )

        # Setup corpus
        setup_success = await evaluator.setup_corpus()
        if not setup_success:
            logger.error("Failed to setup LegalBench-RAG corpus")
            return {"error": "Failed to setup corpus"}

        # Find benchmark files
        benchmarks_dir = Path(benchmarks_path)
        if benchmark_file:
            benchmark_files = [benchmark_file]
        else:
            benchmark_files = [f.name for f in benchmarks_dir.glob("*.json")]

        if not benchmark_files:
            logger.error("No benchmark files found")
            return {"error": "No benchmark files found"}

        # Run evaluations
        all_results = {}

        for bench_file in benchmark_files:
            logger.info(f"Running benchmark: {bench_file}")

            result = await evaluator.run_benchmark(
                bench_file,
                k=10,
                max_test_cases=max_test_cases
            )

            all_results[bench_file] = result

            # Save individual results if output file specified
            if output_file:
                individual_output = output_file.replace('.json', f'_{bench_file}')
                evaluator.save_results(result, individual_output)

        # Save combined results
        if output_file and len(all_results) > 1:
            combined_results = {
                "evaluation_type": "full_legalbench_rag",
                "total_benchmarks": len(all_results),
                "benchmark_results": {}
            }

            for bench_file, result in all_results.items():
                combined_results["benchmark_results"][bench_file] = {
                    "avg_precision": result.avg_precision,
                    "avg_recall": result.avg_recall,
                    "avg_f1_score": result.avg_f1_score,
                    "success_rate": result.success_rate,
                    "total_test_cases": result.total_test_cases
                }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Combined results saved to {output_file}")

        return all_results


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*80)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    # Quick evaluation results
    if "avg_keyword_coverage" in results:
        print(f"📈 Average Keyword Coverage: {results['avg_keyword_coverage']:.2%}")
        print(f"🎯 Average Relevance Score: {results['avg_relevance_score']:.2%}")
        print(f"⏱️  Average Retrieval Time: {results['avg_retrieval_time']:.3f}s")
        print(f"🤖 Average Generation Time: {results['avg_generation_time']:.3f}s")
        print(f"📝 Total Test Cases: {results['total_test_cases']}")

        print(f"\n📋 Detailed Results:")
        for i, result in enumerate(results['detailed_results'], 1):
            print(f"\n  {i}. {result['query']}")
            print(f"     ✅ Keyword Coverage: {result['keyword_coverage']:.2%}")
            print(f"     🎯 Relevance Score: {result['relevance_score']:.2%}")
            print(f"     📄 Retrieved: {result['retrieved_count']} documents")

            if result['missing_keywords']:
                print(f"     ❌ Missing Keywords: {', '.join(result['missing_keywords'])}")

    # Full evaluation results
    elif isinstance(results, dict) and any("avg_precision" in str(v) for v in results.values()):
        for benchmark_name, result in results.items():
            if hasattr(result, 'avg_precision'):
                print(f"\n📊 Benchmark: {benchmark_name}")
                print(f"   Precision: {result.avg_precision:.3f}")
                print(f"   Recall: {result.avg_recall:.3f}")
                print(f"   F1 Score: {result.avg_f1_score:.3f}")
                print(f"   Success Rate: {result.success_rate:.2%}")
                print(f"   Test Cases: {result.total_test_cases}")

    print("\n" + "="*80)


def main():
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system using LegalBench-RAG or quick evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation using sample documents
  python evaluate_rag.py --mode quick

  # Full evaluation with custom paths
  python evaluate_rag.py --mode full --corpus-path data/legalbench_corpus

  # Evaluate specific benchmark with limited test cases
  python evaluate_rag.py --mode full --benchmark-file cuad.json --max-cases 10

  # Save results to file
  python evaluate_rag.py --mode quick --output results/evaluation.json
        """
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Evaluation mode: 'quick' for sample documents, 'full' for LegalBench-RAG"
    )

    parser.add_argument(
        "--corpus-path",
        default="data/legalbench_corpus",
        help="Path to LegalBench-RAG corpus directory (for full mode)"
    )

    parser.add_argument(
        "--benchmarks-path",
        default="data/legalbench_benchmarks",
        help="Path to LegalBench-RAG benchmarks directory (for full mode)"
    )

    parser.add_argument(
        "--benchmark-file",
        help="Specific benchmark JSON file to evaluate (for full mode)"
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of test cases to evaluate (for testing)"
    )

    parser.add_argument(
        "--output",
        help="Output file path for results (JSON format)"
    )

    parser.add_argument(
        "--data-directory",
        default="data",
        help="Directory for sample documents (for quick mode)"
    )

    parser.add_argument(
        "--use-database",
        action="store_true",
        help="Use database-backed vector store instead of local FAISS"
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

    # Check dependencies
    if not FAISS_AVAILABLE and not args.use_database:
        print("❌ FAISS not available. Please install with: pip install faiss-cpu")
        print("   Or use --use-database to use database-backed vector store")
        return 1

    if not LANGCHAIN_OPENAI_AVAILABLE:
        print("❌ LangChain OpenAI not available. Please install with: pip install langchain-openai")
        return 1

    # Check API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return 1

    async def run_evaluation():
        """Run the evaluation asynchronously."""
        runner = RAGEvaluationRunner(
            openai_api_key=openai_api_key,
            use_local_vectorstore=not args.use_database,
            data_directory=args.data_directory
        )

        if args.mode == "quick":
            results = await runner.run_quick_evaluation(args.output)
        else:
            results = await runner.run_full_evaluation(
                corpus_path=args.corpus_path,
                benchmarks_path=args.benchmarks_path,
                benchmark_file=args.benchmark_file,
                max_test_cases=args.max_cases,
                output_file=args.output
            )

        print_results_summary(results)
        return results

    try:
        results = asyncio.run(run_evaluation())
        return 0 if "error" not in str(results) else 1

    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        logger.exception("Evaluation failed with exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
