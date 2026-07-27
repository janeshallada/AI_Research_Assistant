"""
Generates three realistic, multi-page sample PDFs into data/sample_documents/
for demoing upload, classification, summarization, comparison, and Q&A.

Run once: python generate_sample_pdfs.py
"""
import fitz
import os

OUT_DIR = "data/sample_documents"
os.makedirs(OUT_DIR, exist_ok=True)

DOCS = {
    "retrieval_augmented_generation.pdf": [
        (
            "Retrieval-Augmented Generation for Enterprise Knowledge Systems\n\n"
            "Abstract\n"
            "Large language models are powerful general-purpose reasoners, but they are "
            "prone to hallucination when asked about domain-specific or proprietary "
            "information that was not present in their training data. Retrieval-Augmented "
            "Generation (RAG) addresses this limitation by retrieving relevant passages "
            "from an external knowledge base at query time and conditioning the model's "
            "response on that retrieved context. This paper describes a production RAG "
            "architecture built around dense vector retrieval, hybrid search, and "
            "citation-grounded generation."
        ),
        (
            "1. Methodology\n\n"
            "Our pipeline ingests PDF documents, extracts text with page-level metadata, "
            "and splits the text into overlapping chunks of approximately 1000 characters "
            "with a 150 character overlap. Each chunk is embedded using a sentence "
            "transformer model and stored in a vector database alongside its source "
            "document and page number. At query time, the system supports three retrieval "
            "modes: semantic (dense vector similarity), keyword (sparse term matching), "
            "and hybrid, which fuses both signals. Retrieved chunks are inserted into a "
            "structured prompt that instructs the language model to answer strictly from "
            "the provided context and to decline to answer when the context is "
            "insufficient."
        ),
        (
            "2. Advantages and Limitations\n\n"
            "The primary advantage of this approach is grounding: every answer can be "
            "traced back to a specific document and page, which builds user trust and "
            "supports auditability. A further advantage is that the knowledge base can be "
            "updated without retraining the underlying language model. A limitation is "
            "that retrieval quality bounds answer quality; if the relevant passage is not "
            "retrieved, the model cannot answer correctly even if it 'knows' the fact from "
            "pretraining. Chunking strategy therefore has an outsized effect on system "
            "performance."
        ),
        (
            "3. Conclusion\n\n"
            "We conclude that hybrid retrieval combined with strict context-grounding and "
            "explicit citations is an effective architecture for enterprise question "
            "answering over private document collections, and that conversation memory is "
            "necessary to correctly resolve follow-up questions that refer back to earlier "
            "turns in a session."
        ),
    ],
    "cloud_native_microservices.pdf": [
        (
            "Cloud-Native Microservices: Architecture and Scaling Strategies\n\n"
            "Abstract\n"
            "Organizations are migrating monolithic applications to cloud-native "
            "microservice architectures to improve scalability, deployment velocity, and "
            "fault isolation. This paper surveys container orchestration, service mesh, "
            "and autoscaling techniques used to operate microservices reliably in "
            "production cloud environments."
        ),
        (
            "1. Methodology\n\n"
            "We containerize each service using Docker and orchestrate deployment using "
            "Kubernetes, which manages scheduling, self-healing, and horizontal pod "
            "autoscaling based on CPU and custom metrics. A service mesh handles "
            "inter-service traffic routing, retries, and observability. Infrastructure is "
            "provisioned declaratively using infrastructure-as-code, enabling reproducible "
            "environments across staging and production."
        ),
        (
            "2. Advantages and Limitations\n\n"
            "The primary advantage of this architecture is elastic scalability: individual "
            "services can be scaled independently based on demand, avoiding the need to "
            "scale an entire monolith. Fault isolation is also improved, since a failure "
            "in one service does not necessarily cascade to others. A limitation is "
            "increased operational complexity: distributed tracing, service discovery, and "
            "network policy management require dedicated tooling and expertise that a "
            "monolithic deployment does not need."
        ),
        (
            "3. Conclusion\n\n"
            "We conclude that cloud-native microservices provide substantial scalability "
            "and deployment benefits for organizations with sufficient operational "
            "maturity, but that the added complexity is a meaningful cost that should be "
            "weighed against the specific scaling requirements of the application."
        ),
    ],
    "network_intrusion_detection.pdf": [
        (
            "Anomaly-Based Network Intrusion Detection Using Traffic Classification\n\n"
            "Abstract\n"
            "Signature-based intrusion detection systems are effective against known "
            "attack patterns but fail to detect novel or zero-day threats. This paper "
            "presents an anomaly-based detection approach that models normal network "
            "traffic behavior and flags statistically significant deviations as "
            "potential intrusions."
        ),
        (
            "1. Methodology\n\n"
            "We collect network flow features including packet size distribution, "
            "connection duration, and protocol usage patterns, and train a classifier to "
            "distinguish normal traffic from anomalous traffic. The model is evaluated on "
            "held-out traffic captures containing both benign activity and simulated "
            "attack traffic, including port scanning and denial-of-service patterns."
        ),
        (
            "2. Advantages and Limitations\n\n"
            "The primary advantage of anomaly-based detection is the ability to catch "
            "previously unseen attack patterns that signature-based systems would miss. A "
            "limitation is a higher false-positive rate, since legitimate but unusual "
            "traffic can resemble an anomaly; careful threshold tuning and analyst review "
            "workflows are required to keep the system operationally usable."
        ),
        (
            "3. Conclusion\n\n"
            "We conclude that anomaly-based detection is a valuable complement to "
            "signature-based systems, particularly for identifying novel threats, and "
            "recommend deploying both approaches together as part of a layered network "
            "security strategy."
        ),
    ],
}


def main():
    for filename, pages_text in DOCS.items():
        doc = fitz.open()
        for text in pages_text:
            page = doc.new_page(width=612, height=792)  # US Letter
            rect = fitz.Rect(72, 72, 540, 720)
            page.insert_textbox(rect, text, fontsize=11, fontname="helv")
        out_path = os.path.join(OUT_DIR, filename)
        doc.save(out_path)
        doc.close()
        print(f"Created {out_path} ({len(pages_text)} pages)")


if __name__ == "__main__":
    main()
