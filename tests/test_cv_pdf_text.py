from app.services.cv_pdf import markdown_to_text


def test_docling_entities_are_decoded_for_evidence_matching():
    markdown = "Tools &amp; Deployment: Docker, Git\nC++ &lt;3 &quot;RAG&quot;  "
    assert markdown_to_text(markdown) == 'Tools & Deployment: Docker, Git\nC++ <3 "RAG"'
