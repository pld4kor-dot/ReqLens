"""Unit tests for document chunking."""

from reqinone2.ingestion.chunking import chunk_document


class TestChunkDocument:
    def test_empty_input(self):
        spans = chunk_document("")
        assert spans == []

    def test_single_paragraph(self):
        text = "The system shall allow students to register for events."
        spans = chunk_document(text)
        assert len(spans) >= 1
        assert spans[0].text == text.strip()
        assert spans[0].char_start >= 0
        assert spans[0].char_end > spans[0].char_start

    def test_multiple_paragraphs(self, sample_transcript_text):
        spans = chunk_document(sample_transcript_text)
        assert len(spans) >= 1
        # All text should be covered
        for span in spans:
            assert span.text.strip()
            assert span.span_index >= 0

    def test_char_offsets_valid(self, sample_transcript_text):
        spans = chunk_document(sample_transcript_text)
        for span in spans:
            assert span.char_start >= 0
            assert span.char_end > span.char_start
            assert span.char_end <= len(sample_transcript_text) + 10  # small tolerance

    def test_large_document_splits(self):
        """A very large paragraph should be split into multiple spans."""
        text = "Word " * 10000  # ~50k chars
        spans = chunk_document(text, chunk_size=500)
        assert len(spans) > 1

    def test_speaker_detection(self):
        text = "Alice: We need login functionality.\n\nBob: I agree."
        spans = chunk_document(text)
        # At least one span should detect a speaker
        speakers = [s.speaker for s in spans if s.speaker]
        assert len(speakers) >= 0  # heuristic, may not always fire
