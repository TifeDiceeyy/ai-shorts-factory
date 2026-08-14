import json

from shorts_factory.book_ingest import build_private_index, read_book, research_terms, retrieve_private_chunks


def test_private_book_index_and_retrieval(tmp_path):
    book = tmp_path / "book.txt"
    book.write_text(
        "Soap making has a long history involving fats and alkaline materials.\n\n"
        "Roman builders used durable concrete in marine structures.\n\n"
        "More discussion of soap and historical cleaning practices.",
        encoding="utf-8",
    )
    output = tmp_path / "private" / "index.json"
    index = build_private_index(book, output)
    matches = retrieve_private_chunks(index, "soap")

    assert output.exists()
    assert index["source"]["private"] is True
    assert matches and all(chunk["internal_only"] for chunk in matches)
    assert "book.txt" not in json.dumps(matches)  # chunks do not expose a filesystem path
    assert research_terms(matches, "soap")


def test_private_book_retrieval_returns_no_unrelated_chunks(tmp_path):
    book = tmp_path / "book.txt"
    book.write_text("Pottery is made from clay.", encoding="utf-8")
    index = build_private_index(book, tmp_path / "index.json")
    assert retrieve_private_chunks(index, "soap") == []


def test_pdf_reader_path_is_operational(tmp_path):
    from pypdf import PdfWriter

    pdf = tmp_path / "book.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as output:
        writer.write(output)
    assert read_book(pdf) == ""
