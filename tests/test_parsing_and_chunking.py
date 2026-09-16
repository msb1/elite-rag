from __future__ import annotations

from elite_rag.chunking import HierarchicalChunker, approximate_token_count
from elite_rag.models import RawDocument
from elite_rag.parsing.communications import EmailParser, SlackParser
from elite_rag.parsing.html import HtmlParser
from elite_rag.parsing.markdown import MarkdownParser


def document(source: str, content: str) -> RawDocument:
    return RawDocument(
        document_id="dsid_test",
        source=source,
        title="Architecture",
        fields={"content": content},
        metadata={"project": "Hydra", "updated_at": "2026-09-01"},
    )


def test_markdown_preserves_heading_hierarchy() -> None:
    blocks = MarkdownParser().parse(
        document(
            "markdown", "intro\n# Platform\nplatform body\n## Auth\nauth body\n## Logs\nlog body"
        )
    )
    assert [block.path for block in blocks] == [
        ("Content",),
        ("Platform",),
        ("Platform", "Auth"),
        ("Platform", "Logs"),
    ]
    assert [block.text for block in blocks] == ["intro", "platform body", "auth body", "log body"]


def test_markdown_does_not_treat_fenced_code_as_a_heading() -> None:
    blocks = MarkdownParser().parse(
        document("markdown", "## Real section\n```md\n## Not a section\n```\nAfter fence")
    )
    assert len(blocks) == 1
    assert blocks[0].path == ("Real section",)
    assert "## Not a section" in blocks[0].text


def test_html_tracks_h2_and_h3_without_tags() -> None:
    raw = "<div>Intro</div><h2>Auth</h2><p>Use RBAC.</p><h3>Keys</h3><p>Rotate keys.</p>"
    blocks = HtmlParser().parse(document("confluence", raw))
    assert [(block.path, block.text) for block in blocks] == [
        (("Content",), "Intro"),
        (("Auth",), "Use RBAC."),
        (("Auth", "Keys"), "Rotate keys."),
    ]


def test_email_and_slack_use_natural_message_boundaries() -> None:
    email = EmailParser().parse(document("gmail", "New answer.\nOn Sep 11 wrote:\nOld answer."))
    assert len(email) == 2
    assert email[0].path[-1] == "Current message"
    assert email[1].path[-1] == "Quoted reply 1"

    slack_doc = RawDocument("s1", "slack", "eng", {"messages": "a: one\n\nb: two"})
    slack = SlackParser().parse(slack_doc)
    assert [block.text for block in slack] == ["a: one", "b: two"]

    moved_doc = RawDocument("m1", "slack", "Moved", {"description": "Still indexed"})
    assert SlackParser().parse(moved_doc)[0].text == "Still indexed"


def test_hierarchical_chunks_are_bounded_stable_and_zero_overlap() -> None:
    source_words = [f"word{i}" for i in range(90)]
    raw = document(
        "markdown",
        "## One\n" + " ".join(source_words[:45]) + "\n## Two\n" + " ".join(source_words[45:]),
    )
    blocks = MarkdownParser().parse(raw)
    chunker = HierarchicalChunker(
        approximate_token_count, child_max_tokens=55, parent_max_tokens=90
    )
    parents, children = chunker.chunk(raw, blocks)

    assert len(parents) >= 2
    assert all(approximate_token_count(parent.text) <= 90 for parent in parents)
    assert all(approximate_token_count(child.vector_text) <= 55 for child in children)
    assert len({child.child_id for child in children}) == len(children)
    assert {child.parent_id for child in children} <= {parent.parent_id for parent in parents}

    child_words = [
        word for child in children for word in child.text.split() if word.startswith("word")
    ]
    assert child_words == source_words
    _, same_children = chunker.chunk(raw, blocks)
    assert [child.child_id for child in children] == [child.child_id for child in same_children]
