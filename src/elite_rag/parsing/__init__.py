from elite_rag.parsing.base import FieldParser, ParserRegistry
from elite_rag.parsing.communications import EmailParser, SlackParser, TranscriptParser
from elite_rag.parsing.html import HtmlParser
from elite_rag.parsing.markdown import MarkdownParser


def default_registry() -> ParserRegistry:
    registry = ParserRegistry(default=FieldParser())
    registry.register({"markdown", "blog", "google_drive"}, MarkdownParser())
    registry.register({"confluence"}, HtmlParser())
    registry.register({"email", "gmail"}, EmailParser())
    registry.register({"slack"}, SlackParser())
    registry.register({"fireflies", "transcript"}, TranscriptParser())
    return registry


__all__ = ["ParserRegistry", "default_registry"]
