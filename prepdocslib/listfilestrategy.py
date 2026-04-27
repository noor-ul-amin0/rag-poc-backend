import logging
import os
from abc import ABC
from collections.abc import AsyncGenerator
from glob import glob
from typing import IO, Optional

logger = logging.getLogger("scripts")


class File:
    """Represents a file stored locally or in cloud storage."""

    def __init__(self, content: IO, acls: Optional[dict[str, list]] = None, url: Optional[str] = None):
        self.content = content
        self.acls = acls or {}
        self.url = url

    def filename(self) -> str:
        """Get the filename from the content object."""
        if hasattr(self.content, "filename"):
            content_name = getattr(self.content, "filename")
            if content_name:
                return os.path.basename(content_name)

        if hasattr(self.content, "name"):
            content_name = getattr(self.content, "name")
            if content_name and content_name != "file":
                return os.path.basename(content_name)

        raise ValueError("The content object does not have a filename or name attribute.")

    def file_extension(self):
        return os.path.splitext(self.filename())[1]

    def close(self):
        if self.content:
            self.content.close()


class ListFileStrategy(ABC):
    """Abstract strategy for listing files."""

    async def list(self) -> AsyncGenerator[File, None]:
        if False:
            yield

    async def list_paths(self) -> AsyncGenerator[str, None]:
        if False:
            yield


class LocalListFileStrategy(ListFileStrategy):
    """Concrete strategy for listing files from local filesystem."""

    def __init__(
        self,
        path_pattern: str,
        enable_global_documents: bool = False,
    ):
        self.path_pattern = path_pattern
        self.enable_global_documents = enable_global_documents
        self.total_files_seen = 0
        self.total_selected = 0

    async def list_paths(self) -> AsyncGenerator[str, None]:
        async for p in self._list_paths(self.path_pattern):
            yield p

    async def _list_paths(self, path_pattern: str) -> AsyncGenerator[str, None]:
        for path in glob(path_pattern, recursive=True):
            if os.path.isdir(path):
                async for p in self._list_paths(f"{path}/*"):
                    yield p
            else:
                yield path

    async def list(self) -> AsyncGenerator[File, None]:
        self.total_files_seen = 0
        self.total_selected = 0

        acls = {"oids": ["all"], "groups": ["all"]} if self.enable_global_documents else {}
        async for path in self.list_paths():
            self.total_files_seen += 1

            self.total_selected += 1
            yield File(content=open(path, mode="rb"), acls=acls, url=path)
