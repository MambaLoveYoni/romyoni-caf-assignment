"""Merge helpers for libcaf."""

from array import array
from collections.abc import Sequence
from contextlib import ExitStack
import mmap
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from merge3 import Merge3

from . import Tree, TreeRecord, TreeRecordType
from .plumbing import (hash_object, load_commit, load_tree,
                       open_content_for_reading, save_file_content, save_tree)
from .ref import HashRef


class MergeError(Exception):
    """Exception raised for merge-related errors."""


@dataclass
class MergeResult:
    """Represents the output of a 3-way merge."""

    tree_hash: HashRef
    conflicts: list[str]


def is_binary_blob(objects_dir: str | Path, blob_hash: str | None, sample_size: int = 8192) -> bool:
    """Detect if a blob contains binary data using multiple heuristics."""
    if blob_hash is None:
        return False

    try:
        with open_content_for_reading(objects_dir, blob_hash) as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                size = min(sample_size, len(mmapped))
                if size == 0:
                    return False
                sample = mmapped[:size]

                # Check for null bytes (strong indicator of binary)
                if b'\x00' in sample:
                    return True

                # Count non-text bytes (control characters except whitespace)
                non_text_count = 0
                for byte in sample:
                    # Allow common text characters: printable ASCII, newline, tab, carriage return
                    if byte < 32 and byte not in (9, 10, 13):  # tab, LF, CR
                        non_text_count += 1
                    elif byte == 127:  # DEL character
                        non_text_count += 1

                # If more than 30% of bytes are non-text, consider it binary
                if non_text_count > size * 0.3:
                    return True

                return False
    except Exception:
        return False


class MmapLineSequence(Sequence[bytes]):
    """List-like random-access view over blob lines without materializing them all."""

    def __init__(self, objects_dir: str | Path, blob_hash: str) -> None:
        self._handle = None
        self._mmapped = None
        self._size = 0
        self._line_offsets = array('Q')

        try:
            handle = open_content_for_reading(objects_dir, blob_hash)
            self._handle = handle
            self._size = os.fstat(handle.fileno()).st_size
            if self._size == 0:
                handle.close()
                self._handle = None
                return

            mmapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmapped = mmapped
            handle.close()
            self._handle = None
            self._line_offsets.append(0)
            search_start = 0
            while True:
                newline_pos = mmapped.find(b'\n', search_start)
                if newline_pos < 0:
                    break
                line_start = newline_pos + 1
                if line_start < self._size:
                    self._line_offsets.append(line_start)
                search_start = line_start
        except Exception as e:
            self.close()
            msg = f'Error reading blob {blob_hash}'
            raise MergeError(msg) from e

    def __len__(self) -> int:
        return len(self._line_offsets)

    def __getitem__(self, index: int | slice) -> bytes | list[bytes]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        line_count = len(self)
        if index < 0:
            index += line_count
        if index < 0 or index >= line_count:
            raise IndexError('line index out of range')

        if self._mmapped is None:
            raise IndexError('line index out of range')

        start = self._line_offsets[index]
        end = self._line_offsets[index + 1] if index + 1 < line_count else self._size
        return self._mmapped[start:end]

    def close(self) -> None:
        if self._mmapped is not None:
            self._mmapped.close()
            self._mmapped = None

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> 'MmapLineSequence':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

def merge_blob_text(objects_dir: str | Path, base_hash: str | None, ours_hash: str | None, theirs_hash: str | None) -> tuple[HashRef, bool]:
    """Merge three versions of a blob using merge3."""
    with ExitStack() as stack:
        base_lines = stack.enter_context(MmapLineSequence(objects_dir, base_hash)) if base_hash else []
        ours_lines = stack.enter_context(MmapLineSequence(objects_dir, ours_hash)) if ours_hash else []
        theirs_lines = stack.enter_context(MmapLineSequence(objects_dir, theirs_hash)) if theirs_hash else []

        merger = Merge3(base_lines, ours_lines, theirs_lines)
        conflict = False

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_file:
            for group in merger.merge_groups():
                if group[0] == 'unchanged':
                    tmp_file.writelines(group[1])
                elif group[0] == 'a':
                    tmp_file.writelines(group[1])
                elif group[0] == 'b':
                    tmp_file.writelines(group[1])
                elif group[0] == 'conflict':
                    conflict = True
                    tmp_file.write(b'<<<<<<< ours\n')
                    tmp_file.writelines(group[2])  # a_lines (ours)
                    tmp_file.write(b'=======\n')
                    tmp_file.writelines(group[3])  # b_lines (theirs)
                    tmp_file.write(b'>>>>>>> theirs\n')
            tmp_path = tmp_file.name

        try:
            blob = save_file_content(objects_dir, tmp_path)
            return HashRef(blob.hash), conflict
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def merge_blob_binary(objects_dir: str | Path, base_hash: str | None, ours_hash: str | None, theirs_hash: str | None) -> tuple[HashRef, bool]:
    """Merge binary blobs by selecting a version or marking as conflict."""
    if ours_hash == theirs_hash:
        return HashRef(ours_hash), False

    if base_hash == ours_hash and theirs_hash is not None:
        return HashRef(theirs_hash), False

    if base_hash == theirs_hash and ours_hash is not None:
        return HashRef(ours_hash), False

    if ours_hash is not None:
        return HashRef(ours_hash), True
    if theirs_hash is not None:
        return HashRef(theirs_hash), True

    msg = 'Cannot merge binary blobs without any valid version'
    raise MergeError(msg)


def merge_blob(objects_dir: str | Path, base_hash: str | None, ours_hash: str | None, theirs_hash: str | None) -> tuple[HashRef, bool]:
    """Merge two blob versions using their common ancestor."""
    if is_binary_blob(objects_dir, ours_hash) or is_binary_blob(objects_dir, theirs_hash):
        return merge_blob_binary(objects_dir, base_hash, ours_hash, theirs_hash)

    return merge_blob_text(objects_dir, base_hash, ours_hash, theirs_hash)


def merge_trees_core(objects_dir: str | Path, base_tree: Tree | None, ours_tree: Tree | None, theirs_tree: Tree | None, path_prefix: str, conflicts: list[str]) -> HashRef:
    """Recursively merge three trees using 3-way merge logic."""
    base_records = base_tree.records if base_tree else {}
    ours_records = ours_tree.records if ours_tree else {}
    theirs_records = theirs_tree.records if theirs_tree else {}

    all_names = sorted(set(base_records) | set(ours_records) | set(theirs_records))
    merged_records: dict[str, TreeRecord] = {}

    for name in all_names:
        base = base_records.get(name)
        ours = ours_records.get(name)
        theirs = theirs_records.get(name)
        path = str(Path(path_prefix) / name) if path_prefix else name

        if ours == theirs:
            if ours is not None:
                merged_records[name] = ours
            continue

        if base == ours:
            if theirs is not None:
                merged_records[name] = theirs
            continue

        if base == theirs:
            if ours is not None:
                merged_records[name] = ours
            continue

        if (ours is not None and theirs is not None
                and ours.type == TreeRecordType.TREE
                and theirs.type == TreeRecordType.TREE):
            base_subtree = load_tree(objects_dir, base.hash) if (base is not None and base.type == TreeRecordType.TREE) else None
            merged_hash = merge_trees_core(
                objects_dir,
                base_subtree,
                load_tree(objects_dir, ours.hash),
                load_tree(objects_dir, theirs.hash),
                path,
                conflicts,
            )
            merged_records[name] = TreeRecord(TreeRecordType.TREE, merged_hash, name)
            continue

        if (ours is not None and theirs is not None
                and ours.type == TreeRecordType.BLOB
                and theirs.type == TreeRecordType.BLOB):
            base_hash = base.hash if (base is not None and base.type == TreeRecordType.BLOB) else None
            merged_hash, conflict = merge_blob(objects_dir, base_hash, ours.hash, theirs.hash)
            if conflict:
                conflicts.append(path)
            merged_records[name] = TreeRecord(TreeRecordType.BLOB, merged_hash, name)
            continue

        chosen = ours or theirs
        if chosen is not None:
            merged_records[name] = chosen
        conflicts.append(path)

    merged_tree = Tree(merged_records)
    save_tree(objects_dir, merged_tree)
    return HashRef(hash_object(merged_tree))


def find_common_ancestor_core(objects_dir: str, hash1: str, hash2: str) -> HashRef | None:
    """Helper function to run the ancestor search algorithm independent of the Repository class."""
    try:
        ancestors: set[str] = set()
        current_hash: str | None = hash1
        while current_hash:
            ancestors.add(current_hash)
            commit = load_commit(objects_dir, current_hash)
            current_hash = commit.parent

        current_hash2: str | None = hash2
        while current_hash2:
            if current_hash2 in ancestors:
                return HashRef(current_hash2)
            commit = load_commit(objects_dir, current_hash2)
            current_hash2 = commit.parent

    except Exception as e:
        msg = 'Error loading commit during ancestor search'
        raise MergeError(msg) from e

    return None


def merge_commits_core(objects_dir: str | Path, ours_hash: str, theirs_hash: str) -> MergeResult:
    """Perform a 3-way merge between two commits using their common ancestor."""
    ancestor_hash = find_common_ancestor_core(objects_dir, ours_hash, theirs_hash)
    if ancestor_hash is None:
        msg = 'No common ancestor found for merge'
        raise MergeError(msg)

    try:
        ours_commit = load_commit(objects_dir, ours_hash)
        theirs_commit = load_commit(objects_dir, theirs_hash)
        ancestor_commit = load_commit(objects_dir, ancestor_hash)

        ours_tree = load_tree(objects_dir, ours_commit.tree_hash)
        theirs_tree = load_tree(objects_dir, theirs_commit.tree_hash)
        ancestor_tree = load_tree(objects_dir, ancestor_commit.tree_hash)
    except Exception as e:
        msg = 'Error preparing commits for merge'
        raise MergeError(msg) from e

    conflicts: list[str] = []
    merged_tree_hash = merge_trees_core(
        objects_dir,
        ancestor_tree,
        ours_tree,
        theirs_tree,
        '',
        conflicts,
    )

    return MergeResult(merged_tree_hash, conflicts)
