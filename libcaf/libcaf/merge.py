"""Merge helpers for libcaf."""

import mmap
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from merge3 import Merge3

from . import Tree, TreeRecord, TreeRecordType
from .plumbing import (hash_object, hash_string, load_commit, load_tree,
                       open_content_for_reading, open_content_for_writing, save_file_content, save_tree)
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


def read_blob_lines(objects_dir: str | Path, blob_hash: str) -> list[bytes]:
    """Load blob content as a list of byte lines, reading incrementally."""
    try:
        with open_content_for_reading(objects_dir, blob_hash) as handle:
            return handle.readlines()
    except Exception as e:
        msg = f'Error reading blob {blob_hash}'
        raise MergeError(msg) from e


def save_blob_text(objects_dir: str | Path, content: str) -> HashRef:
    """Save UTF-8 text content as a blob and return its hash."""
    try:
        blob_hash = HashRef(hash_string(content))
        with open_content_for_writing(objects_dir, blob_hash) as handle:
            handle.write(content.encode('utf-8'))
    except Exception as e:
        msg = 'Error saving merged blob content'
        raise MergeError(msg) from e

    return blob_hash


def merge_blob_text(
    objects_dir: str | Path,
    base_hash: str | None,
    ours_hash: str | None,
    theirs_hash: str | None) -> tuple[HashRef, bool]:
    """Merge three versions of a blob using merge3."""
    base_lines = read_blob_lines(objects_dir, base_hash) if base_hash else []
    ours_lines = read_blob_lines(objects_dir, ours_hash) if ours_hash else []
    theirs_lines = read_blob_lines(objects_dir, theirs_hash) if theirs_hash else []

    merger = Merge3(base_lines, ours_lines, theirs_lines)

    # temporary file to avoid keeping the entire result in memory
    conflict = False
    tmp_fd, tmp_path = tempfile.mkstemp()

    try:
        with open(tmp_fd, 'wb') as tmp_file:
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

        blob = save_file_content(objects_dir, tmp_path)
        return HashRef(blob.hash), conflict
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def merge_blob_binary(
    objects_dir: str | Path,
    base_hash: str | None,
    ours_hash: str | None,
    theirs_hash: str | None) -> tuple[HashRef, bool]:
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


def merge_blob(
    objects_dir: str | Path,
    base_hash: str | None,
    ours_hash: str | None,
    theirs_hash: str | None) -> tuple[HashRef, bool]:
    """Merge two blob versions using their common ancestor."""
    if is_binary_blob(objects_dir, ours_hash) or is_binary_blob(objects_dir, theirs_hash):
        return merge_blob_binary(objects_dir, base_hash, ours_hash, theirs_hash)

    try:
        return merge_blob_text(objects_dir, base_hash, ours_hash, theirs_hash)
    except MergeError as e:
        if 'not valid UTF-8' in str(e):
            return merge_blob_binary(objects_dir, base_hash, ours_hash, theirs_hash)
        raise


def merge_trees_core(
    objects_dir: str | Path,
    base_tree: Tree | None,
    ours_tree: Tree | None,
    theirs_tree: Tree | None,
    path_prefix: str,
    conflicts: list[str]) -> HashRef:
    """Recursively merge three trees using 3-way merge logic."""
    base_records = base_tree.records if base_tree else {}
    ours_records = ours_tree.records if ours_tree else {}
    theirs_records = theirs_tree.records if theirs_tree else {}

    all_names = sorted(set(base_records) | set(ours_records) | set(theirs_records))
    merged_records: dict[str, TreeRecord] = {}

    for name in all_names:
        base_record = base_records.get(name)
        ours_record = ours_records.get(name)
        theirs_record = theirs_records.get(name)
        path = os.path.join(path_prefix, name) if path_prefix else name

        # no conflict
        if ours_record and theirs_record and ours_record.type == theirs_record.type and ours_record.hash == theirs_record.hash:
            merged_records[name] = ours_record
            continue

        # take theirs
        if base_record and ours_record and base_record.type == ours_record.type and base_record.hash == ours_record.hash:
            if theirs_record is not None:
                merged_records[name] = theirs_record
            continue

        # take ours
        if base_record and theirs_record and base_record.type == theirs_record.type and base_record.hash == theirs_record.hash:
            if ours_record is not None:
                merged_records[name] = ours_record
            continue

        # only on one side
        if base_record is None and ours_record is not None and theirs_record is None:
            merged_records[name] = ours_record
            continue

        if base_record is None and ours_record is None and theirs_record is not None:
            merged_records[name] = theirs_record
            continue

        # Both trees
        if (ours_record and theirs_record
                and ours_record.type == TreeRecordType.TREE
                and theirs_record.type == TreeRecordType.TREE):
            base_subtree = (
                load_tree(objects_dir, base_record.hash)
                if base_record and base_record.type == TreeRecordType.TREE
                else None
            )
            merged_hash = merge_trees_core(
                objects_dir,
                base_subtree,
                load_tree(objects_dir, ours_record.hash),
                load_tree(objects_dir, theirs_record.hash),
                path,
                conflicts,
            )
            merged_records[name] = TreeRecord(TreeRecordType.TREE, merged_hash, name)
            continue

        # 3 way blob merge
        if (ours_record and theirs_record
                and ours_record.type == TreeRecordType.BLOB
                and theirs_record.type == TreeRecordType.BLOB):
            base_hash = base_record.hash if base_record and base_record.type == TreeRecordType.BLOB else None
            merged_hash, conflict = merge_blob(objects_dir, base_hash, ours_record.hash, theirs_record.hash)
            if conflict:
                conflicts.append(path)
            merged_records[name] = TreeRecord(TreeRecordType.BLOB, merged_hash, name)
            continue

        chosen = ours_record or theirs_record
        if chosen is not None:
            merged_records[name] = chosen
        conflicts.append(path)

    merged_tree = Tree(merged_records)
    save_tree(objects_dir, merged_tree)
    return HashRef(hash_object(merged_tree))


def find_common_ancestor_core(objects_dir: str, hash1: str, hash2: str) -> HashRef | None:
    """Helper function to run the ancestor search algorithm independent of the Repository class."""
    try:
        ancestors: set[HashRef] = set()
        current_hash = hash1
        while current_hash:
            ancestors.add(HashRef(current_hash))
            commit = load_commit(objects_dir, current_hash)
            parent = commit.parent
            current_hash = HashRef(parent) if parent else parent

        current_hash2 = hash2
        while current_hash2:
            if current_hash2 in ancestors:
                return HashRef(current_hash2)
            commit = load_commit(objects_dir, current_hash2)
            parent = commit.parent
            current_hash2 = HashRef(parent) if parent else parent

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
