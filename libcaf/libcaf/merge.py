"""Merge helpers for libcaf."""

from dataclasses import dataclass
from pathlib import Path

from . import Tree, TreeRecord, TreeRecordType
from .plumbing import (hash_object, hash_string, load_commit, load_tree,
                       open_content_for_reading, open_content_for_writing, save_tree)
from .ref import HashRef


class MergeError(Exception):
    """Exception raised for merge-related errors."""


@dataclass
class MergeResult:
    """Represents the output of a 3-way merge."""

    tree_hash: HashRef
    conflicts: list[str]


def read_blob_text(objects_dir: str | Path, blob_hash: str) -> str:
    """Load blob content as UTF-8 text."""
    try:
        with open_content_for_reading(objects_dir, blob_hash) as handle:
            content = handle.read()
        return content.decode('utf-8')
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
    theirs_hash: str | None,
) -> tuple[HashRef, bool]:
    """Merge three versions of a blob using merge3."""
    try:
        from merge3 import Merge3
    except Exception as e:
        msg = 'merge3 library is required for 3-way merge'
        raise MergeError(msg) from e

    base_text = read_blob_text(objects_dir, base_hash) if base_hash else ''
    ours_text = read_blob_text(objects_dir, ours_hash) if ours_hash else ''
    theirs_text = read_blob_text(objects_dir, theirs_hash) if theirs_hash else ''

    base_lines = base_text.splitlines(keepends=True)
    ours_lines = ours_text.splitlines(keepends=True)
    theirs_lines = theirs_text.splitlines(keepends=True)

    merger = Merge3(base_lines, ours_lines, theirs_lines)
    merged_lines = list(merger.merge_lines())
    merged_text = ''.join(merged_lines)

    conflict = False
    try:
        for group in merger.merge_groups():
            if group and group[0] == 'conflict':
                conflict = True
                break
    except Exception:
        conflict_markers = ('<<<<<<<', '=======', '>>>>>>>')
        conflict = any(marker in line for line in merged_lines for marker in conflict_markers)

    return save_blob_text(objects_dir, merged_text), conflict


def records_match(record1: TreeRecord | None, record2: TreeRecord | None) -> bool:
    if record1 is None or record2 is None:
        return False
    return record1.type == record2.type and record1.hash == record2.hash


def merge_trees_core(
    objects_dir: str | Path,
    base_tree: Tree | None,
    ours_tree: Tree | None,
    theirs_tree: Tree | None,
    path_prefix: str,
    conflicts: list[str],
) -> HashRef:
    """Merge trees recursively and return the merged tree hash."""
    merged_records: dict[str, TreeRecord] = {}
    base_records = base_tree.records if base_tree else {}
    ours_records = ours_tree.records if ours_tree else {}
    theirs_records = theirs_tree.records if theirs_tree else {}

    for name in sorted(set(base_records) | set(ours_records) | set(theirs_records)):
        base_record = base_records.get(name)
        ours_record = ours_records.get(name)
        theirs_record = theirs_records.get(name)
        path = f'{path_prefix}/{name}' if path_prefix else name

        if records_match(ours_record, theirs_record):
            merged_records[name] = TreeRecord(ours_record.type, ours_record.hash, name)
            continue

        if records_match(base_record, ours_record):
            if theirs_record is not None:
                merged_records[name] = TreeRecord(theirs_record.type, theirs_record.hash, name)
            continue

        if records_match(base_record, theirs_record):
            if ours_record is not None:
                merged_records[name] = TreeRecord(ours_record.type, ours_record.hash, name)
            continue

        if base_record is None and (ours_record is None) != (theirs_record is None):
            chosen = ours_record or theirs_record
            merged_records[name] = TreeRecord(chosen.type, chosen.hash, name)
            continue

        if (
            ours_record
            and theirs_record
            and ours_record.type == TreeRecordType.TREE
            and theirs_record.type == TreeRecordType.TREE
        ):
            base_subtree = (
                load_tree(objects_dir, base_record.hash)
                if base_record and base_record.type == TreeRecordType.TREE
                else None
            )
            ours_subtree = load_tree(objects_dir, ours_record.hash)
            theirs_subtree = load_tree(objects_dir, theirs_record.hash)
            subtree_hash = merge_trees_core(
                objects_dir,
                base_subtree,
                ours_subtree,
                theirs_subtree,
                path,
                conflicts,
            )
            merged_records[name] = TreeRecord(TreeRecordType.TREE, subtree_hash, name)
            continue

        if (
            ours_record
            and theirs_record
            and ours_record.type == TreeRecordType.BLOB
            and theirs_record.type == TreeRecordType.BLOB
        ):
            base_hash = base_record.hash if base_record and base_record.type == TreeRecordType.BLOB else None
            merged_hash, conflict = merge_blob_text(objects_dir, base_hash, ours_record.hash, theirs_record.hash)
            if conflict:
                conflicts.append(path)
            merged_records[name] = TreeRecord(TreeRecordType.BLOB, merged_hash, name)
            continue

        chosen = ours_record or theirs_record
        if chosen is not None:
            merged_records[name] = TreeRecord(chosen.type, chosen.hash, name)
        conflicts.append(path)

    merged_tree = Tree(merged_records)
    save_tree(objects_dir, merged_tree)
    return hash_object(merged_tree)


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
