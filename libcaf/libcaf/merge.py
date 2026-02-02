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


def read_blob_text(objects_dir: str | Path, blob_hash: str) -> str:
    """Load blob content as UTF-8 text."""
    try:
        with open_content_for_reading(objects_dir, blob_hash) as handle:
            content = handle.read()
    except Exception as e:
        msg = f'Error reading blob {blob_hash}'
        raise MergeError(msg) from e

    try:
        return content.decode('utf-8')
    except UnicodeDecodeError as e:
        msg = f'Blob {blob_hash} is not valid UTF-8 text'
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
    base_text = read_blob_text(objects_dir, base_hash) if base_hash else ''
    ours_text = read_blob_text(objects_dir, ours_hash) if ours_hash else ''
    theirs_text = read_blob_text(objects_dir, theirs_hash) if theirs_hash else ''

    base_lines = base_text.splitlines(keepends=True)
    ours_lines = ours_text.splitlines(keepends=True)
    theirs_lines = theirs_text.splitlines(keepends=True)

    merger = Merge3(base_lines, ours_lines, theirs_lines)

    # Write merged content to a temporary file to avoid keeping the entire result in memory
    conflict = False
    tmp_fd, tmp_path = tempfile.mkstemp(text=True)

    try:
        with open(tmp_fd, 'w', encoding='utf-8') as tmp_file:
            for group in merger.merge_groups():
                if group[0] == 'unchanged':
                    tmp_file.writelines(group[1])
                elif group[0] == 'a':
                    tmp_file.writelines(group[1])
                elif group[0] == 'b':
                    tmp_file.writelines(group[1])
                elif group[0] == 'conflict':
                    conflict = True
                    # Add conflict markers
                    tmp_file.write('<<<<<<< ours\n')
                    tmp_file.writelines(group[2])  # a_lines (ours)
                    tmp_file.write('=======\n')
                    tmp_file.writelines(group[3])  # b_lines (theirs)
                    tmp_file.write('>>>>>>> theirs\n')

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

    # Try text merge first, but fall back to binary merge if UTF-8 decoding fails
    try:
        return merge_blob_text(objects_dir, base_hash, ours_hash, theirs_hash)
    except MergeError as e:
        # If the error is due to invalid UTF-8, treat as binary
        if 'not valid UTF-8' in str(e):
            return merge_blob_binary(objects_dir, base_hash, ours_hash, theirs_hash)
        # Re-raise other types of merge errors
        raise


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
    conflicts: list[str]) -> HashRef:
    """Merge trees iteratively using an explicit stack to avoid unbounded recursion."""
    from collections import deque
    from dataclasses import dataclass as dc, field

    @dc
    class MergeTask:
        """Represents a tree merge operation."""
        base_tree: Tree | None
        ours_tree: Tree | None
        theirs_tree: Tree | None
        path_prefix: str
        merged_records: dict[str, TreeRecord] = field(default_factory=dict)
        names_to_process: list[str] = field(default_factory=list)
        current_index: int = 0
        parent_records: dict[str, TreeRecord] | None = None
        record_name: str | None = None

    # Cache for completed tree merges: (base_hash, ours_hash, theirs_hash) -> result_hash
    completed: dict[tuple[str | None, str | None, str | None], HashRef] = {}

    def get_tree_key(base_t: Tree | None, ours_t: Tree | None, theirs_t: Tree | None) -> tuple[str | None, str | None, str | None]:
        base_h = hash_object(base_t) if base_t else None
        ours_h = hash_object(ours_t) if ours_t else None
        theirs_h = hash_object(theirs_t) if theirs_t else None
        return (base_h, ours_h, theirs_h)

    # Stack for iterative processing
    stack: deque[MergeTask] = deque()
    root_task = MergeTask(base_tree, ours_tree, theirs_tree, path_prefix)
    stack.append(root_task)

    while stack:
        task = stack[-1]  # Peek at top

        # Check if this tree was already completed
        tree_key = get_tree_key(task.base_tree, task.ours_tree, task.theirs_tree)
        if tree_key in completed and task.current_index == 0:
            stack.pop()
            if task.parent_records is not None and task.record_name is not None:
                task.parent_records[task.record_name] = TreeRecord(
                    TreeRecordType.TREE, completed[tree_key], task.record_name
                )
            continue

        # Initialize names to process on first visit
        if task.current_index == 0 and not task.names_to_process:
            base_records = task.base_tree.records if task.base_tree else {}
            ours_records = task.ours_tree.records if task.ours_tree else {}
            theirs_records = task.theirs_tree.records if task.theirs_tree else {}
            task.names_to_process = sorted(set(base_records) | set(ours_records) | set(theirs_records))

        # Process records one by one
        if task.current_index < len(task.names_to_process):
            name = task.names_to_process[task.current_index]
            task.current_index += 1

            base_records = task.base_tree.records if task.base_tree else {}
            ours_records = task.ours_tree.records if task.ours_tree else {}
            theirs_records = task.theirs_tree.records if task.theirs_tree else {}

            base_record = base_records.get(name)
            ours_record = ours_records.get(name)
            theirs_record = theirs_records.get(name)
            path = os.path.join(task.path_prefix, name) if task.path_prefix else name

            # Handle simple cases
            if records_match(ours_record, theirs_record):
                task.merged_records[name] = ours_record
                continue

            if records_match(base_record, ours_record):
                if theirs_record is not None:
                    task.merged_records[name] = theirs_record
                continue

            if records_match(base_record, theirs_record):
                if ours_record is not None:
                    task.merged_records[name] = ours_record
                continue

            # File added in only one branch
            if base_record is None and ours_record is not None and theirs_record is None:
                task.merged_records[name] = ours_record
                continue

            if base_record is None and ours_record is None and theirs_record is not None:
                task.merged_records[name] = theirs_record
                continue

            # Both are trees - need to merge subtrees
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

                # Check if subtree already merged
                subtree_key = get_tree_key(base_subtree, ours_subtree, theirs_subtree)
                if subtree_key in completed:
                    task.merged_records[name] = TreeRecord(TreeRecordType.TREE, completed[subtree_key], name)
                else:
                    # Push subtree merge onto stack
                    subtask = MergeTask(
                        base_subtree, ours_subtree, theirs_subtree, path,
                        parent_records=task.merged_records, record_name=name
                    )
                    stack.append(subtask)
                continue

            # Both are blobs - merge them
            if (
                ours_record
                and theirs_record
                and ours_record.type == TreeRecordType.BLOB
                and theirs_record.type == TreeRecordType.BLOB
            ):
                base_hash = base_record.hash if base_record and base_record.type == TreeRecordType.BLOB else None
                merged_hash, conflict = merge_blob(objects_dir, base_hash, ours_record.hash, theirs_record.hash)
                if conflict:
                    conflicts.append(path)
                task.merged_records[name] = TreeRecord(TreeRecordType.BLOB, merged_hash, name)
                continue

            # Conflict case
            chosen = ours_record or theirs_record
            if chosen is not None:
                task.merged_records[name] = chosen
            conflicts.append(path)

        else:
            # All records processed - finalize this tree
            stack.pop()
            merged_tree = Tree(task.merged_records)
            save_tree(objects_dir, merged_tree)
            tree_hash = hash_object(merged_tree)
            completed[tree_key] = tree_hash

            # Update parent if this is a subtree
            if task.parent_records is not None and task.record_name is not None:
                task.parent_records[task.record_name] = TreeRecord(
                    TreeRecordType.TREE, tree_hash, task.record_name
                )

    # Return the root tree hash
    root_key = get_tree_key(base_tree, ours_tree, theirs_tree)
    return completed[root_key]


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
